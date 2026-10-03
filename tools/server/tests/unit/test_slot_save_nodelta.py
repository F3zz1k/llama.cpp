import glob
import os
import shutil
import struct
import time

import pytest
from utils import *

# F1 regression: on a memory type that cannot write a position-range delta (the probe answers
# NO), a parented auto-save must be published as a WHOLE root, never dropped.
#
# The models are the dummy GGUFs test-llama-archs writes for the generate-models ctest fixture
# (build/tests/test-models). LLAMA_TEST_MODELS_DIR names that directory; when it is unset the test
# looks next to the server binary under test (<build>/bin/llama-server -> <build>/tests/test-models),
# so a run against a build whose ctest has generated the models exercises F1 by default. It skips,
# naming the path it tried, only when neither holds the model. Prompts are token ids because the
# dummy vocab has 128 tokens and no meaningful text tokenizer.
#   mamba-dense, mamba2-dense: pure recurrent memory, where the whole state IS the only correct
#   form of every save (a fixed-size fold of the prefix), so the probe always answers NO
#
# The second test is the converse for the memory types that gained state_write_range: the probe
# must answer YES, parented saves must be v3 deltas, and a fresh instance must compose the chain.

def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
CACHE_DIR = "./tmp/slot_save_nodelta"
IDLE_SECONDS = 2

SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE

BASE = [((i * 7) % 100) + 10 for i in range(64)]
EXT = BASE + [((i * 11) % 100) + 10 for i in range(48)]
EXT2 = EXT + [((i * 13) % 100) + 10 for i in range(32)]
EXT3 = EXT2 + [((i * 17) % 100) + 10 for i in range(16)]


def _meta_version(path: str) -> int:
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
    assert magic == SLOT_META_MAGIC
    return version


def _metas():
    return sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


def _wait_for_metas(n: int, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = _metas()
        if len(m) >= n:
            return m
        time.sleep(0.25)
    return _metas()


def _make_server(model: str, log_path: str) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = True
    s.slot_save_block = 16
    s.slot_save_min_tokens = 0
    s.slot_save_context_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_idle_seconds = IDLE_SECONDS
    s.log_path = log_path
    return s


def _complete(s, prompt):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


@pytest.fixture(autouse=True)
def clean_cache_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.parametrize("model_name", ["mamba-dense", "mamba2-dense"])
def test_nodelta_parented_save_is_whole_root(model_name, tmp_path):
    model = os.path.join(MODELS_DIR, f"{model_name}.gguf")
    if not os.path.isfile(model):
        pytest.skip(f"{model} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")

    log1 = str(tmp_path / "server1.log")
    s = _make_server(model, log1)
    s.start()
    _complete(s, BASE)
    assert len(_wait_for_metas(1, IDLE_SECONDS + 12)) == 1, "the base prompt must be flushed"
    _complete(s, EXT)
    assert len(_wait_for_metas(2, IDLE_SECONDS + 12)) == 2, "the first parented save (probe path) must be published"
    # a third, deeper save takes the CACHED path: the probe already said NO, so this save never
    # probes again and must still fall back to a whole root (server-context.cpp, the
    # `is_node && delta_capable == delta_cap::no` branch)
    _complete(s, EXT2)
    metas = _wait_for_metas(3, IDLE_SECONDS + 12)
    fallback = _metric(s, "auto_cache_save_whole_fallback_total")
    roots = _metric(s, "auto_cache_save_root_total")
    deltas = _metric(s, "auto_cache_save_delta_total")
    failed = _metric(s, "auto_cache_save_failed_total")
    s.stop()

    # before F1 every parented save was dropped, leaving only the 64-token root
    assert len(metas) == 3, f"every parented save must be published, got {metas}"
    assert all(_meta_version(m) == 1 for m in metas), "a no-delta class must publish whole v1 roots only"
    assert fallback == 2, "every whole-root fallback is counted, probe path and cached path"
    assert roots == 3 and deltas == 0, f"three whole roots and no deltas, got roots={roots} deltas={deltas}"
    assert failed == 0, "no save was dropped"
    with open(log1) as f:
        log = f.read()
    assert "delta capability probed = NO" in log
    assert log.count("cannot write deltas (probe said NO)") == 1, "the fallback WRN is logged once per instance"

    # a fresh instance restores the DEEPER whole root, not just the first one
    log2 = str(tmp_path / "server2.log")
    s2 = _make_server(model, log2)
    s2.start()
    res = s2.make_request("POST", "/completion", data={
        "prompt": EXT3,
        "n_predict": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200
    hits = _metric(s2, "auto_cache_restore_hit_total")
    restored = _metric(s2, "auto_cache_restore_tokens_total")
    s2.stop()
    with open(log2) as f:
        log = f.read()
    assert f"auto-restore: reused {len(EXT2)} tokens from disk" in log, \
        "the restore must find the deepest whole root, published on the cached fallback path"
    assert hits == 1 and restored == len(EXT2), f"hit counters: hits={hits} tokens={restored}"
    assert res.body["timings"].get("cache_disk_n") == len(EXT2), \
        f"timings.cache_disk_n must report the disk-restored prefix, got {res.body['timings']}"


# llama_kv_cache_dsa (glm-dsa, deepseek32), llama_kv_cache_msa (minimax-m3), llama_kv_cache_dsv4
# (deepseek4) and llama_memory_hybrid_idx with an indexer (glm5-next, qwen4exp) used to write whole
# only. deepseek4 also exercises the server's DSV4-prefix skip in the delta cell-count check.
@pytest.mark.parametrize("model_name", [
    "glm-dsa-moe", "deepseek32-moe", "minimax-m3-moe", "deepseek4-moe", "glm5-next-moe", "qwen4exp-moe",
])
def test_delta_capable_parented_saves_are_deltas(model_name, tmp_path):
    model = os.path.join(MODELS_DIR, f"{model_name}.gguf")
    if not os.path.isfile(model):
        pytest.skip(f"{model} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")

    log1 = str(tmp_path / "server1.log")
    s = _make_server(model, log1)
    s.start()
    _complete(s, BASE)
    assert len(_wait_for_metas(1, IDLE_SECONDS + 12)) == 1, "the base prompt must be flushed"
    _complete(s, EXT)
    assert len(_wait_for_metas(2, IDLE_SECONDS + 12)) == 2
    _complete(s, EXT2)
    metas = _wait_for_metas(3, IDLE_SECONDS + 12)
    roots = _metric(s, "auto_cache_save_root_total")
    deltas = _metric(s, "auto_cache_save_delta_total")
    fallback = _metric(s, "auto_cache_save_whole_fallback_total")
    s.stop()

    with open(log1) as f:
        log = f.read()
    assert "delta capability probed = YES" in log, "this memory type must honour position ranges"
    assert "cell-count check failed" not in log, "the delta cell-count check must accept these deltas"
    assert len(metas) == 3, f"three units expected, got {metas}"
    versions = sorted(_meta_version(m) for m in metas)
    assert versions == [1, 3, 3], f"one v1 root and two v3 deltas expected, got {versions}"
    assert roots == 1 and deltas == 2 and fallback == 0, f"roots={roots} deltas={deltas} fallback={fallback}"

    # a fresh instance composes root + delta + delta (NO_CLEAR) and resumes from the deepest node
    log2 = str(tmp_path / "server2.log")
    s2 = _make_server(model, log2)
    s2.start()
    res = s2.make_request("POST", "/completion", data={
        "prompt": EXT3,
        "n_predict": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200
    hits = _metric(s2, "auto_cache_restore_hit_total")
    failed = _metric(s2, "auto_cache_restore_failed_total")
    s2.stop()
    with open(log2) as f:
        log = f.read()
    assert f"auto-restore: reused {len(EXT2)} tokens from disk" in log, \
        "the restore must compose the delta chain up to the deepest node"
    assert hits == 1 and failed == 0, f"hits={hits} failed={failed}"

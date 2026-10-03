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
# (build/tests/test-models). Point LLAMA_TEST_MODELS_DIR at that directory; the test skips when
# it is unset or the model is missing. Prompts are token ids because the dummy vocab has 128
# tokens and no meaningful text tokenizer.
#   mamba-dense:  pure recurrent memory, no state_write_range override (base-class whole write)
#   qwen4exp-moe: llama_memory_hybrid_idx with an indexer, which writes whole on purpose

MODELS_DIR = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
CACHE_DIR = "./tmp/slot_save_nodelta"
IDLE_SECONDS = 2

SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE

BASE = [((i * 7) % 100) + 10 for i in range(64)]
EXT = BASE + [((i * 11) % 100) + 10 for i in range(48)]
EXT2 = EXT + [((i * 13) % 100) + 10 for i in range(16)]


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


@pytest.mark.parametrize("model_name", ["mamba-dense", "qwen4exp-moe"])
def test_nodelta_parented_save_is_whole_root(model_name, tmp_path):
    model = os.path.join(MODELS_DIR, f"{model_name}.gguf")
    if not MODELS_DIR or not os.path.isfile(model):
        pytest.skip(f"LLAMA_TEST_MODELS_DIR does not hold {model_name}.gguf")

    log1 = str(tmp_path / "server1.log")
    s = _make_server(model, log1)
    s.start()
    _complete(s, BASE)
    assert len(_wait_for_metas(1, IDLE_SECONDS + 12)) == 1, "the base prompt must be flushed"
    _complete(s, EXT)
    metas = _wait_for_metas(2, IDLE_SECONDS + 12)
    fallback = _metric(s, "auto_cache_save_whole_fallback_total")
    s.stop()

    # before F1 the parented save was dropped, leaving only the 64-token root
    assert len(metas) == 2, f"the parented save must be published, got {metas}"
    assert all(_meta_version(m) == 1 for m in metas), "a no-delta class must publish whole v1 roots only"
    assert fallback == 1, "every whole-root fallback is counted"
    with open(log1) as f:
        log = f.read()
    assert "delta capability probed = NO" in log
    assert log.count("cannot write deltas (probe said NO)") == 1, "the fallback WRN is logged once per instance"

    # a fresh instance restores the DEEPER whole root, not just the first one
    log2 = str(tmp_path / "server2.log")
    s2 = _make_server(model, log2)
    s2.start()
    _complete(s2, EXT2)
    s2.stop()
    with open(log2) as f:
        log = f.read()
    assert f"auto-restore: reused {len(EXT)} tokens from disk" in log, \
        "the restore must find the deeper whole root published by the fallback"

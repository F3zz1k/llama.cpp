import glob
import os
import shutil
import time

import pytest
from utils import *

# Draft (.dft) sidecars: a disk restore must bring the speculative draft context back warm, or the
# draft attends over a hole [0, L) for the rest of that conversation and acceptance drops silently.
#
# The draft here is the target model itself (draft-simple), so a warm draft agrees with the target on
# every greedy token and accepts everything, while a cold one drafts from the restored suffix alone.
# The same request is replayed against a copy of the cache with the .dft sidecars deleted, as the
# control. Uses the generate-models dummy GGUFs, found as in test_slot_save_nodelta.py.
#
# What this can and cannot show: the warm/cold counters and the identical output are the
# discriminating checks. The acceptance comparison is weak on these dummies: their random weights
# make greedy decoding nearly context-insensitive, so a cold draft also accepted 24/24 when this was
# written. Measure the acceptance gain on a real MTP model before quoting one.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
CACHE_DIR = "./tmp/slot_save_draft"
CACHE_DIR_COLD = "./tmp/slot_save_draft_cold"
IDLE_SECONDS = 2

BASE = [((i * 7) % 100) + 10 for i in range(64)]
NEXT = BASE + [((i * 11) % 100) + 10 for i in range(16)]


def _make_server(model: str, cache_dir: str, log_path: str) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_draft = model
    s.spec_type = "draft-simple"
    s.spec_draft_n_max = 4
    s.model_alias = "dummy"
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.slot_save_path = cache_dir
    s.slot_save_auto = True
    s.slot_save_incremental = True
    s.slot_save_block = 16
    s.slot_save_min_tokens = 0
    s.slot_save_context_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_idle_seconds = IDLE_SECONDS
    s.log_path = log_path
    return s


def _complete(s, prompt, n_predict):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "id_slot": 0,
        "temperature": 0.0,
        "top_k": 1,
    })
    assert res.status_code == 200
    return res.body


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _wait_for(pattern: str, n: int, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = sorted(glob.glob(pattern))
        if len(m) >= n:
            return m
        time.sleep(0.25)
    return sorted(glob.glob(pattern))


@pytest.fixture(autouse=True)
def clean_cache_dirs():
    for d in (CACHE_DIR, CACHE_DIR_COLD):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    for d in (CACHE_DIR, CACHE_DIR_COLD):
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("model_name", ["llama-dense", "qwen3-dense"])
def test_draft_sidecar_restores_warm_draft(model_name, tmp_path):
    model = os.path.join(MODELS_DIR, f"{model_name}.gguf")
    if not os.path.isfile(model):
        pytest.skip(f"{model} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")

    # 1) a conversation is saved with its draft sidecar
    s = _make_server(model, CACHE_DIR, str(tmp_path / "server1.log"))
    s.start()
    _complete(s, BASE, 8)
    # two units: the prompt node written mid-prefill (--slot-save-node-prompt, cold by default) and
    # the conversation flushed on idle; both must carry a draft sidecar
    metas = _wait_for(os.path.join(CACHE_DIR, "auto-*.bin.meta"), 2, IDLE_SECONDS + 12)
    drafts_saved = _metric(s, "auto_cache_save_draft_total")
    s.stop()
    assert len(metas) == 2, f"the prompt node and the conversation must be on disk: {metas}"
    dfts = sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.dft")))
    assert len(dfts) == len(metas) and drafts_saved == len(metas), \
        f"every unit carries a draft sidecar: metas={metas} dfts={dfts} counter={drafts_saved}"

    # the control: the same store without any draft sidecar
    shutil.copytree(CACHE_DIR, CACHE_DIR_COLD)
    for p in glob.glob(os.path.join(CACHE_DIR_COLD, "*.dft")):
        os.remove(p)

    def resume(cache_dir: str, log_name: str):
        s2 = _make_server(model, cache_dir, str(tmp_path / log_name))
        s2.start()
        body = _complete(s2, NEXT, 32)
        warm = _metric(s2, "auto_cache_restore_draft_warm_total")
        cold = _metric(s2, "auto_cache_restore_draft_cold_total")
        hits = _metric(s2, "auto_cache_restore_hit_total")
        s2.stop()
        return body, warm, cold, hits

    body_w, warm_w, cold_w, hits_w = resume(CACHE_DIR, "server_warm.log")
    body_c, warm_c, cold_c, hits_c = resume(CACHE_DIR_COLD, "server_cold.log")

    assert hits_w == 1 and hits_c == 1, f"both resumes restore from disk (warm {hits_w}, cold {hits_c})"
    assert (warm_w, cold_w) == (1, 0), f"sidecars present: the draft must come back warm, got warm={warm_w} cold={cold_w}"
    assert (warm_c, cold_c) == (0, 1), f"sidecars deleted: the draft must be cold, got warm={warm_c} cold={cold_c}"

    # draft state can change speed, never the output
    assert body_w["content"] == body_c["content"], "warm and cold drafts must produce the same text"

    tw, tc = body_w["timings"], body_c["timings"]
    n_w, a_w = tw.get("draft_n", 0), tw.get("draft_n_accepted", 0)
    n_c, a_c = tc.get("draft_n", 0), tc.get("draft_n_accepted", 0)
    print(f"{model_name}: warm draft accepted {a_w}/{n_w}, cold draft accepted {a_c}/{n_c}")
    assert n_w > 0, "the warm resume must have drafted"
    # the draft IS the target, so with its full context every greedy draft token is accepted
    assert a_w == n_w, f"a warm self-draft accepts every token, got {a_w}/{n_w}"
    assert n_c == 0 or a_w / n_w >= a_c / n_c, "a warm draft must never accept less than a cold one"

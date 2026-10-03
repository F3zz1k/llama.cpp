import glob
import os
import shutil

import pytest
from utils import *

# Resend, extend and reclaim after a restart, per memory class (user design 2026-10-03):
#  - truncatable classes (plain attention, and SWA whose snapshot fits one window) must serve a
#    request that is a PREFIX of a saved unit by restoring it and trimming to the request;
#  - non-rewindable classes (RS / hybrid, SWA past one window, FULL) serve it only from a node at
#    or before the divergence; with the default triggers that miss is expected, and it must be
#    reported (WRN + auto_cache_restore_not_prefix_total), never silent;
#  - every class must serve a request that EXTENDS the saved unit (previous response included);
#  - a conversation preempted by a different one on a --parallel 1 server reaches disk.
#
# Dummy models from test-llama-archs (build/tests/test-models), token-id prompts:
#   llama-dense   PART, plain attention
#   gemma3-dense  iSWA, n_swa = 32: every unit below is longer than one window
#   qwen35-dense  RS (hybrid gated delta net), the qwen3.8-27b class


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
CACHE_DIR = "./tmp/slot_save_resend"
B = 16
PROMPT = [((i * 7) % 100) + 10 for i in range(150)]
TAIL = [((i * 11) % 100) + 10 for i in range(40)]


def _model(name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.exists(path):
        pytest.skip(f"dummy model not found: {path}")
    return path


def _server(model: str, incr: bool = False) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    s.n_ctx = 256
    s.n_batch = 256
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = incr
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    # no mid-prefill node: these tests isolate the release / reclaim units. (The cold prompt-end
    # node that SWA arms today, and the after-user-message trigger, get their own tests.)
    s.slot_save_context_min_tokens = 100000
    s.slot_save_idle_seconds = 3600
    return s


def _complete(s: ServerProcess, prompt):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 8, "temperature": 0, "cache_prompt": True,
        "id_slot": 0, "return_tokens": True,
    })
    assert res.status_code == 200
    t = res.body["timings"]
    return t["prompt_n"], t["cache_n"], res.body["tokens"]


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _units():
    return sorted(int(os.path.basename(p)[:-len(".bin.meta")].split("-")[-1])
                  for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


@pytest.fixture(autouse=True)
def clean_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.parametrize("incr", [False, True])
def test_resend_truncatable_restores_and_trims(incr):
    """Plain attention: the release unit (prompt + 7 generated cells) is LONGER than the resend.
    The restore loads it and trims to the request; only the logits token is re-decoded."""
    model = _model("llama-dense")
    s = _server(model, incr)
    s.start()
    _, _, gen = _complete(s, PROMPT)
    s.stop()
    assert _units() == [len(PROMPT) + len(gen) - 1]

    s = _server(model, incr)
    s.start()
    prompt_n, cache_n, _ = _complete(s, PROMPT)
    s.stop()
    assert prompt_n <= B + 1
    assert cache_n >= len(PROMPT) - B


@pytest.mark.parametrize("name", ["gemma3-dense", "qwen35-dense"])
def test_resend_non_rewindable_miss_is_reported(name):
    """Non-rewindable classes, default triggers: the same-prompt resend after a restart is a miss
    (no node at or before the divergence exists), and the miss is classified, not silent."""
    model = _model(name)
    s = _server(model)
    s.start()
    _complete(s, PROMPT)
    s.stop()

    s = _server(model)
    s.start()
    prompt_n, cache_n, _ = _complete(s, PROMPT)
    not_prefix = _metric(s, "auto_cache_restore_not_prefix_total")
    s.stop()
    assert cache_n == 0 and prompt_n == len(PROMPT)
    assert not_prefix == 1


@pytest.mark.parametrize("incr", [False, True])
@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "qwen35-dense"])
def test_extend_after_restart_hits_every_class(name, incr):
    """The supported shape for every class: the next request carries the previous response, so it
    extends the release unit. Only the new tail (plus at most one block) is prefilled."""
    model = _model(name)
    s = _server(model, incr)
    s.start()
    _, _, gen = _complete(s, PROMPT)
    s.stop()

    follow = PROMPT + gen + TAIL
    s = _server(model, incr)
    s.start()
    prompt_n, cache_n, _ = _complete(s, follow)
    s.stop()
    assert cache_n >= len(PROMPT) + len(gen) - 1 - B
    assert prompt_n <= len(TAIL) + B + 1


def test_reclaim_saves_preempted_conversation():
    """--parallel 1 with the default --cache-idle-slots: conversation A must reach disk when a
    different conversation B takes the only slot, before any idle flush could have fired."""
    model = _model("llama-dense")
    other = [((i * 13) % 100) + 10 for i in range(150)]
    s = _server(model)
    s.start()
    _, _, gen_a = _complete(s, PROMPT)
    assert _units() == []
    _complete(s, other)
    units_after_b = _units()
    s.stop()
    assert len(PROMPT) + len(gen_a) - 1 in units_after_b

    s = _server(model)
    s.start()
    prompt_n, _, _ = _complete(s, PROMPT)
    s.stop()
    assert prompt_n <= B + 1

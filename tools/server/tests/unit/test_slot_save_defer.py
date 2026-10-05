import glob
import os
import shutil
import signal
import time

import pytest
from utils import *

# Stage 2 of the background writer (docs/disk-cache.md, "Deferred positional copy"): a save copies only the
# side-state off the device at the capture and leaves the positional K/V of append-only caches in place,
# copied later on the server thread. Every test here runs the same scenario twice, with --slot-save-defer
# and with --no-slot-save-defer (stage 1, everything copied at the capture), and requires the two stores
# to hold the same files, byte for byte. The scenarios that matter hold the deferred copies pending on
# purpose (test hooks below) so that the cache changes the captured cells before they are copied: slot
# reuse, a context shift, sleep, a restore into the slot, shutdown. The positive control disables the
# engine's flush-on-mutate hook and must then publish a unit that differs.
# Test hooks read from the environment:
#   LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE=1       deferred copies are not emitted at idle (held until forced)
#   LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS=-1   ... nor trickled while busy
#   LLAMA_TEST_SLOT_SAVE_DEFER_NO_HOOK=1       the engine's flush-on-mutate hook does nothing (positive control)
# Prompts are token ids: the dummy vocab has no meaningful text tokenizer.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
MTP_MODEL = os.environ.get("LLAMA_TEST_MTP_MODEL", "")
ROOT = "./tmp/slot_save_defer"
B = 16
IDLE = 1
HOOKS = ("LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE", "LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS",
         "LLAMA_TEST_SLOT_SAVE_DEFER_NO_HOOK", "LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS")

# memory class of each dummy, and whether it has positional cells to defer
CLASSES = {
    "llama-dense":  ("llama",    True),   # kv
    "gemma3-dense": ("gemma3",   True),   # iswa: the global layers defer, the window is side-state
    "qwen35-dense": ("qwen35",   True),   # hybrid: attention defers, the recurrent fold is side-state
    "qwen4exp-moe": ("qwen4exp", True),   # hybrid_idx
    "mamba-dense":  ("mamba",    False),  # recurrent only: nothing positional, stage 1 throughout
}


def _toks(n: int, seed: int):
    return [((i * (7 + 2 * seed) + 13 * seed) % 100) + 10 for i in range(n)]


P = _toks(200, 1)
EXT = P + _toks(80, 2)
Q = _toks(200, 3)


def _model(name: str) -> str:
    if name == "mtp":
        if not MTP_MODEL or not os.path.isfile(MTP_MODEL):
            pytest.skip("LLAMA_TEST_MTP_MODEL not set (see test_slot_save_mtp.py)")
        return MTP_MODEL
    m = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.isfile(m):
        pytest.skip(f"{m} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")
    return m


def _server(name: str, cache: str, log_path: str, defer: bool, port_off: int = 0, n_slots: int = 1,
            idle=IDLE, node_prompt="on", n_ctx=None, ctx_shift=False, sleep_s=None) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = _model(name)
    s.model_alias = "dummy"
    if name == "mtp":
        s.spec_type = "draft-mtp"
        s.spec_draft_n_max = 3
        s.n_ctx = n_ctx or 512
    else:
        # the dummies declare a 256-token training context; the server caps n_ctx at it
        s.override_kv = [f"{CLASSES[name][0]}.context_length=int:4096"]
        s.n_ctx = n_ctx or 2048 * n_slots
    s.n_batch = 512
    s.n_slots = n_slots
    s.temperature = 0.0
    s.server_metrics = True
    s.slot_save_path = cache
    s.slot_save_auto = True
    s.slot_save_incremental = True
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_save_context_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_idle_seconds = idle
    s.slot_save_node_prompt = node_prompt
    s.slot_save_defer = defer
    s.enable_ctx_shift = ctx_shift
    if sleep_s is not None:
        s.sleep_idle_seconds = sleep_s
    s.server_port = s.server_port + port_off
    s.log_path = log_path
    return s


def _complete(s, prompt, slot=0, n_predict=8):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "id_slot": slot, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    return res.body


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _wait(pred, timeout_s: float, step: float = 0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def _terminate(s, timeout_s: float = 120) -> int:
    # graceful stop that waits for the shutdown flush (utils' stop() kills after 5 s)
    if s in server_instances:
        server_instances.remove(s)
    s.process.send_signal(signal.SIGTERM)
    rc = s.process.wait(timeout=timeout_s)
    s.process = None
    s._log.close()
    return rc


def _store(cache: str):
    out = {}
    for p in sorted(glob.glob(os.path.join(cache, "*"))):
        b = os.path.basename(p)
        assert ".tmp" not in b, f"a temp was left behind: {b}"
        with open(p, "rb") as f:
            out[b] = f.read()
    return out


def _compare(ref: dict, got: dict):
    """(identical, description)"""
    if set(ref) != set(got):
        return False, f"file sets differ: only stage 1 {sorted(set(ref) - set(got))}, only deferred {sorted(set(got) - set(ref))}"
    diff = [n for n in ref if ref[n] != got[n]]
    if diff:
        return False, f"{len(diff)} of {len(ref)} files differ: {diff}"
    return True, f"{len(ref)} files identical"


@pytest.fixture(autouse=True)
def clean_root(monkeypatch):
    for h in HOOKS:
        monkeypatch.delenv(h, raising=False)
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    yield
    shutil.rmtree(ROOT, ignore_errors=True)


def _queue_empty(s):
    return _metric(s, "auto_cache_save_queue_depth") == 0 and _metric(s, "auto_cache_save_deferred_pending") == 0


# ---- scenarios: each returns the server's counters just before the graceful stop ----------------------

def _sc_basic(name, cache, log, defer, hold):
    """Two turns (a prompt node and an idle flush each), then a graceful stop."""
    p, ext = (P[:96], P[:96] + EXT[200:240]) if name == "mtp" else (P, EXT)  # the MTP dummy keeps 256 tokens
    s = _server(name, cache, log, defer)
    s.start()
    _complete(s, p)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 15)
    if not hold:
        assert _wait(lambda: _queue_empty(s), 20)
    _complete(s, ext)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 4, IDLE + 15)
    if not hold:
        assert _wait(lambda: _queue_empty(s), 20)
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending", "auto_cache_save_queued_total")}
    assert _terminate(s) == 0
    return c


def _sc_reassign(name, cache, log, defer, hold):
    """The slot is reused by another conversation while the first one's copies are pending."""
    s = _server(name, cache, log, defer)
    s.start()
    _complete(s, P)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 15)
    pend = _metric(s, "auto_cache_save_deferred_pending")
    _complete(s, Q)  # takes the slot: its cells are removed and overwritten
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 4, IDLE + 15)
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending")}
    c["pending_before_reuse"] = pend
    assert _terminate(s) == 0
    return c


def _sc_ctxshift(name, cache, log, defer, hold):
    """A generation overflows the slot's context while the prompt node's copy is pending: the shift removes
    and moves captured cells."""
    s = _server(name, cache, log, defer, n_ctx=512, ctx_shift=True, idle=-1)
    s.start()
    long_p = _toks(400, 4)
    body = _complete(s, long_p, n_predict=200)
    assert body.get("truncated") is True, "the generation must have shifted the context"
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending")}
    assert _terminate(s) == 0
    return c


def _sc_sleep(name, cache, log, defer, hold):
    """The server goes to sleep (its contexts are freed) while copies are pending."""
    s = _server(name, cache, log, defer, sleep_s=2)
    s.start()
    _complete(s, P)
    assert _wait(lambda: "entering sleeping state" in open(log).read(), 30)
    # the copies forced at sleep entry are published before the next request looks the store up, so both
    # runs restore from the same units (which unit a restore picks is not what this test is about)
    assert _wait(lambda: open(log).read().count("auto-save: persisted") >= 2, 30)
    _complete(s, EXT)  # wakes it up; nothing in memory, so a restore
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending")}
    assert _terminate(s) == 0
    return c


def _sc_restore(name, cache, log, defer, hold):
    """A restore into the slot replaces its cells while that slot's copies are pending. The store is seeded
    first by an instance that persists P (identically in both runs)."""
    seed = _server(name, cache, log + ".seed", defer=False, idle=-1, node_prompt="off", port_off=1)
    seed.start()
    _complete(seed, P, n_predict=0)
    assert _terminate(seed) == 0  # the shutdown save persists P
    s = _server(name, cache, log, defer, idle=-1)
    s.start()
    _complete(s, Q)        # a prompt node of Q, pending
    body = _complete(s, EXT)  # restores P from disk into the same slot
    assert body["timings"].get("cache_disk_n", 0) >= len(P) - B, body["timings"]
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending")}
    assert _terminate(s) == 0
    return c


def _sc_shutdown(name, cache, log, defer, hold):
    """Copies still pending at a graceful stop are emitted and published by the shutdown flush."""
    s = _server(name, cache, log, defer, idle=-1)
    s.start()
    _complete(s, EXT, n_predict=4)
    c = {k: _metric(s, k) for k in ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total",
                                     "auto_cache_save_deferred_pending")}
    assert _terminate(s) == 0
    return c


SCENARIOS = {"basic": _sc_basic, "reassign": _sc_reassign, "ctxshift": _sc_ctxshift, "sleep": _sc_sleep,
             "restore": _sc_restore, "shutdown": _sc_shutdown}


def _run_pair(name, scenario, tmp_path, monkeypatch, hold, no_hook=False):
    """Runs `scenario` with stage 1 and with deferred copies; returns (stage-1 store, deferred store, counters)."""
    fn = SCENARIOS[scenario]
    ref_dir = os.path.join(ROOT, "stage1")
    def_dir = os.path.join(ROOT, "deferred")
    os.makedirs(ref_dir)
    os.makedirs(def_dir)
    fn(name, ref_dir, str(tmp_path / "stage1.log"), False, hold)
    if hold:
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE", "1")
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "-1")
    if no_hook:
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_NO_HOOK", "1")
    c = fn(name, def_dir, str(tmp_path / "deferred.log"), True, hold)
    return _store(ref_dir), _store(def_dir), c, str(tmp_path / "deferred.log")


@pytest.mark.parametrize("name", list(CLASSES) + ["mtp"])
def test_deferred_units_are_byte_identical(name, tmp_path, monkeypatch):
    """Two turns and a graceful stop: every published file (.bin, .logits, .dft, .meta) is byte-identical
    to the stage-1 run's. A class with positional cells defers its saves; one without (recurrent only)
    defers nothing and is stage 1 throughout."""
    ref, got, c, log = _run_pair(name, "basic", tmp_path, monkeypatch, hold=False)
    ok, why = _compare(ref, got)
    assert ok, why
    assert any(n.endswith(".bin") for n in ref), "the scenario published nothing"
    positional = CLASSES[name][1] if name in CLASSES else True
    if positional:
        assert c["auto_cache_save_deferred_total"] >= 1, c
    else:
        assert c["auto_cache_save_deferred_total"] == 0, c
    if name == "mtp":
        assert any(n.endswith(".dft") for n in ref), "the MTP run published no draft sidecar"
    text = open(log).read()
    if positional:
        assert "mode deferred, positional" in text


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
@pytest.mark.parametrize("scenario", ["reassign", "ctxshift", "sleep", "restore", "shutdown"])
def test_mutation_flushes_pending_copy_first(name, scenario, tmp_path, monkeypatch):
    """The deferred copies are held pending (no idle emission, no trickle) while the cache changes the
    captured cells: the flush-on-mutate hook (or the server's own force before sleep, shutdown and a
    restore) must copy them first, so the store equals the stage-1 run's."""
    if scenario == "ctxshift" and name == "qwen35-dense":
        pytest.skip("a recurrent state cannot be shifted (the context shift is refused for this class)")
    ref, got, c, log = _run_pair(name, scenario, tmp_path, monkeypatch, hold=True)
    ok, why = _compare(ref, got)
    assert ok, why
    assert c["auto_cache_save_deferred_total"] >= 1, c
    if scenario in ("reassign", "ctxshift"):
        # these reach the engine hook (the others are forced by the server before the context is touched)
        assert c["auto_cache_save_deferred_forced_total"] >= 1, c
    text = open(log).read()
    assert "mode deferred, positional" in text


def test_disabled_hook_corrupts_the_unit(tmp_path, monkeypatch):
    """Positive control: with the engine hook disabled, the reused slot overwrites the captured cells before
    the pending copy runs, and the published unit no longer equals the stage-1 one. Without this the
    equality above could not tell a working hook from a missing one."""
    ref, got, c, log = _run_pair("llama-dense", "reassign", tmp_path, monkeypatch, hold=True, no_hook=True)
    assert c["pending_before_reuse"] >= 1, c
    assert c["auto_cache_save_deferred_forced_total"] == 0, c
    ok, why = _compare(ref, got)
    assert not ok, f"with the hook disabled the store still matches stage 1 ({why}): the control is broken"
    assert set(ref) == set(got), "the control must corrupt bytes, not the file set"

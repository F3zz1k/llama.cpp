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
# reuse, a context shift (of the captured slot, or of ANOTHER slot under YaRN), sleep, a restore into the
# slot, shutdown, and a small staging budget. The positive controls disable the engine's flush-on-mutate
# hook and must then publish a unit that differs.
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
            idle=IDLE, node_prompt="on", n_ctx=None, ctx_shift=False, sleep_s=None, yarn=False,
            staging_mb=None) -> ServerProcess:
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
        if yarn:
            # a rope by 0 then scales keys (mscale): the K-shift graph changes every cell it ropes
            s.override_kv += [f"{CLASSES[name][0]}.rope.scaling.type=str:yarn",
                              f"{CLASSES[name][0]}.rope.scaling.factor=float:4",
                              f"{CLASSES[name][0]}.rope.scaling.original_context_length=int:256"]
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
    s.slot_save_staging_mb = staging_mb
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


def _prompts(name):
    """(P, EXT, Q) for a model: the MTP dummy keeps 256 tokens, so its prompts are shorter"""
    if name == "mtp":
        return P[:96], P[:96] + EXT[200:240], Q[:96]
    return P, EXT, Q


COUNTERS = ("auto_cache_save_deferred_total", "auto_cache_save_deferred_forced_total", "auto_cache_save_deferred_pending")


def _counters(s, extra=()):
    return {k: _metric(s, k) for k in COUNTERS + tuple(extra)}


def _queue_empty(s):
    return _metric(s, "auto_cache_save_queue_depth") == 0 and _metric(s, "auto_cache_save_deferred_pending") == 0


# ---- scenarios: each returns the server's counters just before the graceful stop ----------------------

def _sc_basic(name, cache, log, defer, hold):
    """Two turns (a prompt node and an idle flush each), then a graceful stop."""
    p, ext, _ = _prompts(name)
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
    """The slot is reused by another conversation while the first one's copies are pending. The reclaim save
    finds the queued delta covering the slot and waits for it, which needs the pending copies emitted first."""
    p, _, q = _prompts(name)
    s = _server(name, cache, log, defer)
    s.start()
    _complete(s, p)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 15)
    pend = _metric(s, "auto_cache_save_deferred_pending")
    t0 = time.time()
    _complete(s, q)  # takes the slot: its cells are removed and overwritten
    reuse_s = time.time() - t0
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 4, IDLE + 15)
    c = _counters(s)
    c["pending_before_reuse"] = pend
    c["reuse_s"] = reuse_s
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
    c = _counters(s)
    assert _terminate(s) == 0
    return c


def _sc_othershift(name, cache, log, defer, hold):
    """Two slots, YaRN: slot 0's prompt node is pending while slot 1 overflows its context and shifts. The
    K-shift graph ropes slot 0's cells too (by 0, which YaRN's mscale makes a change), so slot 0's copy must
    be taken first although nothing touched slot 0."""
    s = _server(name, cache, log, defer, n_slots=2, n_ctx=1024, ctx_shift=True, idle=-1, yarn=True)
    s.start()
    _complete(s, P, slot=0, n_predict=4)
    body = _complete(s, _toks(400, 4), slot=1, n_predict=200)
    assert body.get("truncated") is True, "slot 1 must have shifted its context"
    c = _counters(s)
    assert _terminate(s) == 0
    return c


def _sc_sleep(name, cache, log, defer, hold):
    """The server goes to sleep (its contexts are freed) while copies are pending."""
    p, ext, _ = _prompts(name)
    s = _server(name, cache, log, defer, sleep_s=2)
    s.start()
    _complete(s, p)
    assert _wait(lambda: "entering sleeping state" in open(log).read(), 30)
    # the copies forced at sleep entry are published before the next request looks the store up, so both
    # runs restore from the same units (which unit a restore picks is not what this test is about)
    assert _wait(lambda: open(log).read().count("auto-save: persisted") >= 2, 30)
    _complete(s, ext)  # wakes it up; nothing in memory, so a restore
    c = _counters(s)
    assert _terminate(s) == 0
    return c


def _sc_restore(name, cache, log, defer, hold):
    """A restore into the slot replaces its cells while that slot's copies are pending. The store is seeded
    first by an instance that persists P (identically in both runs)."""
    p, ext, q = _prompts(name)
    seed = _server(name, cache, log + ".seed", defer=False, idle=-1, node_prompt="off", port_off=1)
    seed.start()
    _complete(seed, p, n_predict=0)
    assert _terminate(seed) == 0  # the shutdown save persists P
    s = _server(name, cache, log, defer, idle=-1)
    s.start()
    _complete(s, q)        # a prompt node of Q, pending
    body = _complete(s, ext)  # restores P from disk into the same slot
    assert body["timings"].get("cache_disk_n", 0) >= len(p) - B, body["timings"]
    c = _counters(s)
    assert _terminate(s) == 0
    return c


def _sc_shutdown(name, cache, log, defer, hold):
    """Copies still pending at a graceful stop are emitted and published by the shutdown flush."""
    _, ext, _ = _prompts(name)
    s = _server(name, cache, log, defer, idle=-1)
    s.start()
    _complete(s, ext, n_predict=4)
    c = _counters(s)
    assert _terminate(s) == 0
    return c


def _sc_budget(name, cache, log, defer, hold):
    """A 1 MiB staging budget and a conversation growing over eight turns, every prompt node held pending:
    the side-state of the pending captures reaches the budget, so older captures are forced first (and a
    forced remainder too large for the budget streams to the writer)."""
    conv = _toks(1200, 5)
    s = _server(name, cache, log, defer, idle=-1, staging_mb=1)
    s.start()
    peak = 0.0
    for n in range(150, 1201, 150):
        _complete(s, conv[:n], n_predict=4)
        peak = max(peak, _metric(s, "auto_cache_save_deferred_host_bytes"))
    c = _counters(s, ("auto_cache_save_dropped_staging_total", "auto_cache_save_queued_total"))
    c["deferred_host_peak"] = peak
    assert _terminate(s) == 0
    return c


SCENARIOS = {"basic": _sc_basic, "reassign": _sc_reassign, "ctxshift": _sc_ctxshift, "othershift": _sc_othershift,
             "sleep": _sc_sleep, "restore": _sc_restore, "shutdown": _sc_shutdown, "budget": _sc_budget}


def _run_pair(name, scenario, tmp_path, monkeypatch, hold, no_hook=False):
    """Runs `scenario` with stage 1 and with deferred copies; returns (stage-1 store, deferred store, counters)."""
    fn = SCENARIOS[scenario]
    ref_dir = os.path.join(ROOT, "stage1")
    def_dir = os.path.join(ROOT, "deferred")
    os.makedirs(ref_dir)
    os.makedirs(def_dir)
    c_ref = fn(name, ref_dir, str(tmp_path / "stage1.log"), False, hold)
    if hold:
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE", "1")
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "-1")
    if no_hook:
        monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_NO_HOOK", "1")
    c = fn(name, def_dir, str(tmp_path / "deferred.log"), True, hold)
    c["stage1"] = c_ref
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


# kv, hybrid, iSWA, hybrid_idx and the MTP draft (a .dft sidecar per unit)
MUTATION_MODELS = ["llama-dense", "qwen35-dense", "gemma3-dense", "qwen4exp-moe", "mtp"]
# classes that cannot shift: a recurrent state (qwen35, qwen4exp, the qwen35-based MTP dummy) refuses it
NO_SHIFT = ("qwen35-dense", "qwen4exp-moe", "mtp")


@pytest.mark.parametrize("name", MUTATION_MODELS)
@pytest.mark.parametrize("scenario", ["reassign", "ctxshift", "sleep", "restore", "shutdown"])
def test_mutation_flushes_pending_copy_first(name, scenario, tmp_path, monkeypatch):
    """The deferred copies are held pending (no idle emission, no trickle) while the cache changes the
    captured cells: the flush-on-mutate hook (or the server's own force before sleep, shutdown, a restore
    and a slot reuse) must copy them first, so the store equals the stage-1 run's."""
    if scenario == "ctxshift" and name in NO_SHIFT:
        pytest.skip("a recurrent state cannot be shifted (the context shift is refused for this class)")
    ref, got, c, log = _run_pair(name, scenario, tmp_path, monkeypatch, hold=True)
    ok, why = _compare(ref, got)
    assert ok, why
    assert c["auto_cache_save_deferred_total"] >= 1, c
    text = open(log).read()
    assert "mode deferred, positional" in text
    if scenario == "ctxshift":
        # reaches the engine hook (the others are forced by the server before the context is touched)
        assert c["auto_cache_save_deferred_forced_total"] >= 1, c
    if scenario == "reassign":
        # the reclaim save waits for the queued delta covering the slot: the copies ahead of it are emitted
        # first, so the wait ends when the writer publishes, not at the 60 s stall timeout
        assert c["pending_before_reuse"] >= 1, c
        assert "covers this slot (still queued)" not in text
        assert c["reuse_s"] < 15, c


def test_other_slot_shift_flushes_pending_copy_first(tmp_path, monkeypatch):
    """Two slots under YaRN: slot 1's context shift ropes slot 0's captured cells too, so slot 0's pending
    copy must be forced by the K-shift although slot 0 itself was never touched."""
    ref, got, c, log = _run_pair("llama-dense", "othershift", tmp_path, monkeypatch, hold=True)
    ok, why = _compare(ref, got)
    assert ok, why
    text = open(log).read()
    forced0 = [l for l in text.splitlines() if "slot 0: auto-save: persisted" in l and "forced by a cache mutation" in l]
    assert forced0, "slot 0's unit was not forced by slot 1's shift"


@pytest.mark.parametrize("scenario", ["ctxshift", "othershift"])
def test_disabled_hook_corrupts_the_unit(scenario, tmp_path, monkeypatch):
    """Positive controls: with the engine hook disabled, the shift changes the captured cells before the
    pending copy runs, and the published unit no longer equals the stage-1 one (for othershift only because
    of YaRN). Without these the equalities above could not tell a working hook from a missing one."""
    ref, got, c, log = _run_pair("llama-dense", scenario, tmp_path, monkeypatch, hold=True, no_hook=True)
    assert c["auto_cache_save_deferred_total"] >= 1, c
    assert c["auto_cache_save_deferred_forced_total"] == 0, c
    ok, why = _compare(ref, got)
    assert not ok, f"with the hook disabled the store still matches stage 1 ({why}): the control is broken"
    assert set(ref) == set(got), "the control must corrupt bytes, not the file set"


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
def test_small_budget_with_defer_is_byte_identical(name, tmp_path, monkeypatch):
    """--slot-save-staging-mb 1 with deferred copies held pending: the deferred side-state stays within the
    budget (older captures are forced first), nothing is dropped, and the store equals stage 1's under the
    same budget."""
    ref, got, c, log = _run_pair(name, "budget", tmp_path, monkeypatch, hold=True)
    ok, why = _compare(ref, got)
    assert ok, why
    assert c["auto_cache_save_deferred_total"] >= 1, c
    assert c["auto_cache_save_dropped_staging_total"] == 0, c
    assert c["stage1"]["auto_cache_save_dropped_staging_total"] == 0, c["stage1"]
    assert c["deferred_host_peak"] <= 1 << 20, c
    text = open(log).read()
    assert "over the" not in text, "a deferred copy exceeded the budget outside a store-lock scope"
    if name == "qwen35-dense":
        # its side-state (the recurrent fold) is large enough for the budget to bind: captures were forced
        assert ", wait)" in text or ", wait, forced" in text, "the budget never forced an older capture"


@pytest.mark.parametrize("defer", [False, True])
def test_kill_after_response_loses_only_pending_units(defer, tmp_path, monkeypatch):
    """The loss window of a SIGKILL right after a response. With the deferred copy held pending (no idle
    emission, as if the kill landed before the first idle wakeup), the prompt node's unit is lost, as a
    stage-1 unit still in the writer's queue would be; the stage-1 run (the control) publishes it. Either
    way the store holds only complete units, everything published before the kill survives it, and a
    restarted server serves from the store."""
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE", "1")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "-1")
    cache = os.path.join(ROOT, "kill")
    os.makedirs(cache)
    log = str(tmp_path / "kill.log")
    s = _server("llama-dense", cache, log, defer, idle=-1)
    s.start()
    _complete(s, P, n_predict=4)  # a prompt node of P
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, 15)
    if not defer:
        assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 20)
    pend = _metric(s, "auto_cache_save_deferred_pending")
    n_pub = _metric(s, "auto_cache_save_root_total") + _metric(s, "auto_cache_save_delta_total")
    before = sorted(glob.glob(os.path.join(cache, "auto-*.bin")))
    if s in server_instances:
        server_instances.remove(s)
    s.process.send_signal(signal.SIGKILL)
    s.process.wait(timeout=30)
    s._log.close()
    s.process = None
    after = sorted(glob.glob(os.path.join(cache, "auto-*.bin")))
    assert len(before) == n_pub and set(before) <= set(after), (before, after, n_pub)
    for b in after:
        assert os.path.exists(b + ".meta"), f"torn unit: {os.path.basename(b)}"
    if defer:
        assert pend >= 1, "the prompt node's copy must have been pending at the kill"
        assert after == [], f"a unit whose copy was pending cannot have been published: {after}"
    else:
        assert pend == 0 and len(after) >= 1, (pend, after)
    s2 = _server("llama-dense", cache, log + ".2", defer, idle=-1, port_off=1)
    s2.start()
    body = _complete(s2, EXT, n_predict=4)
    got = body["timings"].get("cache_disk_n", 0)
    assert (got == 0) if defer else (got >= len(P) - B), body["timings"]
    _terminate(s2)

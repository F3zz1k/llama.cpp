import glob
import os
import shutil
import signal
import struct
import subprocess
import time

import pytest
from utils import *

# The background writer of the auto disk cache (docs/disk-cache.md, "What runs in the background").
# A save is captured (copied off the device) on the server thread and published by a writer thread;
# these tests slow the writer down or make it fail through test hooks read from the environment:
#   LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS     the writer waits this long after a unit's temps are written
#   LLAMA_TEST_SLOT_SAVE_FAIL_META_AT        the k-th .meta rename fails (1-based)
#   LLAMA_TEST_SLOT_SAVE_TMP_REAP_AGE_S      age after which a dead writer's temps are reaped (default 600)
#   LLAMA_TEST_SLOT_SAVE_SHUTDOWN_DEADLINE_MS the shutdown flush deadline (default 90000)
#   LLAMA_TEST_SLOT_SAVE_WRITER_STALL_MS     a writer silent this long counts as hung (default 60000)
# Prompts are token ids: the dummy vocab has no meaningful text tokenizer.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
CACHE_DIR = "./tmp/slot_save_writer"
B = 16
IDLE = 1
SLOT_META_MAGIC = 0x544D4B4C  # "LKMT"
HOOKS = ("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "LLAMA_TEST_SLOT_SAVE_FAIL_META_AT",
         "LLAMA_TEST_SLOT_SAVE_TMP_REAP_AGE_S", "LLAMA_TEST_SLOT_SAVE_SHUTDOWN_DEADLINE_MS",
         "LLAMA_TEST_SLOT_SAVE_WRITER_STALL_MS")


def _toks(n: int, seed: int):
    return [((i * (7 + 2 * seed) + 13 * seed) % 100) + 10 for i in range(n)]


ARCH = {"llama-dense": "llama", "qwen35-dense": "qwen35"}
P = _toks(200, 1)
EXT = P + _toks(80, 2)


def _model(name: str) -> str:
    m = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.isfile(m):
        pytest.skip(f"{m} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")
    return m


def _server(model: str, log_path: str, n_slots: int = 1, staging_mb=None, idle=IDLE, cache=CACHE_DIR) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    # the dummies declare a 256-token training context; the server caps n_ctx at it
    s.override_kv = [f"{ARCH[os.path.basename(model)[:-len('.gguf')]]}.context_length=int:4096"]
    s.n_ctx = 2048 * n_slots
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
    s.slot_save_node_prompt = "off"
    s.slot_save_staging_mb = staging_mb
    if staging_mb is not None:
        # the budget tests exercise the admission of saves copied at the capture (staged, streamed, waiting,
        # dropped); a deferred save (stage 2, test_slot_save_defer.py) takes none of those paths
        s.slot_save_defer = False
    s.log_path = log_path
    return s


def _complete(s, prompt, slot=0, n_predict=0):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "id_slot": slot,
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


def _site_metric(s, name: str, site: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    key = f'llamacpp:{name}{{site="{site}"}} '
    for line in res.body.splitlines():
        if line.startswith(key):
            return float(line.split()[1])
    raise AssertionError(f"metric {name}{{site={site}}} not found")


def _lens(cache=CACHE_DIR):
    return sorted(_node_parent(m)[1] for m in _metas(cache))


def _wait(pred, timeout_s: float, step: float = 0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def _metas(cache=CACHE_DIR):
    return sorted(glob.glob(os.path.join(cache, "auto-*.bin.meta")))


def _temps(cache=CACHE_DIR):
    return sorted(p for p in glob.glob(os.path.join(cache, "*")) if ".tmp" in os.path.basename(p))


def _bins_without_meta(cache=CACHE_DIR):
    return [p for p in glob.glob(os.path.join(cache, "auto-*.bin")) if not os.path.exists(p + ".meta")]


def _node_parent(meta: str):
    # (version, n_tokens, parent_id or None) from the .meta header: v1 text root / v3 text delta
    with open(meta, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == SLOT_META_MAGIC
    name = os.path.basename(meta)[:-len(".bin.meta")]
    n = int(name.split("-")[-1])
    if version != 3:
        return version, n, None
    parent_id, lo, hi = struct.unpack_from("<QII", data, len(data) - 16)
    return version, n, parent_id


def _id_of(meta: str) -> int:
    return int(os.path.basename(meta).split("-")[2], 16)


def _log(path: str) -> str:
    with open(path) as f:
        return f.read()


def _terminate(s, timeout_s: float = 120) -> int:
    # graceful stop that waits for the shutdown flush (utils' stop() kills after 5 s)
    if s in server_instances:
        server_instances.remove(s)
    s.process.send_signal(signal.SIGTERM)
    rc = s.process.wait(timeout=timeout_s)
    s.process = None
    s._log.close()
    return rc


@pytest.fixture(autouse=True)
def clean_cache_dir(monkeypatch):
    for h in HOOKS:
        monkeypatch.delenv(h, raising=False)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.parametrize("model_name", ["llama-dense", "qwen35-dense"])
def test_pending_unit_is_invisible_until_published(model_name, tmp_path, monkeypatch):
    """A unit the writer still holds is not visible to a peer: a peer's request cold-prefills (no error),
    and the same request restores the unit once its .meta is in place. The renames land in the order
    .bin, .logits, .dft, .meta (rename times, which a rename updates)."""
    model = _model(model_name)
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "4000")
    a = _server(model, str(tmp_path / "a.log"))
    a.start()
    _complete(a, P)
    # the idle flush captured it and the writer holds it: queued, nothing published yet
    assert _wait(lambda: _metric(a, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    assert _metas() == []
    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS")
    peer = _server(model, str(tmp_path / "peer.log"), idle=-1)
    peer.server_port = a.server_port + 1
    peer.slot_save_min_tokens = 100000  # restores, never saves (not even at its shutdown)
    peer.start()
    cold = _complete(peer, EXT)
    assert cold["timings"].get("cache_disk_n", 0) == 0, cold["timings"]
    peer.stop()
    assert _wait(lambda: len(_metas()) == 1, 15)
    a.stop()
    meta = _metas()[0]
    order = [p for p in (meta[:-len(".meta")], meta[:-len(".meta")] + ".logits", meta[:-len(".meta")] + ".dft", meta)
             if os.path.exists(p)]
    ctimes = [os.stat(p).st_ctime_ns for p in order]
    assert ctimes == sorted(ctimes), list(zip(order, ctimes))
    assert "failed after the slot was cleared" not in _log(str(tmp_path / "peer.log"))

    hit = _server(model, str(tmp_path / "hit.log"), idle=-1)
    hit.start()
    body = _complete(hit, EXT)
    hit.stop()
    assert body["timings"].get("cache_disk_n", 0) >= len(P) - B, body["timings"]


def test_restore_waits_for_a_queued_prefix(tmp_path, monkeypatch):
    """In the same instance, a request on another slot whose prefix is still queued waits for the
    publish and restores it instead of re-prefilling."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "3000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=2)
    s.start()
    _complete(s, P, slot=0)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    assert _metas() == []
    body = _complete(s, EXT, slot=1)
    s.stop()
    log = _log(str(tmp_path / "s.log"))
    assert "queued unit(s) that prefix this request" in log
    assert body["timings"].get("cache_disk_n", 0) >= len(P) - B, body["timings"]


@pytest.mark.parametrize("model_name", ["llama-dense", "qwen35-dense"])
def test_child_delta_on_a_queued_parent(model_name, tmp_path, monkeypatch):
    """A delta whose parent is still queued is written against it, published after it, and a fresh
    instance restores the two-unit chain."""
    model = _model(model_name)
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "3000")
    s = _server(model, str(tmp_path / "s.log"))
    s.start()
    _complete(s, P)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    _complete(s, EXT)  # extends the slot in memory: no restore
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 10)
    assert len(_metas()) == 0, "the parent must still be queued when the child is captured"
    assert _wait(lambda: len(_metas()) == 2, 20)
    deltas = _metric(s, "auto_cache_save_delta_total")
    s.stop()
    assert deltas == 1
    info = {m: _node_parent(m) for m in _metas()}
    (root,) = [m for m, (v, n, par) in info.items() if par is None]
    (child,) = [m for m, (v, n, par) in info.items() if par is not None]
    assert info[child][2] == _id_of(root)
    assert os.stat(root).st_ctime_ns <= os.stat(child).st_ctime_ns

    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS")
    r = _server(model, str(tmp_path / "r.log"), idle=-1)
    r.start()
    body = _complete(r, EXT + [42] * B)
    r.stop()
    assert body["timings"].get("cache_disk_n", 0) >= len(EXT) - B, body["timings"]


def test_failed_parent_drops_its_queued_child(tmp_path, monkeypatch):
    """The parent's .meta rename fails: the child queued on it is dropped (it is a suffix only), counted,
    and no .bin is left without its .meta."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "2000")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_FAIL_META_AT", "1")
    s = _server(model, str(tmp_path / "s.log"))
    s.start()
    _complete(s, P)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    _complete(s, EXT)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 10)
    # the orphan's slot still holds the conversation: its idle flush is re-armed and saves it again, now
    # as a root (the failed parent is neither published nor queued)
    assert _wait(lambda: _lens() == [len(EXT)], 20), _lens()
    assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 20)
    orphans = _metric(s, "auto_cache_save_orphan_dropped_total")
    failed = _metric(s, "auto_cache_save_failed_total")
    s.stop()
    assert orphans == 1
    assert failed >= 1
    (meta,) = _metas()
    assert _node_parent(meta)[2] is None, "the re-save must be a root"
    assert _bins_without_meta() == []
    assert _temps() == []
    log = _log(str(tmp_path / "s.log"))
    assert "its parent failed to publish" in log
    assert "re-arming the idle flush" in log


def test_crash_during_write_publishes_nothing_and_temps_are_reaped(tmp_path, monkeypatch):
    """SIGKILL while the writer holds a unit: nothing is published, the next instance ignores and then
    reaps the dead writer's temps, and its request cold-prefills correctly."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "30000")
    s = _server(model, str(tmp_path / "s.log"))
    s.start()
    gen_ref = _complete(s, P, n_predict=8)["tokens"]
    assert _wait(lambda: len(_temps()) > 0 and _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    time.sleep(0.5)  # the temps are complete; the writer is in its delay
    s.process.kill()
    s.process.wait()
    server_instances.discard(s)
    s.process = None
    s._log.close()
    assert _metas() == []
    assert len(_temps()) > 0

    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_TMP_REAP_AGE_S", "0")
    r = _server(model, str(tmp_path / "r.log"), idle=-1)
    r.start()
    body = _complete(r, P, n_predict=8)
    r.stop()
    assert _temps() == []
    assert "which is no longer running" in _log(str(tmp_path / "r.log"))
    assert body["timings"].get("cache_disk_n", 0) == 0
    assert body["tokens"] == gen_ref


def test_staging_full_idle_flush_defers_then_streams(tmp_path, monkeypatch):
    """--slot-save-staging-mb 1: an idle flush larger than the free staging while the writer is busy is
    DEFERRED (nothing dropped, generation unaffected and not delayed), then streamed and published once the
    writer is idle. A later large save on an idle writer streams too, and the staging returns to 0."""
    model = _model("llama-dense")
    big_a = _toks(900, 3)   # about 1.8 MB of state: larger than the whole budget
    big_b = _toks(900, 4)
    ctl = _server(model, str(tmp_path / "ctl.log"), idle=-1)
    ctl.slot_save_auto = False
    ctl.slot_save_path = None
    ctl.slot_save_incremental = False
    ctl.slot_save_idle_seconds = None
    ctl.start()
    ref = _complete(ctl, big_a, n_predict=8)["tokens"]
    ctl.stop()

    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "4000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=2, staging_mb=1)
    s.start()
    _complete(s, P, slot=0)              # small: staged, then the writer sleeps in its delay
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    t0 = time.time()
    out = _complete(s, big_a, slot=1, n_predict=8)["tokens"]
    assert out == ref
    assert time.time() - t0 < 4.0, "the request must not wait on the busy writer"
    # big_a's unit holds its 8 generated tokens too, less the last one sampled
    assert _wait(lambda: any(n >= len(big_a) for n in _lens()), 20), _lens()
    assert _metric(s, "auto_cache_save_streamed_total") >= 1
    assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 20)
    # idle writer: the large save streams at once
    _complete(s, big_b, slot=0)
    assert _wait(lambda: _metric(s, "auto_cache_save_streamed_total") >= 2, IDLE + 10)
    assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 20)
    staging = _metric(s, "auto_cache_save_staging_bytes")
    dropped = _metric(s, "auto_cache_save_dropped_staging_total")
    req = _site_metric(s, "auto_cache_save_site_requested_total", "idle")
    pub = _site_metric(s, "auto_cache_save_site_published_total", "idle")
    s.stop()
    log = _log(str(tmp_path / "s.log"))
    assert "idle flush deferred" in log
    assert dropped == 0
    assert staging == 0
    assert req == pub == 3, (req, pub)
    assert len(big_b) in _lens() and sum(n >= len(big_a) for n in _lens()) == 2, _lens()
    assert _bins_without_meta() == []


def test_reclaim_waits_for_a_busy_writer(tmp_path, monkeypatch):
    """One slot, staging 1 MB: a large conversation reclaimed while the writer is busy with the previous
    unit waits for the writer instead of being dropped, and a fresh instance can restore it."""
    model = _model("llama-dense")
    big = _toks(900, 3)
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "3000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=1, staging_mb=1, idle=-1)
    s.start()
    _complete(s, P)
    _complete(s, big)            # reclaim save of P: small, staged; the writer sleeps in its delay
    _complete(s, _toks(300, 7))  # reclaim save of big while the writer is busy: waits, then streams
    assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 20)
    dropped = _metric(s, "auto_cache_save_dropped_staging_total")
    waits = _metric(s, "auto_cache_save_admission_waits_total")
    req = _site_metric(s, "auto_cache_save_site_requested_total", "reclaim")
    pub = _site_metric(s, "auto_cache_save_site_published_total", "reclaim")
    s.process.send_signal(signal.SIGKILL)
    s.process.wait()
    server_instances.discard(s)
    s.process = None
    s._log.close()
    assert dropped == 0
    assert waits >= 1
    assert req == pub == 2, (req, pub)
    assert len(big) in _lens(), _lens()
    assert "for the writer to take a" in _log(str(tmp_path / "s.log"))

    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS")
    r = _server(model, str(tmp_path / "r.log"), idle=-1)
    r.start()
    body = _complete(r, big + [42] * B)
    r.stop()
    assert body["timings"].get("cache_disk_n", 0) >= len(big) - B, body["timings"]


def test_stalled_writer_drops_instead_of_hanging(tmp_path, monkeypatch):
    """A writer that shows no sign of life for the stall time (here: its test delay, 1 s stall) makes the
    waiting capture give up with a WRN and the drop counter, rather than holding the server loop for good."""
    model = _model("llama-dense")
    big = _toks(900, 3)
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "6000")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_STALL_MS", "1000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=1, staging_mb=1, idle=-1)
    s.start()
    _complete(s, P)
    _complete(s, big)
    t0 = time.time()
    _complete(s, _toks(300, 7))
    dt = time.time() - t0
    dropped = _metric(s, "auto_cache_save_dropped_staging_total")
    s.stop()
    assert dropped == 1
    assert dt < 5.0, dt
    assert "made no progress" in _log(str(tmp_path / "s.log"))


def test_reclaim_saves_again_when_the_covering_queued_delta_is_orphaned(tmp_path, monkeypatch):
    """The idle flush queued the conversation as a delta on a queued parent, and a reclaim arrives while
    both are queued. The parent's publish fails, which orphans the delta. The reclaim waited for the delta's
    outcome instead of trusting the queue, so it writes the conversation itself (as a root) and nothing is
    lost."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "2500")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_FAIL_META_AT", "1")
    s = _server(model, str(tmp_path / "s.log"))
    s.start()
    _complete(s, P)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 1, IDLE + 10)
    _complete(s, EXT)  # extends the slot; its idle flush queues a delta on the queued P
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 10)
    assert _metas() == []
    _complete(s, _toks(300, 7))  # reclaims EXT's slot while the delta is queued
    assert _wait(lambda: len(EXT) in _lens(), 20), _lens()
    orphans = _metric(s, "auto_cache_save_orphan_dropped_total")
    s.stop()
    log = _log(str(tmp_path / "s.log"))
    assert orphans == 1
    assert "not published, saving it again" in log
    meta = [m for m in _metas() if _node_parent(m)[1] == len(EXT)][0]
    assert _node_parent(meta)[2] is None, "the re-save must be a root"
    assert _bins_without_meta() == []


def test_pending_wait_does_not_stall_other_slots(tmp_path, monkeypatch):
    """A task waiting for its queued prefix is held in its own slot only: a request on another slot that
    arrives during the wait is served at its normal speed, and the waiting task restores the prefix."""
    import threading
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "6000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=3)
    s.start()
    t_ref0 = time.time()
    _complete(s, _toks(40, 9), slot=1, n_predict=16)
    t_ref = time.time() - t_ref0
    _complete(s, P, slot=0)
    assert _wait(lambda: _metric(s, "auto_cache_save_queued_total") >= 2, IDLE + 10)
    out = {}
    def waiter():
        out["body"] = _complete(s, EXT, slot=2)  # waits for the queued prefix
    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.5)
    t0 = time.time()
    _complete(s, _toks(40, 11), slot=1, n_predict=16)
    t_other = time.time() - t0
    th.join()
    s.stop()
    log = _log(str(tmp_path / "s.log"))
    assert "queued unit(s) that prefix this request" in log
    assert t_other < t_ref + 1.0, (t_ref, t_other)
    assert out["body"]["timings"].get("cache_disk_n", 0) >= len(P) - B, out["body"]["timings"]


def test_restore_of_a_unit_evicted_after_indexing_falls_back_cleanly(tmp_path):
    """The writer thread now runs the LRU while the server thread restores. Deletes take no store lock (by
    design, as for a peer's LRU), so the worst interleave is a unit this instance has indexed whose .bin is
    gone when the restore opens it. That restore must fall back to a cold prefill with the cold output,
    and count the failure."""
    model = _model("llama-dense")
    ctl = _server(model, str(tmp_path / "ctl.log"), idle=-1)
    ctl.slot_save_auto = False
    ctl.slot_save_path = None
    ctl.slot_save_incremental = False
    ctl.slot_save_idle_seconds = None
    ctl.start()
    ref = _complete(ctl, EXT, n_predict=8)["tokens"]
    ctl.stop()

    s = _server(model, str(tmp_path / "s.log"), n_slots=2)
    s.start()
    _complete(s, P, slot=0)
    assert _wait(lambda: len(_metas()) == 1, IDLE + 10)
    assert _wait(lambda: _metric(s, "auto_cache_save_queue_depth") == 0, 10)
    os.remove(_metas()[0][:-len(".meta")])  # the "eviction": the .meta stays, the state file is gone
    body = _complete(s, EXT, slot=1, n_predict=8)
    s.stop()
    assert body["tokens"] == ref
    assert body["timings"].get("cache_disk_n", 0) == 0, body["timings"]
    assert "failed after the slot was cleared" in _log(str(tmp_path / "s.log")), "the restore must have been tried"


def test_shutdown_flush_publishes_the_queue(tmp_path, monkeypatch):
    """SIGTERM with three slots to save and a slow writer: all three are published before the exit."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "1000")
    s = _server(model, str(tmp_path / "s.log"), n_slots=3, idle=-1)
    s.start()
    for i in range(3):
        _complete(s, _toks(150 + 10 * i, 5 + i), slot=i)
    t0 = time.time()
    rc = _terminate(s)
    assert time.time() - t0 >= 2.5, "the writer's delays must have been waited for"
    assert len(_metas()) == 3
    assert _bins_without_meta() == []
    assert _temps() == []
    assert rc == 0


def test_shutdown_deadline_abandons_without_partial_units(tmp_path, monkeypatch):
    """The same with a deadline shorter than the queue: what is left is abandoned (logged), no partial
    unit is published, no temp is left, and the exit is clean."""
    model = _model("llama-dense")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", "1000")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_SHUTDOWN_DEADLINE_MS", "1500")
    s = _server(model, str(tmp_path / "s.log"), n_slots=3, idle=-1)
    s.start()
    for i in range(3):
        _complete(s, _toks(150 + 10 * i, 5 + i), slot=i)
    rc = _terminate(s)
    log = _log(str(tmp_path / "s.log"))
    assert "abandoned a" in log
    assert len(_metas()) < 3
    assert _bins_without_meta() == []
    assert _temps() == []
    assert rc == 0

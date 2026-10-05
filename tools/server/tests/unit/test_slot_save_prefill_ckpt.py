import glob
import json
import os
import socket
import shutil
import signal
import struct
import threading
import time

import pytest
from utils import *

# Periodic prefill checkpoints (--slot-save-prefill-interval N, default 0 = off; docs/disk-cache.md).
# While a prompt prefills, the server stops its batch at every multiple of N and publishes the slot's state
# there through the ordinary save path, so each checkpoint is the true whole state at that position (sound
# for every memory class) and a delta on the previous one under --slot-save-incremental. These tests check:
#  - every memory class writes the checkpoints, and a request that changes a long prompt after its start
#    restores to within N tokens of the change, with output equal to a cold prefill;
#  - a prefill interrupted by a crash (SIGKILL), a client that goes away (a timeout) or a graceful stop
#    resumes from what was published, by log line, and the resumed prompt still gets its cold prompt node;
#  - a new question at the end of the same long user message restores the last checkpoint inside it;
#  - off (absent or 0) publishes the same store, byte for byte, and no checkpoint;
#  - the checkpoints respect the store's count cap like any other unit, a delta chain included (the new delta
#    that the cap cannot hold is not kept);
#  - on a class that writes no deltas (or with --slot-save-incremental off), each whole checkpoint replaces the
#    previous one of the same prefill, so one prefill keeps its deepest checkpoint only;
#  - a warm reuse that happens to end on a multiple of N is not taken for a resumed prefill.
# Test hooks read from the environment:
#   LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS   every prefill batch sleeps this long (so a test can interrupt it)
#   LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS   a deferred copy is trickled once it waited this long (default 10 s)
# Prompts are token ids: the dummy vocab has no meaningful text tokenizer.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
ROOT = "./tmp/slot_save_prefill_ckpt"
B = 16
N = 64
TOL = 2e-3
SLOT_META_MAGIC = 0x544D4B4C  # "LKMT"
HOOKS = ("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS", "LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS")

# the dummy of every memory class the cache knows, with its architecture prefix (for the context override)
CLASSES = {
    "llama-dense":    "llama",       # plain attention (rewinds)
    "gemma3-dense":   "gemma3",      # iSWA, n_swa = 32: every unit here is past one window
    "mamba-dense":    "mamba",       # recurrent only
    "qwen35-dense":   "qwen35",      # hybrid (gated delta net), the qwen3.8-27b class
    "glm5-next-moe":  "glm5-next",   # hybrid_idx (k-pool indexer)
    "qwen4exp-moe":   "qwen4exp",    # hybrid_idx, the Flash-Next class
    "glm-dsa-moe":    "glm-dsa",     # DSA
    "deepseek32-moe": "deepseek32",  # DSA
    "hy_v4-moe":      "hy_v4",
    "minimax-m3-moe": "minimax-m3",
    "dots3note-moe":  "dots3note",
    "deepseek4-moe":  "deepseek4",   # compressed KV (DeepSeek-V4)
}


def _toks(n: int, seed: int):
    return [((i * (7 + 2 * seed) + 13 * seed) % 100) + 10 for i in range(n)]


P = _toks(300, 1)


def _model(name: str) -> str:
    m = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.isfile(m):
        pytest.skip(f"{m} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")
    return m


def _server(name: str, cache: str | None, log_path: str | None = None, interval=N, n_batch: int = 512,
            node_prompt: str | None = "off", port_off: int = 0, max_count=None, incr: bool = True) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = _model(name)
    s.model_alias = "dummy"
    # the dummies declare a 256-token training context; the server caps n_ctx at it
    s.override_kv = [f"{CLASSES[name]}.context_length=int:4096"]
    s.n_ctx = 2048
    s.n_batch = n_batch
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.server_port += port_off
    s.log_path = log_path
    if cache is None:
        return s
    s.slot_save_path = cache
    s.slot_save_auto = True
    s.slot_save_incremental = incr
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_context_min_tokens = 100000   # no system node unless a test asks for one
    s.slot_save_idle_seconds = -1             # no idle flush: the store holds what the prefill wrote
    s.slot_save_node_prompt = node_prompt
    s.slot_save_prefill_interval = interval
    s.slot_save_max_count = max_count
    return s


def _req(s, prompt, n_predict: int = 4, extra: dict | None = None, timeout=None):
    data = {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": True,
        "id_slot": 0, "return_tokens": True, "n_probs": 8, "post_sampling_probs": False,
    }
    if extra:
        data.update(extra)
    res = s.make_request("POST", "/completion", data=data, timeout=timeout)
    assert res.status_code == 200, res.body
    return res.body


def _dist(body):
    return [{t["id"]: t["logprob"] for t in p["top_logprobs"]} for p in body["completion_probabilities"]]


def _assert_equals_cold(name: str, prompt, body, extra: dict | None = None):
    s = _server(name, None)
    s.start()
    cold = _req(s, prompt, extra=extra)
    s.stop()
    assert body["tokens"] == cold["tokens"], f"restored {body['tokens']} vs cold {cold['tokens']}"
    worst = 0.0
    for dr, dc in zip(_dist(body), _dist(cold)):
        for k, v in dr.items():
            assert k in dc, f"token {k} in the restored top-8 but not in the cold one"
            worst = max(worst, abs(v - dc[k]))
    assert worst < TOL, f"restored logprobs differ from cold by {worst}"


def _metric(s, name: str, site: str | None = None) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    key = f"llamacpp:{name} " if site is None else f'llamacpp:{name}{{site="{site}"}} '
    for line in res.body.splitlines():
        if line.startswith(key):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} {site or ''} not found")


def _settled(s):
    deadline = time.time() + 30
    while _metric(s, "auto_cache_save_queue_depth") > 0 and time.time() < deadline:
        time.sleep(0.05)
    assert _metric(s, "auto_cache_save_queue_depth") == 0


def _metas(cache):
    return sorted(glob.glob(os.path.join(cache, "auto-*.bin.meta")))


def _version(meta: str) -> int:
    with open(meta, "rb") as f:
        magic, version = struct.unpack_from("<II", f.read(8), 0)
    assert magic == SLOT_META_MAGIC
    return version


def _len(meta: str) -> int:
    return int(os.path.basename(meta)[:-len(".bin.meta")].split("-")[-1])


def _units(cache):
    return sorted(_len(m) for m in _metas(cache))


def _log(path: str) -> str:
    with open(path) as f:
        return f.read()


def _wait(pred, timeout_s: float, step: float = 0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def _kill(s):
    # a crash: no shutdown save, no writer drain
    server_instances.discard(s)
    s.process.send_signal(signal.SIGKILL)
    s.process.wait()
    s.process = None
    s._log.close()


def _terminate(s, timeout_s: float = 120) -> int:
    # graceful stop that waits for the shutdown flush (utils' stop() kills after 5 s)
    server_instances.discard(s)
    s.process.send_signal(signal.SIGTERM)
    rc = s.process.wait(timeout=timeout_s)
    s.process = None
    s._log.close()
    return rc


@pytest.fixture(autouse=True)
def clean_root(monkeypatch):
    for h in HOOKS:
        monkeypatch.delenv(h, raising=False)
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    yield
    shutil.rmtree(ROOT, ignore_errors=True)


def _cache(label: str) -> str:
    d = os.path.join(ROOT, label)
    os.makedirs(d, exist_ok=True)
    return d


@pytest.mark.parametrize("name", list(CLASSES))
def test_every_class_checkpoints_and_restores_within_n(name, tmp_path):
    """A cold 300-token prefill with N = 64 publishes units at exactly 64, 128, 192 and 256, each the true
    state there. A class that writes deltas keeps all four (a root and a chain of deltas); a class that writes
    whole units only keeps the deepest, each whole checkpoint replacing the previous one. After a crash, a
    request that shares a prefix and then changes restores to within N of the change (on the deepest kept
    checkpoint at or before it, up to the change itself on plain attention), and its output equals a cold
    prefill."""
    cache = _cache("c")
    s = _server(name, cache, str(tmp_path / "s.log"))
    s.start()
    _req(s, P)
    _settled(s)
    assert _metric(s, "auto_cache_save_site_requested_total", "prefill_checkpoint") == 4
    assert _metric(s, "auto_cache_save_site_published_total", "prefill_checkpoint") == 4
    deltas = _metric(s, "auto_cache_delta_capable") == 1
    superseded = _metric(s, "auto_cache_prefill_checkpoint_superseded_total")
    _kill(s)
    log = _log(str(tmp_path / "s.log"))
    assert "4 prefill checkpoint(s) armed every 64 tokens, first at 64" in log
    for p in (64, 128, 192, 256):
        assert f"prefill checkpoint at {p} of 300 prompt tokens" in log
    if deltas:
        assert _units(cache) == [64, 128, 192, 256]
        assert superseded == 0
        # a root and then a chain of deltas, each on the previous checkpoint
        vs = [_version(m) for m in sorted(_metas(cache), key=_len)]
        assert vs[0] == 1 and all(v == 3 for v in vs[1:]), vs
        div = 200
    else:
        assert _units(cache) == [256]
        assert superseded == 3
        assert "replaces the shallower whole checkpoint" in log
        div = 280
    if name in ("llama-dense", "qwen35-dense"):
        assert deltas, f"{name} is expected to write deltas"

    req = P[:div] + _toks(60, 5)
    r = _server(name, cache, str(tmp_path / "r.log"))
    r.start()
    body = _req(r, req)
    r.stop()
    t = body["timings"]
    disk = t.get("cache_disk_n", 0)
    assert disk <= div and div - disk < N and disk >= (div // N) * N, t
    assert t["cache_n"] == disk, t
    _assert_equals_cold(name, req, body)


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
def test_off_publishes_the_same_store_and_no_checkpoint(name, tmp_path):
    """--slot-save-prefill-interval absent and 0 publish byte-identical stores (and the same output), with no
    checkpoint requested; the same scenario with the interval on publishes more units."""
    ext = P + _toks(200, 2)
    stores = {}
    outs = {}
    for label, interval in (("absent", None), ("zero", 0), ("on", N)):
        cache = _cache(label)
        s = _server(name, cache, str(tmp_path / f"{label}.log"), interval=interval, node_prompt=None)
        s.start()
        a = _req(s, P)["tokens"]
        b = _req(s, ext)["tokens"]
        _settled(s)
        if interval != N:
            assert _metric(s, "auto_cache_save_site_requested_total", "prefill_checkpoint") == 0
        assert _terminate(s) == 0
        files = {}
        for p in sorted(glob.glob(os.path.join(cache, "*"))):
            with open(p, "rb") as f:
                files[os.path.basename(p)] = f.read()
        stores[label] = files
        outs[label] = (a, b)
        if interval != N:
            assert "prefill checkpoint" not in _log(str(tmp_path / f"{label}.log"))
    assert stores["absent"] == stores["zero"]
    assert outs["absent"] == outs["zero"]
    assert len(stores["on"]) > len(stores["absent"])


def _background(fn):
    out = {}

    def run():
        try:
            out["v"] = fn()
        except Exception as e:  # the interrupted request is expected to fail
            out["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, out


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense", "mamba-dense"])
def test_crash_mid_prefill_resumes_from_the_last_checkpoint(name, tmp_path, monkeypatch):
    """SIGKILL while a long prompt prefills: the next instance restores the deepest published checkpoint (by
    log line), says it resumes an interrupted prefill, prefills only the rest, writes the cold prompt node
    the interrupted prefill never reached, and its output equals a cold prefill."""
    cache = _cache("c")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS", "250")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "0")
    s = _server(name, cache, str(tmp_path / "s.log"), n_batch=16, node_prompt="cold")
    s.start()
    th, _ = _background(lambda: _req(s, P, timeout=120))
    # (counted, not listed: a class without deltas keeps only its deepest whole checkpoint on disk)
    assert _wait(lambda: _metric(s, "auto_cache_save_site_published_total", "prefill_checkpoint") >= 2, 60), \
        "no checkpoint was published"
    _kill(s)
    th.join(timeout=10)
    units = _units(cache)
    assert units and all(u % N == 0 for u in units) and max(units) < 288, units
    deepest = max(units)

    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS")
    r = _server(name, cache, str(tmp_path / "r.log"), node_prompt="cold")
    r.start()
    body = _req(r, P)
    _settled(r)
    _kill(r)
    t = body["timings"]
    assert t.get("cache_disk_n", 0) == deepest, t
    assert t["prompt_n"] == len(P) - deepest, t
    log = _log(str(tmp_path / "r.log"))
    assert f"auto-restore: reused {deepest} tokens from disk" in log
    assert "resuming an interrupted prefill" in log
    # the raw prompt has no user span: the prompt node sits one block below its end, block-aligned
    assert 288 in _units(cache), _units(cache)
    _assert_equals_cold(name, P, body)


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
def test_disconnect_mid_prefill_resumes(name, tmp_path, monkeypatch):
    """A client that goes away during a long prefill (closes its connection, as on a timeout): the task is
    cancelled with checkpoints already published. A peer instance on the same store resumes from the deepest
    published checkpoint; a resend on the same instance continues the slot's partial prefill and still writes
    the cold prompt node (the slot's prefill counts as resumed)."""
    cache = _cache("c")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS", "250")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "0")
    a = _server(name, cache, str(tmp_path / "a.log"), n_batch=8, node_prompt="cold")
    a.start()
    # a raw connection, so the test decides exactly when it goes away
    body = json.dumps({"prompt": P, "n_predict": 4, "temperature": 0, "top_k": 1, "cache_prompt": True,
                       "id_slot": 0}).encode()
    sock = socket.create_connection((a.server_host, a.server_port), timeout=10)
    sock.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    assert _wait(lambda: len(_metas(cache)) >= 2, 60), "no checkpoint was published"
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    # the server notices the closed connection and cancels the task. Polling /metrics meanwhile is deliberate:
    # each poll posts a result, which used to restart the completion's 1 s wait for its own result, so a
    # disconnected task was never cancelled while other requests kept the server answering
    assert _wait(lambda: _metric(a, "requests_processing") == 0, 30)
    _settled(a)
    units = _units(cache)
    assert units and all(u % N == 0 for u in units) and max(units) < 288, units
    deepest = max(units)

    # a peer instance (no access to A's slot) resumes from the deepest published checkpoint
    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS")
    # (no checkpoints or prompt node of its own, and a crash instead of a stop, so B adds nothing to the store
    # that would serve A's resend below from disk)
    b = _server(name, cache, str(tmp_path / "b.log"), node_prompt="off", port_off=1, interval=0)
    b.start()
    body_b = _req(b, P)
    _settled(b)
    _kill(b)
    assert body_b["timings"].get("cache_disk_n", 0) == deepest, body_b["timings"]
    assert f"auto-restore: reused {deepest} tokens from disk" in _log(str(tmp_path / "b.log"))
    _assert_equals_cold(name, P, body_b)

    # the same instance still holds the partial prefill: the resend continues it warm, as a resumed cold prompt
    body_a = _req(a, P)
    _settled(a)
    a.stop()
    t = body_a["timings"]
    assert t["cache_n"] >= deepest and t.get("cache_disk_n", 0) == 0, t
    assert "resuming an interrupted prefill" in _log(str(tmp_path / "a.log"))
    assert 288 in _units(cache), _units(cache)
    assert body_a["tokens"] == body_b["tokens"]


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
def test_graceful_stop_mid_prefill_resumes_at_least_from_the_last_checkpoint(name, tmp_path, monkeypatch):
    """SIGTERM while a long prompt prefills: the shutdown save also publishes the partial prefill, so the next
    instance resumes from it, at least as deep as the last checkpoint, and its output equals a cold prefill."""
    cache = _cache("c")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS", "250")
    monkeypatch.setenv("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", "0")
    s = _server(name, cache, str(tmp_path / "s.log"), n_batch=16, node_prompt="cold")
    s.start()
    th, _ = _background(lambda: _req(s, P, timeout=120))
    assert _wait(lambda: len(_metas(cache)) >= 2, 60), "no checkpoint was published"
    ckpts = [u for u in _units(cache) if u % N == 0]
    assert _terminate(s) == 0
    th.join(timeout=10)
    units = _units(cache)
    deepest = max(units)
    assert deepest >= max(ckpts) and deepest < len(P), units

    monkeypatch.delenv("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS")
    r = _server(name, cache, str(tmp_path / "r.log"), node_prompt="cold")
    r.start()
    body = _req(r, P)
    r.stop()
    assert body["timings"].get("cache_disk_n", 0) == deepest, body["timings"]
    assert f"auto-restore: reused {deepest} tokens from disk" in _log(str(tmp_path / "r.log"))
    _assert_equals_cold(name, P, body)


# --- a new question inside the same long user message, with the system and prompt nodes ---------------
DELIMS = [
    {"role": "system",    "delimiter": "<|im_start|>system"},
    {"role": "user",      "delimiter": "<|im_start|>user"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant"},
]


def _tok(s, text: str):
    res = s.make_request("POST", "/tokenize", data={"content": text, "add_special": False, "parse_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "qwen35-dense", "qwen4exp-moe"])
def test_new_question_in_the_same_long_message(name, tmp_path):
    """A system prompt and one long user message (a document) with a question at its end. The cold prefill
    writes the system node, the checkpoints inside the message and the prompt node at its end, none twice.
    After a restart, the same document with a different question restores to within N of where the questions
    differ, counts as a resumed cold prompt (its prompt node is written), and its output equals a cold
    prefill."""
    cache = _cache("c")
    s = _server(name, cache, str(tmp_path / "s.log"), node_prompt="cold")
    s.slot_save_context_min_tokens = 0    # the system node's floor becomes the block size
    s.start()
    S, U, A = (_tok(s, d["delimiter"]) for d in DELIMS)
    used = set(S) | set(U) | set(A)
    pool = [t for t in range(10, 110) if t not in used]
    body_sys = [pool[(i * 7) % len(pool)] for i in range(40)]
    doc = [pool[(i * 11 + 3) % len(pool)] for i in range(330)]
    q1 = [pool[(i * 13 + 1) % len(pool)] for i in range(20)]
    q2 = [pool[(i * 17 + 5) % len(pool)] for i in range(25)]
    nl = [pool[0]]
    first = S + body_sys + U + doc + q1 + A + nl
    sys_end = len(S) + len(body_sys)
    user_end = len(first) - len(A) - len(nl)
    extra = {"message_delimiters": DELIMS}
    _req(s, first, extra=extra)
    _settled(s)
    _kill(s)
    units = _units(cache)
    ckpts = [p for p in range(N, len(first), N) if p != sys_end and p != user_end]
    assert units == sorted(set([sys_end, user_end] + ckpts)), (units, sys_end, user_end)

    second = S + body_sys + U + doc + q2 + A + nl
    div = next(i for i, (x, y) in enumerate(zip(first, second)) if x != y)
    r = _server(name, cache, str(tmp_path / "r.log"), node_prompt="cold")
    r.slot_save_context_min_tokens = 0
    r.start()
    body = _req(r, second, extra=extra)
    _settled(r)
    _kill(r)
    t = body["timings"]
    disk = t.get("cache_disk_n", 0)
    assert disk <= div and div - disk < N, (t, div)
    if name != "llama-dense":
        assert disk == (div // N) * N, (t, div)
    assert "resuming an interrupted prefill" in _log(str(tmp_path / "r.log"))
    assert len(second) - len(A) - len(nl) in _units(cache)
    _assert_equals_cold(name, second, body, extra)


def test_whole_checkpoints_keep_only_the_deepest(tmp_path):
    """With --slot-save-incremental off every checkpoint is a whole unit of the prefix so far: a 600-token prefill
    (nine checkpoints) publishes all nine, each replacing the previous one, and leaves only the deepest, which
    restores correctly. (Without the replacement the store would hold 64 + 128 + ... + 576 tokens of whole units,
    growing with the square of the prompt.)"""
    name = "llama-dense"
    cache = _cache("c")
    p = _toks(600, 3)
    s = _server(name, cache, str(tmp_path / "s.log"), incr=False)
    s.start()
    _req(s, p)
    _settled(s)
    assert _metric(s, "auto_cache_save_site_published_total", "prefill_checkpoint") == 9
    assert _metric(s, "auto_cache_prefill_checkpoint_superseded_total") == 8
    _kill(s)
    assert _units(cache) == [576]
    req = p[:590] + _toks(30, 7)
    r = _server(name, cache, str(tmp_path / "r.log"), incr=False)
    r.start()
    body = _req(r, req)
    r.stop()
    assert body["timings"].get("cache_disk_n", 0) >= 576, body["timings"]
    _assert_equals_cold(name, req, body)


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
def test_checkpoints_respect_the_count_cap_with_a_delta_chain(name, tmp_path):
    """Under --slot-save-incremental the checkpoints form one delta chain, which is evictable only from its tip.
    With --slot-save-max-count 3 a 600-token prefill (nine checkpoints) keeps the store at three units: once the
    chain fills the cap, a new delta that only its own chain blocks is not kept (counted, with a WRN), and the
    chain left in the store still restores correctly."""
    cache = _cache("c")
    p = _toks(600, 3)
    s = _server(name, cache, str(tmp_path / "s.log"), max_count=3, incr=True)
    s.start()
    _req(s, p)
    _settled(s)
    assert _metric(s, "auto_cache_save_site_requested_total", "prefill_checkpoint") == 9
    assert _metric(s, "auto_cache_save_site_published_total", "prefill_checkpoint") == 3
    # each later checkpoint is a delta the cap cannot hold (or, queued on one of those, dropped with it)
    assert _metric(s, "auto_cache_evict_bound_exceeded_total") >= 1
    _kill(s)
    units = _units(cache)
    assert units == [64, 128, 192], units
    assert "cannot hold this conversation's delta chain" in _log(str(tmp_path / "s.log"))
    req = p[:200] + _toks(30, 7)
    r = _server(name, cache, str(tmp_path / "r.log"), max_count=3, incr=True)
    r.start()
    body = _req(r, req)
    r.stop()
    assert body["timings"].get("cache_disk_n", 0) == 192, body["timings"]
    _assert_equals_cold(name, req, body)


def test_a_warm_reuse_on_a_multiple_of_n_is_not_a_resume(tmp_path):
    """A request that reuses exactly 128 tokens (a multiple of N) of the slot's previous, finished prompt is a
    normal warm reuse: it is not taken for a resumed prefill, so the cold prompt node is not armed for it."""
    name = "llama-dense"
    cache = _cache("c")
    s = _server(name, cache, str(tmp_path / "s.log"), node_prompt="cold")
    s.start()
    _req(s, P[:150])
    _settled(s)
    before = _metric(s, "auto_cache_save_site_requested_total", "prompt_node")
    body = _req(s, P[:128] + _toks(100, 9))
    _settled(s)
    after = _metric(s, "auto_cache_save_site_requested_total", "prompt_node")
    _kill(s)
    assert body["timings"]["cache_n"] == 128, body["timings"]
    assert "resuming an interrupted prefill" not in _log(str(tmp_path / "s.log"))
    assert after == before


# --- a model with a draft (MTP) ---------------------------------------------------------------------------------
MTP_MODEL = os.environ.get("LLAMA_TEST_MTP_MODEL", "")


@pytest.mark.parametrize("n_batch", [32, 512])
def test_mtp_checkpoints_carry_the_draft(n_batch, tmp_path):
    """On the MTP test model (qwen35 with a draft-mtp head, a hybrid), every periodic checkpoint carries its
    .dft draft sidecar, including when the prompt prefills in several batches, and a later request whose chain
    runs through the checkpoints restores the draft warm, its output equal to the same request on a server
    without the cache."""
    if not os.path.isfile(MTP_MODEL):
        pytest.skip("no MTP test model (LLAMA_TEST_MTP_MODEL)")
    long = [((i * 13) % 97) + 10 for i in range(200)]
    extra = [((i * 11) % 100) + 10 for i in range(16)]

    def mtp_server(cache, log_path, idle):
        s = ServerProcess()
        s.model_hf_repo = None
        s.model_hf_file = None
        s.model_file = MTP_MODEL
        s.spec_type = "draft-mtp"
        s.spec_draft_n_max = 3
        s.model_alias = "dummy"
        s.n_ctx = 512
        s.n_batch = n_batch
        s.n_slots = 1
        s.temperature = 0.0
        s.server_metrics = True
        s.log_path = log_path
        if cache is None:
            return s
        s.slot_save_path = cache
        s.slot_save_auto = True
        s.slot_save_incremental = True
        s.slot_save_block = B
        s.slot_save_min_tokens = 0
        s.slot_save_context_min_tokens = 100000
        s.slot_restore_min_tokens = 0
        s.slot_save_idle_seconds = idle
        s.slot_save_node_prompt = "off"
        s.slot_save_prefill_interval = 48
        return s

    def comp(srv, prompt):
        res = srv.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 8, "cache_prompt": True, "id_slot": 0,
            "temperature": 0.0, "top_k": 1, "return_tokens": True})
        assert res.status_code == 200, res.body
        return res.body

    cache = _cache("mtp")
    s = mtp_server(cache, str(tmp_path / "s.log"), 1)
    s.start()
    gen = comp(s, long)["tokens"]
    _wait(lambda: _metric(s, "auto_cache_save_site_published_total", "idle") >= 1, 30)
    _settled(s)
    s.stop()
    units = _units(cache)
    ckpts = [u for u in units if u % 48 == 0]
    assert ckpts == [48, 96, 144, 192], units
    for m in _metas(cache):
        assert os.path.isfile(m[:-len(".meta")] + ".dft"), f"{m} has no .dft"

    req = long + gen + extra
    r = mtp_server(cache, str(tmp_path / "r.log"), -1)
    r.start()
    body = comp(r, req)
    warm = _metric(r, "auto_cache_restore_draft_warm_total")
    cold = _metric(r, "auto_cache_restore_draft_cold_total")
    r.stop()
    t = body["timings"]
    assert t.get("cache_disk_n", 0) >= len(long) and t["cache_disk_nodes"] >= 5, t
    assert warm == 1 and cold == 0, (warm, cold)
    ref = mtp_server(None, None, -1)
    ref.start()
    cold_body = comp(ref, req)
    ref.stop()
    assert body["tokens"] == cold_body["tokens"]

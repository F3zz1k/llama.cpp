import glob
import os
import shutil
import signal
import time

import pytest
from utils import *

# Auto disk cache items from the 2026-10-05 audit of the upstream-merge plan (docs/disk-cache.md):
#  - restore mode 2: on the same instance, a node at or below a divergence inside the slot's conversation
#    restores only its side-state, over the slot's own positional cells (classes that support it: recurrent
#    and hybrid); the margin gate compares against what the slot can really keep, not the raw match;
#  - the per-request timings say where the prompt came from (cache_source) and which unit was restored;
#  - the logits sidecar on every class, so an exact resend of a saved unit prefills nothing;
#  - a miss that a unit of the same model under another identity would have served is counted and logged;
#  - an eviction pass that cannot get under a cap is counted;
#  - /props reports the memory class as the cache handles it, checked here against a table of every class.
# Prompts are token ids: the dummy vocab has no meaningful text tokenizer.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
ROOT = "./tmp/slot_save_dropped"
B = 16
TOL = 2e-3

ARCH = {
    "llama-dense": "llama", "gemma3-dense": "gemma3", "mamba-dense": "mamba", "qwen35-dense": "qwen35",
    "glm5-next-moe": "glm5-next", "qwen4exp-moe": "qwen4exp", "glm-dsa-moe": "glm-dsa",
    "deepseek32-moe": "deepseek32", "hy_v4-moe": "hy_v4", "minimax-m3-moe": "minimax-m3",
    "dots3note-moe": "dots3note", "deepseek4-moe": "deepseek4",
}


def _toks(n: int, seed: int):
    return [((i * (7 + 2 * seed) + 13 * seed) % 100) + 10 for i in range(n)]


P = _toks(300, 1)


def _model(name: str) -> str:
    m = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.isfile(m):
        pytest.skip(f"{m} not found (set LLAMA_TEST_MODELS_DIR or run the generate-models ctest)")
    return m


def _server(name: str, cache: str | None, log_path: str | None = None, node_prompt: str | None = "off",
            ctk: str | None = None, max_count=None, interval=None, incr: bool = True) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = _model(name)
    s.model_alias = "dummy"
    s.override_kv = [f"{ARCH[name]}.context_length=int:4096"]
    s.n_ctx = 2048
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.log_path = log_path
    s.ctk = ctk
    # no in-memory context checkpoints: what the slot keeps after a divergence is then only what the class rewinds
    s.ctx_checkpoints = 0
    if cache is None:
        return s
    s.slot_save_path = cache
    s.slot_save_auto = True
    s.slot_save_incremental = incr
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_context_min_tokens = 100000
    s.slot_save_idle_seconds = -1
    s.slot_save_node_prompt = node_prompt
    s.slot_save_max_count = max_count
    s.slot_save_prefill_interval = interval
    return s


def _req(s, prompt, n_predict: int = 8):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": True,
        "id_slot": 0, "return_tokens": True, "n_probs": 8, "post_sampling_probs": False,
    })
    assert res.status_code == 200, res.body
    return res.body


def _dist(body):
    return [{t["id"]: t["logprob"] for t in p["top_logprobs"]} for p in body["completion_probabilities"]]


def _assert_equals_cold(name: str, prompt, body, n_predict: int = 8):
    s = _server(name, None)
    s.start()
    cold = _req(s, prompt, n_predict)
    s.stop()
    assert body["tokens"] == cold["tokens"], f"restored {body['tokens']} vs cold {cold['tokens']}"
    worst = 0.0
    for dr, dc in zip(_dist(body), _dist(cold)):
        for k, v in dr.items():
            assert k in dc, f"token {k} in the restored top-8 but not in the cold one"
            worst = max(worst, abs(v - dc[k]))
    assert worst < TOL, f"restored logprobs differ from cold by {worst}"


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _settled(s):
    deadline = time.time() + 30
    while _metric(s, "auto_cache_save_queue_depth") > 0 and time.time() < deadline:
        time.sleep(0.05)
    assert _metric(s, "auto_cache_save_queue_depth") == 0


def _units(cache):
    return sorted(int(os.path.basename(m)[:-len(".bin.meta")].split("-")[-1])
                  for m in glob.glob(os.path.join(cache, "auto-*.bin.meta")))


def _log(path: str) -> str:
    with open(path) as f:
        return f.read()


def _kill(s):
    server_instances.discard(s)
    s.process.send_signal(signal.SIGKILL)
    s.process.wait()
    s.process = None
    s._log.close()


@pytest.fixture(autouse=True)
def clean_root():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    yield
    shutil.rmtree(ROOT, ignore_errors=True)


def _cache(label: str = "c") -> str:
    d = os.path.join(ROOT, label)
    os.makedirs(d, exist_ok=True)
    return d


# --- restore mode 2 and the usable-reuse margin gate --------------------------------------------------------

@pytest.mark.parametrize("name", ["qwen35-dense", "mamba-dense", "llama-dense", "gemma3-dense"])
def test_divergence_inside_the_slot_restores_the_side_state_only(name, tmp_path):
    """A cold prompt writes its prompt node (block-aligned at 288 for this raw 300-token prompt), the response
    follows in the slot. A request on the same instance that diverges at 290, inside the slot's conversation:
     - hybrid and recurrent: the slot cannot rewind to 290 and has no in-memory checkpoint, so it can keep
       nothing; the node at 288 beats that (before the margin fix it was refused against the raw 290), and only
       its side-state is loaded over the slot's own cells (mode 2: cache_disk_mode "side");
     - plain attention: the slot rewinds to 290 itself and nothing is restored;
     - sliding window (gemma3, n_swa 32): no side-state-only load (the window is not a recurrent fold); the
       slot keeps 290 when its window still covers it, otherwise the node restores whole.
    Every output equals a cold prefill."""
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"), node_prompt="cold")
    s.start()
    _req(s, P)
    _settled(s)
    assert 288 in _units(cache), _units(cache)
    req = P[:290] + _toks(30, 4)
    body = _req(s, req)
    _settled(s)
    side = _metric(s, "auto_cache_restore_side_only_total")
    s.stop()
    t = body["timings"]
    if name in ("qwen35-dense", "mamba-dense"):
        assert side == 1
        assert t.get("cache_disk_n") == 288 and t["cache_source"] == "disk", t
        assert t["cache_disk_mode"] == "side" and t["cache_disk_unit_n"] == 288 and t["cache_disk_nodes"] == 1, t
        assert t["prompt_n"] == len(req) - 288, t
        assert "side-state only" in _log(str(tmp_path / "s.log"))
    elif name == "llama-dense":
        assert side == 0
        assert t.get("cache_disk_n", 0) == 0 and t["cache_source"] == "warm", t
        assert t["cache_n"] >= 288, t
    else:
        assert side == 0
        assert t["cache_n"] >= 288, t
        assert t["cache_source"] == "warm" or (t["cache_source"] == "disk" and t["cache_disk_mode"] == "whole"), t
    _assert_equals_cold(name, req, body)


def test_side_only_restore_keeps_working_across_turns(tmp_path):
    """Mode 2 leaves a normal slot behind: after it, the conversation continues warm, and a second divergence
    restores a side-state again; every turn equals a cold prefill."""
    name = "qwen35-dense"
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"), node_prompt="on")
    s.start()
    _req(s, P)
    _settled(s)
    t1 = P[:290] + _toks(40, 4)
    b1 = _req(s, t1)
    _settled(s)
    t2 = t1 + b1["tokens"][:-1] + _toks(30, 5)
    b2 = _req(s, t2)
    _settled(s)
    assert b2["timings"]["cache_source"] == "warm", b2["timings"]
    # diverge inside the second turn, after its own prompt node
    t3 = t1[:320] + _toks(20, 6)
    b3 = _req(s, t3)
    side = _metric(s, "auto_cache_restore_side_only_total")
    s.stop()
    assert side == 2, side
    assert b3["timings"]["cache_disk_mode"] == "side", b3["timings"]
    for prompt, body in ((t1, b1), (t2, b2), (t3, b3)):
        _assert_equals_cold(name, prompt, body)


# --- per-request cache source --------------------------------------------------------------------------------

def test_timings_cache_source(tmp_path):
    """cold, then warm, then (after a restart) disk with the restored unit's length and chain length."""
    name = "llama-dense"
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"), interval=64)
    s.start()
    t = _req(s, P)["timings"]
    assert t["cache_source"] == "cold" and t["cache_n"] == 0, t
    assert "cache_disk_unit_n" not in t
    t = _req(s, P + _toks(20, 2))["timings"]
    assert t["cache_source"] == "warm", t
    _settled(s)
    _kill(s)
    # the prefill checkpoints at 64..256 form one root and three deltas
    r = _server(name, cache, str(tmp_path / "r.log"))
    r.start()
    t = _req(r, P[:260] + _toks(10, 3))["timings"]
    r.stop()
    assert t["cache_source"] == "disk" and t["cache_disk_n"] == 256, t
    assert t["cache_disk_unit_n"] == 256 and t["cache_disk_nodes"] == 4 and t["cache_disk_mode"] == "whole", t


# --- logits sidecar on every class -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "deepseek4-moe", "glm-dsa-moe"])
def test_exact_resend_prefills_nothing_on_every_class(name, tmp_path):
    """A conversation saved at shutdown and resent exactly after a restart emits its first token from the logits
    sidecar on every class (it used to cover only FULL and RS): nothing is prefilled, and the output equals a
    cold prefill."""
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"))
    s.start()
    seed = _req(s, P)
    s.stop()   # the shutdown save: P plus the response but its last token
    saved = P + seed["tokens"][:-1]
    assert len(saved) in _units(cache), _units(cache)
    assert os.path.exists(glob.glob(os.path.join(cache, f"auto-*-{len(saved)}.bin"))[0] + ".logits")
    r = _server(name, cache, str(tmp_path / "r.log"))
    r.start()
    body = _req(r, saved)
    r.stop()
    t = body["timings"]
    assert t["prompt_n"] == 0 and t.get("cache_disk_n") == len(saved), t
    assert body["tokens"][0] == seed["tokens"][-1]
    _assert_equals_cold(name, saved, body)


# --- identity misses and the eviction bound -------------------------------------------------------------------

def test_a_unit_under_another_identity_is_a_counted_miss(tmp_path):
    """A unit written with f16 KV is never restored by an instance running q8_0 KV (another identity): that miss
    is counted and logged with the differing field. Positive control: an instance with the writer's settings
    restores it."""
    name = "llama-dense"
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"))
    s.start()
    _req(s, P)
    s.stop()
    q = _server(name, cache, str(tmp_path / "q.log"), ctk="q8_0")
    q.start()
    t = _req(q, P + _toks(10, 2))["timings"]
    ident = _metric(q, "auto_cache_restore_miss_identity_total")
    q.stop()
    assert t.get("cache_disk_n", 0) == 0, t
    assert ident == 1
    assert "written under another identity (KV cache types)" in _log(str(tmp_path / "q.log"))
    c = _server(name, cache, str(tmp_path / "c.log"))
    c.start()
    t = _req(c, P + _toks(10, 2))["timings"]
    ident = _metric(c, "auto_cache_restore_miss_identity_total")
    c.stop()
    assert t.get("cache_disk_n", 0) >= 288 and ident == 0, t


def test_a_cap_the_store_cannot_meet_is_counted(tmp_path):
    """--slot-save-max-count 1 with a delta chain: the newest node cannot be evicted (just written) and its parent
    has a live child, so the pass leaves the store above the cap, and says so in a counter."""
    name = "llama-dense"
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"), max_count=1, interval=64)
    s.start()
    _req(s, P[:200])
    _settled(s)
    n = _metric(s, "auto_cache_evict_bound_exceeded_total")
    s.stop()
    assert n >= 1


# --- the class table -------------------------------------------------------------------------------------------
# seq_rm, n_swa, rewinds, side_only_restore, delta probe (1 deltas, 2 whole roots only), per memory class
CLASS_TABLE = {
    "llama-dense":    ("part", 0,  True,  False, 1),
    "gemma3-dense":   ("part", 32, False, False, 1),
    "mamba-dense":    ("full", 0,  False, True,  2),
    "qwen35-dense":   ("full", 0,  False, True,  1),
    "glm5-next-moe":  ("full", 0,  False, False, 1),
    "qwen4exp-moe":   ("full", 0,  False, False, 1),
    "glm-dsa-moe":    ("part", 0,  True,  False, 1),
    "deepseek32-moe": ("part", 0,  True,  False, 1),
    "hy_v4-moe":      ("part", 0,  True,  False, 1),
    "minimax-m3-moe": ("part", 0,  True,  False, 1),
    "dots3note-moe":  ("part", 32, False, False, 1),
    "deepseek4-moe":  ("full", 0,  False, False, 1),
}


@pytest.mark.parametrize("name", list(CLASS_TABLE))
def test_props_reports_the_memory_class(name, tmp_path):
    """/props carries the cache's view of the memory class, and the delta probe (a gauge) is settled by the first
    save. A class that silently changes what it can do fails here."""
    seq_rm, n_swa, rewinds, side_only, delta = CLASS_TABLE[name]
    cache = _cache()
    s = _server(name, cache, str(tmp_path / "s.log"), interval=64)
    s.start()
    props = s.make_request("GET", "/props").body
    assert _metric(s, "auto_cache_delta_capable") == 0
    _req(s, P[:200], 2)
    _settled(s)
    probed = _metric(s, "auto_cache_delta_capable")
    s.stop()
    caps = props["auto_cache"]
    got = (caps["seq_rm"], caps["n_swa"], caps["rewinds"], caps["side_only_restore"], int(probed))
    assert got == (seq_rm, n_swa, rewinds, side_only, delta), got
    assert caps["logits_sidecar"] is True and caps["prefill_interval"] == 64 and caps["block"] == B


# --- decision tasks (a shared prompt prefix) -------------------------------------------------------------------

def test_shared_prefix_tasks_are_reported_not_silently_skipped(tmp_path):
    """A decision request (/v1/systemone) runs its variants as a parent and children sharing a prompt prefix.
    The cache neither restores nor saves such tasks; that is now counted, with a rate-limited WRN."""
    s = ServerPreset.tinyopenjev()   # its questions run as separate tasks, grouped on a shared prefix
    cache = _cache()
    s.slot_save_path = cache
    s.slot_save_auto = True
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_save_idle_seconds = -1
    s.server_metrics = True
    s.log_path = str(tmp_path / "s.log")
    s.start()
    res = s.make_request("POST", "/v1/systemone", data={
        "state": "I was charged twice for my order last week and nobody has replied.",
        "questions": {
            "route": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["can wait", "this week", "today", "right now"]},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        },
    })
    assert res.status_code == 200, res.body
    n = _metric(s, "auto_cache_skipped_shared_total")
    s.stop()
    assert n >= 1
    assert "a task with a shared prompt prefix" in _log(str(tmp_path / "s.log"))

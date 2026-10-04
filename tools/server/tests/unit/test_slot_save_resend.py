import glob
import os
import shutil
import time

import pytest
from utils import *

# Resend, extend and reclaim after a restart, per memory class (user design 2026-10-03):
#  - truncatable classes (plain attention, and SWA whose snapshot fits one window) must serve a
#    request that is a PREFIX of a saved unit by restoring it and trimming to the request;
#  - non-rewindable classes (RS / hybrid, SWA past one window, FULL) serve it only from a node at
#    or before the divergence; the default prompt node (--slot-save-node-prompt cold) provides one,
#    and with the node off the miss must be reported (WRN + auto_cache_restore_not_prefix_total),
#    never silent;
#  - every class must serve a request that EXTENDS the saved unit (previous response included);
#  - a conversation preempted by a different one on a --parallel 1 server reaches disk;
#  - --slot-save-node-prompt (cold / on) gives the non-rewindable classes a node at the end of the
#    prompt, so the resend and a follow-up whose history diverges at the response hit too;
#  - --slot-save-node-response saves as soon as a response completes, and the defaults do not;
#  - a system-prompt-only request caches the whole system prompt (the system node, default on), and a
#    later system + user request restores exactly that much.
# Every hit is also checked for CORRECTNESS against a cold oracle (a server with no cache directory
# prefilling the same prompt): the greedy tokens must be equal and the top-8 logprobs within TOL. A
# restore that loads a wrong state "successfully" would otherwise pass on prompt_n / cache_n alone.
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
TOL = 2e-3


def _model(name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.exists(path):
        pytest.skip(f"dummy model not found: {path}")
    return path


DEFAULT = None   # node_prompt value meaning "leave the server default"


def _server(model: str, incr: bool = False, node_prompt: str | None = "off", cache: bool = True) -> ServerProcess:
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
    if not cache:
        return s
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = incr
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    # no mid-prefill system node, and no prompt node unless a test asks for one (node_prompt; DEFAULT
    # leaves the server default, cold): these tests isolate the release / reclaim units and the
    # prompt node, which has its own floor (max(block, min-tokens) = B here)
    s.slot_save_context_min_tokens = 100000
    s.slot_save_idle_seconds = 3600
    s.slot_save_node_prompt = node_prompt
    return s


def _req(s: ServerProcess, prompt, extra: dict | None = None):
    data = {
        "prompt": prompt, "n_predict": 8, "temperature": 0, "top_k": 1, "cache_prompt": True,
        "id_slot": 0, "return_tokens": True, "n_probs": 8, "post_sampling_probs": False,
    }
    if extra:
        data.update(extra)
    res = s.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body


def _complete(s: ServerProcess, prompt):
    body = _req(s, prompt)
    t = body["timings"]
    _complete.last = body
    return t["prompt_n"], t["cache_n"], body["tokens"]


def _dist(body):
    return [{t["id"]: t["logprob"] for t in p["top_logprobs"]} for p in body["completion_probabilities"]]


def _assert_equals_cold(model: str, prompt, body, extra: dict | None = None):
    """The oracle: the same prompt prefilled by a server with no cache at all. Greedy tokens equal and
    every top-8 logprob of the restored run within TOL of the cold one."""
    s = _server(model, cache=False)
    s.start()
    cold = _req(s, prompt, extra)
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
    # saves are published by a background writer: read the store once its queue is empty
    deadline = time.time() + 30
    while _metric(s, "auto_cache_save_queue_depth") > 0 and time.time() < deadline:
        time.sleep(0.05)
    assert _metric(s, "auto_cache_save_queue_depth") == 0


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
    body = _complete.last
    s.stop()
    assert prompt_n <= B + 1
    assert cache_n >= len(PROMPT) - B
    # every reused cell came from disk (the slot was empty after the restart), none from a "warm slot"
    assert body["timings"].get("cache_disk_n", 0) == cache_n
    _assert_equals_cold(model, PROMPT, body)


@pytest.mark.parametrize("name", ["gemma3-dense", "qwen35-dense"])
def test_resend_non_rewindable_miss_is_reported(name):
    """Non-rewindable classes with the prompt node off: the same-prompt resend after a restart is a miss
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
    body = _complete.last
    s.stop()
    assert cache_n >= len(PROMPT) + len(gen) - 1 - B
    assert prompt_n <= len(TAIL) + B + 1
    _assert_equals_cold(model, follow, body)


@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "qwen35-dense"])
def test_two_turn_delta_chain_equals_cold(name):
    """Three units chained with incremental deltas (P, P+g+T, P+g+T+g2+T2): the deepest restore composes
    the whole chain and must equal a cold prefill."""
    model = _model(name)
    s = _server(model, True)
    s.start()
    _, _, g1 = _complete(s, PROMPT)
    s.stop()
    p2 = PROMPT + g1 + TAIL
    s = _server(model, True)
    s.start()
    _, _, g2 = _complete(s, p2)
    s.stop()
    p3 = p2 + g2 + TAIL[:20]
    s = _server(model, True)
    s.start()
    prompt_n, cache_n, _ = _complete(s, p3)
    body = _complete.last
    s.stop()
    assert cache_n >= len(p2) + len(g2) - 1 - B
    _assert_equals_cold(model, p3, body)


def test_reclaim_saves_preempted_conversation():
    """--parallel 1 with the default --cache-idle-slots: conversation A must reach disk when a
    different conversation B takes the only slot, before any idle flush could have fired."""
    model = _model("llama-dense")
    other = [((i * 13) % 100) + 10 for i in range(150)]
    s = _server(model)
    s.start()
    _, _, gen_a = _complete(s, PROMPT)
    _settled(s)
    assert _units() == []
    _complete(s, other)
    _settled(s)
    units_after_b = _units()
    s.stop()
    assert len(PROMPT) + len(gen_a) - 1 in units_after_b

    s = _server(model)
    s.start()
    prompt_n, _, _ = _complete(s, PROMPT)
    body = _complete.last
    s.stop()
    assert prompt_n <= B + 1
    _assert_equals_cold(model, PROMPT, body)


@pytest.mark.parametrize("name", ["llama-dense", "qwen35-dense"])
@pytest.mark.parametrize("shared", [0, 120])
def test_reclaim_with_shared_leading_prefix(name, shared):
    """Conversation B takes the only slot from A while sharing a long leading prefix with it (a common
    system prompt, the normal agent / RAG case). The RAM cache's f_keep rule stays >= 0.5 there, but A's
    own tail is still overwritten, so A must reach disk before B prefills. A request that EXTENDS the slot
    must not trigger a reclaim write."""
    model = _model(name)
    sys_p = [((i * 3) % 100) + 10 for i in range(shared)]
    a = sys_p + [((i * 7) % 100) + 10 for i in range(160 - shared)]
    b = sys_p + [((i * 13) % 100) + 10 for i in range(170 - shared)]
    s = _server(model)
    s.start()
    _, _, ga = _complete(s, a)
    a2 = a + ga + TAIL[:8]
    _, _, ga2 = _complete(s, a2)   # extends the slot: nothing is lost, nothing is written
    _settled(s)
    assert _units() == []
    _complete(s, b)
    _settled(s)
    units = _units()
    s.stop()
    assert len(a2) + len(ga2) - 1 in units, f"units after B: {units}"

    follow = a2 + ga2 + TAIL[8:24]
    s = _server(model)
    s.start()
    prompt_n, cache_n, _ = _complete(s, follow)
    body = _complete.last
    s.stop()
    assert cache_n >= len(a2) + len(ga2) - 1 - B
    _assert_equals_cold(model, follow, body)


@pytest.mark.parametrize("mode", ["cold", "on"])
@pytest.mark.parametrize("name", ["gemma3-dense", "qwen35-dense"])
def test_resend_hits_with_prompt_node(name, mode):
    """With the after-user-message trigger, the non-rewindable classes hit on the same resend: the
    prompt node (end of the prompt, block-aligned down) is a whole prefix of the request, and only the
    tail past it is prefilled."""
    model = _model(name)
    s = _server(model, node_prompt=mode)
    s.start()
    _, _, gen = _complete(s, PROMPT)
    s.stop()
    node = (len(PROMPT) - 1) // B * B
    assert node in _units()
    assert len(PROMPT) + len(gen) - 1 in _units()

    s = _server(model, node_prompt=mode)
    s.start()
    prompt_n, cache_n, _ = _complete(s, PROMPT)
    body = _complete.last
    not_prefix = _metric(s, "auto_cache_restore_not_prefix_total")
    s.stop()
    assert cache_n >= node
    assert prompt_n <= B + 1
    assert not_prefix == 0
    _assert_equals_cold(model, PROMPT, body)


@pytest.mark.parametrize("incr", [False, True])
@pytest.mark.parametrize("name", ["gemma3-dense", "qwen35-dense"])
def test_divergent_followup_needs_the_prompt_node(name, incr):
    """A follow-up whose history does not re-render the previous response (here: the response is
    dropped) diverges inside the release unit. With the prompt node off: a reported must-extend miss. With
    --slot-save-node-prompt on, the node at the end of the previous prompt serves it, and the new
    node it writes for the follow-up is a delta under --slot-save-incremental."""
    model = _model(name)
    follow = PROMPT + TAIL

    s = _server(model, incr)
    s.start()
    _complete(s, PROMPT)
    s.stop()
    s = _server(model, incr)
    s.start()
    prompt_n, cache_n, _ = _complete(s, follow)
    not_prefix = _metric(s, "auto_cache_restore_not_prefix_total")
    s.stop()
    assert cache_n == 0 and prompt_n == len(follow)
    assert not_prefix == 1

    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)
    s = _server(model, incr, node_prompt="on")
    s.start()
    _complete(s, PROMPT)
    s.stop()
    s = _server(model, incr, node_prompt="on")
    s.start()
    prompt_n, cache_n, _ = _complete(s, follow)
    body = _complete.last
    _settled(s)
    delta = _metric(s, "auto_cache_save_delta_total")
    s.stop()
    node = (len(PROMPT) - 1) // B * B
    assert cache_n >= node
    _assert_equals_cold(model, follow, body)
    assert prompt_n <= len(follow) - node + 1
    # the follow-up's own prompt node, past the restored one by at least a block
    assert (len(follow) - 1) // B * B in _units()
    if incr:
        assert delta >= 1


@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "qwen35-dense"])
def test_prompt_node_cold_by_default(name):
    """User decision (2026-10-03, round 4): with --slot-save-auto on, the prompt node defaults to 'cold'
    for every model. A cold prompt writes the node at the end of the prompt (block-aligned down) next to
    the release unit, and a same-prompt resend after a restart hits on every class, equal to a cold run.
    --slot-save-node-prompt off writes only the release unit."""
    model = _model(name)
    s = _server(model, node_prompt=DEFAULT)
    s.start()
    _, _, gen = _complete(s, PROMPT)
    s.stop()
    node = (len(PROMPT) - 1) // B * B
    assert _units() == [node, len(PROMPT) + len(gen) - 1]

    s = _server(model, node_prompt=DEFAULT)
    s.start()
    prompt_n, cache_n, _ = _complete(s, PROMPT)
    body = _complete.last
    not_prefix = _metric(s, "auto_cache_restore_not_prefix_total")
    s.stop()
    assert cache_n >= node and prompt_n <= B + 1
    assert not_prefix == 0
    _assert_equals_cold(model, PROMPT, body)

    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)
    s = _server(model, node_prompt="off")
    s.start()
    _, _, gen = _complete(s, PROMPT)
    s.stop()
    assert _units() == [len(PROMPT) + len(gen) - 1]


def test_response_node_saves_without_idle_or_reclaim():
    """--slot-save-node-response publishes the conversation as soon as the response completes; the
    default waits for idle, reclaim or shutdown."""
    model = _model("llama-dense")
    s = _server(model)
    s.start()
    _complete(s, PROMPT)
    _settled(s)
    assert _units() == []
    s.stop()

    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)
    s = _server(model)
    s.slot_save_node_response = True
    s.start()
    _, _, gen = _complete(s, PROMPT)
    _settled(s)
    units = _units()
    s.stop()
    assert units == [len(PROMPT) + len(gen) - 1]


def test_reclaim_save_can_be_disabled():
    """--no-slot-save-on-reclaim: a conversation preempted on a --parallel 1 server is not saved when
    the other one takes the slot (only the slot's last conversation reaches disk, at shutdown)."""
    model = _model("llama-dense")
    other = [((i * 13) % 100) + 10 for i in range(150)]
    s = _server(model)
    s.slot_save_on_reclaim = False
    s.start()
    _complete(s, PROMPT)
    _complete(s, other)
    _settled(s)
    assert _units() == []
    s.stop()
    assert len(_units()) == 1


# --- system-prompt-only pre-cache (the system node, on by default) ---------------------------------
# The test-models dummies have no chat template with a system role, so the request carries the role
# delimiters itself (/completion's "message_delimiters", exactly what the chat path forwards) and the
# prompt is built from their token ids. Body ids are drawn from ids the delimiters do not use.

DELIMS = [
    {"role": "system",    "delimiter": "<|im_start|>system"},
    {"role": "user",      "delimiter": "<|im_start|>user"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant"},
]


def _tok(s: ServerProcess, text: str):
    res = s.make_request("POST", "/tokenize", data={"content": text, "add_special": False, "parse_special": True})
    assert res.status_code == 200
    toks = res.body["tokens"]
    assert len(toks) > 0
    return toks


@pytest.mark.parametrize("incr", [False, True])
@pytest.mark.parametrize("name", ["llama-dense", "gemma3-dense", "qwen35-dense"])
def test_system_only_request_precaches_the_whole_system_prompt(name, incr):
    """User scenario 1: a request carrying only a system prompt (plus the generation prompt) caches the
    whole system prompt, at its exact (unaligned) end. After a restart, a system + user request restores
    exactly that much from disk, on every class, and its output equals a cold prefill."""
    model = _model(name)
    s = _server(model, incr)
    s.slot_save_context_min_tokens = 0   # the system node's floor becomes the block size
    s.start()
    S, U, A = (_tok(s, d["delimiter"]) for d in DELIMS)
    used = set(S) | set(U) | set(A)
    pool = [t for t in range(10, 110) if t not in used]   # the id range PROMPT already uses
    assert len(pool) >= 32
    body_sys = [pool[(i * 7) % len(pool)] for i in range(70)]
    body_usr = [pool[(i * 11) % len(pool)] for i in range(40)]
    nl = [pool[0]]
    sys_only = S + body_sys + A + nl
    sys_end = len(S) + len(body_sys)
    assert sys_end % B != 0, "the test wants an unaligned system end"
    extra = {"message_delimiters": DELIMS}
    _req(s, sys_only, extra)
    s.stop()
    assert sys_end in _units(), f"units: {_units()}"

    follow = S + body_sys + U + body_usr + A + nl
    s = _server(model, incr)
    s.slot_save_context_min_tokens = 0
    s.start()
    body = _req(s, follow, extra)
    s.stop()
    t = body["timings"]
    # non-rewindable classes restore the system node, exactly sys_end; plain attention restores the
    # longer release unit and trims it to the real common prefix, which runs a few tokens past sys_end
    # when the user and assistant delimiters share leading tokens. Either way every reused cell is disk.
    expect = sys_end
    if name == "llama-dense":
        expect = next(i for i, (x, y) in enumerate(zip(sys_only, follow)) if x != y)
        assert expect >= sys_end
    assert t["cache_n"] == expect, t
    assert t.get("cache_disk_n", 0) == expect, t
    assert t["prompt_n"] == len(follow) - expect
    _assert_equals_cold(model, follow, body, extra)

import glob
import os
import time
import shutil

import pytest
from utils import *

# The system-prompt node through the real chat path, template-generic (user decision 2026-10-03,
# round 4: no per-model exceptions). The node sits at the end of the system preamble that the
# leading system messages, the tools and the template decide on their own (common_chat_preamble_end
# + server_preamble_cache), so it fires on templates whose parser marks no system role (Gemma 4,
# DeepSeek, the generic autoparser: Laguna, GLM) as well as on those that do (Qwen3.5 family).
#
# For each template, on a plain-attention and a hybrid (recurrent) dummy:
#  - a system + user request writes one node inside the prompt, whose tokens decode to text that
#    contains the system prompt;
#  - after a restart, conversations with the same system prompt and a different first message
#    (one starting with a newline, which merges with a trailing newline in byte-level BPE) restore
#    that node, and the output equals a cold prefill;
#  - a request carrying only the system prompt (a pre-cache) writes a node that later system + user
#    requests restore, and they do not write a near-duplicate root a few tokens further on;
#  - an assistant prefill (continue_final_message) does not stop the node;
#  - nothing is counted as a probe failure or a seam mismatch.
#
# The dummies' vocabularies have none of these templates' special tokens, so every role marker is
# plain text: the hardest case for the token seam (like Laguna's "<user>" on the real model).

MODELS_DIR = os.environ.get("LLAMA_TEST_MODELS_DIR", "") or os.path.normpath(os.path.join(
    os.path.dirname(os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")), "..", "tests", "test-models"))
TEMPLATES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "templates"))
CACHE_DIR = "./tmp/slot_save_sysnode"
B = 16
TOL = 2e-3

SYSTEM = ("You are a careful assistant for a small engineering team. Keep every answer short, cite the "
          "document you used, say so plainly when you do not know, and never invent numbers or names.")

TEMPLATES = [
    "google-gemma-4-31B-it",          # gemma4 parser, no system span
    "deepseek-ai-DeepSeek-V4",        # deepseek parser, no system span
    "poolside-Laguna-S-2.1",          # generic autoparser, plain-text role headers
    "GLM-4.7-Flash",                  # generic autoparser
    "Qwen3.5-4B",                     # qwen3-coder parser, has a system span
]
MODELS = ["llama-dense", "qwen35-dense", "gemma3-dense"]


def _paths(model: str, tmpl: str):
    m = os.path.join(MODELS_DIR, f"{model}.gguf")
    t = os.path.join(TEMPLATES_DIR, f"{tmpl}.jinja")
    if not os.path.exists(m):
        pytest.skip(f"dummy model not found: {m}")
    if not os.path.exists(t):
        pytest.skip(f"template not found: {t}")
    return m, t


def _server(model: str, tmpl: str, cache: bool = True, node_prompt: str | None = "off") -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    s.jinja = True
    s.chat_template_file = tmpl
    s.n_ctx = 1024
    s.n_batch = 1024
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    if not cache:
        return s
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_context_min_tokens = 0     # the system node's floor becomes the block size
    s.slot_save_idle_seconds = 3600
    s.slot_save_node_prompt = node_prompt  # "off": only the system node and the release units; None: the default
    return s


def _chat(s: ServerProcess, messages, extra: dict | None = None):
    data = {"messages": messages, "max_tokens": 4, "temperature": 0, "top_k": 1, "id_slot": 0,
            "logprobs": True, "top_logprobs": 8}
    if extra:
        data.update(extra)
    res = s.make_request("POST", "/chat/completions", data=data)
    assert res.status_code == 200, res.body
    return res.body


def _units():
    return sorted(int(os.path.basename(p)[:-len(".bin.meta")].split("-")[-1])
                  for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


def _unit_files():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


def _unit_len(f: str) -> int:
    return int(f[:-len(".bin.meta")].split("-")[-1])


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _assert_sysnode_counters(s):
    # the callers stop the server right after this (a kill, no shutdown flush): wait until what was captured
    # is on disk, including a node whose positional copy was deferred to the idle loop
    deadline = time.time() + 30
    while time.time() < deadline and (_metric(s, "auto_cache_save_queue_depth") != 0 or
                                      _metric(s, "auto_cache_save_deferred_pending") != 0):
        time.sleep(0.05)
    assert _metric(s, "auto_cache_sysnode_probed_total") >= 1
    assert _metric(s, "auto_cache_sysnode_probe_failed_total") == 0
    assert _metric(s, "auto_cache_sysnode_seam_mismatch_total") == 0


def _prompt_tokens(s: ServerProcess, messages):
    res = s.make_request("POST", "/apply-template", data={"messages": messages})
    assert res.status_code == 200
    res = s.make_request("POST", "/tokenize", data={"content": res.body["prompt"], "add_special": True, "parse_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def _lcp(a, b) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def _assert_node_brackets_system(model, tmpl, X):
    """The node lies past the whole system prompt and before the user's text, checked on tokens (the
    dummies' vocabularies do not detokenize to readable text): it is longer than the common prefix of
    two prompts whose system prompts differ only in their last word, and no longer than the common
    prefix of two prompts whose user messages differ from their first character."""
    s = _server(model, tmpl, cache=False)
    s.start()
    a = _prompt_tokens(s, _sys("What is the capital of France?"))
    b = _prompt_tokens(s, [{"role": "system", "content": SYSTEM[:-6] + "places."},
                           {"role": "user", "content": "What is the capital of France?"}])
    c = _prompt_tokens(s, _sys("Zebras are striped."))
    s.stop()
    assert _lcp(a, b) < X <= _lcp(a, c), f"node {X}, system ends after {_lcp(a, b)}, user starts by {_lcp(a, c)}"


def _dist(body):
    return [{t["token"]: t["logprob"] for t in c["top_logprobs"]} for c in body["choices"][0]["logprobs"]["content"]]


def _assert_equals_cold(model, tmpl, messages, body):
    s = _server(model, tmpl, cache=False)
    s.start()
    cold = _chat(s, messages)
    s.stop()
    assert body["choices"][0]["message"]["content"] == cold["choices"][0]["message"]["content"]
    worst = 0.0
    for dr, dc in zip(_dist(body), _dist(cold)):
        for k, v in dr.items():
            if k in dc:
                worst = max(worst, abs(v - dc[k]))
    assert worst < TOL, f"restored logprobs differ from cold by {worst}"


@pytest.fixture(autouse=True)
def clean_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _sys(user: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@pytest.mark.parametrize("node_prompt", ["off", None])
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("tmpl_name", TEMPLATES)
def test_system_node_restores_for_other_conversations(tmpl_name, model_name, node_prompt):
    """node_prompt None is the server default (cold): the prompt node is written too, at least one
    block past the system node, and the system node still serves the other conversations."""
    model, tmpl = _paths(model_name, tmpl_name)
    first = _sys("What is the capital of France?")
    s = _server(model, tmpl, node_prompt=node_prompt)
    s.start()
    body = _chat(s, first)
    n_prompt = body["timings"]["prompt_n"]
    _assert_sysnode_counters(s)
    s.stop()
    sys_units = [u for u in _units() if u < n_prompt]
    if node_prompt is None and len(sys_units) == 2:
        assert sys_units[1] >= sys_units[0] + B, f"prompt node inside the system node's block: {sys_units}"
        sys_units = sys_units[:1]
    assert len(sys_units) == 1, f"units {_units()}, prompt {n_prompt}"
    X = sys_units[0]
    assert B <= X < n_prompt
    _assert_node_brackets_system(model, tmpl, X)

    for user in ["\nA question that starts with a newline", "Hello there, how are you?"]:
        msgs = _sys(user)
        before = set(_units())
        s = _server(model, tmpl, node_prompt=node_prompt)
        s.start()
        body = _chat(s, msgs)
        _assert_sysnode_counters(s)
        s.stop()
        t = body["timings"]
        if model_name == "qwen35-dense":
            # cannot rewind: exactly the node
            assert t.get("cache_disk_n", 0) == X, t
        else:
            # plain attention may restore a longer unit trimmed to the real common prefix
            assert t.get("cache_disk_n", 0) >= X, t
        # the restored node is reused, not re-written a few tokens further on: this run wrote only its
        # release unit (prompt + generation), nothing inside the prompt
        assert not [u for u in set(_units()) - before if u < t["prompt_n"] + t.get("cache_n", 0)], (_units(), before)
        _assert_equals_cold(model, tmpl, msgs, body)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("tmpl_name", TEMPLATES)
def test_system_only_precache(tmpl_name, model_name):
    model, tmpl = _paths(model_name, tmpl_name)
    s = _server(model, tmpl)
    s.start()
    res = s.make_request("POST", "/chat/completions", data={
        "messages": [{"role": "system", "content": SYSTEM}], "max_tokens": 4, "temperature": 0, "id_slot": 0})
    if res.status_code != 200:
        s.stop()
        # e.g. Qwen3.5 raises "No user query found in messages": nothing to pre-cache with
        pytest.skip(f"{tmpl_name} does not render a system-only request: {res.body}")
    body = res.body
    n_prompt = body["timings"]["prompt_n"]
    _assert_sysnode_counters(s)
    s.stop()
    sys_units = [u for u in _units() if u < n_prompt]
    assert len(sys_units) == 1, f"units {_units()}, prompt {n_prompt}"
    Xs = sys_units[0]

    _assert_node_brackets_system(model, tmpl, Xs)
    msgs = _sys("Summarise the attached report in two lines.")
    before = set(_units())
    s = _server(model, tmpl)
    s.start()
    body = _chat(s, msgs)
    s.stop()
    t = body["timings"]
    if model_name == "qwen35-dense":
        assert t.get("cache_disk_n", 0) == Xs, t
    else:
        assert t.get("cache_disk_n", 0) >= Xs, t
    # dedup: the system + user boundary is a few tokens past Xs, not worth a second root; this run
    # wrote only its release unit
    assert not [u for u in set(_units()) - before if u < t["prompt_n"] + t.get("cache_n", 0)], (_units(), before)
    _assert_equals_cold(model, tmpl, msgs, body)


@pytest.mark.parametrize("tmpl_name", TEMPLATES)
def test_system_node_survives_prefill(tmpl_name):
    """Probes are rendered without the request's continuation: an assistant prefill still gets the
    node, at the same position as a plain request."""
    model, tmpl = _paths("qwen35-dense", tmpl_name)
    s = _server(model, tmpl)
    s.start()
    n_plain = _chat(s, _sys("Plain question?"))["timings"]["prompt_n"]
    s.stop()
    plain = [u for u in _units() if u < n_plain]
    assert len(plain) == 1, _units()

    # (a response_format schema is covered by test-chat-preamble: the dummies' vocabularies cannot
    # build a JSON grammar)
    for extra, msgs in [
        (None, _sys("Continue this sentence.") + [{"role": "assistant", "content": "Sure, the"}]),
    ]:
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        s = _server(model, tmpl)
        s.start()
        res = s.make_request("POST", "/chat/completions", data={
            "messages": msgs, "max_tokens": 2, "temperature": 0, "id_slot": 0, **(extra or {})})
        probed = _metric(s, "auto_cache_sysnode_probed_total")
        failed = _metric(s, "auto_cache_sysnode_probe_failed_total")
        s.stop()
        if res.status_code != 200:
            pytest.skip(f"{tmpl_name} does not serve this request shape: {res.body}")
        assert probed >= 1 and failed == 0
        n = res.body["timings"]["prompt_n"]
        assert [u for u in _units() if u < n] == plain, (extra, _units(), plain)


# Same-length system prompts that differ only after their last whole block (a date at the end: the same
# length every day, a different tail). Each must get its own node and restore it: the node is named and
# deduplicated by its identity over every token, not by the hash of its last whole block. On a model that
# cannot rewind (qwen35-dense) the other day's node is useless, so a suppressed node is a full miss.
DAYS = ["Monday.", "Friday.", "Sunday."]


@pytest.mark.parametrize("model_name", ["qwen35-dense", "gemma3-dense"])
@pytest.mark.parametrize("tmpl_name", ["GLM-4.7-Flash", "Qwen3.5-4B", "google-gemma-4-31B-it"])
def test_same_length_system_prompts_differing_in_the_tail(tmpl_name, model_name):
    model, tmpl = _paths(model_name, tmpl_name)
    nodes = {}
    for d in DAYS:
        msgs = [{"role": "system", "content": SYSTEM + " Today is " + d},
                {"role": "user", "content": "What is the capital of France?"}]
        before = set(_unit_files())
        s = _server(model, tmpl)
        s.start()
        body = _chat(s, msgs)
        s.stop()
        n = body["timings"]["prompt_n"] + body["timings"].get("cache_n", 0)
        new_inside = sorted(_unit_len(f) for f in set(_unit_files()) - before if _unit_len(f) < n)
        assert len(new_inside) == 1, f"{d}: new units inside the prompt {new_inside}, all {_units()}"
        nodes[d] = new_inside[0]
    assert len(set(nodes.values())) == 1, f"the days' system prompts are not the same length: {nodes}"
    for d in DAYS:
        msgs = [{"role": "system", "content": SYSTEM + " Today is " + d},
                {"role": "user", "content": "Another question here?"}]
        s = _server(model, tmpl)
        s.start()
        body = _chat(s, msgs)
        s.stop()
        t = body["timings"]
        if model_name == "qwen35-dense":
            assert t.get("cache_disk_n", 0) == nodes[d], (d, t)
        else:
            assert t.get("cache_disk_n", 0) >= nodes[d], (d, t)
        _assert_equals_cold(model, tmpl, msgs, body)


# With the default prompt node a short first message ends inside the system node's last block. The prompt
# node must then not be armed: it would be a strict prefix of the system node that no request can use, and a
# second synchronous save before the first token of every cold first prompt.
@pytest.mark.parametrize("user", ["Hi", "Ok", "Hi there", "Hello, how are you today?"])
@pytest.mark.parametrize("tmpl_name", TEMPLATES)
def test_prompt_node_not_inside_system_node(tmpl_name, user):
    model, tmpl = _paths("qwen35-dense", tmpl_name)
    s = _server(model, tmpl, node_prompt=None)
    s.start()
    body = _chat(s, _sys(user))
    s.stop()
    n = body["timings"]["prompt_n"]
    inside = sorted(u for u in _units() if u < n)
    assert inside, _units()
    assert all(b >= a + B for a, b in zip(inside, inside[1:])), f"prompt node within a block of the system node: {inside}"
    _assert_node_brackets_system(model, tmpl, inside[0])


# The token seam on a REAL byte-level BPE vocabulary with real special tokens (the dummies have none of
# the templates' special tokens, and the ggml-vocab files in this checkout are LFS placeholders). Point
# LLAMA_TEST_REAL_VOCAB_MODEL at a small chat GGUF (on the rig: the Qwen3-4B behind tasks-tiny, run on
# CPU with 4 threads); skipped without it. With the model's own template the role markers are special
# tokens (the is_special branch of the seam); with the others they are plain text in this vocabulary,
# which byte-level BPE can merge with the message (the re-tokenisation branch).
REAL_MODEL = os.environ.get("LLAMA_TEST_REAL_VOCAB_MODEL", "")
REAL_TOL = 1e-4


@pytest.mark.parametrize("tmpl_name", [None] + TEMPLATES)
def test_system_node_real_vocab(tmpl_name):
    if not REAL_MODEL or not os.path.isfile(REAL_MODEL):
        pytest.skip("no real-vocabulary model (set LLAMA_TEST_REAL_VOCAB_MODEL)")
    tmpl = None
    if tmpl_name is not None:
        tmpl = os.path.join(TEMPLATES_DIR, f"{tmpl_name}.jinja")
        if not os.path.exists(tmpl):
            pytest.skip(f"template not found: {tmpl}")

    def srv(cache=True):
        s = _server(REAL_MODEL, tmpl, cache=cache)
        s.n_ctx = 2048
        s.n_batch = 512
        s.n_threads = 4
        return s

    s = srv()
    s.start()
    body = _chat(s, _sys("What is the capital of France?"))
    n_prompt = body["timings"]["prompt_n"]
    _assert_sysnode_counters(s)
    s.stop()
    inside = [u for u in _units() if u < n_prompt]
    assert len(inside) == 1, f"units {_units()}, prompt {n_prompt}"
    X = inside[0]

    # the node lies past the whole system prompt and before the user's text
    s = srv(cache=False)
    s.start()
    a = _prompt_tokens(s, _sys("What is the capital of France?"))
    b = _prompt_tokens(s, [{"role": "system", "content": SYSTEM[:-6] + "places."},
                           {"role": "user", "content": "What is the capital of France?"}])
    s.stop()
    assert _lcp(a, b) < X <= n_prompt

    def first_probs(s, prompt):
        # /completion on the rendered prompt: the restore goes by tokens, and the first token's
        # distribution is reported whatever the template's response parser would make of the text
        res = s.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 1, "n_probs": 8,
                                                          "temperature": 0, "id_slot": 0, "cache_prompt": True,
                                                          "post_sampling_probs": False})
        assert res.status_code == 200, res.body
        return res.body, {t["id"]: t["logprob"] for t in res.body["completion_probabilities"][0]["top_logprobs"]}

    for user in ["\nA question that starts with a newline", "Hello there, how are you?", " leading space"]:
        msgs = _sys(user)
        s = srv(cache=False)
        s.start()
        res = s.make_request("POST", "/apply-template", data={"messages": msgs})
        assert res.status_code == 200
        prompt = res.body["prompt"]
        c = _prompt_tokens(s, msgs)
        s.stop()
        s = srv()
        s.start()
        body, dr = first_probs(s, prompt)
        s.stop()
        assert body["timings"].get("cache_disk_n", 0) >= X, (user, body["timings"])
        # the oracle: a process with no disk cache that prefills the same split in memory, [0, k) then
        # [k, N), where k is what the restore reused. A cold one-batch prefill is NOT a valid oracle on
        # this quantised model: the split alone moves the first token's top-8 logprobs by up to 0.58 and
        # changes which tokens are in the top 8, measured with no disk cache involved at all
        k = body["timings"]["cache_n"]
        s = srv(cache=False)
        s.start()
        res = s.make_request("POST", "/completion", data={"prompt": c[:k], "n_predict": 0, "id_slot": 0,
                                                          "cache_prompt": True})
        assert res.status_code == 200, res.body
        ctl, dc = first_probs(s, prompt)
        s.stop()
        assert X <= _lcp(a, c), f"node {X} past the point where {user!r} diverges ({_lcp(a, c)})"
        assert ctl["timings"]["cache_n"] == k, (ctl["timings"], k)
        worst = max(abs(dr[k_] - dc[k_]) if k_ in dr else float("inf") for k_ in dc)
        print(f"{tmpl_name} {user!r}: node {X}, reused {k}, restored vs same-split control differ by {worst:.3g}")
        assert worst < REAL_TOL, f"restored state differs from the same-split control by {worst}"

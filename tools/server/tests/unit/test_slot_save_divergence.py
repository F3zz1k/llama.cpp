import glob
import os
import shutil

import pytest
from utils import *

# Multi-turn chats whose client does NOT re-render the previous response token for token, on the memory
# classes that cannot rewind a restored unit (RS / hybrid, hybrid with an indexer), with the conversation
# forced out of the slot between turns (graceful restart: the shutdown flush saves it).
#
# The Qwen3.5 template drops the previous turns' reasoning from the history, so every follow-up diverges
# inside the previous response (and the dummies' random tokens do not re-tokenise identically either).
# The unit saved after a response (prompt + every generated token but the last) is then never a prefix of
# the next turn. What a turn can restore after a restart:
#   - default (--slot-save-node-prompt cold, a node only for a prompt that got no reuse): the turn-1 prompt
#     node, so the re-prefill grows with the conversation. That is by design; such clients should run with
#     'on'. Before the lookup fix not even that survived: from turn 5 the four earlier turns' units, which
#     are shorter than the request but diverge inside it, filled AUTO_MAX_RESTORE_ATTEMPTS first and the turn
#     prefilled cold. The lookup now skips a unit met below its deepest boundary before it takes a place;
#   - 'on': the previous turn's prompt node, so the re-prefill is one response plus one user message, flat
#     over the turns.
# A unit whose response stays inside its trailing partial block (a short answer) is met by the next turn at
# its DEEPEST boundary, so the boundary rule alone cannot tell that the request leaves it; the lookup also
# compares the unit's identity, which commits every cell, with the request's chain at the same length.
# Plain attention (llama-dense) is the control: it restores the longer unit and trims it, and gets no
# prompt node on any turn after the first. A client that EXTENDS the conversation (token ids, the previous
# response included) gets none either.

def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
TEMPLATES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "templates"))
CACHE_DIR = "./tmp/slot_save_divergence"
B = 16
TURNS = 6
N_GEN = 24
ARCH = {"qwen35-dense": "qwen35", "qwen4exp-moe": "qwen4exp", "llama-dense": "llama"}

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
SYSTEM = "You are a careful assistant. " + " ".join(WORDS[i % len(WORDS)] for i in range(120)) + "."
USERS = [f"Question {i}: list three facts about the number {7 * i + 3}." for i in range(1, TURNS + 1)]


def _paths(name: str):
    m = os.path.join(MODELS_DIR, f"{name}.gguf")
    t = os.path.join(TEMPLATES_DIR, "Qwen3.5-4B.jinja")
    if not os.path.exists(m):
        pytest.skip(f"dummy model not found: {m}")
    if not os.path.exists(t):
        pytest.skip(f"template not found: {t}")
    return m, t


def _server(name: str, node_system: bool = False, node_prompt: str | None = None, cache: bool = True) -> ServerProcess:
    m, t = _paths(name)
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = m
    s.model_alias = "dummy"
    s.jinja = True
    s.chat_template_file = t
    s.override_kv = [f"{ARCH[name]}.context_length=int:4096"]
    s.n_ctx = 4096
    s.n_batch = 1024
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.cache_ram = 0                        # the conversation must come back from disk, not from RAM
    if not cache:
        return s
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = True
    s.slot_save_block = B
    s.slot_save_min_tokens = 0
    s.slot_restore_min_tokens = 0
    s.slot_save_context_min_tokens = 0
    s.slot_save_idle_seconds = 3600
    s.slot_save_node_system = node_system
    s.slot_save_node_prompt = node_prompt  # None: the server default (cold)
    return s


@pytest.fixture(autouse=True)
def _clean():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _units():
    return sorted(int(os.path.basename(p)[:-len(".bin.meta")].split("-")[-1])
                  for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _chat(s, messages):
    res = s.make_request("POST", "/chat/completions", data={
        "messages": messages, "max_tokens": N_GEN, "temperature": 0, "top_k": 1, "id_slot": 0,
        "logprobs": True, "top_logprobs": 8})
    assert res.status_code == 200, res.body
    return res.body


def _conversation(name: str, **kw):
    """Run TURNS turns with a graceful restart between them; return per-turn (n_prompt, cache_n,
    cache_disk_n, prompt_n) and the store after each turn."""
    rows, stores = [], []
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in range(TURNS):
        messages.append({"role": "user", "content": USERS[turn]})
        s = _server(name, **kw)
        s.start()
        body = _chat(s, messages)
        t = body["timings"]
        rows.append((t["prompt_n"] + t["cache_n"], t["cache_n"], t.get("cache_disk_n", 0), t["prompt_n"]))
        msg = body["choices"][0]["message"]
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            am["reasoning_content"] = msg["reasoning_content"]
        messages.append(am)
        s.stop()  # graceful: the shutdown flush saves the conversation
        stores.append(_units())
    return rows, stores, messages


@pytest.mark.parametrize("name", ["qwen35-dense", "qwen4exp-moe"])
def test_default_restores_the_deepest_usable_node_every_turn(name):
    """Default (cold): only turn 1 writes a prompt node, and every later turn restores it from disk. From the
    fifth turn the earlier turns' divergent units used to exhaust AUTO_MAX_RESTORE_ATTEMPTS and the turn
    prefilled cold."""
    rows, stores, _ = _conversation(name)
    node = min(stores[0])  # turn 1: the cold prompt node and the conversation saved at shutdown
    assert len(stores[0]) == 2, stores
    for turn in range(1, TURNS):
        n_prompt, cache_n, disk_n, prompt_n = rows[turn]
        assert disk_n == cache_n == node, f"turn {turn + 1}: {rows[turn]} (turn-1 node {node}), {rows}"
        assert prompt_n == n_prompt - node, f"turn {turn + 1}: {rows}"
    # 'cold' writes no node on a turn that restored one: one unit per turn (the shutdown save), nothing more
    growth = [len(stores[i + 1]) - len(stores[i]) for i in range(len(stores) - 1)]
    assert all(g == 1 for g in growth), (growth, stores)


@pytest.mark.parametrize("name", ["qwen35-dense", "qwen4exp-moe"])
def test_node_prompt_on_restores_the_previous_prompt_node(name):
    """--slot-save-node-prompt on: every turn writes a node at the end of its user message, and the next
    turn restores it, so the re-prefill stays flat over the turns."""
    rows, stores, _ = _conversation(name, node_prompt="on")
    for turn in range(1, TURNS):
        n_prompt, cache_n, disk_n, prompt_n = rows[turn]
        prev_prompt = rows[turn - 1][0]
        # the previous turn's prompt node sits at the end of its last user message, exactly: only the
        # generation prompt, the response and the new user message are prefilled again
        assert disk_n == cache_n, f"turn {turn + 1}: {rows[turn]}"
        assert cache_n >= prev_prompt - B, f"turn {turn + 1} restored {cache_n}, previous prompt {prev_prompt}: {rows}"
        assert prompt_n <= n_prompt - prev_prompt + B, f"turn {turn + 1}: {rows}"
    # flat, not growing with the turn number, and never a cold prefill (the pre-fix turn 5)
    assert max(r[3] for r in rows[1:]) <= min(r[3] for r in rows[1:]) + 2 * B, rows


def test_extending_client_writes_no_prompt_node():
    """A client that re-sends the conversation token for token (the previous response included) extends
    the saved unit: every turn restores all of it and no prompt node is armed after turn 1."""
    _paths("qwen35-dense")
    prompt = [((i * 7) % 100) + 10 for i in range(150)]
    s = _server("qwen35-dense")
    s.jinja = False
    s.chat_template_file = None
    n_units = []
    prev_len = 0
    for turn in range(4):
        s.start()
        res = s.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 8, "temperature": 0, "top_k": 1, "cache_prompt": True,
            "id_slot": 0, "return_tokens": True})
        assert res.status_code == 200, res.body
        t = res.body["timings"]
        if turn > 0:
            # the unit saved at the restart holds the previous prompt and every generated token but the
            # last; this request extends it, so all of it comes back from disk
            assert t.get("cache_disk_n", 0) == t["cache_n"] >= prev_len + 8 - 1, (turn, t)
        s.stop()
        n_units.append(len(_units()))
        prev_len = len(prompt)
        prompt = prompt + res.body["tokens"] + [((i * 11) % 100) + 10 for i in range(3 * B)]
    # turn 1: the cold prompt node + the conversation; then one unit per turn and nothing more
    assert [n_units[i + 1] - n_units[i] for i in range(len(n_units) - 1)] == [1, 1, 1], n_units


def test_plain_attention_restores_and_trims():
    """Control: plain attention restores the longer unit and trims it, and the default writes no prompt
    node on a turn that got reuse."""
    rows, stores, _ = _conversation("llama-dense")
    for turn in range(1, TURNS):
        assert rows[turn][1] >= rows[turn - 1][0] - B, rows
    # one unit per turn after the first (the conversation), no per-turn prompt node
    growth = [len(stores[i + 1]) - len(stores[i]) for i in range(len(stores) - 1)]
    assert all(g == 1 for g in growth), (growth, stores)


@pytest.mark.parametrize("name", ["qwen35-dense", "qwen4exp-moe"])
def test_divergent_units_do_not_starve_the_lookup(name):
    """Prompt node off, system node on: every turn can restore only the system node. The earlier turns'
    units share its boundaries, are shorter than the request and diverge inside it; from the fifth turn
    they used to fill AUTO_MAX_RESTORE_ATTEMPTS and turn the hit into a cold prefill."""
    rows, stores, _ = _conversation(name, node_system=True, node_prompt="off")
    sys_node = min(stores[0])
    for turn in range(1, TURNS):
        assert rows[turn][2] == sys_node, f"turn {turn + 1}: {rows[turn]} (system node {sys_node}), {rows}"


@pytest.mark.parametrize("name", ["qwen35-dense", "qwen4exp-moe"])
def test_short_divergent_responses_do_not_starve_the_lookup(name):
    """Default (cold), raw token prompts, 4-token responses the client re-renders differently, every prompt
    4 tokens past a block boundary. Each earlier turn's unit ends inside the block after the divergence, so
    the next request meets it at its deepest boundary and only its identity shows it is not a whole prefix.
    Without that check the fifth turn's four predecessors took every AUTO_MAX_RESTORE_ATTEMPTS place and it
    prefilled cold."""
    _paths(name)
    prompt = [((i * 7) % 100) + 10 for i in range(9 * B + 4)]
    rows, stores = [], []
    for turn in range(TURNS):
        s = _server(name)
        s.jinja = False
        s.chat_template_file = None
        s.start()
        res = s.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 4, "temperature": 0, "top_k": 1, "cache_prompt": True,
            "id_slot": 0, "return_tokens": True})
        assert res.status_code == 200, res.body
        t = res.body["timings"]
        rows.append((len(prompt), t["cache_n"], t.get("cache_disk_n", 0), t["prompt_n"]))
        s.stop()
        stores.append(_units())
        gen = res.body["tokens"]
        alt = [x for x in range(10, 40) if x not in gen][:3]  # the client re-renders the response differently
        prompt = prompt + alt + [((i * 11 + turn) % 100) + 10 for i in range(2 * B - 3)]  # stays 4 past a block
    node = min(stores[0])  # the turn-1 cold prompt node
    assert len(stores[0]) == 2, stores
    for turn in range(1, TURNS):
        n_prompt, cache_n, disk_n, prompt_n = rows[turn]
        assert disk_n == cache_n == node, f"turn {turn + 1}: {rows[turn]} (turn-1 node {node}), {rows}"
        assert prompt_n == n_prompt - node, f"turn {turn + 1}: {rows}"
    growth = [len(stores[i + 1]) - len(stores[i]) for i in range(len(stores) - 1)]
    assert all(g == 1 for g in growth), (growth, stores)

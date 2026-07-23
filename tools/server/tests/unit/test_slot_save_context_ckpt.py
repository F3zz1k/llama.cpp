import glob
import os
import shutil
import struct
import time

import pytest
from utils import *

# Shared-context BASE integration tests (Option A). The feature whole-saves the leading shared context
# [0, B_ctx) — everything before the first user turn — ONCE as a deduplicated v1 base, so N chats that
# share that prefix RESTORE it instead of re-prefilling (and, with --slot-save-incremental, each collapse
# their own save to a small [B_ctx, N) v3 delta parented on that one base). B_ctx is the first-user-
# message token offset, block-aligned down. Unlike the earlier design — which saved a [0,B) SUB-RANGE at
# idle-flush with the slot sitting at N and was therefore hard-gated to dense/PART, n_swa == 0 — the base
# is now written MID-PREFILL as a WHOLE state save, taken at the instant a cold prefill is resident at
# exactly B_ctx. That is sound for EVERY model class (dense, SWA and recurrent/hybrid), so there is NO
# model-class gate: the production qwen3.6-27b (qwen35 hybrid) writes and reuses a base. The old SWA/
# recurrent "no base may ever be written" SOUNDNESS test is therefore obsolete (see the note below).


# --- .meta parsing (shared with test_slot_save_incr.py's on-disk format) -------

SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE
SLOT_META_TOKS_OFF = 104      # offset of the trailing-token-count u32 in the header
SLOT_META_VERSION_NODE = 3


def parse_meta(path: str):
    """Parse a .meta sidecar (v1 whole/partial-root snapshot or v3 delta node)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == SLOT_META_MAGIC, f"bad magic in {path}"
    assert version in (1, 3), f"unexpected .meta version {version} in {path}"
    tok_count = struct.unpack_from("<I", data, SLOT_META_TOKS_OFF)[0]
    chain_hash = struct.unpack_from("<Q", data, SLOT_META_TOKS_OFF + 4)[0]
    off = SLOT_META_TOKS_OFF + 4 + 8  # skip tok_count + chain_hash
    toks = list(struct.unpack_from(f"<{tok_count}i", data, off))
    off += 4 * tok_count
    parent_id, range_lo, range_hi = 0, 0, tok_count
    if version == SLOT_META_VERSION_NODE:
        parent_id, range_lo, range_hi = struct.unpack_from("<QII", data, off)
        off += 16
    assert off == len(data), f"trailing bytes in {path}"
    return {
        "version": version,
        "toks": toks,
        "tok_count": tok_count,
        "chain_hash": chain_hash,
        "parent_id": parent_id,
        "range_lo": range_lo,
        "range_hi": range_hi,
    }


CACHE_DIR = "./tmp/slot_save_context_ckpt"

IDLE_SECONDS = 2

# The positive base-write path needs a model with a separate-system-role chat template whose role
# delimiters are atomic special tokens (so the first-user boundary is found and clears the floor). No
# test-suite preset qualifies: stories260K (tinyllama2) lacks the chatml markers as special tokens, so
# the delimiters do not align and the boundary is never found; tinygemma3 has aligned special-token
# delimiters but its template merges the system prompt INTO the first user turn (boundary == 1, below the
# floor). The positive base-write and the restore-with-a-real-match are therefore validated ON-RIG with
# qwen3.6-27b (chatml, separate system role, atomic specials) — which, being a qwen35 HYBRID, also
# exercises the Option-A soundness for the recurrent/hybrid class the old design excluded. CPU CI still
# covers boundary detection (test-chat), arg validation, and the no-boundary no-op.
_NO_CPU_POSITIVE_MODEL = (
    "no CPU preset has separate-system-role + special-token delimiters clearing the floor; "
    "the shared-context base-write path is validated on-rig (qwen3.6-27b, a qwen35 hybrid)"
)

BLOCK = 16
CONTEXT_MIN = 32  # 2 blocks; effective base floor is max(BLOCK, CONTEXT_MIN) == 32


# A shared leading context long enough that the first-user boundary, block-aligned down, clears the
# max(BLOCK, CONTEXT_MIN) base floor. Two distinct user turns share it verbatim (the base) then
# diverge (their own deltas). A trailing newline is avoided so the rendered system block's token ids
# stay a clean strict prefix of the full prompt.
SHARED_SYSTEM = " ".join(
    ["You are a meticulous assistant grounded in the following reference material."] * 8
    + ["The village by the river kept careful records of every harvest for two hundred years."] * 8
)
USER_A = "Summarise the reference material in one sentence."
USER_B = "List three facts drawn from the reference material."


def _metas():
    return sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.meta")))


def _bin_for(meta_path: str) -> str:
    assert meta_path.endswith(".meta")
    return meta_path[: -len(".meta")]


def _wait_for_metas(n: int, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = _metas()
        if len(m) >= n:
            return m
        time.sleep(0.25)
    return _metas()


def _roots(metas):
    return [m for m in metas if parse_meta(m)["version"] == 1]


def _deltas(metas):
    return [m for m in metas if parse_meta(m)["version"] == SLOT_META_VERSION_NODE]


def _has_strict_prefix_base(metas) -> bool:
    """True iff some v1 root's token ids are a STRICT prefix of another saved snapshot's tokens —
    i.e. a shared-context base was written. This is the direct, delta-mechanics-independent definition
    of "a base was written", used to assert the base is absent when there is no user boundary."""
    parsed = [parse_meta(m) for m in metas]
    roots = [p for p in parsed if p["version"] == 1]
    for r in roots:
        rt = r["toks"]
        for other in parsed:
            ot = other["toks"]
            if len(ot) > len(rt) and ot[: len(rt)] == rt:
                return True
    return False


@pytest.fixture(autouse=True)
def clean_cache_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _mk_attn_server(context_min: int = CONTEXT_MIN, restore_min: int = 0):
    """Pure-attention chatml server (stories260K: PART, n_swa == 0). --jinja + --chat-template chatml
    so the chat completions path renders user/assistant delimiters and the server computes the
    first-user boundary B."""
    s = ServerPreset.tinyllama2()
    s.n_ctx = 1024
    s.n_batch = 1024
    s.n_slots = 1
    s.temperature = 0.0
    s.seed = 42
    s.jinja = True
    s.chat_template = "chatml"
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = True
    s.slot_save_block = BLOCK
    s.slot_save_min_tokens = 0
    s.slot_save_context_min_tokens = context_min
    s.slot_restore_min_tokens = restore_min
    s.slot_save_idle_seconds = IDLE_SECONDS
    return s


def _chat(s, system, user, max_tokens=1):
    res = s.make_request("POST", "/chat/completions", data={
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    })
    assert res.status_code == 200, res.body
    return res.body


# --- (1) two shared-context chats -> exactly ONE base + two v3 deltas ----------

@pytest.mark.skip(reason=_NO_CPU_POSITIVE_MODEL)
def test_two_chats_share_one_base_plus_two_deltas():
    """RECOMMENDATION §3/§6: two chats that share a >context-min leading context each persist their
    own small [B, N) delta, but the shared prefix [0, B) is written to disk EXACTLY ONCE as a v1 base
    at the block-aligned first-user boundary. Result on disk: one v1 root (the base, tok_count == B,
    block-aligned, a STRICT prefix of both full prompts) + two v3 deltas, both chaining to that base."""
    global server
    server = _mk_attn_server()
    server.start()

    _chat(server, SHARED_SYSTEM, USER_A)
    m1 = _wait_for_metas(2, IDLE_SECONDS + 12)   # base (checkpoint) + chat A's own delta
    assert len(m1) == 2, f"first chat should write a base + its delta, got {len(m1)}"

    _chat(server, SHARED_SYSTEM, USER_B)
    metas = _wait_for_metas(3, IDLE_SECONDS + 12)
    server.stop()

    roots = _roots(metas)
    deltas = _deltas(metas)
    assert len(roots) == 1, \
        f"the shared context must be ONE base root (2nd chat's checkpoint is deduped), got {len(roots)}"
    assert len(deltas) == 2, f"each chat's own save collapses to a v3 delta, got {len(deltas)}"

    base = parse_meta(roots[0])
    assert base["tok_count"] % BLOCK == 0, "the base is block-aligned (B_ctx = B - B % block)"
    assert base["tok_count"] >= max(BLOCK, CONTEXT_MIN), "the base clears the max(block, context-min) floor"

    for d in deltas:
        dm = parse_meta(d)
        assert dm["parent_id"] == base["chain_hash"], "each delta chains to the one base (parent_id == base hash)"
        assert dm["range_lo"] == base["tok_count"], "each delta's KV range begins exactly at the base length B"
        assert dm["tok_count"] > base["tok_count"], "the base is a STRICT prefix of each full chat"

    assert _has_strict_prefix_base(metas), "a shared-context base checkpoint must have been written"


# --- (2) no user boundary -> whole prefix, NO base ----------------------------

def test_no_user_boundary_writes_whole_prefix_no_base():
    """RECOMMENDATION §2/§6: a request with no first-user boundary (a raw /completion carries no
    message delimiters, so ctx_boundary stays -1) writes the whole prefix unchanged and NO base
    checkpoint — the checkpoint is a clean no-op, not a behaviour change to the existing save."""
    global server
    server = _mk_attn_server()
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": SHARED_SYSTEM + " " + USER_A,
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    metas = _wait_for_metas(1, IDLE_SECONDS + 12)
    server.stop()

    assert len(_deltas(metas)) == 0, "no boundary => no base => nothing to parent a delta on"
    roots = _roots(metas)
    assert len(roots) == 1, f"a boundary-less request saves exactly one whole snapshot, got {len(roots)}"
    assert not _has_strict_prefix_base(metas), "no base checkpoint may be written without a user boundary"


# --- (3) restore-min above the match skips the disk load ----------------------

@pytest.mark.skip(reason=_NO_CPU_POSITIVE_MODEL + " (needs a base+delta on disk first)")
def test_restore_min_above_match_skips_disk_load():
    """RECOMMENDATION §4/§6: --slot-restore-min-tokens gates the disk restore on the byte-verified
    matched prefix. Set above any available match, a cold slot must REPROCESS (no disk load) rather
    than pay the multi-GB read — observable as ~zero cached prompt tokens after a restart."""
    global server

    # produce a base + delta on disk for chat A (restore-min 0 so saving is unaffected).
    server = _mk_attn_server(restore_min=0)
    server.start()
    body_a = _chat(server, SHARED_SYSTEM, USER_A)
    metas = _wait_for_metas(2, IDLE_SECONDS + 12)
    assert len(metas) >= 2, "need a base + delta on disk before testing the restore guard"
    full_len = body_a["usage"]["prompt_tokens"]
    server.stop()

    # control: a fresh process with restore-min 0 DOES restore the full prefix from disk.
    server = _mk_attn_server(restore_min=0)
    server.start()
    ctrl = _chat(server, SHARED_SYSTEM, USER_A)
    server.stop()
    ctrl_cached = ctrl["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert ctrl_cached >= full_len - BLOCK, \
        f"control: disk restore should reuse ~the whole prefix, cached={ctrl_cached} of {full_len}"

    # guard: restore-min above the whole match forces a cold reprocess (no disk load).
    server = _mk_attn_server(restore_min=full_len + BLOCK)
    server.start()
    guarded = _chat(server, SHARED_SYSTEM, USER_A)
    server.stop()
    guarded_cached = guarded["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert guarded_cached == 0, \
        f"restore-min above the match must skip the disk load (reprocess); cached={guarded_cached}"


# --- (4) OBSOLETE: the model-class SOUNDNESS gate is gone (Option A) -----------
#
# The former test here — test_swa_model_never_writes_base_checkpoint — asserted that a sliding-window
# (SWA) model, and by extension a recurrent/hybrid (FULL/RS) model, could NEVER write a shared-context
# base, because the earlier design saved a [0, B) SUB-RANGE with the slot sitting at N: for SWA the
# window had already evicted cells [0, B), and for a recurrent fold the persisted bytes were the state-
# after-N mislabelled as a B-length prefix — both silent wrong output. That was a real correctness
# boundary, and the code hard-gated the save to dense/PART, n_swa == 0.
#
# Option A REMOVES that gate. The base is now whole-saved MID-PREFILL, at the instant a cold prefill is
# resident at EXACTLY B_ctx (nothing decoded beyond it): the resident sequence IS the true whole state
# at B_ctx, so llama_state_seq_save_file serialises the correct state for dense, SWA AND recurrent/
# hybrid alike. There is consequently no model-class gate left to test — an SWA or hybrid model with a
# real first-user boundary above the floor now legitimately WRITES and reuses a base, which is exactly
# the behaviour the old test forbade. Asserting "no base on SWA" would now be asserting a bug.
#
# The positive base-write on a class the old design excluded (the qwen35 HYBRID qwen3.6-27b) is
# validated ON-RIG — see _NO_CPU_POSITIVE_MODEL and test_two_chats_share_one_base_plus_two_deltas. CPU
# CI retains the model-class-independent coverage: boundary detection (test-chat), arg validation, and
# the no-boundary no-op (test_no_user_boundary_writes_whole_prefix_no_base above).

import glob
import os
import shutil
import struct
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


# --- prompts ------------------------------------------------------------------
# BASE is deliberately much longer than either continuation so a delta node's
# .bin (which covers only the appended tail) is strictly smaller than the base
# root's .bin. BASE ends WITHOUT a trailing space and each TAIL begins with a
# space, so the concatenation BASE + TAIL keeps BASE's token ids as an exact
# strict prefix (a trailing space in BASE would merge with the tail's first word
# and break the prefix — the property the incremental parent-find relies on).
def _join(sentence: str, n: int) -> str:
    return " ".join([sentence] * n)


BASE = _join("Once upon a time there was a little dog named Spot who loved to run.", 10)
TAIL_A = " " + _join("The dog ran fast across the green field every single morning.", 3)
TAIL_B = " " + _join("A quiet cat sat on the warm windowsill all afternoon long.", 3)

P1 = BASE
P2A = BASE + TAIL_A
P2B = BASE + TAIL_B


CACHE_DIR = "./tmp/slot_save_incr"

IDLE_SECONDS = 2


SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE
SLOT_META_TOKS_OFF = 104      # offset of the trailing-token-count u32 in the header
SLOT_META_VERSION_NODE = 3


def parse_meta(path: str):
    """Parse a .meta sidecar (v1 whole snapshot or v3 delta node) per slot_meta_write.

    Returns a dict: version, toks, chain_hash, and (v3 only) parent_id/range_lo/range_hi.
    A v1 meta reports parent_id=0, range_lo=0, range_hi=len(toks) — the implicit root
    covering [0, tok_count). Asserts exact EOF (no trailing bytes)."""
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


def _make_server(incremental: bool, fa: str = "off", n_ctx: int = 512, n_slots: int = 1):
    s = ServerPreset.tinyllama2()
    s.n_ctx = n_ctx
    s.n_batch = 512
    s.n_slots = n_slots
    s.temperature = 0.0
    s.seed = 42
    s.fa = fa
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = incremental
    s.slot_save_block = 16       # small block so short test prompts clear the boundary
    s.slot_save_min_tokens = 0   # keep the floor at the hash block size
    s.slot_save_idle_seconds = IDLE_SECONDS
    return s


@pytest.fixture(autouse=True)
def clean_cache_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _complete(s, prompt, n_predict=8, id_slot=0):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": id_slot,
    })
    assert res.status_code == 200
    return res.body


# --- (1) incremental-ON writes a v3 delta node --------------------------------

def test_incremental_writes_v3_delta_node():
    """(U7.1) With --slot-save-incremental, a continuation of an already-saved prompt is
    persisted as a v3 delta node: parent_id == the root's chain_hash, range_lo == the root's
    token count (the delta's KV .bin starts where the base ended), range_hi == the full prompt
    length, and the delta's .bin is strictly smaller than the base root's .bin (it holds only
    the appended tail's KV, not the whole prefix)."""
    global server
    server = _make_server(incremental=True)
    server.start()

    _complete(server, P1, n_predict=0)
    m1 = _wait_for_metas(1, IDLE_SECONDS + 12)
    assert len(m1) == 1, "the base prompt must be flushed as one root snapshot"
    root = parse_meta(m1[0])
    assert root["version"] == 1, "the first (parentless) save is a whole v1 root"

    _complete(server, P2A)
    m2 = _wait_for_metas(2, IDLE_SECONDS + 12)
    assert len(m2) == 2, "the continuation must be flushed as a second (delta) node"
    server.stop()

    roots = _roots(m2)
    deltas = _deltas(m2)
    assert len(roots) == 1 and len(deltas) == 1, \
        f"expected one v1 root + one v3 delta, got roots={len(roots)} deltas={len(deltas)}"

    root_meta = parse_meta(roots[0])
    delta = parse_meta(deltas[0])
    assert delta["parent_id"] == root_meta["chain_hash"], \
        "the delta must chain to the root via parent_id == root chain_hash"
    assert delta["range_lo"] == root_meta["tok_count"], \
        "the delta's KV range must begin exactly at the root's token count"
    assert delta["range_hi"] == delta["tok_count"] > root_meta["tok_count"], \
        "the delta covers [root_len, full_len) and is longer than the root"

    root_bin = os.path.getsize(_bin_for(roots[0]))
    delta_bin = os.path.getsize(_bin_for(deltas[0]))
    assert delta_bin < root_bin, \
        f"delta .bin ({delta_bin}) must be smaller than the base root .bin ({root_bin})"


# --- (2) base+delta cold-restore continues token-identical to a no-cache ref --

@pytest.mark.parametrize("fa", ["off", "on"])
def test_base_delta_cold_restore_token_identical(fa):
    """(U7.2) A base+delta chain on disk is cold-restored across a server restart: after
    reconstructing the [0, N) prefix by loading root + delta in position order, the
    continuation is TOKEN-IDENTICAL to a from-scratch, no-cache reference. Parametrised over
    flash-attn on/off (fa off exercises the transposed-V delta path)."""
    global server

    # reference: a pristine server with NO disk cache generates the continuation of P2A.
    ref = ServerPreset.tinyllama2()
    ref.n_ctx = 512
    ref.n_batch = 512
    ref.n_slots = 1
    ref.temperature = 0.0
    ref.seed = 42
    ref.fa = fa
    ref.start()
    ref_body = ref.make_request("POST", "/completion", data={
        "prompt": P2A, "n_predict": 24, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    }).body
    ref.stop()
    ref_content = ref_body["content"]

    # produce base (root) + delta on disk with incremental saving.
    server = _make_server(incremental=True, fa=fa)
    server.start()
    _complete(server, P1, n_predict=0)
    _wait_for_metas(1, IDLE_SECONDS + 12)
    _complete(server, P2A)
    metas = _wait_for_metas(2, IDLE_SECONDS + 12)
    assert len(metas) == 2 and len(_deltas(metas)) == 1, "need a root + a delta on disk"
    server.stop()

    # cold restart: empty KV. Resending P2A must restore the base+delta chain from disk.
    server = _make_server(incremental=True, fa=fa)
    server.start()
    body = server.make_request("POST", "/completion", data={
        "prompt": P2A, "n_predict": 24, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    }).body
    server.stop()

    full_len = parse_meta(_deltas(metas)[0])["tok_count"]
    reused = body["timings"]["cache_n"]
    assert reused >= full_len - server.slot_save_block, \
        f"disk restore should reuse ~the whole prefix; cache_n={reused} of {full_len}"
    assert body["content"] == ref_content, \
        "continuation after a base+delta restore must be token-identical to the no-cache reference"


# --- (3) a fork shares one base on disk ---------------------------------------

def test_fork_shares_one_base_on_disk():
    """(U7.3) Two continuations of the same base prompt (a fork) each persist their own delta
    node, but the shared base prefix is written to disk exactly once: one v1 root, two v3
    deltas, and both deltas point at that single root."""
    global server
    server = _make_server(incremental=True)
    server.start()

    _complete(server, P1, n_predict=0)
    _wait_for_metas(1, IDLE_SECONDS + 12)
    _complete(server, P2A)
    _wait_for_metas(2, IDLE_SECONDS + 12)
    _complete(server, P2B)
    metas = _wait_for_metas(3, IDLE_SECONDS + 12)
    server.stop()

    roots = _roots(metas)
    deltas = _deltas(metas)
    assert len(roots) == 1, f"the shared base must be one root on disk, got {len(roots)}"
    assert len(deltas) == 2, f"each fork tail is its own delta, got {len(deltas)}"

    root_hash = parse_meta(roots[0])["chain_hash"]
    for d in deltas:
        dm = parse_meta(d)
        assert dm["parent_id"] == root_hash, "both fork deltas must share the one base"
        assert dm["range_lo"] == parse_meta(roots[0])["tok_count"]


# --- (4) incremental-OFF: no v3 meta + golden byte-identity of the root -------

def test_incremental_off_no_v3_and_root_bytes_golden():
    """(U7.4) Golden lock: with incremental saving OFF every snapshot is a whole v1 meta (no
    v3 node ever appears), and turning incremental ON does NOT change the bytes of a whole
    (root) snapshot — the v1 .bin captured for the base prompt is byte-identical whether or not
    --slot-save-incremental is set (roots stay whole snapshots on both paths)."""
    global server

    # incremental OFF: two turns, both whole snapshots, no v3 node.
    server = _make_server(incremental=False)
    server.start()
    _complete(server, P1, n_predict=0)
    _wait_for_metas(1, IDLE_SECONDS + 12)
    _complete(server, P2A)
    metas_off = _wait_for_metas(2, IDLE_SECONDS + 12)
    assert len(metas_off) == 2, "incremental OFF still persists both turns"
    server.stop()
    assert _deltas(metas_off) == [], "incremental OFF must never write a v3 delta node"
    for m in metas_off:
        assert parse_meta(m)["version"] == 1

    # identify the P1 root (shorter of the two whole snapshots)
    off_by_len = sorted(metas_off, key=lambda m: parse_meta(m)["tok_count"])
    off_p1_bin = _bin_for(off_by_len[0])
    off_p1_bytes = open(off_p1_bin, "rb").read()
    off_p1_len = parse_meta(off_by_len[0])["tok_count"]

    # fresh dir, incremental ON: the P1 root is still a whole v1 snapshot.
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    server = _make_server(incremental=True)
    server.start()
    _complete(server, P1, n_predict=0)
    metas_on = _wait_for_metas(1, IDLE_SECONDS + 12)
    server.stop()
    assert len(metas_on) == 1
    on_root_meta = parse_meta(metas_on[0])
    assert on_root_meta["version"] == 1, "the first (root) save is a whole v1 snapshot even with incremental ON"
    assert on_root_meta["tok_count"] == off_p1_len
    on_p1_bytes = open(_bin_for(metas_on[0]), "rb").read()
    assert on_p1_bytes == off_p1_bytes, \
        "a whole (root) snapshot must be byte-identical with incremental ON vs OFF (golden lock)"


# --- (5) a context shift rebases to a fresh root ------------------------------

def test_context_shift_rebases_to_fresh_root():
    """(U7.5) When generation triggers a context shift, the already-saved positions are mutated,
    so the next save's parent-find byte-verify fails and it falls back to a whole (root) save —
    an automatic rebase. Without the shift the same superset continuation would be a v3 delta
    (see test 1); with the shift no v3 delta is produced."""
    global server
    # single slot; the prompt fits the 512-token slot context but a long generation overflows it.
    server = _make_server(incremental=True, n_ctx=512, n_slots=1)
    server.enable_ctx_shift = True
    server.start()

    # turn 1: a base that fits and is saved as a root.
    _complete(server, P1, n_predict=0)
    _wait_for_metas(1, IDLE_SECONDS + 12)

    # turn 2: same base + tail (fits), then generate enough that prompt+generation overflows the
    # 512-token slot context and triggers a shift mid-generation (rebasing already-saved positions).
    body = server.make_request("POST", "/completion", data={
        "prompt": P2A, "n_predict": 256, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    }).body
    assert body["truncated"] is True, "the second turn must have shifted the context"
    metas = _wait_for_metas(2, IDLE_SECONDS + 12)
    server.stop()

    assert len(_roots(metas)) >= 1, "at least the base root must be present"
    assert _deltas(metas) == [], \
        "a context-shifted conversation must rebase to a fresh root, never a cross-shift delta"


# --- (6) --slot-save-incremental requires --slot-save-auto --------------------

def test_incremental_requires_auto_at_startup():
    """(U7.6) --slot-save-incremental is meaningless without the auto disk cache it modifies, so
    it is rejected at startup rather than silently ignored."""
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = False
    server.slot_save_incremental = True
    with pytest.raises(Exception):
        server.start(timeout_seconds=10)
    server.stop()


# --- (7) eviction never removes a node with a live child ----------------------

def test_eviction_never_removes_node_with_live_child():
    """(U7.7) A base whose delta child is still on disk is never evicted, even when the count cap
    would otherwise be exceeded. With a cap of 1 and a root+delta pair, the just-written delta is
    protected as just_written and the root is protected as its live parent, so the store is left
    (correctly) above the cap with BOTH files intact rather than deleting the base out from under
    its child."""
    global server
    server = _make_server(incremental=True)
    server.slot_save_max_count = 1
    server.start()

    _complete(server, P1, n_predict=0)
    m1 = _wait_for_metas(1, IDLE_SECONDS + 12)
    assert len(m1) == 1

    _complete(server, P2A)
    metas = _wait_for_metas(2, IDLE_SECONDS + 12)
    server.stop()

    roots = _roots(metas)
    deltas = _deltas(metas)
    assert len(roots) == 1 and len(deltas) == 1, \
        "cap=1 must NOT evict the base while its delta child is live (both survive)"
    assert os.path.exists(_bin_for(roots[0])), "the base root .bin was wrongly evicted"
    assert os.path.exists(_bin_for(deltas[0])), "the delta .bin was wrongly evicted"


# --- (8) SWA model: incremental save+restore spanning > the sliding window -----

@pytest.mark.parametrize("fa", ["off", "on"])
def test_swa_incremental_save_restore(fa):
    """(U7.8) An iSWA model (tinygemma3) with text-only turns whose combined length spans more
    than the sliding window: the incremental save decomposes into a global-attention delta plus a
    whole sliding-window blob, and a cold base+delta restore continues token-identical to a
    no-cache reference. Exercises the SWA state_write_range / compose-restore path end-to-end."""
    global server

    # tinygemma3's sliding window is 4096 tokens, so the BASE alone (~4400 tokens) already spans
    # past it: the base's sliding-attention layers hold only the last window, while the global
    # layers hold the whole prefix. This is exactly the regime the decomposition targets — the
    # delta re-saves the (bounded, slid) window whole and only appends the new global-attention
    # cells. A cold base+delta restore must recompose both correctly.
    swa_base = _join("The history of the small mountain village stretched back many centuries and its "
                     "people told long stories every evening.", 230)
    swa_tail = " " + _join("In recent years the village welcomed travellers from distant lands.", 6)
    swa_p1 = swa_base
    swa_p2 = swa_base + swa_tail

    def _mk():
        s = ServerPreset.tinygemma3()
        s.n_ctx = 8192
        s.n_batch = 2048
        s.n_slots = 1
        s.temperature = 0.0
        s.seed = 42
        s.fa = fa
        return s

    # no-cache reference continuation of the full prompt.
    ref = _mk()
    ref.start()
    ref_content = ref.make_request("POST", "/completion", data={
        "prompt": swa_p2, "n_predict": 16, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    }).body["content"]
    ref.stop()

    # produce base+delta on disk.
    server = _mk()
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_incremental = True
    server.slot_save_block = 16
    server.slot_save_min_tokens = 0
    server.slot_save_idle_seconds = IDLE_SECONDS
    server.start()
    _complete(server, swa_p1, n_predict=0)
    _wait_for_metas(1, IDLE_SECONDS + 15)
    _complete(server, swa_p2, n_predict=4)
    metas = _wait_for_metas(2, IDLE_SECONDS + 15)
    assert len(metas) == 2, "SWA base + delta must both be persisted"
    assert len(_deltas(metas)) == 1, "the SWA continuation must be a v3 delta node"
    assert parse_meta(_roots(metas)[0])["tok_count"] > 4096, \
        "the SWA base must span past the 4096-token sliding window (else the slid-window path is untested)"
    server.stop()

    # cold restore: reconstruct the SWA prefix from disk and continue token-identical.
    server = _mk()
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_incremental = True
    server.slot_save_block = 16
    server.slot_save_min_tokens = 0
    server.slot_save_idle_seconds = IDLE_SECONDS
    server.start()
    body = server.make_request("POST", "/completion", data={
        "prompt": swa_p2, "n_predict": 16, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    }).body
    server.stop()

    assert body["content"] == ref_content, \
        "SWA base+delta cold restore must continue token-identical to the no-cache reference"

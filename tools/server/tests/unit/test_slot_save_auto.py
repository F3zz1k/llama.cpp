import glob
import hashlib
import os
import shutil
import struct
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()

# Frozen golden fixture: an auto-cache unit (.bin state + v1 .meta sidecar) captured from
# the pre-media base build (auto-disk-kvcache @ 9683b5b7b) with the server flags below and
# GOLDEN_PROMPT. What it locks is the FORK's side of the format: the v1 .meta layout (a v1
# .meta carries no fp_mmproj, the reader backfills it) and the fork's own cell-record
# handling, so units written by an older build keep indexing and restoring unmodified.
#
# It cannot lock the .bin across a change of LLAMA_STATE_SEQ_VERSION (include/llama.h). That
# word is the first thing llama_state_seq_load_file checks and a mismatch refuses the file
# outright, so on an upstream bump the pair must be RECAPTURED (same flags, same
# GOLDEN_PROMPT, graceful stop) and FIXTURE_SHA256, EMITTED_SHA256 and
# fixtures/golden-v1/SHA256SUMS updated with it. The version word is asserted directly below
# so that bump reports itself instead of surfacing as an opaque sha mismatch or, worse, as a
# restore test that quietly cold-prefills.
STATE_SEQ_VERSION = 3          # must track LLAMA_STATE_SEQ_VERSION in include/llama.h
STATE_SEQ_MAGIC   = 0x67677371 # LLAMA_STATE_SEQ_MAGIC ("ggsq")
FIXTURE_DIR = "./fixtures/golden-v1"
FIXTURE_SHA256 = {
    "auto-f414107cff91a49e-5efede8f57f0c198-377.bin":
        "b5c5f419ef0ad9d609c1c92e64850e2d79f69102a02182c4ef2dc66d827cc9ba",
    "auto-f414107cff91a49e-5efede8f57f0c198-377.bin.meta":
        "ecbac54e17454e180c4b50abe0845822996cbca207df95719f6e0a9328f54f90",
}

# The SAME frozen unit under the name THIS binary emits for it. F4 widened the auto-snapshot
# filename PREFIX from fp_model alone to a full-identity hash (model_fp::identity_hash — every
# operator== field) so mixed-geometry peers sharing one dir get disjoint names instead of
# clobbering. Only the derived prefix changed (f414107cff91a49e -> de633b3190d4d950): the
# block chain keeps its fp_model salt, so the chain-hash/token-count tail (5efede8f57f0c198-377)
# and the .bin/.meta CONTENT are byte-identical to the pre-media base build's (the shas below
# equal FIXTURE_SHA256's — invariant 0 for the on-disk BYTES is preserved). The old-prefix
# fixture above is still what the restore tests feed in, and it still indexes and restores
# (the scan verifies fp == cur_fp after reading the sidecar and never parses the prefix).
#
# merge-upstream-20261003 (e1998922a) then changed the middle hash for a unit whose length is not a
# whole number of blocks, as this one is (377 = 23 blocks of 16 + 9): a unit is named by its identity
# over EVERY token (the chain value after its last cell), not by its last whole block, so two units of
# equal length that differ only in that partial block no longer share a name. The .meta records the
# same value in its chain-hash field (a delta's parent_id is the parent's file id), so that one 8-byte
# field changes; every other byte of the .meta, the v1 layout, and the .bin are unchanged, and no
# reader uses the field (an older build reads this unit as before). Asserted field by field below.
#
# The same branch then took n_ctx out of the identity (it is a capacity record, not a restore gate), so
# identity_hash folds the LongRoPE regime bit where it folded fp_n_ctx and the prefix moved again
# (de633b3190d4d950 -> 4c354f5acbfd2d60). The .meta still records fp_n_ctx, so its bytes do not change.
EMITTED_SHA256 = {
    "auto-4c354f5acbfd2d60-a028e20ed7d8f6b7-377.bin":
        FIXTURE_SHA256["auto-f414107cff91a49e-5efede8f57f0c198-377.bin"],
    "auto-4c354f5acbfd2d60-a028e20ed7d8f6b7-377.bin.meta":
        FIXTURE_SHA256["auto-f414107cff91a49e-5efede8f57f0c198-377.bin.meta"],
}
EMITTED_CHAIN_HASH = 0xa028e20ed7d8f6b7

# The .bin holds the KV numbers themselves, and those depend on the backend's arithmetic: the
# frozen fixture is a GPU (SYCL) capture, and a CPU build emits a .bin of the same size and layout
# whose K/V values differ (about 42k bytes, all inside the tensor data). One frozen capture per
# backend, each taken with this build's predecessor on that backend (the CPU one from main-patched
# 593c2d0d7, re-emitted unchanged by every later CPU build). The .meta carries no numbers and must
# match exactly on every backend.
EMITTED_BIN_SHA256_PER_BACKEND = {
    "gpu-sycl": FIXTURE_SHA256["auto-f414107cff91a49e-5efede8f57f0c198-377.bin"],
    "cpu":      "c97f4c568102edae478b21e7ac7937e2a437e0e76e8f9410a313d59f0ebc8cc7",
}


def _server_backend() -> str:
    """The backend the server under test computes on, so the .bin is checked against THAT backend's
    capture only. LLAMA_TEST_BACKEND overrides; otherwise the GPU backend libraries next to the binary
    (build tree: bin/, release: ../lib) decide, and a build with none of them is the CPU backend."""
    env = os.environ.get("LLAMA_TEST_BACKEND", "")
    if env:
        assert env in EMITTED_BIN_SHA256_PER_BACKEND, f"LLAMA_TEST_BACKEND={env} has no frozen capture"
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    names: set[str] = set()
    for d in (os.path.dirname(server_bin), os.path.join(os.path.dirname(server_bin), "..", "lib")):
        if os.path.isdir(d):
            names |= set(os.listdir(d))
    if any(n.startswith("libggml-sycl") for n in names):
        return "gpu-sycl"
    other_gpu = [n for n in names if n.startswith(("libggml-cuda", "libggml-vulkan", "libggml-hip", "libggml-metal", "libggml-opencl"))]
    assert not other_gpu, f"no frozen capture for this backend ({other_gpu}); set LLAMA_TEST_BACKEND"
    assert any(n.startswith("libggml-cpu") for n in names) or not names, \
        "cannot tell the backend from the files next to the server binary; set LLAMA_TEST_BACKEND"
    return "cpu"

# exact prompt the fixture was captured with (tokenizes to >= 1 hash block of 256)
GOLDEN_PROMPT = "Once upon a time there was a little dog named Spot. " * 24
GOLDEN_REQUEST = {
    "prompt": GOLDEN_PROMPT,
    "n_predict": 16,
    "temperature": 0,
    "cache_prompt": True,
    "id_slot": 0,
}

CACHE_DIR = "./tmp/slot_save_auto"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # geometry equals the fixture capture's (-c 512 -np 1). fp_n_ctx is no longer identity (it is a
    # capacity record), but the golden test checks the emitted bytes, which carry it.
    server.n_ctx = 512
    server.n_batch = 512
    server.n_slots = 1
    server.temperature = 0.0
    # the test prompts sit well under the 1024-token default minimum-snapshot floor; drop it so
    # the auto cache behaves as it did before --slot-save-min-tokens (floor = the hash block size).
    server.slot_save_min_tokens = 0
    # this file tests the unit format, identity, media and flush paths: no mid-prefill prompt node
    # (cold by default), so each test sees exactly the units it writes (the node: test_slot_save_resend.py)
    server.slot_save_node_prompt = "off"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)


def assert_state_seq_version(path: str):
    """Check a state .bin's magic + version word against what this build reads. A .bin whose
    version word is not LLAMA_STATE_SEQ_VERSION is refused outright by llama_state_seq_load_file,
    which makes every assertion built on that file meaningless. Paths that are not a .bin (the
    ".bin.meta" sidecar, which carries its own independent version) are ignored."""
    if not path.endswith(".bin"):
        return
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
    assert magic == STATE_SEQ_MAGIC, f"{path}: not a llama state-seq file"
    assert version == STATE_SEQ_VERSION, (
        f"{path}: state file is version {version}, this build reads only version "
        f"{STATE_SEQ_VERSION}. LLAMA_STATE_SEQ_VERSION was bumped: recapture the golden fixture "
        "(same flags, same GOLDEN_PROMPT, graceful stop) and update FIXTURE_SHA256, "
        "EMITTED_SHA256 and fixtures/golden-v1/SHA256SUMS."
    )


def verify_and_copy_fixture(dst: str, names=None):
    for name, expected in FIXTURE_SHA256.items():
        path = os.path.join(FIXTURE_DIR, name)
        assert_state_seq_version(path)
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected, f"golden fixture {name} changed on disk — it must stay frozen"
        if names is None or name in names:
            shutil.copy(path, dst)


def test_text_only_meta_byte_identical():
    """The current binary re-emits the golden unit for the capture prompt with byte-identical
    contents (invariant 0: text-only on-disk BYTES are frozen at the pre-media base build's).
    Since F4 the filename PREFIX is the full-identity hash rather than fp_model alone, and since
    e1998922a the middle hash of this unaligned unit is its identity over every token, so the
    emitted name is EMITTED_SHA256's; the .bin bytes are unchanged and the .meta differs only in
    its chain-hash field (see the EMITTED_SHA256 comment)."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    # graceful stop -> auto_save_slots_at_shutdown publishes the unit
    server.stop()

    # exact emitted filenames (identity-hash prefix, then the fp_model-salted block-chain hash
    # and the token count — a drift in any of them shows up here first)
    # (plus the unit's .logits sidecar, written for every memory class since 2026-10-05; not part of the
    # frozen format, so only its presence is checked)
    files = sorted(os.listdir(CACHE_DIR))
    assert [n for n in files if not n.endswith(".logits")] == sorted(EMITTED_SHA256)
    assert [n for n in files if n.endswith(".logits")] == [n + ".logits" for n in EMITTED_SHA256 if n.endswith(".bin")]
    # the sha comparison below is against the frozen fixture, so it only carries meaning while
    # both files are readable by this build: name a version bump for what it is first
    for name in FIXTURE_SHA256:
        assert_state_seq_version(os.path.join(FIXTURE_DIR, name))
    for name in EMITTED_SHA256:
        assert_state_seq_version(os.path.join(CACHE_DIR, name))
    for name, expected in EMITTED_SHA256.items():
        with open(os.path.join(CACHE_DIR, name), "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if name.endswith(".bin"):
            backend = _server_backend()
            assert actual == EMITTED_BIN_SHA256_PER_BACKEND[backend], \
                f"{name}: emitted bytes differ from the {backend} frozen capture ({actual})"
        else:
            # the frozen .meta with only its chain-hash field replaced by the unit's id (see above)
            with open(os.path.join(FIXTURE_DIR, "auto-f414107cff91a49e-5efede8f57f0c198-377.bin.meta"), "rb") as f:
                frozen = bytearray(f.read())
            assert hashlib.sha256(frozen).hexdigest() == expected
            off = SLOT_META_TOKS_OFF + 4
            struct.pack_into("<Q", frozen, off, EMITTED_CHAIN_HASH)
            with open(os.path.join(CACHE_DIR, name), "rb") as f:
                emitted = f.read()
            assert emitted == bytes(frozen), f"{name}: emitted .meta differs from the golden one beyond its chain-hash field"


def test_v1_meta_still_indexed():
    """The frozen v1 fixture unit is indexed and restored by the current binary."""
    global server

    # cold baseline: no auto cache, full prefill
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_cold = res.body["timings"]["prompt_n"]
    content_cold = res.body["content"]
    assert prompt_n_cold >= 256  # sanity: the prompt spans at least one hash block
    server.stop()

    # warm run: a fresh process pointed at a cache dir holding ONLY the fixture unit must
    # index the v1 .meta at startup and restore one 256-token block from the .bin
    verify_and_copy_fixture(CACHE_DIR)
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_warm = res.body["timings"]["prompt_n"]
    # the snapshot (377 tokens = prompt + 16 generated) extends past the resent prompt;
    # the restore loads it and the normal prefix-reuse logic then trims to the request, so
    # at most a handful of tokens are re-decoded (never anywhere near a full block less
    # than the cold prefill — the gate for "the v1 unit was indexed and restored")
    assert prompt_n_warm <= prompt_n_cold - 256
    # greedy continuation from the restored KV must match the cold run
    assert res.body["content"] == content_cold


def test_snapshot_below_min_tokens_not_saved():
    """A snapshot smaller than --slot-save-min-tokens is skipped — the state-file write buys too
    little prefill against the later restore. A prompt above the floor is persisted as usual.
    The effective floor is max(slot_save_block, slot_save_min_tokens)."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_block = 16      # keep the block floor low so min-tokens is the binding gate
    server.slot_save_min_tokens = 200
    server.start()

    # a mid-size prompt that clears the 16-token block floor but not the 200-token min-tokens
    # floor: it is worth a hash block but not worth persisting to disk -> no unit emitted.
    small_prompt = "The quick brown fox jumps over the lazy dog. " * 4
    res = server.make_request("POST", "/completion", data={"prompt": small_prompt, "n_predict": 4})
    assert res.status_code == 200
    prompt_n_small = res.body["timings"]["prompt_n"]
    assert 16 < prompt_n_small < 200  # above the block floor, below the min-tokens floor
    server.stop()  # the shutdown flush must skip the sub-floor snapshot
    assert os.listdir(CACHE_DIR) == []

    # the golden prompt spans >= 256 tokens -> above the floor -> persisted as normal
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] >= 200
    server.stop()
    assert len(os.listdir(CACHE_DIR)) > 0


def _text_cache_server(n_ctx: int, ctk: str | None = None) -> ServerProcess:
    s = ServerPreset.tinyllama2()
    s.n_ctx = n_ctx
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_min_tokens = 0  # short-prompt test: keep the floor at the hash block size
    s.slot_save_node_prompt = "off"  # see create_server
    if ctk is not None:
        s.ctk = ctk
    return s


def _golden_completion(s: ServerProcess):
    res = s.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    t = res.body["timings"]
    return t["prompt_n"], res.body["content"], t.get("cache_disk_n", 0)


def _bins():
    return sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".bin"))


def test_mixed_geometry_peers_get_disjoint_filenames():
    """G3/F4: two peers sharing one --slot-save-path that agree on the token prefix but
    differ in an identity field absent from the block-chain salt used to mint the SAME
    filename and atomically rename over each other (destructive clobber: a K=f32 blob
    landing under a K=f16 peer's name). The full-identity filename prefix gives them
    disjoint names so both units coexist and each restores its OWN; a same-config peer
    still reuses one deterministic name (no file proliferation).

    Discriminator is --cache-type-k (f16 vs f32). It must be an identity field: n_ctx is
    not one (peers differing only in it share one name and one unit, see
    test_cross_ctx_peers_share_one_unit), so a ctx-only pair cannot exercise the naming.
    f32 is not a quantised type, so the tiny CI model initialises it."""
    a = _text_cache_server(512)
    a.start()
    a_prompt_cold, a_content_cold, a_disk_cold = _golden_completion(a)
    a.stop()   # shutdown flush publishes A's unit
    assert a_disk_cold == 0
    assert len(_bins()) == 1

    b = _text_cache_server(512, ctk="f32")
    b.start()  # B's startup scan reads A's sidecar, fp_cache_k differs -> not indexed
    b_prompt_cold, b_content_cold, b_disk_cold = _golden_completion(b)
    b.stop()   # shutdown flush publishes B's unit (old code: renames over A's)
    assert b_disk_cold == 0                      # B never touched A's K=f16 unit
    assert b_prompt_cold == a_prompt_cold        # and really prefilled cold

    bins = _bins()
    # no clobber: two distinct units with disjoint identity-hash prefixes but an IDENTICAL
    # chain-hash + token-count tail (the chain keeps its fp_model salt).
    assert len(bins) == 2, bins
    assert len({f.split("-")[1] for f in bins}) == 2, bins
    assert len({"-".join(f.split("-")[2:]) for f in bins}) == 1, bins

    # both restore from the shared dir (fresh processes). Each peer can only ever index
    # its OWN unit (the other is refused by restore_compatible before any load), so a
    # disk hit here is necessarily a hit on its own file.
    a2 = _text_cache_server(512)
    a2.start()
    a_prompt_warm, a_content_warm, a_disk_warm = _golden_completion(a2)
    a2.stop()
    assert a_disk_warm >= 256
    assert a_prompt_warm <= a_prompt_cold - 256
    assert a_content_warm == a_content_cold

    b2 = _text_cache_server(512, ctk="f32")
    b2.start()
    b_prompt_warm, b_content_warm, b_disk_warm = _golden_completion(b2)
    b2.stop()
    assert b_disk_warm >= 256
    assert b_prompt_warm <= b_prompt_cold - 256
    assert b_content_warm == b_content_cold

    # uniform-config idempotence: re-running A's exact config adds no third unit
    a3 = _text_cache_server(512)
    a3.start()
    _golden_completion(a3)
    a3.stop()
    assert len(_bins()) == 2


def test_cross_ctx_peers_share_one_unit():
    """fp_n_ctx is neither an identity field nor a restore gate: a unit restores into any peer
    whose n_ctx holds its cells, in either direction, and peers differing only in n_ctx name it
    alike. Each way round the second peer restores the first peer's unit and, finding it already
    covering the prefix, writes no copy of its own."""
    for first, second in ((512, 1024), (1024, 512)):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        a = _text_cache_server(first)
        a.start()
        a_prompt_cold, a_content_cold, a_disk_cold = _golden_completion(a)
        a.stop()
        assert a_disk_cold == 0
        assert len(_bins()) == 1
        b = _text_cache_server(second)
        b.start()
        b_prompt, b_content, b_disk = _golden_completion(b)
        b.stop()
        assert b_disk >= 256, (first, second)         # cross-ctx reuse of A's unit
        assert b_prompt <= a_prompt_cold - 256
        assert b_content == a_content_cold
        assert len(_bins()) == 1, _bins()             # one unit, one name, both rungs

def test_missing_meta_is_transient_not_rejected():
    """A final-named .bin with no .meta sidecar yet is a peer mid-publish (the publish
    sequence renames the .bin first, the .meta last) or a writer that crashed in between.
    A scan that runs inside that window must NOT permanently blind this process to the
    unit: once the sidecar lands, a later lookup's rescan must index and restore it."""
    global server

    # cold baseline: no auto cache, full prefill
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_cold = res.body["timings"]["prompt_n"]
    content_cold = res.body["content"]
    server.stop()

    # simulate the publish window: only the fixture's .bin is visible, no sidecar
    bin_name = next(n for n in FIXTURE_SHA256 if n.endswith(".bin"))
    meta_name = next(n for n in FIXTURE_SHA256 if n.endswith(".meta"))
    verify_and_copy_fixture(CACHE_DIR, names={bin_name})
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.server_slots = True  # for the slot erase below
    server.start()

    # an unrelated block-sized request forces the startup scan + a lookup-miss rescan to
    # run while the sidecar is missing — the exact window that must not cache a rejection
    res = server.make_request("POST", "/completion", data={
        # >= one 256-token hash block (so the lookup path runs) but well under n_ctx 512
        "prompt": "A completely different story about a cat named Tom. " * 10,
        "n_predict": 8,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200

    # the peer finishes publishing: the sidecar lands (same process keeps running)
    verify_and_copy_fixture(CACHE_DIR, names={meta_name})
    # erase the slot so the cat prompt's resident KV cannot trip the restore margin gate
    # (disk must beat the in-memory match by a full block; the shared BOS token would
    # otherwise leave n_keep_mem=1 against a one-block n_keep_disk of 256) — the gate is
    # not under test here, the rescan's indexing of the completed unit is
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_warm = res.body["timings"]["prompt_n"]
    # the lookup-miss rescan must now index the completed unit and restore from it
    assert prompt_n_warm <= prompt_n_cold - 256
    assert res.body["content"] == content_cold


def test_torn_unit_refused():
    """A hand-crafted torn unit — a pristine, indexable .meta over a truncated .bin —
    is refused at restore time without a crash: the state load fails, the slot is left
    cleared and the request falls through to a clean full prefill."""
    global server

    # cold baseline: no auto cache, full prefill
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_cold = res.body["timings"]["prompt_n"]
    content_cold = res.body["content"]
    server.stop()

    # tear the unit: the sidecar is the fixture's (parses, fp-matches, gets indexed),
    # the state file is cut in half (llama_state_seq_load_file must fail cleanly)
    bin_name = next(n for n in FIXTURE_SHA256 if n.endswith(".bin"))
    meta_name = next(n for n in FIXTURE_SHA256 if n.endswith(".meta"))
    verify_and_copy_fixture(CACHE_DIR, names={meta_name})
    with open(os.path.join(FIXTURE_DIR, bin_name), "rb") as f:
        pristine_bin = f.read()
    with open(os.path.join(CACHE_DIR, bin_name), "wb") as f:
        f.write(pristine_bin[: len(pristine_bin) // 2])

    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.cache_ram = 0  # any reuse below could then only have come from the torn unit
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    # the indexed unit was selected, the load failed, and the request cold-prefilled
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["timings"]["prompt_n"] == prompt_n_cold
    assert res.body["content"] == content_cold


SLOT_META_VERSION_OFF = 4  # the version dword sits right after the 4-byte magic


def test_unknown_meta_version_skipped():
    """A sidecar with an unknown future version is skipped cleanly AND remembered by
    filename: published units are immutable after their atomic rename, so a rejection
    is permanent and no rescan in this process may ever re-open the file. The probe
    below violates that immutability on purpose — it swaps a valid v1 sidecar in under
    the SAME name; if any rescan re-opened the file, the unit would restore (exactly
    what test_v1_meta_still_indexed proves for a fresh process)."""
    global server

    # cold baseline: no auto cache, full prefill
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    prompt_n_cold = res.body["timings"]["prompt_n"]
    content_cold = res.body["content"]
    server.stop()

    # the fixture unit with its sidecar's version dword bumped to 99
    bin_name = next(n for n in FIXTURE_SHA256 if n.endswith(".bin"))
    meta_name = next(n for n in FIXTURE_SHA256 if n.endswith(".meta"))
    verify_and_copy_fixture(CACHE_DIR, names={bin_name})
    with open(os.path.join(FIXTURE_DIR, meta_name), "rb") as f:
        pristine_meta = f.read()
    v99 = bytearray(pristine_meta)
    struct.pack_into("<I", v99, SLOT_META_VERSION_OFF, 99)
    with open(os.path.join(CACHE_DIR, meta_name), "wb") as f:
        f.write(bytes(v99))

    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.cache_ram = 0    # reuse below could then only come from the disk unit
    server.server_slots = True  # for the slot erase below
    server.start()

    # skipped cleanly: full prefill, unchanged output, no crash
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["timings"]["prompt_n"] == prompt_n_cold
    assert res.body["content"] == content_cold

    # the sidecar becomes valid IN PLACE (same filename); the resident KV is erased so
    # only a disk restore could shrink the next prefill
    with open(os.path.join(CACHE_DIR, meta_name), "wb") as f:
        f.write(pristine_meta)
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    # remembered by filename: the lookup-miss rescan must NOT re-open the rejected
    # unit, so the request cold-prefills again
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["timings"]["prompt_n"] == prompt_n_cold
    assert res.body["content"] == content_cold


# deterministic 32x32 PNG generated once and frozen (no network, no fixture file): the
# mtmd bitmap id is the FNV-1a hash of these exact bytes, so the id in the .meta below
# is stable across runs
IMG_DATA_URI = "data:image/png;base64," + (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAhElEQVR42rXCkVYFAAAFwcUwDMPFMAzDMAzD"
    "8GEYhmEYhmEYhmEYhtntK3bOMEyzI9Ps2DQ7Mc1OTTNNszPT7Nw0uzDNLk2zK9Ps2jS7Mc1uTbM70+xgmt2b"
    "Zg+m2aNp9mSaPZtmL6bZq2n2Zpq9m2YfptmnafZlmn2bZj+m2a9p9mf6H5c0jFuX69A9AAAAAElFTkSuQmCC"
)

SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE
# byte offset of tok_count in a .meta sidecar: magic(4) + version(4) + fp fields(96)
SLOT_META_TOKS_OFF = 104
LLAMA_TOKEN_NULL = -1


def parse_meta(path: str):
    """Parse a .meta sidecar (v1 or v2) per the slot_meta_write layout; asserts exact EOF."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == SLOT_META_MAGIC
    tok_count = struct.unpack_from("<I", data, SLOT_META_TOKS_OFF)[0]
    off = SLOT_META_TOKS_OFF + 4 + 8  # skip tok_count + chain_hash
    toks = list(struct.unpack_from(f"<{tok_count}i", data, off))
    off += 4 * tok_count
    media = []
    if version == 2:
        fp_mmproj = struct.unpack_from("<Q", data, off)[0]
        assert fp_mmproj != 0
        off += 8
        n_media = struct.unpack_from("<I", data, off)[0]
        off += 4
        for _ in range(n_media):
            start_idx, n_tokens, n_pos, nx, ny, is_audio, id_len = struct.unpack_from("<7I", data, off)
            off += 28
            assert id_len > 0
            media.append({
                "start_idx": start_idx, "n_tokens": n_tokens, "n_pos": n_pos,
                "nx": nx, "ny": ny, "is_audio": is_audio,
                "id": data[off:off + id_len],
            })
            off += id_len
    assert off == len(data), f"trailing bytes in {path}"
    return version, toks, media


def test_vision_save_writes_v2_unit():
    """A media turn on a vision server persists a v2 unit (cell-aligned tokens + identity
    records tiling the NULL cells); a text-only turn on the same server stays a
    byte-layout v1 unit (invariant 0)."""
    vserver = ServerPreset.tinygemma3()
    vserver.slot_save_path = CACHE_DIR
    vserver.slot_save_auto = True
    # small hash block so the pre-image text prefix spans chunk-safe boundaries
    vserver.slot_save_block = 16
    vserver.slot_save_min_tokens = 0  # short-prompt test: keep the floor at the hash block size
    vserver.slot_save_node_prompt = "off"  # see create_server
    vserver.start()

    # pin each turn to its own slot: the prefix-similarity slot picker would otherwise route
    # the second request onto the first one's slot (shared chat-template preamble) and
    # overwrite its still-unsaved KV — both prompts must survive to the shutdown flush
    res = vserver.make_request("POST", "/chat/completions", data={
        "temperature": 0,
        "max_tokens": 4,
        "id_slot": 0,
        "messages": [
            {"role": "user", "content": [
                # >= 1 block of plain text BEFORE the image: those boundaries stay chunk-safe
                {"type": "text", "text": "Please describe the picture in as much detail as you possibly can. " * 4},
                {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
            ]},
        ],
    })
    assert res.status_code == 200

    res = vserver.make_request("POST", "/chat/completions", data={
        "temperature": 0,
        "max_tokens": 4,
        "id_slot": 1,
        "messages": [
            {"role": "user", "content": "Tell me a very long story about a dog named Spot. " * 2},
        ],
    })
    assert res.status_code == 200

    # graceful stop -> auto_save_slots_at_shutdown flushes both slots to the store
    vserver.stop()

    metas = {p: parse_meta(p) for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))}
    v1 = [m for m in metas.values() if m[0] == 1]
    v2 = [m for m in metas.values() if m[0] == 2]
    # the text-only turn on the vision server must NOT have become a v2 unit
    assert len(v1) >= 1
    for _, toks, media in v1:
        assert media == []
        assert all(t != LLAMA_TOKEN_NULL for t in toks)
    # the media turn produced a v2 unit whose records exactly tile the NULL cells
    assert len(v2) >= 1
    for _, toks, media in v2:
        assert len(media) >= 1
        n_null = sum(1 for t in toks if t == LLAMA_TOKEN_NULL)
        assert n_null == sum(r["n_tokens"] for r in media)
        for r in media:
            assert r["is_audio"] == 0
            assert r["n_tokens"] > 0 and r["n_pos"] > 0
            cells = toks[r["start_idx"]:r["start_idx"] + r["n_tokens"]]
            assert len(cells) == r["n_tokens"]
            assert all(t == LLAMA_TOKEN_NULL for t in cells)


def test_vision_save_persists_generation():
    """A media turn that GENERATES tokens persists the generated tail, not just the prompt
    prefix — media saves on the same terms as text (F7).

    An earlier build gated the media save path so that, on a FULL-seq-rm model, a snapshot
    was refused once generation had appended tokens (only prompt-prefix media states saved).
    That silently denied a completed image turn any disk cache even though the next
    conversation turn re-renders the assistant history to byte-identical tokens and so
    extends the whole snapshot (the FULL restore condition), exactly as a text turn does.
    The gate is gone; media is aligned with text.

    The gate only ever fired on a FULL-seq-rm context, and no CPU-runnable vision model is
    FULL-seq-rm (tinygemma3 is PART), so this cannot be a fail-before check on CPU — that
    discrimination lives in the rig recipe against qwen3.6-27b. What it DOES lock on CPU is
    the on-disk signature the fix guarantees on every backend: the saved v2 unit spans the
    prompt AND the generated cells, so any re-introduced save-side refusal/truncation of a
    prompt+generation media snapshot is caught here."""
    vs = make_vision_server(auto=True)
    vs.start()
    res = vs.make_request("POST", "/chat/completions", data={
        "temperature": 0,
        "max_tokens": 8,
        "id_slot": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VISION_TEXT_PRE},
            {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
        ]}],
    })
    assert res.status_code == 200
    t = res.body["timings"]
    prompt_n, predicted_n = t["prompt_n"], t["predicted_n"]
    assert predicted_n > 0  # the turn actually generated a tail to persist
    vs.stop()  # shutdown flush publishes the v2 unit

    v2 = read_v2_metas()
    assert len(v2) == 1
    _, toks, media = v2[0]
    # the snapshot is the prompt cells PLUS the generated cells (== prompt+generation);
    # a prompt-prefix-only save would stop at prompt_n. Allow a 1-cell slack for the single
    # in-flight sampled token that a mid-generation boundary can leave unpersisted.
    assert len(toks) > prompt_n
    assert len(toks) >= prompt_n + predicted_n - 1
    # the media records still tile only the image cells; the generated tail is plain tokens
    n_null = sum(1 for x in toks if x == LLAMA_TOKEN_NULL)
    assert n_null == sum(r["n_tokens"] for r in media)
    assert all(x != LLAMA_TOKEN_NULL for x in toks[-predicted_n:])


# two more frozen deterministic 32x32 PNGs (generated once, committed as bytes): distinct
# raw bytes => distinct mtmd bitmap ids, same pixel grid => same cell count per image
IMG2_DATA_URI = "data:image/png;base64," + (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAhUlEQVR42u3NoRrBYACG0b9Ns0ajTaNZ8zba"
    "NJo1Go1Go9FotGnTtkazRqPRaDQa17Hn+c4NHGN+ZD7YL/IPijdKFyon3CPsaUQ0Q9oB/preksGc0ZTJmNmQ"
    "RZ9Vl02HbYudR1znUCOpci5zdbgXeOZ4Z/laGAUKFChQoECBAgUKFKQl+AOBhvw9aGnVvAAAAABJRU5ErkJg"
    "gg=="
)
IMG3_DATA_URI = "data:image/png;base64," + (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAO0lEQVR42mP4/58BjhgY/sMRtcQZRi0gaAEt"
    "DEUVH7WAkPhoMh3NB6MWjOaD0XwwasFoPhjNB6MWgBEAy7P6TA6KJSQAAAAASUVORK5CYII="
)

# >= 3 hash blocks of 16 before the first image, so pre-image reuse spans real boundaries
VISION_TEXT_PRE  = "Please describe the picture in as much detail as you possibly can. " * 4
# >= 2 blocks between the two images of the adjacent test, so the block+chunk-safe clamp
# can land between them and reuse provably extends past the first image
VISION_TEXT_MID  = "Now compare it carefully against this second picture here. " * 3
# a text-only prompt spanning several 16-token blocks (text-on-vision-server tests)
VISION_TEXT_STORY = "Tell me a very long story about a dog named Spot. " * 4


def make_vision_server(auto: bool) -> ServerProcess:
    vs = ServerPreset.tinygemma3()
    # single slot: both images of the adjacent test must fit one sequence (n_ctx is split
    # across slots), and geometry must be identical between the saving and restoring runs
    # (fp_n_ctx is part of the snapshot fingerprint)
    vs.n_slots = 1
    vs.n_ctx = 1024
    vs.temperature = 0.0
    if auto:
        vs.slot_save_path = CACHE_DIR
        vs.slot_save_auto = True
        vs.slot_save_block = 16
        vs.slot_save_min_tokens = 0  # short-prompt test: keep the floor at the hash block size
        vs.slot_save_node_prompt = "off"  # see create_server
    return vs


def vision_request(vs: ServerProcess, contents: list, id_slot: int | None = None):
    parts = []
    for c in contents:
        if c.startswith("data:image/"):
            parts.append({"type": "image_url", "image_url": {"url": c}})
        else:
            parts.append({"type": "text", "text": c})
    data = {
        "temperature": 0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": parts}],
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
    res = vs.make_request("POST", "/chat/completions", data=data)
    assert res.status_code == 200
    timings = res.body["timings"]
    content = res.body["choices"][0]["message"]["content"]
    return timings["prompt_n"], timings["cache_n"], content


def read_v2_metas():
    """Parse all v2 .meta units in CACHE_DIR; returns [(path, toks, media), ...]."""
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))):
        version, toks, media = parse_meta(p)
        if version == 2:
            out.append((p, toks, media))
    return out


def parse_meta_node(path: str):
    """Parse a .meta sidecar of ANY version (v1..v4), returning the media tail AND the delta-node
    tail. Mirrors parse_meta but does not stop at v2: a v3 text-delta / v4 media-delta node appends
    parent_id + range_lo + range_hi (media-then-node order). A whole snapshot reports the implicit
    root (parent_id=0, range_lo=0, range_hi=tok_count). Asserts exact EOF."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == SLOT_META_MAGIC
    tok_count = struct.unpack_from("<I", data, SLOT_META_TOKS_OFF)[0]
    off = SLOT_META_TOKS_OFF + 4
    chain_hash = struct.unpack_from("<Q", data, off)[0]
    off += 8
    toks = list(struct.unpack_from(f"<{tok_count}i", data, off))
    off += 4 * tok_count
    media = []
    if version in (2, 4):
        off += 8  # fp_mmproj
        n_media = struct.unpack_from("<I", data, off)[0]
        off += 4
        for _ in range(n_media):
            start_idx, n_tokens, n_pos, nx, ny, is_audio, id_len = struct.unpack_from("<7I", data, off)
            off += 28
            assert id_len > 0
            media.append({"start_idx": start_idx, "n_tokens": n_tokens, "n_pos": n_pos,
                          "nx": nx, "ny": ny, "is_audio": is_audio, "id": data[off:off + id_len]})
            off += id_len
    parent_id, range_lo, range_hi = 0, 0, tok_count
    if version in (3, 4):
        parent_id, range_lo, range_hi = struct.unpack_from("<QII", data, off)
        off += 16
    assert off == len(data), f"trailing bytes in {path}"
    return {"version": version, "tok_count": tok_count, "toks": toks, "chain_hash": chain_hash,
            "media": media, "parent_id": parent_id, "range_lo": range_lo, "range_hi": range_hi}


def test_vision_incremental_writes_v4_delta_node():
    """End-to-end media delta: with --slot-save-incremental, a conversation turn that EXTENDS an
    already-persisted media prefix is restored from disk and re-saved as a v4 media delta node —
    a media tail (the WHOLE [0,N) record tiling) + a node tail (parent link + [parent_hi, N) range),
    the only format carrying both. Exercises U4 (media parent-find), U5/U6 (the pos_next range save +
    post-save cell-count assert — a mismatch would have fallen back to a whole v2), U7 (v4 emission)
    and U8 (restore/compose: session 2 cold-restores the v2 root before extending it).

    tinygemma3 is PART-seq-rm (no CPU vision model is FULL), so this locks the on-disk delta signature
    and the base restore; M-RoPE delta-compose CORRECTNESS against cold-prefill is the on-rig gate
    (it uses normal positions here). Two server sessions share CACHE_DIR so the base is genuinely on
    disk (not just in the resident slot) when the extending turn's parent-find runs."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)

    # session 1: a media turn -> a v2 media whole root on disk (the base).
    s1 = make_vision_server(auto=True)
    s1.slot_save_incremental = True
    s1.start()
    r1 = s1.make_request("POST", "/chat/completions", data={
        "temperature": 0, "max_tokens": 4, "id_slot": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VISION_TEXT_PRE},
            {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
        ]}],
    })
    assert r1.status_code == 200
    reply = r1.body["choices"][0]["message"]["content"]
    s1.stop()  # shutdown flush publishes the v2 root

    roots = [parse_meta_node(p) for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))]
    assert len(roots) == 1 and roots[0]["version"] == 2, "session 1 must persist exactly one v2 media root"
    root = roots[0]
    assert len(root["media"]) >= 1  # the image chunk is recorded

    # session 2 (fresh process, shared dir): continue the SAME conversation. The re-rendered prefix
    # (user turn + assistant reply) strict-extends the base's cells, so it cold-restores the v2 root
    # from disk (proving U8) and the extending turn is saved as a v4 delta chained to it.
    s2 = make_vision_server(auto=True)
    s2.slot_save_incremental = True
    s2.start()
    r2 = s2.make_request("POST", "/chat/completions", data={
        "temperature": 0, "max_tokens": 4, "id_slot": 0,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": VISION_TEXT_PRE},
                {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
            ]},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": VISION_TEXT_STORY},
        ],
    })
    assert r2.status_code == 200
    # the base restored from disk: the extending turn reused ~the whole base prefix rather than
    # cold-prefilling it (U8 media restore + compose).
    assert r2.body["timings"]["cache_n"] >= root["tok_count"] - s2.slot_save_block, \
        f"session 2 must restore the media base from disk; cache_n={r2.body['timings']['cache_n']}"
    s2.stop()  # shutdown flush publishes the v4 delta

    metas = {p: parse_meta_node(p) for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))}
    v2s = [m for m in metas.values() if m["version"] == 2]
    v4s = [m for m in metas.values() if m["version"] == 4]
    assert len(v2s) == 1 and len(v4s) == 1, \
        f"expected exactly one v2 root + one v4 media delta, got {sorted(m['version'] for m in metas.values())}"
    delta = v4s[0]
    # the v4 delta chains to the root and its .bin covers only [parent_hi, N)
    assert delta["parent_id"] == root["chain_hash"], "the v4 delta must chain to the v2 root"
    assert delta["range_lo"] == root["tok_count"], "the delta's KV range begins at the root's cell count"
    assert delta["range_hi"] == delta["tok_count"] > root["tok_count"], \
        "the delta covers [root_len, full_len) and is longer than the root"
    # the v4 meta carries the WHOLE tiling (media tail present) even though the .bin is a delta
    assert len(delta["media"]) >= 1, "a v4 delta carries the full [0,N) media record tiling"
    n_null = sum(1 for t in delta["toks"] if t == LLAMA_TOKEN_NULL)
    assert n_null == sum(r["n_tokens"] for r in delta["media"]), "records tile every NULL cell of [0,N)"

    # session 3 (decision 2 / U9): a MANUAL /slots restore pointed at the v4 delta tip MUST NOT
    # refuse it — the manual path uses the SAME shared chain-walk helper, so it composes the base
    # v2 root + this v4 delta to the full cell set and rehydrates the media records into stubs. A
    # pre-U9 manual restore would have loaded the partial delta .bin alone (or refused). We assert
    # it succeeds and reports the WHOLE composed token count, not the delta's [parent_hi, N) slice.
    v4_meta_path = [p for p, m in metas.items() if m["version"] == 4][0]
    v4_bin_name = os.path.basename(v4_meta_path[:-len(".meta")])
    s3 = make_vision_server(auto=True)
    s3.slot_save_incremental = True
    s3.server_slots = True  # expose the manual /slots endpoints
    s3.start()
    mres = s3.make_request("POST", "/slots/0?action=restore", data={"filename": v4_bin_name})
    assert mres.status_code == 200, f"manual restore of a v4 media delta tip must not refuse: {mres.body}"
    assert mres.body["n_restored"] == delta["tok_count"], \
        "manual restore composes the whole [0,N) cell set, not just the delta's tail slice"
    s3.stop()

    # (no delta-vs-root .bin size assertion here: tinygemma3 is iSWA, so kv_swa is written WHOLE at
    # every node and grows with the TOTAL sequence length — the delta shrinks only the kv_base
    # portion, so a delta over a longer total prompt can exceed a shorter whole root on disk. The
    # write-amplification win is model-family-dependent, per the design's family analysis; the
    # structural range/parent/tiling asserts above are what pin the v4 delta's correctness.)

    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def test_text_only_on_vision_server():
    """A TEXT-ONLY prompt on a --mmproj server caches and restores across a restart —
    the #21133 case: the media gate is per-request, so merely loading a projector must
    not disable text caching."""
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_cold, cache_n_cold, content_cold = vision_request(vs, [VISION_TEXT_STORY])
    assert cache_n_cold == 0
    assert prompt_n_cold > 32  # sanity: spans several 16-token blocks
    vs.stop()  # shutdown flush publishes the unit

    # the persisted unit keeps the base v1 on-disk shape (no media section)
    metas = glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))
    assert len(metas) == 1
    assert read_v2_metas() == []

    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_warm, cache_n_warm, content_warm = vision_request(vs, [VISION_TEXT_STORY])
    vs.stop()
    assert prompt_n_warm <= 16  # at most a partial block re-decoded
    assert cache_n_warm >= prompt_n_cold - 16
    assert content_warm == content_cold


def test_vision_full_reuse():
    """A media prompt saved at shutdown restores across a full server restart: the
    byte-identical resend reuses (nearly) the whole prompt from disk and the greedy
    answer is unchanged."""
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_cold, cache_n_cold, content_cold = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    assert cache_n_cold == 0
    assert prompt_n_cold > 200  # sanity: the image cells dominate the prompt
    vs.stop()  # shutdown flush publishes the v2 unit

    assert len(read_v2_metas()) == 1

    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_warm, cache_n_warm, content_warm = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    vs.stop()
    # the whole request is a verified prefix of the snapshot; only the [TAG_PROMPT_LOGITS]
    # tail token (plus at most a partial block) is re-decoded
    assert prompt_n_warm <= 16
    assert cache_n_warm >= prompt_n_cold - 16
    assert content_warm == content_cold


def test_vision_zero_reprefill_faithfulness():
    """Byte-identical resend of a WHOLE media snapshot, verifying the zero-re-prefill
    contract is FAITHFULNESS, not bit-identity.

    When the resend covers the entire snapshot, the first generated token is produced by
    a single decode into the restored KV rather than by a prefill batch. On a FULL-seq-rm
    SYCL/flash-attn backend those two paths reduce in a different order, so the greedy run
    can pick a different first token at a near-tie (rig-observed on qwen3.6-27b: it swapped
    one near-synonym then stayed coherent) — the KV round-trips faithfully but the
    continuation is not guaranteed bit-identical to an uninterrupted cold run, exactly like
    upstream /slots. This is the CPU counterpart of rigtest's scen_restore1: on CPU decode
    and prefill are deterministic so identity does hold, but the contract asserted here is
    the weaker faithfulness one (restore fired, whole snapshot reused, a coherent answer)."""
    # cold reference from a server with no cache at all (cross-process determinism control)
    ref = make_vision_server(auto=False)
    ref.start()
    _, cache_n_ref, content_cold = vision_request(ref, [VISION_TEXT_PRE, IMG_DATA_URI])
    ref.stop()
    assert cache_n_ref == 0
    assert len(content_cold) > 0

    # save a whole-prompt media snapshot at shutdown
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_cold, _, _ = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    vs.stop()
    assert len(read_v2_metas()) == 1

    # byte-identical resend: the whole request is a verified prefix of the snapshot
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_warm, cache_n_warm, content_warm = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    vs.stop()
    # faithfulness signal 1: auto-restore fired and reused essentially the whole snapshot,
    # so this really is the (near) zero-re-prefill path and not a partial-prefix reuse
    assert prompt_n_warm <= 16
    assert cache_n_warm >= prompt_n_cold - 16
    # faithfulness signal 2: a coherent, non-empty continuation. Bit-identity is NOT the
    # contract (see docstring) — but CPU decode == prefill deterministically, so here the
    # answer additionally matches the cold reference; a near-tie first-token drift on a
    # SYCL backend would still be faithful.
    assert len(content_warm) > 0
    assert content_warm == content_cold


def test_vision_prefix_reuse_different_image():
    """Same text + a DIFFERENT image: per-record verification truncates reuse to the
    pre-image prefix — the image itself is re-encoded and re-decoded, and the answer
    matches a cold run of the new image."""
    # cold reference for the image-B request, on a server with no cache at all
    ref = make_vision_server(auto=False)
    ref.start()
    _, _, content_b_ref = vision_request(ref, [VISION_TEXT_PRE, IMG2_DATA_URI])
    ref.stop()

    # save an image-A unit
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_cold, _, _ = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    vs.stop()
    metas = read_v2_metas()
    assert len(metas) == 1
    (rec,) = metas[0][2]  # exactly one media record: the image-A chunk
    s1 = rec["start_idx"]

    # byte-identical text, different image bytes => reuse exactly the pre-image prefix
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_warm, cache_n_warm, content_warm = vision_request(vs, [VISION_TEXT_PRE, IMG2_DATA_URI])
    vs.stop()
    assert cache_n_warm >= 16       # at least one whole block restored from disk
    assert cache_n_warm <= s1       # and never a single cell of the mismatched image
    assert prompt_n_warm >= prompt_n_cold - s1  # the new image was fully re-processed
    assert content_warm == content_b_ref


def test_adjacent_images_second_mismatch():
    """Two images in one prompt; on resend the SECOND differs: per-record iteration
    verifies each image separately, so reuse extends past the first image's cells and
    truncates exactly at the second's record."""
    ref = make_vision_server(auto=False)
    ref.start()
    _, _, content_ac_ref = vision_request(
        ref, [VISION_TEXT_PRE, IMG_DATA_URI, VISION_TEXT_MID, IMG3_DATA_URI])
    ref.stop()

    vs = make_vision_server(auto=True)
    vs.start()
    vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI, VISION_TEXT_MID, IMG2_DATA_URI])
    vs.stop()
    metas = read_v2_metas()
    assert len(metas) == 1
    rec1, rec2 = metas[0][2]  # ordered by start_idx: image A, image B
    end1 = rec1["start_idx"] + rec1["n_tokens"]
    s2 = rec2["start_idx"]
    assert s2 - end1 >= 32  # the mid text really spans blocks (test-shape sanity)

    # image A unchanged, image B -> C: the first record verifies, the second mismatches
    vs = make_vision_server(auto=True)
    vs.start()
    _, cache_n_warm, content_warm = vision_request(
        vs, [VISION_TEXT_PRE, IMG_DATA_URI, VISION_TEXT_MID, IMG3_DATA_URI])
    vs.stop()
    assert cache_n_warm > end1  # reuse extends PAST the first image (it was verified)
    assert cache_n_warm <= s2   # and truncates at the mismatched second record
    assert content_warm == content_ac_ref


def test_corrupt_v2_meta_fallback():
    """Corrupt v2 sidecars degrade cleanly: a torn/invalid meta is skipped (full
    prefill), a wrong id hard-mismatches (pre-image reuse only) — never a crash, and
    the answer never changes."""
    vs = make_vision_server(auto=True)
    vs.start()
    prompt_n_cold, _, content_cold = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
    vs.stop()
    metas = read_v2_metas()
    assert len(metas) == 1
    meta_path, toks, media = metas[0]
    s1 = media[0]["start_idx"]
    bin_path = meta_path[:-len(".meta")]
    with open(meta_path, "rb") as f:
        pristine_meta = f.read()
    with open(bin_path, "rb") as f:
        pristine_bin = f.read()
    # offset of the first media record (and of its id) in the v2 layout
    rec_off = SLOT_META_TOKS_OFF + 4 + 8 + 4 * len(toks) + 8 + 4
    id_off = rec_off + 28

    def corrupt_truncated(data: bytes) -> bytes:
        return data[:-10]

    def corrupt_tiling(data: bytes) -> bytes:
        # first record's start_idx += 1: records no longer tile the NULL cells
        (start_idx,) = struct.unpack_from("<I", data, rec_off)
        out = bytearray(data)
        struct.pack_into("<I", out, rec_off, start_idx + 1)
        return bytes(out)

    def corrupt_id(data: bytes) -> bytes:
        # structurally valid meta whose image id no longer matches the request
        out = bytearray(data)
        out[id_off] ^= 0x01
        return bytes(out)

    for corrupt, full_prefill in [
        (corrupt_truncated, True),
        (corrupt_tiling, True),
        (corrupt_id, False),
    ]:
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        with open(bin_path, "wb") as f:
            f.write(pristine_bin)
        with open(meta_path, "wb") as f:
            f.write(corrupt(pristine_meta))
        vs = make_vision_server(auto=True)
        vs.start()
        prompt_n, cache_n, content = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI])
        vs.stop()
        assert content == content_cold, corrupt.__name__
        if full_prefill:
            # unparseable unit: skipped and remembered, clean cold prefill
            assert cache_n == 0 and prompt_n == prompt_n_cold, corrupt.__name__
        else:
            # parseable unit with a foreign id: hard mismatch truncates to the
            # pre-image prefix — reuse stops before the image, which is re-processed
            assert 16 <= cache_n <= s1, corrupt.__name__
            assert prompt_n >= prompt_n_cold - s1, corrupt.__name__


def test_cross_process_share():
    """Two live processes share one cache dir: A publishes a media snapshot (shutdown
    flush), and B — running since BEFORE the unit existed — picks it up through the
    dir-mtime refresh / lookup-miss rescan and restores it."""
    a = make_vision_server(auto=True)
    b = make_vision_server(auto=True)
    b.server_port = 8580  # distinct port: both processes are alive at once
    a.start()
    b.start()  # B's startup scan sees an EMPTY dir — the unit must arrive via refresh

    prompt_n_cold, cache_n_cold, content_cold = vision_request(a, [VISION_TEXT_PRE, IMG_DATA_URI])
    assert cache_n_cold == 0
    a.stop()  # A's shutdown flush publishes the unit
    assert len(read_v2_metas()) == 1

    prompt_n_warm, cache_n_warm, content_warm = vision_request(b, [VISION_TEXT_PRE, IMG_DATA_URI])
    b.stop()
    assert prompt_n_warm <= 16
    assert cache_n_warm >= prompt_n_cold - 16
    assert content_warm == content_cold


def make_parallel_vision_server() -> ServerProcess:
    vs = ServerPreset.tinygemma3()
    vs.n_slots = 2   # --parallel 2: n_ctx is split across slots (1024 each)
    vs.n_ctx = 2048
    vs.temperature = 0.0
    vs.slot_save_path = CACHE_DIR
    vs.slot_save_auto = True
    vs.slot_save_block = 16
    vs.slot_save_min_tokens = 0  # short-prompt test: keep the floor at the hash block size
    vs.slot_save_node_prompt = "off"  # see create_server
    return vs


def test_parallel_slots_media():
    """--parallel 2 with CONCURRENT media + text traffic sharing one cache dir: each
    slot's sequence saves and restores independently (general users run multi-slot —
    prod's --parallel 1 must not be a hidden assumption)."""
    vs = make_parallel_vision_server()
    vs.start()
    media_cold, text_cold = parallel_function_calls([
        (vision_request, (vs, [VISION_TEXT_PRE, IMG_DATA_URI], 0)),
        (vision_request, (vs, [VISION_TEXT_STORY], 1)),
    ])
    assert media_cold is not None and text_cold is not None
    assert media_cold[1] == 0 and text_cold[1] == 0  # both truly cold
    vs.stop()  # shutdown flush persists BOTH slots

    # the media slot published a v2 unit, the text slot a v1 unit, in the same dir
    assert len(glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))) == 2
    assert len(read_v2_metas()) == 1

    vs = make_parallel_vision_server()
    vs.start()
    media_warm, text_warm = parallel_function_calls([
        (vision_request, (vs, [VISION_TEXT_PRE, IMG_DATA_URI], 0)),
        (vision_request, (vs, [VISION_TEXT_STORY], 1)),
    ])
    assert media_warm is not None and text_warm is not None
    vs.stop()
    for (prompt_n_cold, _, content_cold), (prompt_n_warm, cache_n_warm, content_warm) in (
            (media_cold, media_warm), (text_cold, text_warm)):
        assert prompt_n_warm <= 16
        assert cache_n_warm >= prompt_n_cold - 16
        assert content_warm == content_cold


# --- manual /slots endpoints (per-slot media gate) --------------------------------
# a server-wide check_no_mtmd guard used to 501 EVERY manual save/restore/erase as
# soon as --mmproj was loaded. The gate is per-slot now: text slots keep the exact
# pre-media behaviour (state file + optional .logits, no sidecar), media slots write
# a v2 .meta and restore by rehydrating identity-only STUB chunks from it — stubs
# verify and count positions like live chunks but can never be re-encoded (any path
# that would need to refuses and clears the slot instead).
#
# Snapshots here are taken with n_predict=0 (prompt-only): on an SWA model a snapshot
# that extends PAST the follow-up request (e.g. by generated tokens) invalidates the
# restored checkpoint (pos_max > pos_next) and forces a cold re-process, so the
# identical resend below could not demonstrate reuse. Prompt-only manual snapshots
# are exactly the restorable artifact class (mirroring the FULL-model finding).

MEDIA_MARKER = "<__media__>"  # mtmd_default_marker()
IMG_B64 = IMG_DATA_URI.split(",", 1)[1]
IMG2_B64 = IMG2_DATA_URI.split(",", 1)[1]
MEDIA_PROMPT = VISION_TEXT_PRE + MEDIA_MARKER + " Describe it now."
# Batch-split noise on this tiny Q8 model, measured 2026-10-03 on CPU with NO restore involved: a
# 386-token text prompt re-sent in-process (its tail re-decoded from a checkpoint, 5 tokens) moves the
# first token's top-8 logprobs by 0.042 against the cold prefill; a 1-token re-decode after a restore of
# MEDIA_PROMPT moves them by 0.036 (the same on main-patched 593c2d0d7). Short prompts show exactly 0.
MEDIA_SPLIT_TOL = 0.1


def assert_media_close_to_cold(vs_cold: ServerProcess, prompt: str, files: list, warm_body, other_files: list):
    """Independent oracle for a restored media state: the first token's top-8 logprobs against a cold
    prefill of the same prompt in a process that never restored anything. The restored run decodes the
    last prompt token in another batch split, which on this model moves the logprobs by float noise
    (MEDIA_SPLIT_TOL, measured), so equality of the greedy text is not a valid criterion. The oracle is
    shown to discriminate on every run: the same prompt with a different image must differ from the
    cold one by more than the tolerance, or the check proves nothing and fails."""
    cold = raw_media_request_body(vs_cold, prompt, files, n_predict=1, n_probs=8)
    assert cold["timings"]["cache_n"] == 0, "the oracle must prefill cold"
    other = raw_media_request_body(vs_cold, prompt, other_files, n_predict=1, n_probs=8)
    lp_warm, lp_cold, lp_other = (first_token_logprobs(b) for b in (warm_body, cold, other))
    worst = max(abs(lp_warm[k] - lp_cold[k]) if k in lp_warm else float("inf") for k in lp_cold)
    contrast = max(abs(lp_other[k] - lp_cold[k]) if k in lp_other else float("inf") for k in lp_cold)
    print(f"restored vs cold first-token logprobs: max diff {worst:.3g}; other image vs cold {contrast:.3g}")
    assert contrast > MEDIA_SPLIT_TOL, f"the oracle cannot tell two images apart ({contrast:.3g})"
    assert worst <= MEDIA_SPLIT_TOL, f"restored state differs from cold by {worst:.3g}"


def make_manual_vision_server() -> ServerProcess:
    # raw /completion prompts carry media via the plain media marker (no chat template)
    os.environ["LLAMA_MEDIA_MARKER"] = MEDIA_MARKER
    vs = make_vision_server(auto=False)
    vs.slot_save_path = CACHE_DIR
    return vs


def raw_media_request_body(vs: ServerProcess, prompt_string: str, files: list, n_predict: int = 8, n_probs: int = 0):
    data = {
        "prompt": {"prompt_string": prompt_string, "multimodal_data": files},
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": 0,
    }
    if n_probs > 0:
        data.update({"n_probs": n_probs, "post_sampling_probs": False})
    res = vs.make_request("POST", "/completion", data=data)
    assert res.status_code == 200
    return res.body


def raw_media_request(vs: ServerProcess, prompt_string: str, files: list, n_predict: int = 8):
    """/completion with a raw prompt string + media files: byte-stable across resends
    (no chat-template re-render), which is what snapshot-matching follow-ups need."""
    body = raw_media_request_body(vs, prompt_string, files, n_predict)
    t = body["timings"]
    return t["prompt_n"], t["cache_n"], body["content"]


def first_token_logprobs(body) -> dict:
    return {t["id"]: t["logprob"] for t in body["completion_probabilities"][0]["top_logprobs"]}


def _metric_v(s: ServerProcess, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def _make_node_vision_server() -> ServerProcess:
    """The vision server with the DEFAULT node settings (the system node on, the prompt node cold),
    the system node's floor at one block."""
    os.environ["LLAMA_MEDIA_MARKER"] = MEDIA_MARKER
    vs = make_vision_server(auto=True)
    vs.slot_save_node_prompt = None
    vs.slot_save_context_min_tokens = 0
    vs.server_metrics = True
    return vs


def _metas_by_len():
    out = {}
    for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")):
        version, toks, media = parse_meta(p)
        out[len(toks)] = (p[:-len(".meta")], version, toks, media)
    return out


# the text after the image is longer than one block (the prompt node lands in it, after the image) or
# shorter (the block-aligned position falls inside the image and the node moves down to its start)
MEDIA_TAIL_LONG  = " Describe it now, in detail, listing every colour, every shape and every object you can see in it."
MEDIA_TAIL_SHORT = " Describe it now."


@pytest.mark.parametrize("tail", [MEDIA_TAIL_LONG, MEDIA_TAIL_SHORT])
def test_vision_prompt_node_default(tail):
    """The default prompt node on a media prompt (it used to skip every request carrying media). It is
    cut where the previous cell is text, so no chunk is split: after the image when enough text follows
    it (a v2 unit holding the image), else at the image's start (a v1 text unit). A same-prompt resend
    after a restart, with the conversation unit removed, restores exactly that node, and its first token
    matches a cold prefill within the measured split noise."""
    prompt = VISION_TEXT_PRE + MEDIA_MARKER + tail
    vs = _make_node_vision_server()
    vs.start()
    body = raw_media_request_body(vs, prompt, [IMG_B64], n_predict=1)
    n_prompt = body["timings"]["prompt_n"]
    assert _metric_v(vs, "auto_cache_node_media_skipped_total") == 0
    vs.stop()
    metas = _metas_by_len()
    inside = sorted(n for n in metas if n < n_prompt)
    assert len(inside) == 1, f"units {sorted(metas)}, prompt {n_prompt}"
    node = inside[0]
    _, version, toks, media = metas[node]
    assert toks[-1] != LLAMA_TOKEN_NULL, "a node must end on a text cell"
    if tail is MEDIA_TAIL_LONG:
        assert version == 2 and len(media) == 1, (version, media)
    else:
        assert version == 1, version   # all text: the cut moved down to the image's start
    # leave only the node: the conversation unit (prompt + generation) would otherwise be preferred
    for n, (path, _, _, _) in metas.items():
        if n != node:
            for suffix in ("", ".meta", ".logits", ".dft"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)

    vs = _make_node_vision_server()
    vs.start()
    warm = raw_media_request_body(vs, prompt, [IMG_B64], n_predict=1, n_probs=8)
    vs.stop()
    assert warm["timings"].get("cache_disk_n", 0) == node, warm["timings"]
    cold = make_vision_server(auto=False)
    cold.start()
    assert_media_close_to_cold(cold, prompt, [IMG_B64], warm, [IMG2_B64])
    cold.stop()


def test_vision_system_node_default():
    """The system node on a chat request carrying an image (the template-generic preamble end used to
    be skipped for media): a later conversation with the same system prompt and a different image
    restores it after a restart."""
    sys_msg = {"role": "system", "content": "You are a careful assistant who describes pictures for a museum catalogue. "
                                            "Keep every answer short and never invent details."}

    def req(vs, img):
        res = vs.make_request("POST", "/chat/completions", data={
            "temperature": 0, "max_tokens": 4, "id_slot": 0,
            "messages": [sys_msg, {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What is in this picture?"}]}]})
        assert res.status_code == 200, res.body
        return res.body["timings"]

    vs = _make_node_vision_server()
    vs.start()
    t = req(vs, IMG_DATA_URI)
    assert _metric_v(vs, "auto_cache_sysnode_probed_total") >= 1
    assert _metric_v(vs, "auto_cache_sysnode_seam_mismatch_total") == 0
    assert _metric_v(vs, "auto_cache_node_media_skipped_total") == 0
    vs.stop()
    inside = sorted(n for n in _metas_by_len() if n < t["prompt_n"])
    assert inside, f"no node inside the prompt: {sorted(_metas_by_len())}"
    sys_node = inside[0]
    assert _metas_by_len()[sys_node][1] == 1   # the preamble is text

    vs = _make_node_vision_server()
    vs.start()
    t2 = req(vs, IMG2_DATA_URI)
    vs.stop()
    assert t2.get("cache_disk_n", 0) >= sys_node, t2


def test_manual_slots_text_on_vision_server():
    """Manual save/restore/erase of a TEXT slot works under --mmproj (per-slot gate
    replacing the server-wide 501) and keeps the base on-disk shape: no .meta."""
    prompt_a = "Tell me a very long story about a dog named Spot. " * 2

    # prompt-only snapshot (see the n_predict=0 note above)
    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/completion", data={
        "prompt": prompt_a, "n_predict": 0, "temperature": 0,
        "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 0

    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "text.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved == prompt_n_full  # prompt-only snapshot
    # text snapshots keep the pre-media on-disk shape: no identity sidecar
    assert os.path.exists(os.path.join(CACHE_DIR, "text.bin"))
    assert not os.path.exists(os.path.join(CACHE_DIR, "text.bin.meta"))
    vs.stop()

    # fresh process: cold reference, then erase + restore + identical resend reuses it
    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/completion", data={
        "prompt": prompt_a, "n_predict": 8, "temperature": 0,
        "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200
    content_ref = res.body["content"]
    assert res.body["timings"]["cache_n"] == 0

    # erase (also previously 501 under --mmproj) empties the slot
    res = vs.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "text.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = vs.make_request("POST", "/completion", data={
        "prompt": prompt_a, "n_predict": 8, "temperature": 0,
        "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= n_saved - 8  # restored state actually reused
    assert res.body["content"] == content_ref
    vs.stop()


def _file_bytes(name: str) -> bytes:
    with open(os.path.join(CACHE_DIR, name), "rb") as f:
        return f.read()


def test_manual_media_save_restore_continues():
    """Manual SAVE of a media slot writes upstream's packed server_tokens payload (no
    sidecar); a fresh process manually RESTORES it to exactly the saved state, and the
    identical request reuses the restored cells INCLUDING the image's (the image is never
    re-encoded).

    Fidelity is checked on the state itself: saving the restored slot again must give a
    byte-identical .bin. The answer is compared only against a run that decodes
    the same suffix in the same batch split (an in-process erase + restore): on the tiny CI
    model the greedy answer depends on the split alone, e.g. the in-memory continuation
    re-decodes 5 tokens from a prefill checkpoint where a restore re-decodes 1, and those two
    give different text from one and the same state."""
    vs = make_manual_vision_server()
    vs.start()
    prompt_n_full, cache_n_cold, _ = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    assert cache_n_cold == 0
    assert prompt_n_full > 200  # sanity: the image cells dominate the prompt

    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved == prompt_n_full  # prompt-only snapshot
    # reference answer: the same restore-then-continue, inside the process that saved it
    res = vs.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 200
    prompt_n_ref, _, content_ref = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])
    vs.stop()

    # upstream's format: no sidecar, and the token payload opens with the packed marker
    assert not os.path.exists(os.path.join(CACHE_DIR, "media.bin.meta"))
    data = _file_bytes("media.bin")
    assert struct.unpack_from("<i", data, 12)[0] == LLAMA_TOKEN_NULL
    assert struct.unpack_from("<I", data, 20)[0] == n_saved  # packed: marker, version, n_tokens

    # fresh process: a cold run first (the slot then holds other state), then erase + restore
    vs = make_manual_vision_server()
    vs.start()
    raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])

    res = vs.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # the restored state is exactly the saved one
    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media-rt.bin"})
    assert res.status_code == 200
    assert res.body["n_saved"] == n_saved
    assert _file_bytes("media-rt.bin") == _file_bytes("media.bin")
    assert not os.path.exists(os.path.join(CACHE_DIR, "media-rt.bin.meta"))

    warm = raw_media_request_body(vs, MEDIA_PROMPT, [IMG_B64], n_probs=8)
    prompt_n_warm, cache_n_warm, content_warm = warm["timings"]["prompt_n"], warm["timings"]["cache_n"], warm["content"]
    assert cache_n_warm == n_saved - prompt_n_warm        # every saved cell came from the restore
    assert prompt_n_warm <= 8                             # the image was NOT re-processed
    assert prompt_n_warm == prompt_n_ref                  # same split as the reference ...
    assert content_warm == content_ref                    # ... so the same answer
    vs.stop()

    # INDEPENDENT oracle, no restore anywhere: a fresh process prefills the same prompt cold. Its
    # greedy text may differ from the restored run's on this tiny model (the last prompt token is decoded
    # in another batch split), so the criterion is the first token's top-8 distribution, which a wrong
    # restored state would move by far more than float noise.
    vs = make_manual_vision_server()
    vs.start()
    assert_media_close_to_cold(vs, MEDIA_PROMPT, [IMG_B64], warm, [IMG2_B64])
    vs.stop()


def test_manual_media_restore_stub_never_encoded():
    """After a manual media restore the slot's image chunks are identity-only stubs: a
    follow-up with a DIFFERENT image id-mismatches them, so they are dropped (never
    encoded) and the request re-processes its own image. Both that answer and the one for
    the original image afterwards must equal a reference process that ran the same
    sequence from a restore of the same file (same cached prefixes and batch splits; see
    test_manual_media_save_restore_continues for why a cold run is not a valid reference on
    the CI model)."""
    vs = make_manual_vision_server()
    vs.start()
    raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    # reference sequence: erase, restore, the different image, then the original one
    assert vs.make_request("POST", "/slots/0?action=erase").status_code == 200
    assert vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"}).status_code == 200
    ref_b = raw_media_request(vs, MEDIA_PROMPT, [IMG2_B64])
    ref_a = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])
    vs.stop()

    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 200

    # different image bytes => different id => the stub never matches nor encodes: the
    # request processes its own image, exactly as in the reference
    out_b = raw_media_request(vs, MEDIA_PROMPT, [IMG2_B64])
    assert out_b[1] < out_b[0]  # cache_n stops before the image: the stub was not reused
    assert out_b == ref_b

    # and the original image request answers exactly like the reference sequence
    body_a = raw_media_request_body(vs, MEDIA_PROMPT, [IMG_B64], n_probs=8)
    out_a = (body_a["timings"]["prompt_n"], body_a["timings"]["cache_n"], body_a["content"])
    assert out_a == ref_a
    vs.stop()

    # INDEPENDENT oracles, no restore and no stub anywhere. The different image: on this sliding-window
    # model it cannot rewind into the restored state, so the stub run processes it in full, exactly as a
    # fresh cold process does (asserted: same split, same answer). The original image afterwards: its
    # first-token distribution against a cold prefill, within the measured split noise.
    vs = make_manual_vision_server()
    vs.start()
    cold_b = raw_media_request(vs, MEDIA_PROMPT, [IMG2_B64])
    vs.stop()
    assert cold_b[:2] == out_b[:2], f"oracle split {cold_b[:2]} vs stub run {out_b[:2]}"
    assert cold_b[2] == out_b[2]
    vs = make_manual_vision_server()
    vs.start()
    assert_media_close_to_cold(vs, MEDIA_PROMPT, [IMG_B64], body_a, [IMG2_B64])
    vs.stop()


def test_manual_media_restore_refuses_cells_without_meta():
    """A plain cell list with media cells and no .meta sidecar (a pre-2026-10 manual media
    save that lost its sidecar) has nothing to rebuild the media from: the manual restore
    refuses with an explicit error AND leaves the slot cleared-but-usable."""
    vs = make_manual_vision_server()
    vs.start()
    prompt_n_full, _, _ = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    vs.stop()

    # rewrite upstream's packed payload into the old plain cell list (NULL media cells, no records)
    path = os.path.join(CACHE_DIR, "media.bin")
    data = bytearray(_file_bytes("media.bin"))
    payload_end = 12 + struct.unpack_from("<I", data, 8)[0] * 4
    n_tokens = struct.unpack_from("<I", data, 20)[0]
    cells = data[24:24 + n_tokens * 4]
    assert LLAMA_TOKEN_NULL in struct.unpack_from(f"<{n_tokens}i", cells, 0)
    data = data[:12] + cells + data[payload_end:]
    struct.pack_into("<I", data, 8, n_tokens)
    with open(path, "wb") as f:
        f.write(data)

    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 400
    assert "Invalid tokens in slot save file" in str(res.body)

    # the refused restore dropped the loaded state entirely: the slot cold-serves
    prompt_n, cache_n, content = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])
    assert cache_n == 0
    assert prompt_n == prompt_n_full
    vs.stop()


# -- idle-delay flush (C9) -----------------------------------------------------------------
# The two legacy save sites only fire on next-task-arrival (get_available_slot reclaim) or on
# a graceful shutdown, so a lone request's warm KV stays crash-volatile and invisible to peer
# instances until more traffic lands. --slot-save-idle-seconds closes that window: a slot idle
# for N seconds is flushed like any other site (v2 for media), on the main-loop thread, with no
# second request and no shutdown.

IDLE_SECONDS = 3


def _queued(s) -> float:
    # saves handed to the background writer: a unit captured but not yet published is not on disk
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith("llamacpp:auto_cache_save_queued_total "):
            return float(line.split()[1])
    raise AssertionError("auto_cache_save_queued_total not found")


def _units_on_disk():
    return sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta")))


def _wait_for_unit(timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        metas = _units_on_disk()
        if metas:
            return metas
        time.sleep(0.25)
    return _units_on_disk()


def test_idle_flush_text():
    """A single text completion, with no follow-up request and no shutdown, is flushed to the
    auto disk cache after the idle delay: nothing is written while the slot is processing or at
    completion (both legacy sites need a NEXT task), then the timed idle wake persists it while
    the server keeps running, exactly once."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_block = 16
    server.slot_save_idle_seconds = IDLE_SECONDS
    server.server_metrics = True
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "n_predict": 8,
        "temperature": 0,
    })
    assert res.status_code == 200
    # the request itself (processing + completion) writes nothing, and captures nothing either
    assert _units_on_disk() == []
    assert _queued(server) == 0

    metas = _wait_for_unit(IDLE_SECONDS + 10)
    assert len(metas) == 1, "the idle slot must have been flushed to disk after the delay"
    version, toks, media = parse_meta(metas[0])
    assert version == 1 and media == []
    assert all(t != LLAMA_TOKEN_NULL for t in toks)

    # the slot stays idle but is not re-flushed into a second unit (one flush per idle period)
    time.sleep(IDLE_SECONDS + 1)
    assert _units_on_disk() == metas

    server.stop()


def test_idle_flush_media():
    """The idle-delay flush persists a MEDIA slot as a v2 unit too (records tiling the NULL
    cells), after the delay, with no follow-up request and no shutdown."""
    vs = make_vision_server(auto=True)
    vs.slot_save_idle_seconds = IDLE_SECONDS
    vs.server_metrics = True
    vs.start()

    prompt_n, _, _ = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI], id_slot=0)
    assert prompt_n > 0
    # nothing on disk yet, and nothing captured: no second request, no shutdown
    assert _units_on_disk() == []
    assert _queued(vs) == 0

    metas = _wait_for_unit(IDLE_SECONDS + 10)
    v2 = [parse_meta(p) for p in metas]
    v2 = [m for m in v2 if m[0] == 2]
    assert len(v2) == 1, "the idle media slot must have been flushed as a single v2 unit"
    _, toks, media = v2[0]
    assert len(media) >= 1
    n_null = sum(1 for t in toks if t == LLAMA_TOKEN_NULL)
    assert n_null == sum(r["n_tokens"] for r in media)

    vs.stop()


def test_idle_flush_disabled_legacy():
    """--slot-save-idle-seconds -1 restores legacy behaviour: an idle slot is NOT flushed on a
    timer; only the next-task / shutdown sites persist it."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_block = 16
    server.slot_save_idle_seconds = -1
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "n_predict": 8,
        "temperature": 0,
    })
    assert res.status_code == 200

    # well past what the default idle delay would be: still nothing (feature off)
    time.sleep(IDLE_SECONDS + 2)
    assert _units_on_disk() == []

    # the shutdown flush still persists it (legacy path intact)
    server.stop()
    assert len(_units_on_disk()) == 1


# -- idle-delay flush robustness (F3: G1/G2/G4) --------------------------------------------

def _wait_for_units(n: int, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        metas = _units_on_disk()
        if len(metas) >= n:
            return metas
        time.sleep(0.25)
    return _units_on_disk()


def test_idle_flush_requires_auto_at_startup():
    """G4: --slot-save-idle-seconds is inert without --slot-save-auto (its flush is gated on the
    master switch), so an explicit value without --slot-save-auto is rejected at startup rather
    than silently ignored. --slot-save-block <= 0 and --slot-save-min-tokens < 0 are likewise
    rejected unconditionally."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = False
    server.slot_save_idle_seconds = IDLE_SECONDS
    with pytest.raises(Exception):
        server.start(timeout_seconds=10)
    server.stop()

    # a non-positive block size is never meaningful even without the master switch
    server = ServerPreset.tinyllama2()
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = False
    server.slot_save_block = -5
    with pytest.raises(Exception):
        server.start(timeout_seconds=10)
    server.stop()

    # a negative minimum-snapshot floor is never meaningful even without the master switch
    server = ServerPreset.tinyllama2()
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = False
    server.slot_save_min_tokens = -1
    with pytest.raises(Exception):
        server.start(timeout_seconds=10)
    server.stop()


def test_idle_flush_multiple_slots_all_flushed():
    """G1: the idle flush persists at most one slot per wakeup, but every due slot is still
    flushed across successive wakeups — two slots that go idle together (no follow-up request, so
    only the idle timer can persist them) each land as their own unit while the server keeps
    running (the one-per-wakeup refactor must not drop the slots after the first)."""
    global server
    server.n_ctx = 1024  # split across 2 slots -> 512 each, room for both prompts
    server.n_slots = 2
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_block = 16
    server.slot_save_idle_seconds = IDLE_SECONDS
    server.start()

    def _req(i):
        return server.make_request("POST", "/completion", data={
            "prompt": f"Slot {i}: the quick brown fox jumps over the lazy dog. " * 8,
            "n_predict": 8,
            "temperature": 0,
            "id_slot": i,
        })

    # fire both concurrently so both slots go idle together with no subsequent request — the
    # reclaim/next-task save site cannot fire, so only the idle timer can persist them
    results = parallel_function_calls([(_req, (0,)), (_req, (1,))])
    assert all(r.status_code == 200 for r in results)

    metas = _wait_for_units(2, IDLE_SECONDS + 15)
    assert len(metas) == 2, "both idle slots must be flushed, one per wakeup across wakeups"
    assert len(set(metas)) == 2, "the two slots hold distinct prompts -> two distinct units"

    server.stop()


def test_idle_flush_completes_before_sleep():
    """G2: with both --sleep-idle-seconds and a longer --slot-save-idle-seconds set, a pending
    idle flush completes before the server sleeps (sleeping unloads the KV cache, discarding it
    unsaved). Sleep is deferred past its own threshold until the flush lands."""
    global server
    server.n_ctx = 512
    server.n_batch = 512
    server.n_slots = 1
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.slot_save_block = 16
    server.sleep_idle_seconds = 2
    server.slot_save_idle_seconds = 6  # flush deadline strictly later than the sleep threshold
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "n_predict": 8,
        "temperature": 0,
    })
    assert res.status_code == 200

    # past the sleep threshold (2s) but before the flush deadline (6s): sleep is deferred while a
    # flush is still pending, and nothing is on disk yet
    time.sleep(3.5)
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["is_sleeping"] == False, "sleep must be deferred until the pending flush runs"
    assert _units_on_disk() == []

    # the flush lands (KV persisted) before the server is allowed to sleep
    metas = _wait_for_unit(IDLE_SECONDS + 12)
    assert len(metas) == 1, "the idle flush must persist the slot before sleeping discards its KV"

    # having flushed, the server is now free to sleep
    deadline = time.time() + 10
    while time.time() < deadline:
        props = server.make_request("GET", "/props")
        if props.status_code == 200 and props.body["is_sleeping"]:
            break
        time.sleep(0.25)
    assert props.body["is_sleeping"] == True, "server must enter the sleeping state after flushing"

    server.stop()


# --- bounded-store eviction is opt-in (scoped to --slot-save-auto) ---------------------------
# Regression for the M1/D1/D2 finding: the LRU bounded store must NOT touch a plain
# --slot-save-path directory (upstream manual-save semantics = never deletes anything). It runs
# ONLY when --slot-save-auto owns the directory as its cache. The pair below toggles nothing but
# the --slot-save-auto flag with identical caps and identical manual saves to prove the gate.

def _manual_save_server(with_auto: bool):
    s = ServerPreset.tinyllama2()
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = with_auto
    if with_auto:
        s.slot_save_block = 16
        s.slot_save_min_tokens = 0  # short-prompt test: keep the floor at the hash block size
    # a tight count cap: eviction (when it runs) keeps at most 2 snapshots
    s.slot_save_max_count = 2
    return s


def test_manual_save_no_auto_never_evicts():
    """M1/D1/D2: with a cap set but WITHOUT --slot-save-auto, manual /slots saves of more files
    than the cap delete NOTHING — including unrelated user files in the directory. Plain
    --slot-save-path keeps upstream's non-destructive behaviour."""
    s = _manual_save_server(with_auto=False)
    s.start()

    # populate slot 0 with some KV to snapshot
    res = s.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little dog named Spot. " * 4,
        "n_predict": 0, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200

    # an unrelated user file living in the same directory must survive untouched
    unrelated = os.path.join(CACHE_DIR, "notes.txt")
    with open(unrelated, "w") as f:
        f.write("keep me")

    # save 4 hand-named snapshots — well over the count cap of 2
    names = ["session-a.bin", "session-b.bin", "session-c.bin", "session-d.bin"]
    for name in names:
        r = s.make_request("POST", "/slots/0?action=save", data={"filename": name})
        assert r.status_code == 200
        time.sleep(0.05)  # distinct mtimes (harmless; nothing should be evicted anyway)
    s.stop()

    # NOTHING was deleted: all 4 snapshots AND the unrelated file are still present
    for name in names:
        assert os.path.exists(os.path.join(CACHE_DIR, name)), f"{name} was wrongly evicted"
    assert os.path.exists(unrelated), "an unrelated user file was wrongly deleted"


def test_manual_save_with_auto_still_evicts():
    """With --slot-save-auto owning the directory, the same count cap DOES enforce the bounded
    store: after saving 4 snapshots with a cap of 2, only the 2 most-recent survive. Confirms
    F1 scoped eviction to the auto cache without disabling it there."""
    s = _manual_save_server(with_auto=True)
    s.start()

    res = s.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little dog named Spot. " * 4,
        "n_predict": 0, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200

    names = ["session-a.bin", "session-b.bin", "session-c.bin", "session-d.bin"]
    for name in names:
        r = s.make_request("POST", "/slots/0?action=save", data={"filename": name})
        assert r.status_code == 200
        time.sleep(0.05)  # ensure strictly increasing mtimes for the LRU order

    # assert BEFORE stopping: the shutdown auto-flush would add its own auto-* unit and shift the
    # LRU. At this point only the 4 manual saves have run, each enforcing the cap of 2: exactly the
    # last two saved remain, the older two were evicted.
    surviving = sorted(b for b in os.listdir(CACHE_DIR) if b.endswith(".bin"))
    assert surviving == ["session-c.bin", "session-d.bin"], f"unexpected survivors: {surviving}"
    s.stop()


def test_concurrent_publish_temps_not_reaped():
    """F2/M2: a peer process's in-flight sidecar temps (…tmp.logits / …tmp.meta) briefly have no
    base state file — between the state temp's rename to <fname> and the sidecar temps' own
    renames. Another process's enforce_limits pass must NOT reap them as orphaned sidecars (doing
    so unlinks a peer's just-published unit). A genuinely orphaned FINAL-named sidecar (its state
    file gone, not a temp) must still be reaped."""
    s = _manual_save_server(with_auto=True)
    s.slot_save_max_count = 100  # generous: exercise the orphan-reap pass, not cap eviction
    s.start()

    # prime slot 0 so the trigger save below has KV to snapshot
    res = s.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little dog named Spot. " * 4,
        "n_predict": 0, "temperature": 0, "cache_prompt": True, "id_slot": 0,
    })
    assert res.status_code == 200
    time.sleep(0.1)  # let the release-time auto-save settle before planting peer files

    # a peer's just-published unit, mid-publish: the final state file is on disk and its sidecar
    # temps are still awaiting their rename to the final .logits/.meta names.
    peer_state = os.path.join(CACHE_DIR, "auto-1111111111111111-2222222222222222-42")
    with open(peer_state, "wb") as f:
        f.write(b"\x00" * 64)
    peer_logits_tmp = peer_state + ".99999.7.tmp.logits"
    peer_meta_tmp = peer_state + ".99999.7.tmp.meta"
    for p in (peer_logits_tmp, peer_meta_tmp):
        with open(p, "wb") as f:
            f.write(b"\x00" * 16)

    # a genuinely orphaned FINAL-named sidecar (state file evicted/gone) — must still be reaped
    true_orphan = os.path.join(CACHE_DIR, "auto-deadbeefdeadbeef-cafef00dcafef00d-9.logits")
    with open(true_orphan, "wb") as f:
        f.write(b"\x00" * 16)

    # a manual save under --slot-save-auto runs a full enforce_limits pass over the directory
    r = s.make_request("POST", "/slots/0?action=save", data={"filename": "trigger.bin"})
    assert r.status_code == 200
    s.stop()

    assert os.path.exists(peer_logits_tmp), "in-flight sidecar temp .logits was wrongly reaped"
    assert os.path.exists(peer_meta_tmp), "in-flight sidecar temp .meta was wrongly reaped"
    assert not os.path.exists(true_orphan), "a genuinely orphaned final-named sidecar should be reaped"

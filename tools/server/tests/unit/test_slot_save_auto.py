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
# GOLDEN_PROMPT. It must stay bit-frozen: restoring it under every future binary is the
# regression lock for v1 compatibility (a v1 .meta carries no fp_mmproj — the reader
# backfills it — and existing units must keep restoring unmodified).
FIXTURE_DIR = "./fixtures/golden-v1"
FIXTURE_SHA256 = {
    "auto-f414107cff91a49e-5efede8f57f0c198-377.bin":
        "70dfefbd6f550c60ad416599e4ec5adb3ad25b646682c72d7f4aa731bfb16435",
    "auto-f414107cff91a49e-5efede8f57f0c198-377.bin.meta":
        "ecbac54e17454e180c4b50abe0845822996cbca207df95719f6e0a9328f54f90",
}

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
    # geometry must equal the fixture capture's (-c 512 -np 1): fp_n_ctx is part of the
    # snapshot fingerprint and a mismatch refuses the restore.
    server.n_ctx = 512
    server.n_batch = 512
    server.n_slots = 1
    server.temperature = 0.0
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)


def verify_and_copy_fixture(dst: str, names=None):
    for name, expected in FIXTURE_SHA256.items():
        path = os.path.join(FIXTURE_DIR, name)
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected, f"golden fixture {name} changed on disk — it must stay frozen"
        if names is None or name in names:
            shutil.copy(path, dst)


def test_text_only_meta_byte_identical():
    """The current binary re-emits the golden unit for the capture prompt with the same
    deterministic filename and byte-identical contents (invariant 0: text-only on-disk
    behaviour is frozen at the pre-media base build's)."""
    global server
    server.slot_save_path = CACHE_DIR
    server.slot_save_auto = True
    server.start()
    res = server.make_request("POST", "/completion", data=GOLDEN_REQUEST)
    assert res.status_code == 200
    # graceful stop -> auto_save_slots_at_shutdown publishes the unit
    server.stop()

    # same filenames as the fixture capture (the name encodes fp_model, the block-chain
    # hash and the token count — a drift in any of them shows up here first)
    assert sorted(os.listdir(CACHE_DIR)) == sorted(FIXTURE_SHA256)
    for name, expected in FIXTURE_SHA256.items():
        with open(os.path.join(CACHE_DIR, name), "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected, f"{name}: emitted bytes differ from the golden fixture"


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

    metas = {p: parse_meta(p) for p in glob.glob(os.path.join(CACHE_DIR, "auto-*.meta"))}
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
    for p in sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.meta"))):
        version, toks, media = parse_meta(p)
        if version == 2:
            out.append((p, toks, media))
    return out


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
    metas = glob.glob(os.path.join(CACHE_DIR, "auto-*.meta"))
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
    assert len(glob.glob(os.path.join(CACHE_DIR, "auto-*.meta"))) == 2
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


def make_manual_vision_server() -> ServerProcess:
    # raw /completion prompts carry media via the plain media marker (no chat template)
    os.environ["LLAMA_MEDIA_MARKER"] = MEDIA_MARKER
    vs = make_vision_server(auto=False)
    vs.slot_save_path = CACHE_DIR
    return vs


def raw_media_request(vs: ServerProcess, prompt_string: str, files: list, n_predict: int = 8):
    """/completion with a raw prompt string + media files: byte-stable across resends
    (no chat-template re-render), which is what snapshot-matching follow-ups need."""
    res = vs.make_request("POST", "/completion", data={
        "prompt": {"prompt_string": prompt_string, "multimodal_data": files},
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200
    t = res.body["timings"]
    return t["prompt_n"], t["cache_n"], res.body["content"]


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


def test_manual_media_save_restore_continues():
    """Manual SAVE of a media slot writes a v2 .meta sidecar; a fresh process manually
    RESTORES it (stub rehydration) and the identical request reuses the restored
    cells INCLUDING the image's — the image is never re-encoded — with the answer
    identical to a cold run."""
    vs = make_manual_vision_server()
    vs.start()
    prompt_n_full, cache_n_cold, _ = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    assert cache_n_cold == 0
    assert prompt_n_full > 200  # sanity: the image cells dominate the prompt

    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved == prompt_n_full  # prompt-only snapshot
    vs.stop()

    # the sidecar is a v2 meta whose single record tiles the image's NULL cells
    version, toks, media = parse_meta(os.path.join(CACHE_DIR, "media.bin.meta"))
    assert version == 2
    assert len(toks) == n_saved
    assert len(media) == 1 and media[0]["is_audio"] == 0
    n_img = media[0]["n_tokens"]
    assert n_img > 0
    assert sum(1 for t in toks if t == LLAMA_TOKEN_NULL) == n_img

    # fresh process: cold reference first, then erase + restore + identical resend
    vs = make_manual_vision_server()
    vs.start()
    _, _, content_ref = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])

    res = vs.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    prompt_n_warm, cache_n_warm, content_warm = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])
    assert content_warm == content_ref
    assert cache_n_warm >= media[0]["start_idx"] + n_img  # every image cell came from the restore
    assert prompt_n_warm <= 8                             # the image was NOT re-processed
    vs.stop()


def test_manual_media_restore_stub_never_encoded():
    """After a manual media restore the slot's image chunks are identity-only stubs: a
    follow-up with a DIFFERENT image id-mismatches them, so they are dropped (never
    encoded) and the request re-processes its own image — answers stay identical to
    no-cache runs for both the different and the original image."""
    ref = make_manual_vision_server()
    ref.start()
    _, _, content_ref_b = raw_media_request(ref, MEDIA_PROMPT, [IMG2_B64])
    ref.stop()

    vs = make_manual_vision_server()
    vs.start()
    _, _, content_ref_a = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    vs.stop()

    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code == 200

    # different image bytes => different id => the stub never matches nor encodes
    _, _, content_b = raw_media_request(vs, MEDIA_PROMPT, [IMG2_B64])
    assert content_b == content_ref_b

    # and the original image request still answers exactly like its cold run
    ref = make_manual_vision_server()
    ref.start()
    _, _, content_ref_a8 = raw_media_request(ref, MEDIA_PROMPT, [IMG_B64])
    ref.stop()
    _, _, content_a = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64])
    assert content_a == content_ref_a8
    vs.stop()


def test_manual_media_restore_refuses_without_meta():
    """A media state file without its .meta sidecar cannot be rehydrated: the manual
    restore refuses with an explicit error AND leaves the slot cleared-but-usable."""
    vs = make_manual_vision_server()
    vs.start()
    prompt_n_full, _, _ = raw_media_request(vs, MEDIA_PROMPT, [IMG_B64], n_predict=0)
    res = vs.make_request("POST", "/slots/0?action=save", data={"filename": "media.bin"})
    assert res.status_code == 200
    vs.stop()

    os.remove(os.path.join(CACHE_DIR, "media.bin.meta"))

    vs = make_manual_vision_server()
    vs.start()
    res = vs.make_request("POST", "/slots/0?action=restore", data={"filename": "media.bin"})
    assert res.status_code != 200
    assert "sidecar" in str(res.body)

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


def _units_on_disk():
    return sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.meta")))


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
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "n_predict": 8,
        "temperature": 0,
    })
    assert res.status_code == 200
    # the request itself (processing + completion) writes nothing
    assert _units_on_disk() == []

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
    vs.start()

    prompt_n, _, _ = vision_request(vs, [VISION_TEXT_PRE, IMG_DATA_URI], id_slot=0)
    assert prompt_n > 0
    # nothing on disk yet: no second request, no shutdown
    assert _units_on_disk() == []

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

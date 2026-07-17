import glob
import hashlib
import os
import shutil
import struct

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

import glob
import os
import shutil
import struct
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


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
    # the test prompts sit well under the 1024-token default minimum-snapshot floor; drop it so
    # the auto cache behaves as it did before --slot-save-min-tokens (floor = the hash block size).
    server.slot_save_min_tokens = 0
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)


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


SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE


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

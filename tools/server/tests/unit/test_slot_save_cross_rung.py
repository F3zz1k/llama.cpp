import glob
import os
import re
import shutil
import time

import pytest
from utils import *

# Context rungs: instances of ONE model that differ only in inputs which never reach the state bytes
# (n_ctx, ubatch) share their units through one --slot-save-path, in BOTH directions, delta chains
# included. Production case: a lower-ctx vision rung beside the text rung of the same model, which
# could never reuse the text rung's system+tools prefix while n_ctx was an identity field.
#
# n_ctx is out of the identity for every memory class (the per-class audit is on model_fp::fp_n_ctx in
# server-common.h), so the matrix below runs one test-llama-archs dummy per class. The one input that
# does reach the bytes through n_ctx is LongRoPE's long/short factor choice, which stays identity
# (test_rungs_across_the_rope_threshold_do_not_share).
#
# Geometry traps: llama_context pads n_ctx to a multiple of 256 and the server caps n_ctx at
# n_ctx_train, so the rungs are 256 and 512 and every dummy (context_length 256) gets
# --override-kv context_length=1024. The dummies also carry rope.scaling.original_context_length=256;
# it is raised to 1024 so that only the LongRoPE tests below depend on where it sits (on a model without
# LongRoPE factors it no longer splits anything, test_original_context_splits_only_longrope_models).
# The instrument asserts the two live n_ctx values differ, so a collapsed geometry fails loudly.

ROOT = "./tmp/slot_save_cross_rung"
IDLE = 2
MTP_MODEL = os.environ.get("LLAMA_TEST_MTP_MODEL", "")
MODELS_DIR = os.environ.get("LLAMA_TEST_MODELS_DIR", "") or os.path.normpath(os.path.join(
    os.path.dirname(os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")), "..", "tests", "test-models"))

BASE  = [((i * 7) % 100) + 10 for i in range(48)]
EXTRA = [((i * 11) % 100) + 10 for i in range(24)]
BIG, SMALL = (512, 128), (256, 32)                    # (n_ctx, ubatch)

# dummy -> the memory class llama_model::create_memory builds for it
CLASS_MATRIX = [
    ("llama-dense",    "kv"),
    ("gemma3-dense",   "iswa"),
    ("mamba-dense",    "recurrent"),
    ("qwen35-dense",   "hybrid"),
    ("lfm2-dense",     "hybrid-iswa"),
    ("qwen4exp-moe",   "hybrid_idx"),
    ("glm-dsa-moe",    "dsa"),
    ("dots3note-moe",  "dsa-iswa"),
    ("minimax-m3-moe", "msa"),
    ("deepseek4-moe",  "dsv4"),
]


def _arch_of(path):
    # general.architecture is read by the server anyway; for the override keys the GGUF file name of a
    # test-llama-archs dummy is "<arch>-<dense|moe>.gguf", except the MTP dummy (qwen35)
    base = os.path.basename(path)
    if path == MTP_MODEL:
        return "qwen35"
    return re.sub(r"-(dense|moe)\.gguf$", "", base)


def _server(kind, cache_dir, log_path, n_ctx, n_ubatch, auto=True, ctx_train=1024, rope_orig=1024, **extra):
    if kind == "tinyllama":
        s = ServerPreset.tinyllama2()
    else:
        s = ServerProcess()
        s.model_hf_repo = None
        s.model_hf_file = None
        s.model_alias = "dummy"
        if kind == "qwen35-mtp":
            s.model_file = MTP_MODEL
            s.spec_type = "draft-mtp"
            s.spec_draft_n_max = 2
        else:
            s.model_file = os.path.join(MODELS_DIR, kind + ".gguf")
        arch = _arch_of(s.model_file)
        s.override_kv = [f"{arch}.context_length=int:{ctx_train}"]
        if rope_orig is not None:
            s.override_kv.append(f"{arch}.rope.scaling.original_context_length=int:{rope_orig}")
    s.n_ctx = n_ctx
    s.n_batch = 512
    s.n_ubatch = n_ubatch
    s.n_slots = 1
    s.temperature = 0.0
    s.debug = True
    s.log_path = log_path
    if auto:
        s.slot_save_path = cache_dir
        s.slot_save_auto = True
        s.slot_save_incremental = True
        s.slot_save_block = 16
        s.slot_save_min_tokens = 0
        s.slot_restore_min_tokens = 0
        s.slot_save_idle_seconds = IDLE
        s.slot_save_node_prompt = "off"
    for k, v in extra.items():
        assert hasattr(s, k), k
        setattr(s, k, v)
    return s


def _complete(s, prompt, n_predict=8):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "id_slot": 0,
        "temperature": 0.0, "top_k": 1, "return_tokens": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def _wait_meta(d, n, t):
    end = time.time() + t
    while time.time() < end:
        if len(glob.glob(os.path.join(d, "auto-*.bin.meta"))) >= n:
            break
        time.sleep(0.25)
    return sorted(glob.glob(os.path.join(d, "auto-*.bin.meta")))


def _n_ctx(s):
    res = s.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body["default_generation_settings"]["n_ctx"]


def _identities(d):
    return {os.path.basename(p).split("-")[1] for p in glob.glob(os.path.join(d, "auto-*.bin"))}


def _log(path):
    with open(path, errors="replace") as f:
        return f.read()


def _delta_probe(log_text):
    m = re.search(r"delta capability probed = (YES|NO)", log_text)
    return m.group(1) if m else None


@pytest.fixture(autouse=True)
def clean():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    yield


def _need_dummy(kind):
    if kind == "qwen35-mtp":
        if not os.path.isfile(MTP_MODEL):
            pytest.skip("no LLAMA_TEST_MTP_MODEL")
    elif kind != "tinyllama" and not os.path.isfile(os.path.join(MODELS_DIR, kind + ".gguf")):
        pytest.skip(f"no {kind}.gguf in {MODELS_DIR} (build test-llama-archs)")


def _write_chain(kind, cache, tmp_path, cfg, tag="w"):
    """Writer rung seeds a root (BASE + gen) and its continuation (BASE + gen + EXTRA + gen2), saved as
    a delta when the class can (otherwise a whole root). Returns (request, writer n_ctx, probe)."""
    w = _server(kind, cache, str(tmp_path / f"{tag}.log"), *cfg)
    w.start()
    w_ctx = _n_ctx(w)
    gen = _complete(w, BASE)["tokens"]
    _wait_meta(cache, 1, IDLE + 15)
    turn2 = BASE + gen + EXTRA
    gen2 = _complete(w, turn2)["tokens"]
    metas = _wait_meta(cache, 2, IDLE + 15)
    w.stop()
    assert len(metas) >= 2, metas                   # turn-1 unit + turn-2 unit on the writer rung
    assert len(_identities(cache)) == 1, _identities(cache)
    return turn2 + gen2 + [42], len(turn2) + len(gen2) - 1, w_ctx, _delta_probe(_log(str(tmp_path / f"{tag}.log")))


def _cross_rung(kind, big_first, tmp_path):
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    w_cfg, r_cfg = (BIG, SMALL) if big_first else (SMALL, BIG)
    req, tip, w_ctx, probe = _write_chain(kind, cache, tmp_path, w_cfg)
    # the same-rung restore is the trusted path; it is the oracle for the cross-rung one. It reads a
    # copy, so neither run sees units the other writes.
    cache_same = os.path.join(ROOT, "same")
    shutil.copytree(cache, cache_same)

    same = _server(kind, cache_same, str(tmp_path / "same.log"), *w_cfg)
    same.start()
    ref = _complete(same, req)
    same.stop()

    r = _server(kind, cache, str(tmp_path / "r.log"), *r_cfg)
    r.start()
    r_ctx = _n_ctx(r)
    warm = _complete(r, req)
    r.stop()

    c = _server(kind, None, str(tmp_path / "cold.log"), *r_cfg, auto=False)
    c.start()
    cold = _complete(c, req)
    c.stop()

    assert w_ctx != r_ctx, (w_ctx, r_ctx)           # instrument: the rungs really differ in n_ctx
    assert ref["timings"].get("cache_disk_n", 0) >= tip, ref["timings"]   # control: the chain restores
    # every unit on disk carried the writer's rung when R started, so this hit is cross-rung; covering
    # the turn-2 unit proves the whole chain (root, and the delta when the class writes one) resolved
    assert warm["timings"].get("cache_disk_n", 0) >= tip, warm["timings"]
    assert warm["tokens"] == ref["tokens"]
    assert warm["content"] == ref["content"]
    assert "cross-ctx reuse" in _log(str(tmp_path / "r.log"))
    return probe, warm, cold


@pytest.mark.parametrize("big_first", [True, False])
@pytest.mark.parametrize("kind", ["tinyllama", "qwen35-mtp"])
def test_rungs_differing_only_in_ctx_and_ubatch_share_text_units(kind, big_first, tmp_path):
    """The two production restore classes (plain attention rewinding per token, and the hybrid
    recurrent + MTP class that restores whole prefixes only). Rungs differ in n_ctx AND ubatch; the
    reader must restore the writer's chain and produce exactly a cold reader's output too."""
    _need_dummy(kind)
    probe, warm, cold = _cross_rung(kind, big_first, tmp_path)
    assert probe == "YES"                           # the turn-2 unit is a delta: the chain is two files
    assert warm["tokens"] == cold["tokens"]
    assert warm["content"] == cold["content"]


@pytest.mark.parametrize("big_first", [True, False])
@pytest.mark.parametrize("kind,mem_class", CLASS_MATRIX, ids=[c for _, c in CLASS_MATRIX])
def test_every_memory_class_shares_units_across_rungs(kind, mem_class, big_first, tmp_path):
    """One dummy per memory class: the reader rung restores the writer rung's chain and continues
    token-for-token like a same-rung restore of it. A class that cannot write deltas is reported by
    its probe and still shares its whole roots."""
    _need_dummy(kind)
    probe, _, _ = _cross_rung(kind, big_first, tmp_path)
    assert probe in ("YES", "NO"), f"{mem_class}: no delta probe in the writer log"


@pytest.mark.parametrize("kind", ["tinyllama", "llama-dense"])
def test_unit_longer_than_reader_ctx_is_skipped_not_attempted(kind, tmp_path):
    """A larger rung's unit that exceeds the reader's whole context never reaches the state loader:
    the reader prefills cold, matches a cold run, and its log carries no failed restore. Only a class
    that rewinds per token could use such a unit for a shorter request, so plain attention is the case.
    Positive control: a reader with room for the unit restores it, so it WAS a live candidate."""
    _need_dummy(kind)
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    long_prompt = BASE * 6                          # 288 tokens + gen, > the small rung's 256
    w = _server(kind, cache, str(tmp_path / "w.log"), *BIG)
    w.start()
    assert _n_ctx(w) >= len(long_prompt) + 8
    _complete(w, long_prompt)
    _wait_meta(cache, 1, IDLE + 15)
    w.stop()
    req = long_prompt[:100]

    c = _server(kind, None, str(tmp_path / "cold.log"), *SMALL, auto=False)
    c.start()
    cold = _complete(c, req)
    c.stop()

    cache_ctl = os.path.join(ROOT, "ctl")
    shutil.copytree(cache, cache_ctl)
    r = _server(kind, cache, str(tmp_path / "r.log"), *SMALL)
    r.start()
    assert _n_ctx(r) < len(long_prompt)
    body = _complete(r, req)
    again = _complete(r, req)                       # the server still serves after the skip
    r.stop()
    log = _log(str(tmp_path / "r.log"))
    assert body["timings"].get("cache_disk_n", 0) == 0
    assert body["tokens"] == cold["tokens"]
    assert again["content"] == cold["content"]
    assert "not enough cells" not in log            # skipped before any state-file I/O
    assert "failed after the slot was cleared" not in log

    ctl = _server(kind, cache_ctl, str(tmp_path / "ctl.log"), 512 + 256, 32)
    ctl.start()
    hit = _complete(ctl, req)
    ctl.stop()
    assert hit["timings"].get("cache_disk_n", 0) >= 64, hit["timings"]


def test_rungs_across_the_rope_threshold_do_not_share(tmp_path):
    """phi3 (LongRoPE: rope_long/rope_short) picks its factors by n_ctx_seq > n_ctx_orig_yarn, so two
    rungs on opposite sides of the original context bake different K rotations into the same tokens.
    They must not share; two rungs on the same side must. The dummy's original context is 256."""
    kind = "phi3-dense"
    _need_dummy(kind)

    def run(w_cfg, r_cfg, sub):
        cache = os.path.join(ROOT, sub)
        os.makedirs(cache)
        w = _server(kind, cache, str(tmp_path / f"{sub}-w.log"), *w_cfg, rope_orig=None)
        w.start()
        _complete(w, BASE)
        _wait_meta(cache, 1, IDLE + 15)
        w.stop()
        r = _server(kind, cache, str(tmp_path / f"{sub}-r.log"), *r_cfg, rope_orig=None)
        r.start()
        body = _complete(r, BASE + [42])
        r.stop()
        return body["timings"].get("cache_disk_n", 0), _identities(cache)

    # 512 (long factors) and 256 (short): refused, and the reader publishes under its own prefix
    n, ids = run(BIG, SMALL, "straddle")
    assert n == 0
    assert len(ids) == 2, ids
    # 512 and 768, both long: shared under one prefix
    n, ids = run(BIG, (768, 32), "same-side")
    assert n >= len(BASE) - 1
    assert len(ids) == 1, ids


def _seed_then_read(kind, cache, tmp_path, sub, w_cfg, r_cfg, w_kw=None, r_kw=None):
    """Writer saves BASE (+gen); a reader with its own config asks for BASE + [42]. Returns the reader's
    restored token count, its log and the identity prefixes on disk after both ran."""
    w = _server(kind, cache, str(tmp_path / f"{sub}-w.log"), *w_cfg, **(w_kw or {}))
    w.start()
    w_ctx = _n_ctx(w)
    _complete(w, BASE)
    _wait_meta(cache, 1, IDLE + 15)
    w.stop()
    r = _server(kind, cache, str(tmp_path / f"{sub}-r.log"), *r_cfg, **(r_kw or {}))
    r.start()
    r_ctx = _n_ctx(r)
    body = _complete(r, BASE + [42])
    _wait_meta(cache, 2, IDLE + 15)
    r.stop()
    return body["timings"].get("cache_disk_n", 0), _log(str(tmp_path / f"{sub}-r.log")), _identities(cache), w_ctx, r_ctx


def test_rope_threshold_is_the_models_own_not_yarn_orig_ctx(tmp_path):
    """--yarn-orig-ctx sets cparams.n_ctx_orig_yarn only; get_rope_factors compares n_ctx_seq against
    hparams.n_ctx_orig_yarn (the GGUF original context, 256 here). Rungs at 512 (long factors) and 256
    (short) both started with --yarn-orig-ctx 128 are on opposite sides of the REAL threshold and must
    not share, although a threshold rebuilt from --yarn-orig-ctx puts both on the long side."""
    kind = "phi3-dense"
    _need_dummy(kind)
    kw = {"yarn_orig_ctx": 128, "rope_orig": None}   # keep the GGUF original context (256)
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    n, log, ids, w_ctx, r_ctx = _seed_then_read(kind, cache, tmp_path, "yoc", BIG, SMALL, kw, kw)
    assert (w_ctx, r_ctx) == (512, 256), (w_ctx, r_ctx)
    assert n == 0
    assert "cross-ctx reuse" not in log
    assert len(ids) == 2, ids
    # control: two rungs on the long side of the real threshold, same flags, still share
    cache2 = os.path.join(ROOT, "c2")
    os.makedirs(cache2)
    n, _, ids, _, _ = _seed_then_read(kind, cache2, tmp_path, "yoc2", BIG, (768, 32), kw, kw)
    assert n >= len(BASE) - 1
    assert len(ids) == 1, ids


@pytest.mark.parametrize("kind", ["tinyllama", "llama-dense"])
def test_original_context_splits_only_longrope_models(kind, tmp_path):
    """A model without LongRoPE factors may still carry rope.scaling.original_context_length (YaRN baked
    into the GGUF). Its rungs on both sides of that value write identical bytes and must share."""
    _need_dummy(kind)
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    if kind == "tinyllama":
        kw = {"override_kv": ["llama.rope.scaling.original_context_length=int:256"]}
    else:
        kw = {"rope_orig": 256}                     # _server writes the override for a dummy
    n, log, ids, w_ctx, r_ctx = _seed_then_read(kind, cache, tmp_path, "orig", BIG, SMALL, kw, kw)
    assert w_ctx > 256 >= r_ctx, (w_ctx, r_ctx)
    assert n >= len(BASE) - 1
    assert "cross-ctx reuse" in log
    assert len(ids) == 1, ids


@pytest.mark.parametrize("w_kw,r_kw,what", [
    ({"fa": "on"},  {"fa": "off"}, "v_trans"),
    ({"n_slots": 1}, {"n_slots": 2}, "n_stream"),
])
def test_kv_layout_splits_identity(w_kw, r_kw, what, tmp_path):
    """V transposition (-fa off) and the KV stream count are refused on mismatch by the state reader.
    Peers differing in either must name units apart and never attempt each other's units; peers
    agreeing on them share (the second half, a positive control)."""
    kind = "tinyllama"
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    r_cfg = (BIG[0] * 2, BIG[1]) if what == "n_stream" else BIG   # n_ctx is split across the streams
    n, log, ids, _, _ = _seed_then_read(kind, cache, tmp_path, what, BIG, r_cfg, w_kw, r_kw)
    assert n == 0
    assert "failed after the slot was cleared" not in log
    assert "incompatible V transposition" not in log and "n_stream mismatch" not in log
    assert len(ids) == 2, ids
    cache2 = os.path.join(ROOT, "c2")
    os.makedirs(cache2)
    n, _, ids, _, _ = _seed_then_read(kind, cache2, tmp_path, what + "-ctl", BIG, r_cfg, r_kw, r_kw)
    assert n >= len(BASE) - 1
    assert len(ids) == 1, ids


def test_store_lock_pairs_publish_and_restore(tmp_path):
    """The directory flock protocol: a publish waits while a restore holds the lock shared, and a
    restore is skipped (cold, no failed load) while a publish holds it exclusive."""
    import fcntl
    kind = "tinyllama"
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    fd = os.open(cache, os.O_RDONLY)
    try:
        # a "restore in progress": the idle save of the writer must not land while it lasts
        fcntl.flock(fd, fcntl.LOCK_SH)
        w = _server(kind, cache, str(tmp_path / "w.log"), *BIG)
        w.start()
        _complete(w, BASE)
        time.sleep(IDLE + 4)
        assert _wait_meta(cache, 1, 0.1) == []
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert len(_wait_meta(cache, 1, IDLE + 15)) >= 1     # it lands once the lock is free
        w.stop()

        # a "publish in progress": a restore must not read the store while it lasts
        fcntl.flock(fd, fcntl.LOCK_EX)
        r = _server(kind, cache, str(tmp_path / "r.log"), *BIG)
        r.start()
        busy = _complete(r, BASE + [42])
        fcntl.flock(fd, fcntl.LOCK_UN)
        r.stop()
        log = _log(str(tmp_path / "r.log"))
        assert busy["timings"].get("cache_disk_n", 0) == 0
        assert "store lock busy" in log
        assert "failed after the slot was cleared" not in log
    finally:
        os.close(fd)
    # control: with the lock free the same request restores
    r2 = _server(kind, cache, str(tmp_path / "r2.log"), *BIG)
    r2.start()
    hit = _complete(r2, BASE + [42])
    r2.stop()
    assert hit["timings"].get("cache_disk_n", 0) >= len(BASE) - 1, hit["timings"]


def test_chain_named_under_an_earlier_identity_rule_restores_and_is_not_extended(tmp_path):
    """A store written before n_ctx left the identity names its units under another prefix. Restore
    resolves delta parents under the tip's own prefix, so such a chain still restores whole; a new
    save never links a delta to it (the delta would name a parent under this instance's prefix that
    does not exist) and writes a whole root instead, which a fresh instance restores in full."""
    kind = "tinyllama"
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    req, tip, _, probe = _write_chain(kind, cache, tmp_path, BIG)
    assert probe == "YES"
    (live,) = _identities(cache)
    old = "0123456789abcdef"
    for p in glob.glob(os.path.join(cache, f"auto-{live}-*")):
        os.rename(p, p.replace(f"auto-{live}-", f"auto-{old}-"))

    r = _server(kind, cache, str(tmp_path / "r.log"), *BIG)
    r.start()
    warm = _complete(r, req)
    assert warm["timings"].get("cache_disk_n", 0) >= tip, warm["timings"]
    turn3 = req + warm["tokens"] + EXTRA
    gen3 = _complete(r, turn3)["tokens"]
    _wait_meta(cache, 4, IDLE + 15)
    r.stop()
    assert live in _identities(cache)

    f = _server(kind, cache, str(tmp_path / "f.log"), *BIG)
    f.start()
    body = _complete(f, turn3 + gen3 + [42])
    f.stop()
    assert body["timings"].get("cache_disk_n", 0) >= len(turn3) + len(gen3) - 1, body["timings"]

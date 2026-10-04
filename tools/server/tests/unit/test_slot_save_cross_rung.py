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
# --override-kv context_length=1024. The dummies also carry rope.scaling.original_context_length=256,
# which would put the 512 rung on the long side of the rope threshold, so it is raised to 1024 too.
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


def _server(kind, cache_dir, log_path, n_ctx, n_ubatch, auto=True, ctx_train=1024, rope_orig=1024):
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

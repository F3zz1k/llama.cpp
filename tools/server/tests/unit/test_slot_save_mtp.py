import glob
import os
import re
import shutil
import struct
import time

import numpy as np
import pytest
from utils import *

# Checks for the .dft draft sidecar on a real MTP draft: a qwen35 dummy with one nextn block, built
# from the qwen35-dense test-llama-archs dummy by tools/server/tests/mk_mtp_dummy.py (needs gguf-py
# and numpy):
#   python tools/server/tests/mk_mtp_dummy.py build/tests/test-models/qwen35-dense.gguf qwen35-mtp.gguf
# and point LLAMA_TEST_MTP_MODEL at the result. Every test skips without it.
# The instrument is the MTP draft distribution
# itself: with --verbose the server logs every draft candidate with its probability, and a draft
# whose KV and carry-over state are restored exactly must propose the same candidates with the
# same probabilities as a control that prefilled the whole prompt in one process.

MODEL = os.environ.get("LLAMA_TEST_MTP_MODEL", "")
ROOT = "./tmp/slot_save_mtp"
IDLE = 2

BASE = [((i * 7) % 100) + 10 for i in range(64)]
EXTRA = [((i * 11) % 100) + 10 for i in range(16)]

CAND_RE = re.compile(r"draft candidate\s+(\d+), pos\s+(\d+):\s+(\d+) \(\s*([0-9.]+)\)")


def _server(cache_dir, log_path, spec=True, n_max=3, auto=True, node_prompt=None):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = MODEL
    if spec:
        s.spec_type = "draft-mtp"
        s.spec_draft_n_max = n_max
    s.model_alias = "dummy"
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.debug = True
    if auto:
        s.slot_save_path = cache_dir
        s.slot_save_auto = True
        s.slot_save_incremental = True
        s.slot_save_block = 16
        s.slot_save_min_tokens = 0
        s.slot_save_context_min_tokens = 0
        s.slot_restore_min_tokens = 0
        s.slot_save_idle_seconds = IDLE
        s.slot_save_node_prompt = node_prompt  # None: the server default (cold)
    s.log_path = log_path
    return s


def _complete(s, prompt, n_predict):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "id_slot": 0,
        "temperature": 0.0, "top_k": 1, "return_tokens": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def _metric(s, name):
    res = s.make_request("GET", "/metrics")
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(name)


def _cands(log_path, mark):
    txt = open(log_path, errors="replace").read()
    i = txt.find(mark)
    assert i >= 0, f"marker {mark} not in {log_path}"
    return [(int(a), int(b), int(c), float(d)) for a, b, c, d in CAND_RE.findall(txt[i:])]


def _wait_meta(d, n, t):
    end = time.time() + t
    while time.time() < end:
        m = glob.glob(os.path.join(d, "auto-*.bin.meta"))
        if len(m) >= n:
            return m
        time.sleep(0.25)
    return glob.glob(os.path.join(d, "auto-*.bin.meta"))


@pytest.fixture(autouse=True)
def clean():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    yield


def _seed(cache, logp, node_prompt=None):
    s = _server(cache, logp, node_prompt=node_prompt)
    s.start()
    b = _complete(s, BASE, 8)
    # the idle-flushed conversation, plus the prompt node written mid-prefill unless it is off
    _wait_meta(cache, 1 if node_prompt == "off" else 2, IDLE + 15)
    saved = _metric(s, "auto_cache_save_draft_total")
    s.stop()
    return b["tokens"], saved


def test_mtp_warm_restore_matches_uninterrupted(tmp_path):
    if not os.path.isfile(MODEL):
        pytest.skip("no MTP dummy")
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    gen, saved = _seed(cache, str(tmp_path / "seed.log"))
    print("dft files:", sorted(os.path.basename(p) for p in glob.glob(cache + "/*.dft")), "saved", saved)
    nxt = BASE + gen + EXTRA

    # control: one process, whole prompt prefilled, no disk cache
    s0 = _server(None, str(tmp_path / "ctl.log"), auto=False)
    s0.start()
    b0 = _complete(s0, nxt, 24)
    s0.stop()

    cold = os.path.join(ROOT, "cold")
    shutil.copytree(cache, cold)
    for p in glob.glob(cold + "/*.dft"):
        os.remove(p)

    out = {}
    for name, d in (("warm", cache), ("cold", cold)):
        s = _server(d, str(tmp_path / f"{name}.log"))
        s.start()
        b = _complete(s, nxt, 24)
        out[name] = (b, _metric(s, "auto_cache_restore_draft_warm_total"),
                     _metric(s, "auto_cache_restore_hit_total"))
        s.stop()

    c0 = _cands(str(tmp_path / "ctl.log"), "")
    cw = _cands(str(tmp_path / "warm.log"), "")
    cc = _cands(str(tmp_path / "cold.log"), "")
    print("control", b0["timings"].get("draft_n"), b0["timings"].get("draft_n_accepted"), c0[:6])
    for k, (b, w, h) in out.items():
        print(k, "warm_ctr", w, "hit", h, "cache_n", b["timings"].get("cache_n"), "disk", b["timings"].get("cache_disk_n"),
              "draft", b["timings"].get("draft_n"), b["timings"].get("draft_n_accepted"))
    print("warm ", cw[:6])
    print("cold ", cc[:6])
    assert b0["content"] == out["warm"][0]["content"] == out["cold"][0]["content"]
    assert out["warm"][2] == 1 and out["warm"][1] == 1
    n = min(len(c0), len(cw))
    dif = [(i, c0[i], cw[i]) for i in range(n) if c0[i][2] != cw[i][2] or abs(c0[i][3] - cw[i][3]) > 2e-3]
    print("warm vs control: first diffs", dif[:5], "of", n)
    nc = min(len(c0), len(cc))
    difc = [(i, c0[i], cc[i]) for i in range(nc) if c0[i][2] != cc[i][2] or abs(c0[i][3] - cc[i][3]) > 2e-3]
    print("cold vs control (positive control, must differ):", len(difc), difc[:3])
    assert difc, "instrument is blind: a cold draft drafts like the control"
    assert not dif, "a warm MTP restore must draft exactly like an uninterrupted prefill"


def test_mtp_config_changes(tmp_path):
    if not os.path.isfile(MODEL):
        pytest.skip("no MTP dummy")
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    # one root and one delta, compared cell for cell: no prompt node, which would add a second root
    gen, _ = _seed(cache, str(tmp_path / "seed.log"), node_prompt="off")
    nxt = BASE + gen + EXTRA
    s0 = _server(None, str(tmp_path / "ctl.log"), spec=False, auto=False)
    s0.start()
    ref = _complete(s0, nxt, 24)["content"]
    s0.stop()
    for name, spec, nmax in (("nmax1", True, 1), ("nmax5", True, 5), ("nospec", False, 3)):
        d = os.path.join(ROOT, name)
        shutil.copytree(cache, d)
        s = _server(d, str(tmp_path / f"{name}.log"), spec=spec, n_max=nmax)
        s.start()
        b = _complete(s, nxt, 24)
        hit = _metric(s, "auto_cache_restore_hit_total")
        w = _metric(s, "auto_cache_restore_draft_warm_total")
        s.stop()
        print(name, "hit", hit, "warm", w, "cache_disk_n", b["timings"].get("cache_disk_n"))
        assert hit == 1, f"{name}: the target restore must never be refused by a draft config change"
        assert b["content"] == ref, f"{name}: restored output differs from a cold control"
    # MTP off at save time, on at restore: target restores, draft cold
    d = os.path.join(ROOT, "plain")
    os.makedirs(d)
    s = _server(d, str(tmp_path / "plainseed.log"), spec=False)
    s.start()
    b = _complete(s, BASE, 8)
    _wait_meta(d, 1, IDLE + 15)
    s.stop()
    assert not glob.glob(d + "/*.dft")
    s = _server(d, str(tmp_path / "plain2.log"), spec=True)
    s.start()
    nxt2 = BASE + b["tokens"] + EXTRA
    out2 = _complete(s, nxt2, 24)["content"]
    hit = _metric(s, "auto_cache_restore_hit_total")
    cold = _metric(s, "auto_cache_restore_draft_cold_total")
    s.stop()
    print("plain->mtp hit", hit, "cold", cold)
    assert hit == 1 and cold == 1
    s0 = _server(None, str(tmp_path / "ctl2.log"), spec=False, auto=False)
    s0.start()
    assert _complete(s0, nxt2, 24)["content"] == out2
    s0.stop()



def _parse_kv1(path):
    """pos -> K row (float32 array) of a single-layer plain KV state file (MTP draft cache)."""
    b = open(path, "rb").read()
    o = 0
    magic, ver, ntok = struct.unpack_from("<III", b, o); o += 12 + 4 * ntok
    (nstream,) = struct.unpack_from("<I", b, o); o += 4
    out = {}
    for _ in range(nstream):
        (cc,) = struct.unpack_from("<I", b, o); o += 4
        if cc == 0:
            continue
        for ext in (0, 12, 8, 16):
            oo = o
            poss = []
            ok = True
            for _c in range(cc):
                pos, nseq = struct.unpack_from("<iI", b, oo); oo += 8 + ext + 4 * nseq
                if nseq != 1:
                    ok = False
                    break
                poss.append(pos)
            if not ok:
                continue
            vt, nl, kt, ksr = struct.unpack_from("<IIiQ", b, oo)
            if vt in (0, 1) and nl == 1 and 0 < ksr < 1 << 20:
                oo += 20
                dt = np.float16 if kt == 1 else np.float32
                rows = np.frombuffer(b, dtype=dt, count=cc * ksr // np.dtype(dt).itemsize, offset=oo).reshape(cc, -1)
                for p, r in zip(poss, rows):
                    out[p] = r.astype(np.float32)
                return out, ext
        raise AssertionError("could not parse " + path)
    return out, None


def test_mtp_boundary_cell_matches_control(tmp_path):
    if not os.path.isfile(MODEL):
        pytest.skip("no MTP dummy")
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    # one root and one delta, compared cell for cell: no prompt node, which would add a second root
    gen, _ = _seed(cache, str(tmp_path / "seed.log"), node_prompt="off")
    nxt = BASE + gen + EXTRA
    L = len(BASE) + len(gen) - 1  # cells in the seed snapshot (last generated token not decoded)
    roots = sorted(glob.glob(cache + "/*.dft"))
    assert len(roots) == 1

    # warm resume: restore [0, L) + prefill the suffix, idle-save a delta with its own .dft
    s = _server(cache, str(tmp_path / "warm.log"), node_prompt="off")
    s.start()
    _complete(s, nxt, 1)
    _wait_meta(cache, 2, IDLE + 15)
    s.stop()
    deltas = [p for p in sorted(glob.glob(cache + "/*.dft")) if p not in roots]
    assert len(deltas) == 1, deltas

    # control: the same prompt prefilled in one process, root .dft
    cdir = os.path.join(ROOT, "ctl")
    os.makedirs(cdir)
    s = _server(cdir, str(tmp_path / "ctl.log"), node_prompt="off")
    s.start()
    _complete(s, nxt, 1)
    _wait_meta(cdir, 1, IDLE + 15)
    s.stop()
    ctl = sorted(glob.glob(cdir + "/*.dft"))
    ctl_kv, ext = _parse_kv1(ctl[-1])
    d_kv, _ = _parse_kv1(deltas[0])
    r_kv, _ = _parse_kv1(roots[0])
    print("ext bytes", ext, "control cells", min(ctl_kv), max(ctl_kv), "delta cells", min(d_kv), max(d_kv),
          "root cells", min(r_kv), max(r_kv), "L", L)
    rows = []
    for p in sorted(set(d_kv) | set(r_kv)):
        src = d_kv.get(p, r_kv.get(p))
        if p in ctl_kv:
            rows.append((p, float(np.max(np.abs(src - ctl_kv[p])))))
    print("max |dK| per pos around the boundary:", [r for r in rows if L - 3 <= r[0] <= L + 3])
    bad = [r for r in rows if r[1] > 1e-2]
    print("cells that differ from the control:", bad)
    assert not bad, "the restored draft (root + delta sidecars) must equal an uninterrupted prefill cell for cell"


@pytest.mark.parametrize("spec", [True, False])
def test_exact_resend_counts(tmp_path, spec):
    if not os.path.isfile(MODEL):
        pytest.skip("no MTP dummy")
    cache = os.path.join(ROOT, "c")
    os.makedirs(cache)
    s = _server(cache, str(tmp_path / "seed.log"), spec=spec)
    s.start()
    b = _complete(s, BASE, 8)
    _wait_meta(cache, 1, IDLE + 15)
    s.stop()
    saved = BASE + b["tokens"][:-1]
    for name, prompt in (("exact", saved), ("plus1", saved + [42])):
        # the oracle: the same prompt prefilled cold, with no cache directory at all
        s = _server(os.path.join(ROOT, "none"), str(tmp_path / f"{name}-cold.log"), spec=spec, auto=False)
        s.start()
        cold = _complete(s, prompt, 8)
        s.stop()
        d = os.path.join(ROOT, name)
        shutil.copytree(cache, d)
        s = _server(d, str(tmp_path / f"{name}.log"), spec=spec)
        s.start()
        r = _complete(s, prompt, 8)
        hit = _metric(s, "auto_cache_restore_hit_total")
        tok = _metric(s, "auto_cache_restore_tokens_total")
        miss = _metric(s, "auto_cache_restore_miss_total")
        s.stop()
        t = r["timings"]
        print(f"spec={spec} {name}: n_prompt={len(prompt)} hit={hit} restored_tokens={tok} miss={miss} "
              f"cache_n={t.get('cache_n')} cache_disk_n={t.get('cache_disk_n')} prompt_n={t.get('prompt_n')} "
              f"usage={r.get('tokens_cached')}")
        # parity with and without MTP: the restore is a kept hit either way, an exact resend of the
        # saved unit emits its first token from the logits sidecar (nothing prefilled), and one more
        # token prefills exactly that token
        assert hit == 1 and miss == 0
        assert t.get("cache_disk_n", 0) == len(saved)
        assert t["prompt_n"] == (0 if name == "exact" else 1)
        # and the hit is a CORRECT hit: the first token of an exact resend comes from the logits sidecar
        # and must be the token the seed run sampled there; every token must equal the cold oracle
        if name == "exact":
            assert r["tokens"][0] == b["tokens"][-1]
        assert r["tokens"] == cold["tokens"], f"{name}: restored {r['tokens']} vs cold {cold['tokens']}"


def test_ram_disk_warm_classification(tmp_path):
    model = os.path.join(os.environ["LLAMA_TEST_MODELS_DIR"], "llama-dense.gguf")
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    s.server_metrics = True
    s.cache_ram = 64
    s.log_path = str(tmp_path / "ram.log")
    s.start()
    P1 = [((i * 7) % 100) + 10 for i in range(64)]
    P2 = [((i * 13) % 100) + 10 for i in range(64)]
    r1 = _complete(s, P1, 4)
    r2 = _complete(s, P2, 4)              # P1's state goes to the RAM cache
    r3 = _complete(s, P1 + [50, 51], 4)   # comes back from RAM
    r4 = _complete(s, P1 + [50, 51, 52], 4)  # warm slot
    s.stop()
    for n, r in (("cold", r1), ("other", r2), ("ram", r3), ("warm", r4)):
        t = r["timings"]
        print(f"{n}: cache_n={t.get('cache_n')} cache_ram_n={t.get('cache_ram_n')} cache_disk_n={t.get('cache_disk_n')} prompt_n={t.get('prompt_n')}")
    assert r3["timings"].get("cache_ram_n", 0) > 0
    assert r4["timings"].get("cache_ram_n", 0) == 0 and r4["timings"]["cache_n"] > 0

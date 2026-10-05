import glob
import os
import shutil

import pytest
from utils import *

# A18: a text-only unit does not depend on whether a projector (--mmproj) is loaded, so it restores across an
# instance with and an instance without one (model_fp::restore_compatible: the projector fields are identity
# for units that hold media cells only). The text K/V is computed by the language model alone and text positions
# come from the token index on both sides; the projector only embeds media chunks. Checked both ways on the
# tinygemma3 vision dummy, each restore against a cold prefill on the restoring side, by the restore log line and
# the counters. A media unit stays tied to its projector (fp_mmproj), which test_slot_save_auto.py covers.

CACHE_DIR = "./tmp/slot_save_mmproj_identity"
TEXT = "Tell me a very long story about a dog named Spot. " * 4


def _server(with_mmproj: bool, cache: bool, log_path: str | None = None) -> ServerProcess:
    s = ServerPreset.tinygemma3()
    s.n_slots = 1
    s.n_ctx = 1024
    s.temperature = 0.0
    s.server_metrics = True
    s.log_path = log_path
    if not with_mmproj:
        s.no_mmproj = True
    if cache:
        s.slot_save_path = CACHE_DIR
        s.slot_save_auto = True
        s.slot_save_block = 16
        s.slot_save_min_tokens = 0
        s.slot_save_node_prompt = "off"
    return s


def _request(s: ServerProcess):
    data = {
        "temperature": 0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [{"type": "text", "text": TEXT}]}],
    }
    res = s.make_request("POST", "/chat/completions", data=data)
    assert res.status_code == 200, res.body
    t = res.body["timings"]
    return t, res.body["choices"][0]["message"]["content"]


def _metric(s, name: str) -> float:
    res = s.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


@pytest.fixture(autouse=True)
def clean_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.parametrize("save_with_mmproj", [True, False])
def test_text_unit_restores_across_the_projector(save_with_mmproj, tmp_path):
    """A text unit saved by an instance with (or without) --mmproj restores on an instance without (or with) one,
    and the answer equals a cold prefill on the restoring side."""
    load_with_mmproj = not save_with_mmproj

    ref = _server(load_with_mmproj, cache=False)
    ref.start()
    t_cold, content_cold = _request(ref)
    ref.stop()
    assert t_cold.get("cache_n", 0) == 0
    assert t_cold["prompt_n"] > 32

    a = _server(save_with_mmproj, cache=True)
    a.start()
    _request(a)
    a.stop()  # the shutdown flush publishes the unit
    metas = glob.glob(os.path.join(CACHE_DIR, "auto-*.bin.meta"))
    assert len(metas) == 1, metas

    log_path = str(tmp_path / "b.log")
    b = _server(load_with_mmproj, cache=True, log_path=log_path)
    b.start()
    t_warm, content_warm = _request(b)
    ident = _metric(b, "auto_cache_restore_miss_identity_total")
    b.stop()
    with open(log_path) as f:
        log = f.read()
    assert "auto-restore: reused" in log, "no disk restore on the other side of the projector"
    assert t_warm.get("cache_disk_n", 0) >= t_cold["prompt_n"] - 16, t_warm
    assert ident == 0
    assert content_warm == content_cold

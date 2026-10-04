import os
import shutil
import struct

import pytest
from utils import *

# Manual /slots save and restore on upstream's implementation and format (user decision 2026-10-03,
# round 4), per memory class, with the fork's additions around it:
#  - the .bin is upstream's (the token area opens with the packed-tokens marker LLAMA_TOKEN_NULL) and
#    has no .meta, so the auto cache never indexes it;
#  - on the FULL / RS classes a .logits sidecar is written next to it, so an exact resend of the saved
#    prompt after a restore emits its first token at once (prompt_n == 0), the seed's token (a
#    sliding-window state past one window has no sidecar and re-prefills: a known limit);
#  - a restored slot continues exactly like a cold prefill on plain attention, iSWA, hybrid and MTP
#    (with and without speculation);
#  - auto-* filenames are refused for saves (reserved for the auto cache);
#  - a failed restore leaves the slot usable.
#
# Dummy models from test-llama-archs (build/tests/test-models), token-id prompts:
#   llama-dense   plain attention
#   gemma3-dense  iSWA (n_swa = 32; the prompt is longer than one window)
#   qwen35-dense  RS (hybrid gated delta net)
# plus the MTP dummy (LLAMA_TEST_MTP_MODEL), a qwen35 hybrid with an MTP head.


def _default_models_dir() -> str:
    env = os.environ.get("LLAMA_TEST_MODELS_DIR", "")
    if env:
        return env
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.normpath(os.path.join(os.path.dirname(server_bin), "..", "tests", "test-models"))


MODELS_DIR = _default_models_dir()
MTP_MODEL = os.environ.get("LLAMA_TEST_MTP_MODEL", "")
SAVE_DIR = "./tmp/slot_save_manual"
LLAMA_TOKEN_NULL = -1
PROMPT = [((i * 7) % 100) + 10 for i in range(100)]
TAIL = [((i * 11) % 100) + 10 for i in range(24)]


def _model(name: str) -> str:
    if name == "mtp":
        if not MTP_MODEL or not os.path.isfile(MTP_MODEL):
            pytest.skip("no MTP dummy (set LLAMA_TEST_MTP_MODEL)")
        return MTP_MODEL
    path = os.path.join(MODELS_DIR, f"{name}.gguf")
    if not os.path.exists(path):
        pytest.skip(f"dummy model not found: {path}")
    return path


def _server(model: str, spec: bool = False, save: bool = True) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.model_alias = "dummy"
    s.n_ctx = 512
    s.n_batch = 512
    s.n_slots = 1
    s.temperature = 0.0
    if spec:
        s.spec_type = "draft-mtp"
        s.spec_draft_n_max = 3
    if save:
        s.slot_save_path = SAVE_DIR
    return s


def _complete(s: ServerProcess, prompt, n_predict: int = 8):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": True,
        "id_slot": 0, "return_tokens": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def _slots(s: ServerProcess, action: str, filename: str):
    return s.make_request("POST", f"/slots/0?action={action}", data={"filename": filename})


@pytest.fixture(autouse=True)
def clean_dir():
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    os.makedirs(SAVE_DIR)
    yield
    shutil.rmtree(SAVE_DIR, ignore_errors=True)


CASES = [("llama-dense", False), ("gemma3-dense", False), ("qwen35-dense", False), ("mtp", False), ("mtp", True)]


@pytest.mark.parametrize("name,spec", CASES)
def test_manual_round_trip_continues_like_cold(name, spec):
    """Save after a completion, restore in a fresh process, then (1) resend exactly the saved prompt and
    (2) extend it: both must produce what a cold prefill produces. On the FULL / RS classes the exact
    resend emits its first token from the .logits sidecar without prefilling anything."""
    model = _model(name)
    s = _server(model, spec)
    s.start()
    seed = _complete(s, PROMPT)
    res = _slots(s, "save", "snap.bin")
    assert res.status_code == 200, res.body
    n_saved = res.body["n_saved"]
    s.stop()
    saved = PROMPT + seed["tokens"][:-1]
    assert n_saved == len(saved)

    # upstream's format: the token area opens with the packed marker, and there is no .meta
    with open(os.path.join(SAVE_DIR, "snap.bin"), "rb") as f:
        head = f.read(16)
    assert struct.unpack_from("<i", head, 12)[0] == LLAMA_TOKEN_NULL
    assert not os.path.exists(os.path.join(SAVE_DIR, "snap.bin.meta"))
    has_logits = os.path.exists(os.path.join(SAVE_DIR, "snap.bin.logits"))
    if name in ("qwen35-dense", "mtp"):
        assert has_logits, "the RS class needs the logits sidecar for its exact-resend fast path"

    for label, prompt in (("exact", saved), ("extend", saved + TAIL)):
        s = _server(model, spec, save=False)
        s.start()
        cold = _complete(s, prompt)
        s.stop()

        s = _server(model, spec)
        s.start()
        res = _slots(s, "restore", "snap.bin")
        assert res.status_code == 200, res.body
        assert res.body["n_restored"] == n_saved
        body = _complete(s, prompt)
        s.stop()
        t = body["timings"]
        assert body["tokens"] == cold["tokens"], f"{label}: restored {body['tokens']} vs cold {cold['tokens']}"
        if label == "exact":
            assert body["tokens"][0] == seed["tokens"][-1]
            if has_logits:
                assert t["prompt_n"] == 0, t
            elif name == "llama-dense":
                assert t["prompt_n"] <= 1, t   # plain attention trims one token and re-decodes it
            # gemma3-dense: a sliding-window state past one window cannot rewind by one token and the
            # logits sidecar covers only the FULL and RS classes, so the exact resend re-prefills (a
            # known limit, the same before this endpoint moved to upstream's format); still equal to cold
        else:
            assert t["cache_n"] >= n_saved - 1, t
            assert t["prompt_n"] <= len(TAIL) + 1, t


def test_manual_save_refuses_auto_names():
    model = _model("llama-dense")
    s = _server(model)
    s.start()
    _complete(s, PROMPT)
    res = _slots(s, "save", "auto-0123456789abcdef-0123456789abcdef-100.bin")
    s.stop()
    assert res.status_code == 400
    assert "reserved" in str(res.body)
    assert os.listdir(SAVE_DIR) == []


def test_failed_restore_leaves_the_slot_usable():
    """A restore of a file that is not a slot snapshot fails with 400; the slot then serves a request
    exactly like a fresh one (no restore flag or logits left armed from an earlier restore)."""
    model = _model("qwen35-dense")
    s = _server(model)
    s.start()
    _complete(s, PROMPT)
    assert _slots(s, "save", "good.bin").status_code == 200
    with open(os.path.join(SAVE_DIR, "bad.bin"), "wb") as f:
        f.write(b"not a state file at all")
    assert _slots(s, "restore", "good.bin").status_code == 200
    res = _slots(s, "restore", "bad.bin")
    assert res.status_code == 400
    body = _complete(s, PROMPT)
    s.stop()

    s = _server(model, save=False)
    s.start()
    cold = _complete(s, PROMPT)
    s.stop()
    assert body["tokens"] == cold["tokens"]
    assert body["timings"]["prompt_n"] == len(PROMPT)

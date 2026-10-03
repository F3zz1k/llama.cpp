#!/usr/bin/env python3
# Build a tiny qwen35 GGUF with one MTP (nextn) block from the qwen35-dense test-llama-archs dummy,
# for tools/server/tests/unit/test_slot_save_mtp.py. usage: mk_mtp_dummy.py <qwen35-dense.gguf> <out.gguf>
import os
import sys
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = "qwen35"
w = gguf.GGUFWriter(dst, arch)

for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture":
        continue
    vt = f.types[0]
    if k == f"{arch}.block_count":
        w.add_uint32(k, 3); continue
    if k == f"{arch}.nextn_predict_layers":
        w.add_uint32(k, 1); continue
    if k == f"{arch}.attention.recurrent_layers":
        w.add_key_value(k, [1, 0, 0], vt, sub_type=f.types[1]); continue
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[1] if len(f.types) > 1 else None
        vals = f.contents()
        if len(vals) == 0:
            continue
        if sub == gguf.GGUFValueType.STRING:
            w.add_array(k, list(vals))
        else:
            w.add_key_value(k, list(vals), vt, sub_type=sub)
    else:
        w.add_key_value(k, f.contents(), vt)

rng = np.random.default_rng(1234)
for t in r.tensors:
    a = np.array(t.data)
    w.add_tensor(t.name, a)
    if t.name.startswith("blk.1."):
        nm = t.name[len("blk.1."):]
        if nm.endswith(".bias") or nm.endswith(".scale") or nm.endswith(".input_scale"):
            continue
        b = a.copy()
        if nm in ("attn_v.weight", "attn_output.weight"):
            b = (rng.standard_normal(b.shape) * float(os.environ.get("MTP_ATT_STD", "0.2"))).astype(np.float32)
        if nm in ("attn_q.weight", "attn_k.weight"):
            b = (rng.standard_normal(b.shape) * float(os.environ.get("MTP_QK_STD", "0.005"))).astype(np.float32)
        if nm.endswith(".bias"):
            b = np.zeros_like(b)
        w.add_tensor("blk.2." + nm, b)

ne = 256
w.add_tensor("blk.2.nextn.eh_proj.weight", (rng.standard_normal((ne, 2 * ne)) * float(os.environ.get("MTP_EH_STD", "0.2"))).astype(np.float32))
w.add_tensor("blk.2.nextn.enorm.weight", np.ones(ne, dtype=np.float32))
w.add_tensor("blk.2.nextn.hnorm.weight", np.ones(ne, dtype=np.float32))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", dst)

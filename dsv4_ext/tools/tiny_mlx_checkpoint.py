#!/usr/bin/env python3
"""Builds a tiny MTP-only checkpoint in the MLX naming/dtypes, for the dsv4_ext MTP bridge.

    python3 tools/tiny_mlx_checkpoint.py OUTDIR     # writes model-00001-of-00001.safetensors

The names and dtypes mirror what an MLX checkpoint of DeepSeek-V4 actually ships (read from a real
model.safetensors on 2026-10-21, metadata {"format": "mlx"}):

    decoder.attn.wq_a.weight   U32  int8 codes, four per element   + .scales U8 (32-col blocks)
    decoder.attn.wo_a.weight   U32  with leading dims that fold into the row count
    decoder.ffn.switch_mlp.gate_proj.weight U32 [E][rows][cols/8] eight 4-bit codes per element
    decoder.ffn.gate.weight    BF16 no scales
    enorm.weight / hnorm.weight / e_proj.weight / h_proj.weight / hc_head.* at the top level

The shapes are those of the tiny model tools/tiny_model.py writes (DIM=64, HD=64, QL=32, NH=4,
G=2, OL=16, E=8, FF=32, HC=4, ratio 8). The reference file records the float each code means so
the test can check the conversion instead of trusting it:

    dense_reference.txt   "<name> <row> <col> <value>" for sampled dense elements
    experts_reference.txt "<matrix> <e> <row> <col> <value>" for sampled expert elements
"""
import json
import math
import os
import struct
import sys

import numpy as np

rng = np.random.default_rng(7)

DIM, NH, HD, QL, OL, G, E, FF, HC = 64, 4, 64, 32, 16, 2, 8, 32, 4
GIN = NH * HD // G
RATIO = 8

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

T = {}   # name -> (dtype, shape, bytes)


def raw(name, dtype, shape, data):
    T[name] = (dtype, list(shape), data)


def f32(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    T[name] = ("F32", list(a.shape), a.astype("<f4").tobytes())


def bf16(name, arr):
    a = np.asarray(arr, dtype=np.float32).view(np.uint32)
    b = ((a >> 16) & 0xFFFF).astype("<u2")
    T[name] = ("BF16", list(np.asarray(arr).shape), b.tobytes())


def norm(n):
    return 1.0 + 0.1 * rng.normal(0, 1, n)


def dense(name, rows, cols):
    """An MLX dense weight: int8 codes four per U32 element, one e8m0 scale per 32 columns."""
    a = rng.normal(0, 1.0 / math.sqrt(cols), (rows, cols))
    codes = np.zeros((rows, cols), dtype=np.int8)
    scales = np.zeros((rows, cols // 32), dtype=np.uint8)
    for r in range(rows):
        for b in range(cols // 32):
            blk = a[r, b * 32:(b + 1) * 32]
            amax = float(np.abs(blk).max())
            s = 127
            if amax > 0:
                s = 127 + int(math.ceil(math.log2(amax / 127.0)))
            s = max(0, min(255, s))
            scale = 2.0 ** (s - 127)
            q = np.clip(np.round(blk / scale), -127, 127).astype(np.int8)
            codes[r, b * 32:(b + 1) * 32] = q
            scales[r, b] = s
            for k in range(0, 32, 7):        # a few sampled elements per block
                c = b * 32 + k
                ref_dense.append((name + ".weight", r, c, float(int(q[k])) * scale))
    packed = codes.view(np.uint8).reshape(rows, cols // 4, 4)
    u = (packed[:, :, 0].astype(np.uint32) | (packed[:, :, 1].astype(np.uint32) << 8)
         | (packed[:, :, 2].astype(np.uint32) << 16) | (packed[:, :, 3].astype(np.uint32) << 24))
    # mlx stores the codes little-endian inside each U32: element 0 is the low byte
    T[name + ".weight"] = ("U32", [rows, cols // 4], u.astype("<u4").tobytes())
    T[name + ".scales"] = ("U8", [rows, cols // 32], scales.tobytes())


def experts_fused(tag, rows, cols):
    """One fused MLX expert matrix: [E][rows][cols/8] U32, eight e2m1 codes per element."""
    TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]     # e2m1 magnitudes
    enc = {}
    for i, v in enumerate(TABLE):
        enc[v] = i
        enc[-v] = i | 8
    a = rng.normal(0, 1.0 / math.sqrt(cols), (E, rows, cols))
    codes = np.zeros((E, rows, cols), dtype=np.uint8)
    scales = np.zeros((E, rows, cols // 32), dtype=np.uint8)
    for e in range(E):
        for r in range(rows):
            for b in range(cols // 32):
                blk = a[e, r, b * 32:(b + 1) * 32]
                amax = float(np.abs(blk).max())
                s = 127
                if amax > 0:
                    s = 127 + int(math.ceil(math.log2(amax / 6.0)))
                s = max(0, min(255, s))
                scale = 2.0 ** (s - 127)
                for k in range(32):
                    v = blk[k] / scale
                    m = min(range(8), key=lambda i: abs(abs(v) - TABLE[i]))
                    c = enc[0.0] if v == 0 else (m | (8 if v < 0 else 0))
                    codes[e, r, b * 32 + k] = c
                    if k % 8 == 0:
                        ref_exp.append((tag, e, r, b * 32 + k, TABLE[c & 7] * (-1 if c & 8 else 1) * scale))
                scales[e, r, b] = s
    # eight codes per U32: element 2k in the low nibble of byte k
    by = codes.reshape(E, rows, cols // 8, 8)
    bb = np.zeros((E, rows, cols // 8, 4), dtype=np.uint8)
    for k in range(4):
        bb[..., k] = by[..., 2 * k] | (by[..., 2 * k + 1] << 4)
    u = (bb[..., 0].astype(np.uint32) | (bb[..., 1].astype(np.uint32) << 8)
         | (bb[..., 2].astype(np.uint32) << 16) | (bb[..., 3].astype(np.uint32) << 24))
    T["decoder.ffn.switch_mlp." + tag + ".weight"] = ("U32", [E, rows, cols // 8], u.astype("<u4").tobytes())
    T["decoder.ffn.switch_mlp." + tag + ".scales"] = ("U8", [E, rows, cols // 32], scales.tobytes())


ref_dense, ref_exp = [], []

P = "decoder."
f32(P + "attn.attn_sink", rng.normal(0, 1, NH))
bf16(P + "attn.q_norm.weight", norm(QL))
bf16(P + "attn.kv_norm.weight", norm(HD))
bf16(P + "attn_norm.weight", norm(DIM))
bf16(P + "ffn_norm.weight", norm(DIM))
f32(P + "ffn.gate.e_score_correction_bias", rng.normal(0, 0.3, E))
bf16(P + "ffn.gate.weight", rng.normal(0, 1.0 / math.sqrt(DIM), (E, DIM)))
dense(P + "attn.wq_a", QL, DIM)
dense(P + "attn.wq_b", NH * HD, QL)
dense(P + "attn.wkv", HD, DIM)
dense(P + "attn.wo_a", G * OL, GIN)
dense(P + "attn.wo_b", DIM, G * OL)
# the draft layer's compressor (ratio 8, no indexer), as the real checkpoint spells it
bf16(P + "attn.compressor.norm.weight", norm(HD))
dense(P + "attn.compressor.wkv", (1 + (1 if RATIO == 4 else 0)) * HD, DIM)
dense(P + "attn.compressor.wgate", (1 + (1 if RATIO == 4 else 0)) * HD, DIM)
f32(P + "attn.compressor.ape", rng.normal(0, 0.5, (RATIO, (1 + (1 if RATIO == 4 else 0)) * HD)))
mix = (2 + HC) * HC
f32(P + "attn_hc.base", rng.normal(0.5, 0.5, mix))
f32(P + "attn_hc.scale", rng.normal(0.5, 0.2, 3))
f32(P + "attn_hc.fn", rng.normal(0, 1.0 / math.sqrt(HC * DIM), (mix, HC * DIM)))
f32(P + "ffn_hc.base", rng.normal(0.5, 0.5, mix))
f32(P + "ffn_hc.scale", rng.normal(0.5, 0.2, 3))
f32(P + "ffn_hc.fn", rng.normal(0, 1.0 / math.sqrt(HC * DIM), (mix, HC * DIM)))
dense(P + "ffn.shared_experts.gate_proj", FF, DIM)
dense(P + "ffn.shared_experts.up_proj", FF, DIM)
dense(P + "ffn.shared_experts.down_proj", DIM, FF)
experts_fused("gate_proj", FF, DIM)
experts_fused("up_proj", FF, DIM)
experts_fused("down_proj", DIM, FF)

# the fusion module, at the top level exactly as mlx-to-hf leaves it
bf16("enorm.weight", norm(DIM))
bf16("hnorm.weight", norm(DIM))
dense("e_proj", DIM, DIM)
dense("h_proj", DIM, DIM)
f32("hc_head.base", rng.normal(0.5, 0.5, HC))
f32("hc_head.scale", rng.normal(0.5, 0.2, 1))
f32("hc_head.fn", rng.normal(0, 1.0 / math.sqrt(HC * DIM), (HC, HC * DIM)))
bf16("decoder.norm.weight", norm(DIM))
bf16("norm.weight", norm(DIM))

names = sorted(T)
data = b""
header = {}
for n in names:
    dtype, shape, rawb = T[n]
    if len(data) % 8:
        data += b"\0" * (8 - len(data) % 8)
    header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(rawb)]}
    data += rawb
hdr = json.dumps(header, separators=(",", ":")).encode()
path = os.path.join(out, "model-00001-of-00001.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr)))
    f.write(hdr)
    f.write(data)
with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": len(data)}, "weight_map": {n: os.path.basename(path) for n in names}}, f)
with open(os.path.join(out, "dense_reference.txt"), "w") as f:
    for row in ref_dense:
        f.write("%s %d %d %.9g\n" % row)
with open(os.path.join(out, "experts_reference.txt"), "w") as f:
    for row in ref_exp:
        f.write("%s %d %d %d %.9g\n" % row)

print("MLX MTP checkpoint: %d tensors, %d experts x 3 fused matrices | %s" % (len(T), E, path))

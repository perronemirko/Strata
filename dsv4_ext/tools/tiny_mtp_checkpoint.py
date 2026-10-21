#!/usr/bin/env python3
"""Builds a tiny MTP-only Hugging Face checkpoint (safetensors) for the dsv4_ext MTP bridge.

    python3 tools/tiny_mtp_checkpoint.py OUTDIR          # writes model-00001-of-00001.safetensors
    ./build-out/test_mtp_safetensors --model /tmp/dsv4_tiny/tiny.gguf --ckpt OUTDIR

The tensors are the DeepSeek-V4 HF names (mtp.0.attn.wq_a, mtp.0.ffn.experts.<e>.w1, ...) with the
shapes of the tiny model that tools/tiny_model.py writes (DIM=64, HD=64, QL=32, NH=4, G=2, OL=16,
E=8, FF=32, IH=4, ID=32, HC=4, ratio 8 -> compressor, no indexer). The dtypes mirror what the real
checkpoint ships: F32 for the 1-D parameters and the dense weights, and for the routed experts the
packed 4-bit codes (two signed nibbles per byte) plus one e8m0 scale per 32 columns.

The reference files let the test check the conversion instead of trusting it:
    experts_reference.txt   one line per expert tensor: "<matrix> <e> <row> <col> <value>" for a
                            sample of elements, the float the packed code x 2^(scale-127) means.
"""
import json
import math
import os
import struct
import sys

import numpy as np

rng = np.random.default_rng(11)

DIM, NH, HD, RD, QL, OL, G, WIN = 64, 4, 64, 16, 32, 16, 2, 4
HC, E, K, FF, NHASH = 4, 8, 2, 32, 1
IH, ID, ITOPK = 4, 32, 2
RATIO = 8                      # the draft layer: compressor, no indexer (as in tiny_model.py)
VOCAB = 40

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# ------------------------------------------------------------------ the tensors, HF names
T = {}   # name -> (dtype, shape, bytes)


def f32(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    T[name] = ("F32", list(a.shape), a.astype("<f4").tobytes())


def add(name, *shape, scale=1.0):
    """A random dense matrix as F32 (the real checkpoint has these as F8_E4M3 + scales; the
    bridge's dense path is dtype-checked, and the exactness test covers the expert path)."""
    a = rng.normal(0, scale / math.sqrt(shape[-1]), shape)
    f32(name, a)


def norm(n):
    return 1.0 + 0.1 * rng.normal(0, 1, n)


P = "mtp.0."
f32(P + "attn_norm.weight", norm(DIM))
f32(P + "attn.q_norm.weight", norm(QL))
f32(P + "attn.kv_norm.weight", norm(HD))
f32(P + "attn.attn_sink", rng.normal(0, 1, NH))
f32(P + "ffn_norm.weight", norm(DIM))
f32(P + "ffn.gate.bias", rng.normal(0, 0.3, E))
mix = (2 + HC) * HC
f32(P + "hc_attn_base", rng.normal(0.5, 0.5, mix))
f32(P + "hc_attn_scale", rng.normal(0.5, 0.2, 3))
f32(P + "hc_ffn_base", rng.normal(0.5, 0.5, mix))
f32(P + "hc_ffn_scale", rng.normal(0.5, 0.2, 3))
add(P + "attn.wq_a", QL, DIM)
add(P + "attn.wq_b", NH * HD, QL)
add(P + "attn.wkv", HD, DIM)
add(P + "attn.wo_a", G * OL, NH * HD // G)
add(P + "attn.wo_b", DIM, G * OL)
add(P + "ffn.gate.weight", E, DIM)
add(P + "ffn.shared_experts.w1", FF, DIM)
add(P + "ffn.shared_experts.w3", FF, DIM)
add(P + "ffn.shared_experts.w2", DIM, FF)
add(P + "hc_attn_fn", mix, HC * DIM)
add(P + "hc_ffn_fn", mix, HC * DIM)
# the draft layer's compressor (ratio 8, no indexer)
f32(P + "attn.compressor.norm.weight", norm(HD))
add(P + "attn.compressor.wkv", HD, DIM)
add(P + "attn.compressor.wgate", HD, DIM)
f32(P + "attn.compressor.ape", rng.normal(0, 0.5, (RATIO, HD)))
# the additive fusion module (what the DeepSeek-V4 checkpoint ships)
f32(P + "enorm.weight", norm(DIM))
f32(P + "hnorm.weight", norm(DIM))
add(P + "e_proj.weight", DIM, DIM)
add(P + "h_proj.weight", DIM, DIM)
# the real checkpoint spells these three without a .weight suffix
add(P + "hc_head_fn", HC, HC * DIM)
f32(P + "hc_head_base", rng.normal(0.5, 0.5, HC))
f32(P + "hc_head_scale", rng.normal(0.5, 0.2, 1))

# ------------------------------------------------------------------ the routed experts, packed 4-bit + e8m0
# Same container as the real checkpoint: the header calls it I8, two signed 4-bit codes per byte,
# one e8m0 scale byte per 32 columns. The reference records the float each code means.
ref = []


def experts(tag, rows, cols):
    """Writes mtp.0.ffn.experts.<e>.<tag>.{weight,scale} for every expert, the real HF order."""
    prefix = P + "ffn.experts."
    for e in range(E):
        w = rng.normal(0, 1.0 / math.sqrt(cols), (rows, cols))
        packed = bytearray()
        scales = bytearray()
        for r in range(rows):
            for b in range(cols // 32):
                blk = w[r, b * 32:(b + 1) * 32]
                amax = float(np.abs(blk).max())
                # an e8m0 scale is a power of two: take the smallest one that keeps every code in [-8, 7]
                s = 127
                if amax > 0:
                    s = 127 + int(math.ceil(math.log2(amax / 7.0)))
                s = max(0, min(255, s))
                scale = 2.0 ** (s - 127)
                codes = []
                for v in blk:
                    q = int(round(float(v) / scale))
                    q = max(-8, min(7, q))
                    codes.append(q)
                for k in range(16):
                    lo, hi = codes[2 * k], codes[2 * k + 1]
                    packed.append((lo & 0xF) | ((hi & 0xF) << 4))
                scales.append(s)
                for k in range(32):
                    ref.append((P + "ffn.experts.%d.%s.weight" % (e, tag), e, r, b * 32 + k, codes[k] * scale))
        T[P + "ffn.experts.%d.%s.weight" % (e, tag)] = ("I8", [rows, cols // 2], bytes(packed))
        T[P + "ffn.experts.%d.%s.scale" % (e, tag)] = ("F8_E8M0", [rows, cols // 32], bytes(scales))


experts(P + "ffn.experts.0.w1", FF, DIM)      # w1 and w3 are [FF][DIM]: the bridge splits them by name
experts(P + "ffn.experts.0.w3", FF, DIM)
experts(P + "ffn.experts.0.w2", DIM, FF)

# ------------------------------------------------------------------ write one safetensors shard
names = sorted(T)
data = b""
header = {}
for n in names:
    dtype, shape, raw = T[n]
    # a tensor's byte range is exactly its data: the alignment padding goes BEFORE the next tensor,
    # never inside this one (as the safetensors writer does).
    if len(data) % 8:
        data += b"\0" * (8 - len(data) % 8)
    header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
    data += raw
hdr = json.dumps(header, separators=(",", ":")).encode()
path = os.path.join(out, "model-00001-of-00001.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr)))
    f.write(hdr)
    f.write(data)
with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": len(data)}, "weight_map": {n: os.path.basename(path) for n in names}}, f)
with open(os.path.join(out, "experts_reference.txt"), "w") as f:
    for row in ref:
        f.write("%s %d %d %d %.9g\n" % row)

print("MTP checkpoint: %d tensors, %d experts x 3 matrices packed 4-bit | %s" % (len(T), E, path))

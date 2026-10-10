#!/usr/bin/env python3
"""Riferimento numpy di gpt-oss (naive, float64) + checkpoint minuscolo con gli STESSI nomi di tensore del modello vero
(model.layers.N.self_attn.q_proj.weight, ...mlp.router.weight, ...mlp.experts.gate_up_proj_blocks/_scales/_bias, ...).
Pesi degli expert in MXFP4 (blocchi da 32 lungo l'ultima dimensione, scala uint8 con bias 127, nibble basso = elemento pari),
tutto il resto BF16. Scrive config.json, 2 shard, indice e golden.txt.

Le scelte che il modello card e config.json NON fissano sono varianti (docs/gptoss_porting.md, "Non noto"):
  base        gate/up interleaved (gate = righe pari), RoPE half-split, concentrazione YaRN applicata a cos/sin
  glu_halves  gate = prima meta', up = seconda meta'
  rope_pairs  RoPE su coppie adiacenti
  no_conc     senza il fattore 0.1*ln(factor)+1 su cos/sin
Uso: python3 tools/tiny_gptoss.py tests/tiny   (solo numpy)"""
import json, math, os, struct, sys
import numpy as np

CFG = dict(hidden=64, layers=4, heads=4, kv=2, hd=16, E=8, topk=2, I=64, vocab=96, eps=1e-5, theta=150000.0,
           factor=4.0, orig=8, bf=32.0, bs=1.0, window=4, limit=3.0, alpha=1.702, eos=2, pad=11)
VARIANTS = {"base": dict(glu="inter", rope="half", conc=True), "glu_halves": dict(glu="halves", rope="half", conc=True),
            "rope_pairs": dict(glu="inter", rope="pairs", conc=True), "no_conc": dict(glu="inter", rope="half", conc=False)}
T_SEQ = 24
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)
P = "model."


def bf16_round(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f(u):
    return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def mx_quant(w):
    """w[..., K] (K%32==0) -> (nibble codes packed [..., K/32, 16] uint8, scales [..., K/32] uint8, dequantised float64)."""
    sh = w.shape
    b = w.reshape(sh[:-1] + (sh[-1] // 32, 32))
    amax = np.abs(b).max(-1)
    e = np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)).astype(np.int64)
    e = np.clip(e, -127, 128)
    s = np.exp2(e.astype(np.float64))[..., None]
    q = b / s
    idx = np.abs(np.abs(q)[..., None] - FP4[:8]).argmin(-1)
    code = (idx + 8 * (q < 0)).astype(np.uint8)
    deq = FP4[code] * s
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).astype(np.uint8)
    return packed, (e + 127).astype(np.uint8), deq.reshape(sh)


def write_st(path, tensors):
    hdr, off = {}, 0
    for k in sorted(tensors):
        dt, shape, raw = tensors[k]
        hdr[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for k in sorted(tensors):
            f.write(tensors[k][2])


class W:
    def __init__(self, seed=20250805):
        self.r = np.random.default_rng(seed); self.f = {}; self.t = {}

    def bf(self, n, a):
        u = bf16_round(a); self.f[n] = bf16_f(u).reshape(a.shape); self.t[n] = ("BF16", a.shape, u.tobytes())

    def mx(self, n, a):
        p, s, d = mx_quant(a)
        self.f[n] = d; self.t[n + "_blocks"] = ("U8", p.shape, p.tobytes()); self.t[n + "_scales"] = ("U8", s.shape, s.tobytes())


def build(c):
    w = W(); r = w.r; H, I, E = c["hidden"], c["I"], c["E"]; nh, kv, hd = c["heads"], c["kv"], c["hd"]
    n = lambda *s, sd=0.1: r.normal(0, sd, s)
    w.bf(P + "embed_tokens.weight", n(c["vocab"], H, sd=0.6)); w.bf(P + "norm.weight", 1 + n(H, sd=0.1))
    w.bf("lm_head.weight", n(c["vocab"], H, sd=0.3))
    for l in range(c["layers"]):
        b = P + "layers.%d." % l
        w.bf(b + "input_layernorm.weight", 1 + n(H, sd=0.1)); w.bf(b + "post_attention_layernorm.weight", 1 + n(H, sd=0.1))
        for nm, rows in (("q", nh * hd), ("k", kv * hd), ("v", kv * hd)):
            w.bf(b + "self_attn.%s_proj.weight" % nm, n(rows, H, sd=0.2)); w.bf(b + "self_attn.%s_proj.bias" % nm, n(rows, sd=0.1))
        w.bf(b + "self_attn.o_proj.weight", n(H, nh * hd, sd=0.2)); w.bf(b + "self_attn.o_proj.bias", n(H, sd=0.1))
        w.bf(b + "self_attn.sinks", n(nh, sd=1.0))
        w.bf(b + "mlp.router.weight", n(E, H, sd=0.4)); w.bf(b + "mlp.router.bias", n(E, sd=0.2))
        w.mx(b + "mlp.experts.gate_up_proj", n(E, 2 * I, H, sd=0.5)); w.bf(b + "mlp.experts.gate_up_proj_bias", n(E, 2 * I, sd=0.3))
        w.mx(b + "mlp.experts.down_proj", n(E, H, I, sd=0.5)); w.bf(b + "mlp.experts.down_proj_bias", n(E, H, sd=0.1))
    return w


def rms(x, s, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * s


def freqs(c, trunc=False):
    d = c["hd"]; h = d // 2
    f = c["theta"] ** (np.arange(h) / h)
    lo = h * math.log(c["orig"] / (c["bf"] * 2 * math.pi)) / math.log(c["theta"])
    hi = h * math.log(c["orig"] / (c["bs"] * 2 * math.pi)) / math.log(c["theta"])
    if trunc: lo, hi = math.floor(lo), math.ceil(hi)
    ramp = np.clip((np.arange(h) - lo) / (hi - lo), 0, 1)
    return (1 / (c["factor"] * f)) * ramp + (1 / f) * (1 - ramp)


def rope(x, t, fr, conc, mode):
    ang = t * fr; cs, sn = np.cos(ang) * conc, np.sin(ang) * conc
    if mode == "half":
        h = x.shape[-1] // 2; a, b = x[..., :h], x[..., h:]
        return np.concatenate([a * cs - b * sn, b * cs + a * sn], -1)
    a, b = x[..., 0::2], x[..., 1::2]; o = np.empty_like(x); o[..., 0::2] = a * cs - b * sn; o[..., 1::2] = a * sn + b * cs
    return o


def forward(w, c, ids, var):
    F = w.f; T = len(ids); H, I = c["hidden"], c["I"]; nh, kv, hd = c["heads"], c["kv"], c["hd"]
    fr = freqs(c); conc = (0.1 * math.log(c["factor"]) + 1.0) if var["conc"] else 1.0
    x = F[P + "embed_tokens.weight"][ids]; gap = 1e9
    for l in range(c["layers"]):
        b = P + "layers.%d." % l; sliding = (l % 2 == 0)
        h = rms(x, F[b + "input_layernorm.weight"], c["eps"])
        q = (h @ F[b + "self_attn.q_proj.weight"].T + F[b + "self_attn.q_proj.bias"]).reshape(T, nh, hd)
        k = (h @ F[b + "self_attn.k_proj.weight"].T + F[b + "self_attn.k_proj.bias"]).reshape(T, kv, hd)
        v = (h @ F[b + "self_attn.v_proj.weight"].T + F[b + "self_attn.v_proj.bias"]).reshape(T, kv, hd)
        q = np.stack([rope(q[t], t, fr, conc, var["rope"]) for t in range(T)])
        k = np.stack([rope(k[t], t, fr, conc, var["rope"]) for t in range(T)])
        sinks = F[b + "self_attn.sinks"]; att = np.zeros((T, nh * hd))
        for t in range(T):
            lo = max(0, t - c["window"] + 1) if sliding else 0
            for hh in range(nh):
                g = hh // (nh // kv)
                s = (k[lo:t + 1, g] @ q[t, hh]) * hd ** -0.5
                lg = np.concatenate([s, [sinks[hh]]]); p = np.exp(lg - lg.max()); p /= p.sum()
                att[t, hh * hd:(hh + 1) * hd] = p[:-1] @ v[lo:t + 1, g]
        x = x + att @ F[b + "self_attn.o_proj.weight"].T + F[b + "self_attn.o_proj.bias"]
        h = rms(x, F[b + "post_attention_layernorm.weight"], c["eps"])
        rl = h @ F[b + "mlp.router.weight"].T + F[b + "mlp.router.bias"]
        gu_w, gu_b = F[b + "mlp.experts.gate_up_proj"], F[b + "mlp.experts.gate_up_proj_bias"]
        dn_w, dn_b = F[b + "mlp.experts.down_proj"], F[b + "mlp.experts.down_proj_bias"]
        y = np.zeros_like(x)
        for t in range(T):
            o = np.argsort(-rl[t], kind="stable")[:c["topk"] + 1]; gap = min(gap, rl[t][o[c["topk"] - 1]] - rl[t][o[c["topk"]]])
            idx = o[:c["topk"]]; z = np.exp(rl[t][idx] - rl[t][idx].max()); wt = z / z.sum()
            for e, ww in zip(idx, wt):
                gu = gu_w[e] @ h[t] + gu_b[e]
                gate, up = (gu[0::2], gu[1::2]) if var["glu"] == "inter" else (gu[:I], gu[I:])
                gate = np.minimum(gate, c["limit"]); up = np.clip(up, -c["limit"], c["limit"])
                glu = gate / (1 + np.exp(-c["alpha"] * gate))
                y[t] += ww * (dn_w[e] @ ((up + 1) * glu) + dn_b[e])
        x = x + y
    return rms(x, F[P + "norm.weight"], c["eps"]) @ F["lm_head.weight"].T, gap


def hf_config(c):
    return {"architectures": ["GptOssForCausalLM"], "attention_bias": True, "eos_token_id": c["eos"], "pad_token_id": c["pad"],
            "head_dim": c["hd"], "hidden_size": c["hidden"], "initial_context_length": c["orig"], "intermediate_size": c["I"],
            "layer_types": ["sliding_attention" if l % 2 == 0 else "full_attention" for l in range(c["layers"])],
            "max_position_embeddings": 4096, "model_type": "gpt_oss", "num_attention_heads": c["heads"], "num_experts_per_tok": c["topk"],
            "num_hidden_layers": c["layers"], "num_key_value_heads": c["kv"], "num_local_experts": c["E"], "rms_norm_eps": c["eps"],
            "rope_scaling": {"beta_fast": c["bf"], "beta_slow": c["bs"], "factor": c["factor"], "original_max_position_embeddings": c["orig"],
                             "rope_type": "yarn", "truncate": False},
            "rope_theta": c["theta"], "sliding_window": c["window"], "swiglu_limit": c["limit"], "vocab_size": c["vocab"],
            "tie_word_embeddings": False}


def main(out):
    os.makedirs(out, exist_ok=True); c = CFG; w = build(c)
    ids = np.random.default_rng(5).integers(3, c["vocab"], T_SEQ).tolist()
    lines = ["ids %d %s" % (T_SEQ, " ".join(map(str, ids)))]
    for name, var in VARIANTS.items():
        lg, gap = forward(w, c, ids, var); assert gap > 1e-4, "gap router %g: cambia seed" % gap
        lines.append("logits_%s %d %s" % (name, lg.size, " ".join("%.9g" % v for v in lg.ravel())))
    names = sorted(w.t); half = len(names) // 2; wm = {}
    for fn, ns in (("model-00000-of-00001.safetensors", names[:half]), ("model-00001-of-00001.safetensors", names[half:])):
        write_st(os.path.join(out, fn), {k: w.t[k] for k in ns}); wm.update({k: fn for k in ns})
    json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    json.dump(hf_config(c), open(os.path.join(out, "config.json"), "w"), indent=1)
    open(os.path.join(out, "golden.txt"), "w").write("\n".join(lines) + "\n")
    print("ok: %d tensori, %d varianti, gap router min > 1e-4" % (len(w.t), len(VARIANTS)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/tiny")

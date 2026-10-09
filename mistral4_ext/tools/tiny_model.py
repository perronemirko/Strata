#!/usr/bin/env python3
"""Fase 2: riferimento numpy di Mistral Small 4 (model_type "mistral4") + checkpoint minuscolo.

Cosa fa
  * scrive in <out>/ un checkpoint MINUSCOLO con gli STESSI nomi di tensore del modello vero
    (language_model.model.layers.N.self_attn.q_a_proj.weight, ...mlp.experts.gate_up_proj, ...),
    pesi grandi in FP8 E4M3 + *_scale_inv, resto in BF16, due shard + model.safetensors.index.json + config.json;
  * calcola i logit con un forward NAIVE (K e V materializzati, come nel codice HF di DeepSeek-V3/Mistral4),
    in float64, e li scrive in <out>/golden.txt. Il motore C++ usa invece la forma "assorbita" di MLA:
    se i due coincidono, anche l'algebra dell'assorbimento e' verificata.

Le incognite NON risolte dalle fonti lette (vedi docs/mistral_small4_porting.md) sono varianti:
  base      router softmax, nessun mscale^2 nello softmax, scala llama4 su tutto q
  sigmoid   router sigmoid
  mscale    softmax_scale *= mscale^2   (come DeepSeek-V3 HF)
  qpe       scala llama4 solo su q_pe   (come descrive NeMo)

Uso:  python3 tools/tiny_model.py tests/tiny
Richiede solo numpy."""
import json
import math
import os
import struct
import sys

import numpy as np

CFG = dict(hidden=64, layers=3, heads=4, vocab=96, q_lora=32, kv_lora=16, nope=16, rope=8, v_head=16,
           n_exp=8, topk=2, moe_ff=24, n_shared=1, eps=1e-6, rope_theta=10000.0,
           yarn_factor=8.0, yarn_orig=16, beta_fast=32.0, beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0,
           llama4_beta=0.1, norm_topk=True, routed_scale=1.0, bos=1, eos=2, pad=11)
VARIANTS = {
    "base": dict(router="softmax", mscale=False, l4="q"),
    "sigmoid": dict(router="sigmoid", mscale=False, l4="q"),
    "mscale": dict(router="softmax", mscale=True, l4="q"),
    "qpe": dict(router="softmax", mscale=False, l4="qpe"),
}
T_SEQ = 24  # > yarn_orig (16): la scala llama4 e' != 1 per le posizioni >= 16
P = "language_model.model."


# ---------------------------------------------------------------- formati numerici
def fp8_table():
    """Valore decodificato dei 256 codici E4M3FN (bias 7, 0x7f/0xff = NaN)."""
    t = np.zeros(256, dtype=np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        if e == 15 and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[c] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


FP8 = fp8_table()
_FIN = np.array([c for c in range(256) if not np.isnan(FP8[c])])
_FINV = FP8[_FIN]
_ORD = np.argsort(_FINV, kind="stable")
_SORTED_V, _SORTED_C = _FINV[_ORD], _FIN[_ORD]


def fp8_encode(x):
    """Arrotonda al valore E4M3 piu' vicino (saturazione a +-448)."""
    x = np.clip(np.asarray(x, dtype=np.float64), -448.0, 448.0)
    i = np.clip(np.searchsorted(_SORTED_V, x), 1, len(_SORTED_V) - 1)
    lo, hi = _SORTED_V[i - 1], _SORTED_V[i]
    pick = np.where(np.abs(x - lo) <= np.abs(hi - x), i - 1, i)
    return _SORTED_C[pick].astype(np.uint8)


def bf16_round(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return u


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- safetensors (scrittura a mano)
def write_safetensors(path, tensors):
    """tensors: nome -> (dtype_str, shape, bytes)."""
    hdr, off = {}, 0
    for k in sorted(tensors):
        dt, shape, raw = tensors[k]
        hdr[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    hdr["__metadata__"] = {"format": "pt"}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in sorted(tensors):
            f.write(tensors[k][2])


# ---------------------------------------------------------------- pesi
class W:
    def __init__(self, seed=20260316):
        self.rng = np.random.default_rng(seed)
        self.f = {}   # nome -> float64 ESATTO (dopo quantizzazione): quello che il riferimento usa
        self.t = {}   # nome -> (dtype, shape, bytes)

    def bf16(self, name, arr):
        u = bf16_round(arr)
        self.f[name] = bf16_to_f32(u).astype(np.float64).reshape(arr.shape)
        self.t[name] = ("BF16", arr.shape, u.tobytes())

    def fp8(self, name, arr, scale_name, groups=None, parts=1):
        """arr [..]; groups = dim0 per-gruppo (esperti); parts = scale per gruppo (1 o 2: gate/up)."""
        arr = np.asarray(arr, dtype=np.float64)
        if groups is None:
            s = np.array([max(np.abs(arr).max() / 400.0, 1e-8)])
            codes = fp8_encode(arr / s[0])
            deq = FP8[codes] * s[0]
            scales = s
        else:
            g = arr.shape[0]
            scales = np.zeros((g, parts))
            codes = np.zeros(arr.shape, dtype=np.uint8)
            deq = np.zeros(arr.shape)
            rows = arr.shape[1] // parts
            for e in range(g):
                for p in range(parts):
                    blk = arr[e, p * rows:(p + 1) * rows]
                    sc = max(np.abs(blk).max() / 400.0, 1e-8)
                    scales[e, p] = sc
                    c = fp8_encode(blk / sc)
                    codes[e, p * rows:(p + 1) * rows] = c
                    deq[e, p * rows:(p + 1) * rows] = FP8[c] * sc
        self.f[name] = deq
        self.t[name] = ("F8_E4M3", arr.shape, codes.tobytes())
        sc32 = scales.astype(np.float32).ravel()
        self.t[scale_name] = ("F32", (len(sc32),), sc32.tobytes())
        self.t[name.rsplit(".", 1)[0] + ".activation_scale" if scale_name.endswith("weight_scale_inv")
               else name + "_activation_scale"] = ("F32", (1,), np.float32([1.0]).tobytes())
        # il riferimento deve usare gli scale arrotondati a float32, come il motore
        if groups is None:
            self.f[name] = FP8[codes] * float(sc32[0])
        else:
            sc = sc32.reshape(arr.shape[0], parts)
            d2 = np.zeros(arr.shape)
            rows = arr.shape[1] // parts
            for e in range(arr.shape[0]):
                for p in range(parts):
                    d2[e, p * rows:(p + 1) * rows] = FP8[codes[e, p * rows:(p + 1) * rows]] * float(sc[e, p])
            self.f[name] = d2


def build(cfg):
    w = W()
    H, Q, KV = cfg["hidden"], cfg["q_lora"], cfg["kv_lora"]
    nh, nope, rope, vh = cfg["heads"], cfg["nope"], cfg["rope"], cfg["v_head"]
    E, I = cfg["n_exp"], cfg["moe_ff"]
    r = w.rng
    nrm = lambda *s, sd=0.08: r.normal(0, sd, s)
    w.bf16(P + "embed_tokens.weight", nrm(cfg["vocab"], H, sd=0.5))
    w.bf16(P + "norm.weight", 1.0 + nrm(H, sd=0.1))
    w.bf16("language_model.lm_head.weight", nrm(cfg["vocab"], H, sd=0.3))
    for l in range(cfg["layers"]):
        b = P + "layers.%d." % l
        w.bf16(b + "input_layernorm.weight", 1.0 + nrm(H, sd=0.1))
        w.bf16(b + "post_attention_layernorm.weight", 1.0 + nrm(H, sd=0.1))
        w.bf16(b + "self_attn.q_a_layernorm.weight", 1.0 + nrm(Q, sd=0.1))
        w.bf16(b + "self_attn.kv_a_layernorm.weight", 1.0 + nrm(KV, sd=0.1))
        for nm, shape, sd in [("q_a_proj", (Q, H), 0.15), ("q_b_proj", (nh * (nope + rope), Q), 0.2),
                              ("kv_a_proj_with_mqa", (KV + rope, H), 0.15), ("kv_b_proj", (nh * (nope + vh), KV), 0.25),
                              ("o_proj", (H, nh * vh), 0.15)]:
            w.fp8(b + "self_attn.%s.weight" % nm, nrm(*shape, sd=sd), b + "self_attn.%s.weight_scale_inv" % nm)
        w.bf16(b + "mlp.gate.weight", nrm(E, H, sd=0.3))
        parts = 2 if l == 1 else 1   # il layer 1 prova la variante "uno scale per gate e uno per up"
        w.fp8(b + "mlp.experts.gate_up_proj", nrm(E, 2 * I, H, sd=0.2), b + "mlp.experts.gate_up_proj_scale_inv",
              groups=E, parts=parts)
        w.fp8(b + "mlp.experts.down_proj", nrm(E, H, I, sd=0.2), b + "mlp.experts.down_proj_scale_inv", groups=E)
        SI = I * cfg["n_shared"]
        for nm, shape in [("gate_proj", (SI, H)), ("up_proj", (SI, H)), ("down_proj", (H, SI))]:
            w.fp8(b + "mlp.shared_experts.%s.weight" % nm, nrm(*shape, sd=0.2),
                  b + "mlp.shared_experts.%s.weight_scale_inv" % nm)
    return w


# ---------------------------------------------------------------- riferimento (naive, float64)
def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def yarn_freqs(dim, base, factor, orig, bf, bs):
    pos = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extra, inter = 1.0 / pos, 1.0 / (factor * pos)
    fcd = lambda n: dim * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(fcd(bf)), 0), min(math.ceil(fcd(bs)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    return inter * ramp + extra * (1 - ramp)


def rope_pairs(x, pos, freqs):
    """x[..., r]: rotazione complessa sulle coppie adiacenti (2i, 2i+1) (rope_interleave=true)."""
    a, b = x[..., 0::2], x[..., 1::2]
    ang = pos * freqs
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def route(logits, cfg, mode):
    if mode == "softmax":
        e = np.exp(logits - logits.max())
        sc = e / e.sum()
    else:
        sc = 1.0 / (1.0 + np.exp(-logits))
    o = np.argsort(-sc, kind="stable")[:cfg["topk"] + 1]
    margin = logits[o[cfg["topk"] - 1]] - logits[o[cfg["topk"]]]   # gap sui logit (monotoni rispetto a softmax/sigmoid)
    idx = o[:cfg["topk"]]
    w = sc[idx]
    if cfg["norm_topk"]:
        w = w / w.sum()
    return idx, w * cfg["routed_scale"], margin


def forward(w, cfg, ids, var):
    F = w.f
    H, nh, nope, rope, vh = cfg["hidden"], cfg["heads"], cfg["nope"], cfg["rope"], cfg["v_head"]
    E, I, T = cfg["n_exp"], cfg["moe_ff"], len(ids)
    freqs = yarn_freqs(rope, cfg["rope_theta"], cfg["yarn_factor"], cfg["yarn_orig"], cfg["beta_fast"], cfg["beta_slow"])
    sm = (nope + rope) ** -0.5
    if var["mscale"]:
        m = 0.1 * cfg["mscale_all_dim"] * math.log(cfg["yarn_factor"]) + 1.0
        sm *= m * m
    s4 = np.array([1 + cfg["llama4_beta"] * math.log(1 + math.floor(t / cfg["yarn_orig"])) for t in range(T)])
    x = F[P + "embed_tokens.weight"][ids]
    min_margin = 1e9
    for l in range(cfg["layers"]):
        b = P + "layers.%d." % l
        h = rmsnorm(x, F[b + "input_layernorm.weight"], cfg["eps"])
        cq = rmsnorm(h @ F[b + "self_attn.q_a_proj.weight"].T, F[b + "self_attn.q_a_layernorm.weight"], cfg["eps"])
        q = (cq @ F[b + "self_attn.q_b_proj.weight"].T).reshape(T, nh, nope + rope)
        ckv = h @ F[b + "self_attn.kv_a_proj_with_mqa.weight"].T
        c = rmsnorm(ckv[:, :cfg["kv_lora"]], F[b + "self_attn.kv_a_layernorm.weight"], cfg["eps"])
        kpe = np.stack([rope_pairs(ckv[t, cfg["kv_lora"]:], t, freqs) for t in range(T)])
        kvb = (c @ F[b + "self_attn.kv_b_proj.weight"].T).reshape(T, nh, nope + vh)
        k_nope, v = kvb[..., :nope], kvb[..., nope:]
        q_nope = q[..., :nope].copy()
        q_pe = np.stack([rope_pairs(q[t, :, nope:], t, freqs) for t in range(T)])
        if var["l4"] == "q":
            q_nope = q_nope * s4[:, None, None]
        q_pe = q_pe * s4[:, None, None]
        k = np.concatenate([k_nope, np.broadcast_to(kpe[:, None, :], (T, nh, rope))], -1)
        qq = np.concatenate([q_nope, q_pe], -1)
        att = np.zeros((T, nh * vh))
        for t in range(T):
            for hh in range(nh):
                sc = (k[:t + 1, hh] @ qq[t, hh]) * sm
                p = np.exp(sc - sc.max())
                p /= p.sum()
                att[t, hh * vh:(hh + 1) * vh] = p @ v[:t + 1, hh]
        x = x + att @ F[b + "self_attn.o_proj.weight"].T
        h = rmsnorm(x, F[b + "post_attention_layernorm.weight"], cfg["eps"])
        gl = h @ F[b + "mlp.gate.weight"].T
        gu_w, dn_w = F[b + "mlp.experts.gate_up_proj"], F[b + "mlp.experts.down_proj"]
        y = np.zeros_like(x)
        for t in range(T):
            idx, wt, mg = route(gl[t], cfg, var["router"])
            min_margin = min(min_margin, mg)
            for e, ww in zip(idx, wt):
                gu = gu_w[e] @ h[t]
                a = gu[:I] / (1 + np.exp(-gu[:I])) * gu[I:]
                y[t] += ww * (dn_w[e] @ a)
        sg, su = h @ F[b + "mlp.shared_experts.gate_proj.weight"].T, h @ F[b + "mlp.shared_experts.up_proj.weight"].T
        y += (sg / (1 + np.exp(-sg)) * su) @ F[b + "mlp.shared_experts.down_proj.weight"].T
        x = x + y
    return rmsnorm(x, F[P + "norm.weight"], cfg["eps"]) @ F["language_model.lm_head.weight"].T, min_margin


def hf_config(cfg):
    return {"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
            "text_config": {
                "model_type": "mistral4", "hidden_size": cfg["hidden"], "num_hidden_layers": cfg["layers"],
                "num_attention_heads": cfg["heads"], "num_key_value_heads": cfg["heads"], "vocab_size": cfg["vocab"],
                "q_lora_rank": cfg["q_lora"], "kv_lora_rank": cfg["kv_lora"], "qk_nope_head_dim": cfg["nope"],
                "qk_rope_head_dim": cfg["rope"], "qk_head_dim": cfg["nope"] + cfg["rope"], "v_head_dim": cfg["v_head"],
                "n_routed_experts": cfg["n_exp"], "num_experts_per_tok": cfg["topk"], "n_shared_experts": cfg["n_shared"],
                "moe_intermediate_size": cfg["moe_ff"], "intermediate_size": 2 * cfg["moe_ff"], "first_k_dense_replace": 0,
                "n_group": 1, "topk_group": 1, "norm_topk_prob": cfg["norm_topk"], "routed_scaling_factor": cfg["routed_scale"],
                "rms_norm_eps": cfg["eps"], "rope_interleave": True, "bos_token_id": cfg["bos"], "eos_token_id": cfg["eos"],
                "pad_token_id": cfg["pad"], "max_position_embeddings": 4096,
                "rope_parameters": {"rope_type": "yarn", "rope_theta": cfg["rope_theta"], "factor": cfg["yarn_factor"],
                                    "original_max_position_embeddings": cfg["yarn_orig"], "beta_fast": cfg["beta_fast"],
                                    "beta_slow": cfg["beta_slow"], "mscale": cfg["mscale"],
                                    "mscale_all_dim": cfg["mscale_all_dim"], "llama_4_scaling_beta": cfg["llama4_beta"]}}}


def main(out):
    os.makedirs(out, exist_ok=True)
    cfg = CFG
    w = build(cfg)
    ids = (np.random.default_rng(7).integers(3, cfg["vocab"], T_SEQ)).tolist()
    lines = ["ids %d %s" % (T_SEQ, " ".join(map(str, ids))), "fp8_table 256 " + " ".join("nan" if np.isnan(v) else "%.9g" % v for v in FP8)]
    for name, var in VARIANTS.items():
        logits, mg = forward(w, cfg, ids, var)
        assert mg > 1e-4, "margine del router troppo piccolo (%g): cambia seed" % mg
        lines.append("logits_%s %d %s" % (name, logits.size, " ".join("%.9g" % v for v in logits.ravel())))
    names = sorted(w.t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wm = {}
    for fn, ns in shards.items():
        write_safetensors(os.path.join(out, fn), {n: w.t[n] for n in ns})
        wm.update({n: fn for n in ns})
    json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    json.dump(hf_config(cfg), open(os.path.join(out, "config.json"), "w"), indent=1)
    open(os.path.join(out, "golden.txt"), "w").write("\n".join(lines) + "\n")
    # golden delle funzioni singole
    print("ok: %d tensori, %d shard, %d varianti, gap router min > 1e-4" % (len(w.t), len(shards), len(VARIANTS)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/tiny")

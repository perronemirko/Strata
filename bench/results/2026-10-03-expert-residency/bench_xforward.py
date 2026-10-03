"""Do the experts stay resident across forwards?  The measurement the single-forward bench cannot make.

A single forward with keep < 128 experts always pays the file: 128 experts x 14.9 ms = 1.9 s per layer,
whatever the batching.  That says nothing about a real chat turn, where the router picks the same
experts again and again across the 64 decode steps that follow a prefill.  An LRU of blocks can be
empty inside one forward and still be ~free across a turn.

So this runs WHOLE TURNS: one prefill of PREFILL tokens, then DECODE decode steps, repeated TURNS
times, and reports each turn separately so the cold first turn is visible apart from the warm ones.

Routing is not uniform: real routers are skewed.  `--zipf s` weights expert e by 1/(e+1)^s, so a few
experts carry most of the picks (s=0 is uniform, s=1 is very skewed).  Prefill still touches every
expert, because n*k slots over 128 experts with no-repeat forces it - that is the wall.

    python /tmp/bench_xforward.py <mode> [key=value ...] [--zipf 0.9] [--turns 4]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/albus/workspaces/Strata-qwen3.6")

from serve.gemma4 import gguf as G
from serve.gemma4.backends import (BlockedExperts, ScratchExperts, StreamedExperts,
                                   load_layer_weights)
from serve.gemma4.model import Gemma4MoE, Gemma4Router, StackedExperts
from serve.gemma4.ops import make_ops

GGUF = ("/home/albus/.lmstudio/hub/models/lmstudio-community/"
        "gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf")
LAYER = 0


def even_routing(cfg, n_tokens, seed=3):
    """As even a top-k routing as no-repeat allows: n*K slots spread over all the experts."""
    k, ne = cfg.top_k_experts, cfg.num_experts
    base = (n_tokens * k) // ne
    per = [base] * ne
    for i in range(n_tokens * k - base * ne):
        per[i % ne] += 1
    cap, picks = [k] * n_tokens, [[] for _ in range(n_tokens)]
    for e, cnt in enumerate(per):
        for t in sorted(range(n_tokens), key=lambda t: (-cap[t], t))[:cnt]:
            cap[t] -= 1
            picks[t].append(e)
    idx = np.array([sorted(p) for p in picks], np.int64)
    vals = np.random.default_rng(seed).uniform(0.2, 1.0, (n_tokens, k)).astype(np.float32)
    return idx, vals


def skewed_routing(cfg, n_tokens, p, rng):
    """n tokens, each taking k distinct experts drawn from the skewed distribution `p`."""
    k, ne = cfg.top_k_experts, cfg.num_experts
    rows = np.empty((n_tokens, k), np.int64)
    for t in range(n_tokens):
        rows[t] = np.sort(rng.choice(ne, size=k, replace=False, p=p))
    vals = rng.uniform(0.2, 1.0, (n_tokens, k)).astype(np.float32)
    return rows, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("kw", nargs="*")
    ap.add_argument("--zipf", type=float, default=0.9)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=64)
    a = ap.parse_args()
    kw = {}
    for s in a.kw:
        k, v = s.split("=")
        kw[k] = int(v)

    import torch
    xp = make_ops("torch", device="cpu", dtype=torch.float32)
    g = G.Gemma4GGUF.open(GGUF)
    cfg = g.config()
    t = cfg.text
    w = load_layer_weights(g, cfg, LAYER, xp, stack_experts=(a.mode == "stack"))
    router = Gemma4Router(w, t, xp=xp)
    if a.mode == "stack":
        src = StackedExperts(w, t.moe_intermediate_size, xp=xp)
    elif a.mode == "stream":
        src = StreamedExperts(g, LAYER, cfg, xp, **kw)
    elif a.mode == "block":
        src = BlockedExperts(g, LAYER, cfg, xp, **kw)
    else:
        src = ScratchExperts(g, LAYER, cfg, xp, **kw)
    moe = Gemma4MoE(t, w, router, src, xp=xp)

    ne = int(t.num_experts)
    wgt = 1.0 / (np.arange(1, ne + 1, dtype=np.float64) ** a.zipf)
    p = wgt / wgt.sum()
    rng = np.random.default_rng(11)

    pre_idx, pre_vals = even_routing(t, a.prefill)
    dec = [skewed_routing(t, 1, p, rng) for _ in range(a.decode)]
    xp_pre = (xp.asarray(pre_idx, xp.int64), xp.asarray(pre_vals, xp.float32))
    xp_pre_x = xp.asarray(np.random.default_rng(1).standard_normal(
        (a.prefill, t.hidden_size)).astype(np.float32), xp.float32)
    xp_dec = [(xp.asarray(np.random.default_rng(5).standard_normal(
        (1, t.hidden_size)).astype(np.float32), xp.float32),
        xp.asarray(i, xp.int64), xp.asarray(v, xp.float32)) for i, v in dec]

    turns = []
    for turn in range(a.turns):
        t0 = time.perf_counter()
        moe._routed(xp_pre_x, *xp_pre)
        t_pre = time.perf_counter() - t0
        t1 = time.perf_counter()
        for x, idx, vals in xp_dec:
            moe._routed(x, idx, vals)
        t_dec = time.perf_counter() - t1
        turns.append((t_pre, t_dec))

    stats = " ".join(f"{k_}={getattr(src, k_, 0)}"
                     for k_ in ("reads", "hits", "built", "fills") if hasattr(src, k_))
    label = f"{a.mode} {kw}" if kw else a.mode
    per = " | ".join(f"T{i+1} pre {p_*1e3:7.1f} dec {d_*1e3:7.1f}"
                     for i, (p_, d_) in enumerate(turns))
    warm_pre = np.mean([p_ for p_, _ in turns[1:]]) * 1e3
    warm_dec = np.mean([d_ for _, d_ in turns[1:]]) * 1e3
    print(f"{label:<30} zipf={a.zipf} | {per} | warm pre {warm_pre:7.1f} ms "
          f"dec {warm_dec:7.1f} ms | {stats}", flush=True)
    g.close()


if __name__ == "__main__":
    main()

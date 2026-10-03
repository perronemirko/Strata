"""One expert profile per process, so no earlier config's memory can distort the next one's timing.

The combined bench measured `stack` at 307 ms for 128 tokens while an isolated probe of the same
matmuls on the same tensors did 88 ms: the process had already allocated several gigabytes of expert
caches by then.  Numbers that depend on what ran before them are not numbers, so each config now gets
a fresh interpreter.

    python /tmp/bench_one_mode.py <mode> [key=value ...]
"""
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
LAYER, BUDGET = 0, 4.0


def routing(cfg, n_tokens, seed=3):
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


def main():
    mode = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("=")
        kw[k] = int(v)
    import torch
    xp = make_ops("torch", device="cpu", dtype=torch.float32)
    g = G.Gemma4GGUF.open(GGUF)
    cfg = g.config()
    t = cfg.text
    w = load_layer_weights(g, cfg, LAYER, xp, stack_experts=(mode == "stack"))
    router = Gemma4Router(w, t, xp=xp)
    if mode == "stack":
        src = StackedExperts(w, t.moe_intermediate_size, xp=xp)
    elif mode == "stream":
        src = StreamedExperts(g, LAYER, cfg, xp, **kw)
    elif mode == "block":
        src = BlockedExperts(g, LAYER, cfg, xp, **kw)
    else:
        src = ScratchExperts(g, LAYER, cfg, xp, **kw)
    moe = Gemma4MoE(t, w, router, src, xp=xp)

    out = []
    for n in (1, 128, 512):
        idx_np, vals_np = routing(t, n)
        x = xp.asarray(np.random.default_rng(1).standard_normal((n, t.hidden_size)).astype(np.float32),
                       xp.float32)
        idx, vals = xp.asarray(idx_np, xp.int64), xp.asarray(vals_np, xp.float32)
        moe._routed(x, idx, vals)                       # warm
        for attr in ("reads", "hits", "built", "fills"):
            if hasattr(src, attr):
                setattr(src, attr, 0)
        best, reps, total = None, 0, 0.0
        while total < BUDGET and reps < 5:
            t0 = time.perf_counter()
            moe._routed(x, idx, vals)
            dt = time.perf_counter() - t0
            total += dt
            reps += 1
            best = dt if best is None else min(best, dt)
        out.append((n, best, reps))
    label = f"{mode} {kw}" if kw else mode
    per = " | ".join(f"{n:>4} tok {b*1e3:8.1f} ms x{r}" for n, b, r in out)
    stats = " ".join(f"{a}={getattr(src, a, 0)}" for a in ("reads", "built", "fills")
                     if hasattr(src, a))
    print(f"{label:<34} {per} | {stats}", flush=True)
    g.close()


if __name__ == "__main__":
    main()

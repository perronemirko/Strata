"""Perche' `block block=8 keep=16` ha dato 3007 ms a 128 token in un bench e 171 ms in un altro?

Stessa config, stesso layer, stesso n: 19x di differenza.  O e' rumore della macchina (il server
Strata ha la GPU occupata e altri processi torch giravano), o c'e' una pathologia vera in
BlockedExperts.  Questo stampa OGNI singola ripetizione, non solo il min, piu' la RSS del processo:
se le rep sono stabili e' un numero, se ballano di un ordine di grandezza e' la macchina.

    python /tmp/probe_block.py block=8 keep=16 n=128
"""
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "/home/albus/workspaces/Strata-qwen3.6")

from serve.gemma4 import gguf as G
from serve.gemma4.backends import BlockedExperts, StreamedExperts, load_layer_weights
from serve.gemma4.model import Gemma4MoE, Gemma4Router
from serve.gemma4.ops import make_ops

GGUF = ("/home/albus/.lmstudio/hub/models/lmstudio-community/"
        "gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf")
LAYER = 0


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
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = int(v)
    n = kw.pop("n", 128)

    import torch
    print(f"torch threads: intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()}",
          flush=True)
    xp = make_ops("torch", device="cpu", dtype=torch.float32)
    g = G.Gemma4GGUF.open(GGUF)
    cfg = g.config()
    t = cfg.text
    w = load_layer_weights(g, cfg, LAYER, xp, stack_experts=False)
    router = Gemma4Router(w, t, xp=xp)
    if "keep" in kw and "block" not in kw:
        src = StreamedExperts(g, LAYER, cfg, xp, **kw)
    else:
        src = BlockedExperts(g, LAYER, cfg, xp, **kw)
    moe = Gemma4MoE(t, w, router, src, xp=xp)

    idx_np, vals_np = routing(t, n)
    x = xp.asarray(np.random.default_rng(1).standard_normal((n, t.hidden_size)).astype(np.float32),
                   xp.float32)
    idx, vals = xp.asarray(idx_np, xp.int64), xp.asarray(vals_np, xp.float32)

    for rep in range(6):
        t0 = time.perf_counter()
        moe._routed(x, idx, vals)
        dt = (time.perf_counter() - t0) * 1e3
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        st = " ".join(f"{k_}={getattr(src, k_, 0)}"
                      for k_ in ("reads", "hits", "built") if hasattr(src, k_))
        tag = "warm" if rep == 0 else "    "
        print(f"  {tag} rep{rep} {dt:9.1f} ms | rss {rss:8.1f} MiB | {st}", flush=True)
    g.close()


if __name__ == "__main__":
    main()

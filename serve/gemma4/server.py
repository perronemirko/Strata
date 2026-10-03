"""serve/gemma4/server.py - run Gemma 4 behind Strata's existing OpenAI/Anthropic API.

    python -m serve.gemma4.server --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --port 8095

This is a thin launcher.  It opens the GGUF, checks that this build can actually run it, builds a
`Gemma4Engine` over a backend, a `GgufTokenizer` and a `Gemma4ChatTemplate`, then hands them to the
SAME `Service` and `serve()` that run the qwen4exp engine - so every endpoint, the streaming, the
FIFO, the API key and the Monitor behave identically.  Nothing in `serve/server.py` is modified;
this module only imports from it.

Why not `--engine gemma4` inside serve.server
---------------------------------------------
`serve.server.main` is one long function that knows about `strata` and `mock`.  Adding a third
branch there would edit the file every existing test exercises.  Keeping Gemma 4's launcher here
means the existing `main` is untouched, and a user reaches Gemma 4 with a different command.  The
two share their whole HTTP layer by importing it.

What the launcher checks before it starts
-----------------------------------------
A 17 GB download is expensive, and the failure modes are specific: an MTP drafter passed as the
target, a quant this build cannot decode, a split model with a shard missing, a config that
disagrees with the file.  `--check` prints the verdict and exits without loading anything, and a
normal start runs the same check first so the user sees the reason instead of a stack trace.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serve.server import Service, serve  # the existing HTTP layer, unchanged  # noqa: E402

from .backends import (EXPERT_MODE_CHOICES, available_backends,  # noqa: E402
                       backend_supports)
from .config import Gemma4Config  # noqa: E402
from .engine import Gemma4Engine, default_sampling  # noqa: E402
from .template import Gemma4ChatTemplate  # noqa: E402
from .tokenizer import GgufTokenizer  # noqa: E402


def build_engine(args) -> Gemma4Engine:
    config = Gemma4Config.from_json(args.config) if args.config else None
    kwargs: dict = {"expert_mode": args.expert_mode, "expert_stream": args.expert_stream,
                    "expert_block": args.expert_block, "expert_blocks": args.expert_blocks,
                    "expert_group": args.expert_group}
    if args.backend == "torch":
        kwargs.update({"device": args.device, "dtype": args.dtype})
    return Gemma4Engine(
        path=args.model, config=config, backend_name=args.backend,
        max_context=args.max_context, lazy=args.lazy, **kwargs,
    )


def check(path: str) -> int:
    ok, why = backend_supports(path)
    print(("can run: " if ok else "cannot run: ") + why)
    return 0 if ok else 1


def gguf_sampling(path: str) -> dict:
    """The file's own `general.sampling.*` header defaults, read without loading any weights.

    Gemma 4's card says temperature 1.0 / top_p 0.95 / top_k 64 and Unsloth writes exactly those
    into the GGUF. `Service` only applies defaults it is handed through `sampling_defaults`, so a
    launcher that drops them quietly turns every request that omits a temperature greedy.
    """
    from . import gguf as _gguf
    try:
        g = _gguf.Gemma4GGUF.open(path)
    except Exception:
        return {}
    try:
        return default_sampling(g.metadata)
    finally:
        g.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="a Gemma 4 GGUF (single file or a -00001-of-00002 shard)")
    ap.add_argument("--config", help="the model's HF config.json (optional; the GGUF's own "
                                     "metadata is used without it)")
    ap.add_argument("--backend", default="numpy", choices=available_backends(),
                    help=f"what runs the forward pass (available: {', '.join(available_backends())})")
    ap.add_argument("--device", default="auto", help="torch device: auto, cuda, cpu, cuda:0")
    ap.add_argument("--dtype", default="float16", choices=("float32", "float16", "bfloat16"),
                    help="torch compute/weight dtype")
    ap.add_argument("--expert-mode", default="auto", choices=EXPERT_MODE_CHOICES,
                    help="where the routed experts live. auto (the default) picks by regime from the "
                         "memory actually free: stack when all the experts fit, which is the only "
                         "regime measured fast (~0.15 s per layer at 128 tokens), otherwise stream "
                         "sized to what fits, which measured the fastest of the partial profiles "
                         "(~3 s per layer, and block measured 1.55x slower than stream there for the "
                         "same bytes). scratch = one reusable buffer of --expert-group experts per "
                         "layer. See bench/results/2026-10-03-expert-residency/")
    ap.add_argument("--expert-block", type=int, default=8,
                    help="--expert-mode block: experts per block, and the largest group the MoE will "
                         "batch (8 is 182 MiB per layer in float32, 91 MiB in float16). Only "
                         "--expert-block times --expert-blocks >= the expert count keeps a layer fully "
                         "resident; below that, block measured SLOWER than stream for the same bytes")
    ap.add_argument("--expert-blocks", type=int, default=4,
                    help="--expert-mode block: how many blocks per layer to keep")
    ap.add_argument("--expert-group", type=int, default=8,
                    help="--expert-mode scratch: experts in the reusable buffer, and the largest "
                         "group the MoE will batch (8 is 182 MiB per layer in float32)")
    ap.add_argument("--expert-stream", type=int, default=24,
                    help="--expert-mode stream: how many single experts per layer to keep on device")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--model-name", default="gemma-4-26b-a4b")
    ap.add_argument("--enable-thinking", action="store_true", help="turn thinking on by default")
    ap.add_argument("--lazy", action="store_true", help="load the model on the first request")
    ap.add_argument("--check", action="store_true",
                    help="report whether this file can run, then exit (loads no weights)")
    ap.add_argument("--open", action="store_true", help="open the web app when ready")
    a = ap.parse_args()

    ok, why = backend_supports(a.model)
    if not ok:
        print(f"[gemma4] {a.model}: {why}", file=sys.stderr, flush=True)
        return 1
    if a.check:
        print(f"[gemma4] can run: {why}")
        return 0
    print(f"[gemma4] {why}", flush=True)

    print(f"[gemma4] tokenizer from {a.model} ...", flush=True)
    tok = GgufTokenizer.from_gguf(a.model)
    print(f"[gemma4] {tok.n} pieces; building the engine on the '{a.backend}' backend "
          f"({'on first request' if a.lazy else 'now'}) ...", flush=True)
    engine = build_engine(a)

    # the template's BOS: only pass a real one if the tokenizer actually has that piece, else ""
    bos = "<bos>" if "<bos>" in tok.id_of else ""
    # the file's own chat_template when it carries one (Google revised theirs after the first
    # upload; the bundled jinja is the fallback, not the authority)
    template = Gemma4ChatTemplate.from_gguf(a.model, bos_token=bos,
                                            enable_thinking=a.enable_thinking)

    svc = Service(engine, tok, template, model_name=a.model_name,
                  sampling_defaults=gguf_sampling(a.model))
    if svc.sampling_defaults:
        print(f"[gemma4] sampling defaults from the file: {svc.sampling_defaults}", flush=True)
    svc.api_key = a.api_key
    httpd = serve(svc, host=a.host, port=a.port)
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI /v1/chat/completions, Anthropic /v1/messages, "
          f"context {engine.max_context}{', API key required' if svc.api_key else ''})", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{here}:{a.port}/")

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[gemma4] stopping ...", flush=True)
        httpd.shutdown()
        engine.close()
        print("[gemma4] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

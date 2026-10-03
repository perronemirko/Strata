"""serve/gemma4/engine.py - `Gemma4Engine`: the `Engine` protocol, over a Gemma 4 backend.

`serve/server.py`'s `Engine` protocol is one method and one attribute:

    max_context: int
    generate(prompt_ids, max_new, sampling, cancel) -> iterator[int]

`StrataEngine` satisfies it by talking to the C++ engine over a pipe; `MockEngine` by replaying a
script.  `Gemma4Engine` satisfies it by running a `Backend` in-process.  Because it is the same
protocol, the whole existing HTTP layer - OpenAI and Anthropic endpoints, streaming, the FIFO, the
conversation status, the Monitor - works unchanged on a Gemma 4 model.

The contract it honours, read from how `Service.run` uses an engine
-------------------------------------------------------------------
  * `generate` yields token ids one at a time and stops at `max_new` or when `cancel` is set.
  * It checks `cancel` between tokens, so a client disconnect stops the model promptly.
  * It sets `self.last` to a fresh dict when it finishes, with the keys the Monitor reads:
    generated, prompt_tokens, prompt_ms, decode_ms, finish.  `Service` compares the object identity
    of `last` to know whether a DONE happened, so a new dict per request matters.
  * `alive()`, `restart()`, `unload()`, `exit_code()` let `Service.ensure_loaded` bring the model
    back after an unload or a crash, exactly as it does for `StrataEngine`.
  * `self.info` is a dict of facts for the Monitor (here: backend name, quant, layer counts).

Sampling
--------
Gemma 4's documented defaults are temperature=1.0, top_p=0.95, top_k=64, and the GGUF itself
carries them as `general.sampling.temp/top_p/top_k`; `default_sampling()` reads those so a request
that sends nothing gets what the model was tuned with.  `generate` applies temperature, top-k and
top-p and is greedy at temperature 0.  The sampler is deliberately small and readable; it is not
Strata's tuned C++ sampler, and it says so.

Stopping
--------
The text config's `eos_token_id` is 1 (`<eos>`), while the GGUF's `tokenizer.ggml.eos_token_id` is
106 (`<turn|>`, the end of a chat turn).  Both are stop tokens, and a generation that stops only on
one of them either runs past the end of a turn or never terminates on bare text.  `stop_ids()`
returns the union, and the server's own detokeniser handles the `<turn|>` marker.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np

from .backends import Backend, get_backend
from .config import Gemma4Config


class Gemma4Engine:
    """An in-process Gemma 4 model behind Strata's `Engine` protocol."""

    def __init__(self, backend: Backend | None = None, *, path: str | None = None,
                 config: Gemma4Config | None = None, backend_name: str = "numpy",
                 max_context: int = 8192, lazy: bool = False, eos_ids: set[int] | None = None,
                 **backend_kwargs):
        if backend is None:
            backend = get_backend(backend_name, path=path, config=config, **backend_kwargs)
        self.backend = backend
        self.max_context = int(max_context)
        self.eos_ids = set(eos_ids) if eos_ids is not None else {1, 106}
        self.info: dict = {"backend": backend.name}
        self.last: dict = {}
        self.progress = None
        self.unloaded = True
        self._dead = False
        self._lock = threading.Lock()
        if not lazy:
            self._load()

    # ---- lifecycle (what Service.ensure_loaded / unload call)
    def _load(self) -> None:
        self.backend.load()
        cfg = getattr(self.backend, "config", None)
        if cfg is not None:
            self.info.update(n_layers=cfg.text.num_hidden_layers,
                             n_full_layers=cfg.text.n_full_layers(),
                             n_sliding_layers=cfg.text.n_sliding_layers(),
                             n_experts=cfg.text.num_experts, top_k=cfg.text.top_k_experts,
                             vocab=cfg.text.vocab_size,
                             sliding_window=cfg.text.sliding_window,
                             softcap=cfg.text.final_logit_softcapping)
            if cfg.text.max_position_embeddings:
                self.max_context = min(self.max_context, cfg.text.max_position_embeddings)
            # the config's own EOS (106 = `<turn|>`, the end of a chat turn) alongside the text
            # config's 1 (`<eos>`); a generation that stops on only one of them either runs past
            # the end of a turn or never terminates on bare text. Gemma 4's config.json writes
            # `eos_token_id` as the list [1, 106], so honour the whole list, not just its first.
            for v in (getattr(cfg, "eos_token_ids", ()) or
                      (getattr(cfg, "eos_token_id", None),
                       getattr(cfg.text, "eos_token_id", None))):
                if v is not None:
                    self.eos_ids.add(int(v))
        self.unloaded = False
        self._dead = False

    def alive(self) -> bool:
        return not self.unloaded and not self._dead and self.backend.loaded

    def exit_code(self):
        return None if self.alive() else 1

    def restart(self) -> None:
        self._load()

    def unload(self) -> None:
        self.backend.close()
        self.unloaded = True

    def close(self) -> None:
        self.unload()

    # ---- generation
    def generate(self, ids: list[int], max_new: int, sampling: dict,
                 cancel: threading.Event) -> Iterator[int]:
        t0 = time.perf_counter()
        prompt_ms = 0.0
        generated = 0
        finish = "length"
        try:
            if not self.alive():
                self._load()
            self.backend.reset()
            # Prefill: run the whole prompt once, keep only its last-position logits.
            prefill = time.perf_counter()
            next_logits = _row(self.backend.forward(ids, start_pos=0), -1)
            prompt_ms = (time.perf_counter() - prefill) * 1000.0
            pos = len(ids)
            temperature = float(sampling.get("temperature", 1.0) or 0.0)
            top_k = int(sampling.get("top_k", 64) or 0)
            top_p = float(sampling.get("top_p", 0.95) or 0.0)
            seed = sampling.get("seed")
            rng = np.random.default_rng(None if seed in (None, -1) else int(seed))
            for _ in range(max_new):
                if cancel.is_set():
                    finish = "cancel"
                    break
                tok = _sample(next_logits, temperature, top_k, top_p, rng)
                if tok in self.eos_ids:
                    finish = "stop"
                    break
                generated += 1
                yield int(tok)
                if cancel.is_set():
                    finish = "cancel"
                    break
                next_logits = _row(self.backend.forward([tok], start_pos=pos), -1)
                pos += 1
        finally:
            decode_ms = (time.perf_counter() - t0) * 1000.0 - prompt_ms
            # A fresh dict every request: Service compares identity to tell a real DONE from a death.
            self.last = {"generated": generated, "prompt_tokens": len(ids), "prompt_ms": prompt_ms,
                         "decode_ms": max(0.0, decode_ms), "finish": finish}


# ------------------------------------------------------------------ logits -> numpy
def _row(logits, index: int) -> np.ndarray:
    """One row of a backend's logits as float32 numpy, whatever the backend ran on."""
    if hasattr(logits, "detach"):                      # torch
        logits = logits.detach().to("cpu", dtype=torch_float32())
        return np.asarray(logits, dtype=np.float32)[index]
    return np.asarray(logits, dtype=np.float32)[index]


def torch_float32():
    import torch
    return torch.float32


# ------------------------------------------------------------------ sampling
def _sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float,
            rng: np.random.Generator) -> int:
    """Greedy at temperature 0; else temperature -> top-k -> top-p -> multinomial.  Small and
    readable on purpose; the reference path is about correctness, not sampler parity."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_k and top_k < p.size:
        idx = np.argpartition(p, -top_k)[-top_k:]
        mask = np.zeros_like(p, dtype=bool)
        mask[idx] = True
        p[~mask] = 0.0
        p /= p.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        sorted_p = p[order]
        cum = np.cumsum(sorted_p)
        cut = np.searchsorted(cum, top_p) + 1
        keep = order[:cut]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p[~mask] = 0.0
        p /= p.sum()
    return int(rng.choice(p.size, p=p))


def default_sampling(metadata: dict) -> dict:
    """The sampling the GGUF asks for: `general.sampling.temp/top_p/top_k`.  Gemma 4's card says
    temperature 1.0, top_p 0.95, top_k 64, and Unsloth writes exactly those into the file, so a
    launcher that ignores them is quietly changing the model's behaviour."""
    out = {}
    for key, k in (("temperature", "general.sampling.temp"), ("top_p", "general.sampling.top_p"),
                   ("top_k", "general.sampling.top_k")):
        v = metadata.get(k)
        if v is not None:
            out[key] = float(v) if key != "top_k" else int(v)
    return out

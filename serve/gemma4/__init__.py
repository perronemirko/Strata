"""serve/gemma4 - Gemma 4 (gemma-4-26B-A4B) support for Strata, as an add-on.

Why a separate package instead of a flag inside the existing engine
-------------------------------------------------------------------
Strata's C++ engine (`src/`, `include/`) is specialised for ONE architecture: `qwen4exp`, whose
layers are GDN (a gated-delta recurrence) and QSA (a sparse indexer over a shared-head KV cache),
with a 512-expert MoE and a per-layer-embedding n-gram table.  Its geometry is a struct of those
numbers (`include/strata/core/layout.hpp::ModelGeometry`) and its kernels are written against them.

Gemma 4 26B A4B is a different model on every one of those axes:

    qwen4exp (Strata's engine)          gemma-4-26B-A4B
    ----------------------------        ----------------------------------
    GDN 36 + QSA 12 layers              24 sliding + 6 full attention (5/6 pattern)
    512 experts, top-10                 128 experts, top-8, + the dense MLP as a 9th
    head_dim 256, 2 KV heads            head_dim 256 sliding / 512 full, 8 / 2 KV heads
    one RoPE (yarn for long ctx)        two RoPEs: default 1e4, proportional 1e6 with NoPE
    PLE n-gram table on layer 1         none (hidden_size_per_layer_input = 0)
    no logit softcap                    final_logit_softcapping 30.0
    no QK-norm                          attn_q_norm / attn_k_norm every layer

Teaching the existing engine a second architecture would mean editing the geometry struct, the
layer composition, the session state and every kernel that reads them - i.e. the files that today
carry a measured, parity-tested contract.  That is the "don't break the repo" line.

So this package plugs in at the boundary Strata already exposes.  `serve/server.py` says it in its
own docstring:

    The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of
    token ids`.

`Gemma4Engine` implements that protocol, and `serve.gemma4.server` hands it to the existing
`Service`/`serve()`.  The dependency points one way only: gemma4 imports serve, serve never
imports gemma4.  Nothing under `src/`, `include/`, `serve/*.py` or `setup.py` changes, so every
existing test keeps running against exactly the code it ran against before.

What is here
------------
    config.py     the HF `config.json` and the GGUF's `gemma4.*` metadata -> `Gemma4Config`,
                  including the per-layer head widths and the two ropes
    ops.py        the array vocabulary (numpy / torch) the model is written against
    gguf.py       the GGUF side: names, shapes, validation, dequantisation, split shards, and
                  per-expert byte slicing
    quant.py      the matching encoders, so a test can write a real Q4_K/Q5_1/Q8_0 file
    model.py      the architecture as classes: RMSNorm, QK-norm attention, router, MoE, decoder
    tokenizer.py  the tokenizer, from the GGUF's own metadata
    backends.py   what runs the forward pass, behind one interface (a registry): `numpy` and `torch`
    engine.py     `Gemma4Engine`: the `Engine` protocol over a backend
    server.py     `python -m serve.gemma4.server` - the same OpenAI/Anthropic API, Gemma 4 model

Honesty about status
--------------------
`NumpyBackend` is the reference: the architecture written out so the shapes, the layer order, the
two RoPEs and the router can be checked on a tiny model in a unit test, and it will run a real
BF16/Q8_0 file on a machine with enough RAM, slowly.  `TorchBackend` runs the same `model.py` on
GPU with the routed experts streamed from the file.  Neither is Strata's C++ engine and neither
inherits its streaming expert pool or its speculative decoding; nobody should quote a tokens/s
number for them without measuring.  The dequantisers are checked byte-for-byte against ggml's
`dequantize_row_*`, and the tensor names against the real Unsloth header, so what is asserted here
is what the file actually contains.
"""
from __future__ import annotations

from .config import (Gemma4Config, Gemma4TextConfig, LayerKind, LayerSpec, MtpSpec,  # noqa: F401
                     RopeParams)
from .engine import Gemma4Engine  # noqa: F401
from .gguf import Gemma4GGUF, UnsupportedQuant, validate  # noqa: F401

__all__ = ["Gemma4Config", "Gemma4TextConfig", "Gemma4Engine", "Gemma4GGUF", "LayerKind",
           "LayerSpec", "MtpSpec", "RopeParams", "UnsupportedQuant", "validate"]

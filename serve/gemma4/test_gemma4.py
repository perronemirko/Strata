"""Tests for the Gemma 4 add-on.  No GPU and no download are required: a tiny model is written as
a real GGUF - with the tensor names and metadata keys the real Unsloth file uses - and run through
the reference backend end to end.

    python -m pytest serve/gemma4/test_gemma4.py -q
    python -m unittest serve.gemma4.test_gemma4 -v

What is checked, and why each thing is here
-------------------------------------------
  * the dequantisers, against a scalar reference transcribed from ggml's `dequantize_row_*`.  The
    numpy versions are vectorised, and a vectorisation slip in a K-quant moves a weight by one
    nibble while keeping every shape correct.
  * the encoder/dequant round trip, which is what proves the fixture's byte layout is the layout
    the reader expects.
  * expert slicing: `read_expert(n, e)` must equal `read_tensor(n)[:, :, e]`, byte for byte.
  * split shards: a two-shard model must load and read like a single-file one.
  * config: the per-layer head widths, the per-layer KV array, the two ropes, the bool
    `sliding_window_pattern`, and the `rope_freqs.weight` NoPE sentinel.
  * the forward pass: shapes, finiteness, the logit softcap, and that token-by-token decode
    matches one full forward (the KV cache and the sliding window).
  * numpy vs torch parity on the same weights, when torch is importable.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import pathlib
import sys
import tempfile
import threading
import unittest

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serve.gemma4 import gguf as G  # noqa: E402
from serve.gemma4 import quant as Q  # noqa: E402
from serve.gemma4.backends import (NumpyBackend, TorchBackend, available_backends,  # noqa: E402
                                   backend_supports, estimate_bytes, expert_resident_bytes,
                                   get_backend)
from serve.gemma4.config import (ConfigError, Gemma4Config, Gemma4TextConfig, LayerKind,  # noqa: E402
                                 MtpSpec, RopeParams, apply_rope_factors, check_rope_freqs)
from serve.gemma4.engine import Gemma4Engine, _sample, default_sampling  # noqa: E402
from serve.gemma4.model import (Gemma4Attention, Gemma4DecoderLayer, Gemma4ForCausalLM,  # noqa: E402
                                Gemma4Model, Gemma4MoE, Gemma4Router, KVCache, StackedExperts,
                                apply_rotary, gelu_pytorch_tanh, rms_norm, rope_tables)
from serve.gemma4.ops import NumpyOps, available_ops
from serve.gemma4.template import Gemma4ChatTemplate  # noqa: E402
from serve.gemma4.tokenizer import GgufTokenizer  # noqa: E402


# ------------------------------------------------------------------ a tiny model, as a real GGUF
def tiny_text_config(**over) -> Gemma4TextConfig:
    """Small but structurally faithful: 2 layers, one sliding and one full, fused gate/up experts,
    QK-norms, layer scales, the six norms of a MoE layer."""
    kw = dict(
        hidden_size=16, num_hidden_layers=2, vocab_size=32, intermediate_size=16,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
        num_global_key_value_heads=1, kv_heads_per_layer=(1, 1), sliding_window=4,
        max_position_embeddings=64, num_experts=4, top_k_experts=2, moe_intermediate_size=8,
        layer_types=(LayerKind.SLIDING, LayerKind.FULL),
        rope_sliding=RopeParams("default", 10000.0, 1.0),
        rope_full=RopeParams("proportional", 1_000_000.0, 0.25),
    )
    kw.update(over)
    return Gemma4TextConfig(**kw)


def _tensor_shapes(cfg: Gemma4TextConfig, layer: int) -> dict[str, tuple[int, ...]]:
    """The per-layer tensors, named exactly as the real file names them."""
    p = ""
    s = {
        p + "attn_norm.weight": (cfg.hidden_size,),
        p + "attn_q.weight": (cfg.hidden_size, cfg.q_proj_out(layer)),
        p + "attn_q_norm.weight": (cfg.head_dim_for(layer),),
        p + "attn_output.weight": (cfg.q_proj_out(layer), cfg.hidden_size),
        p + "post_attention_norm.weight": (cfg.hidden_size,),
        p + "ffn_norm.weight": (cfg.hidden_size,),
        p + "ffn_gate.weight": (cfg.hidden_size, cfg.intermediate_size),
        p + "ffn_up.weight": (cfg.hidden_size, cfg.intermediate_size),
        p + "ffn_down.weight": (cfg.intermediate_size, cfg.hidden_size),
        p + "post_ffw_norm.weight": (cfg.hidden_size,),
        p + "layer_output_scale.weight": (1,),
    }
    if cfg.has_kv_proj(layer):
        s[p + "attn_k.weight"] = (cfg.hidden_size, cfg.kv_proj_out(layer))
        s[p + "attn_k_norm.weight"] = (cfg.head_dim_for(layer),)
        if cfg.has_v_proj(layer):
            s[p + "attn_v.weight"] = (cfg.hidden_size, cfg.kv_proj_out(layer))
    if cfg.enable_moe_block:
        s[p + "ffn_gate_inp.weight"] = (cfg.hidden_size, cfg.num_experts)
        s[p + "ffn_gate_inp.scale"] = (cfg.hidden_size,)
        s[p + "pre_ffw_norm_2.weight"] = (cfg.hidden_size,)
        s[p + "post_ffw_norm_1.weight"] = (cfg.hidden_size,)
        s[p + "post_ffw_norm_2.weight"] = (cfg.hidden_size,)
        s[p + "ffn_gate_up_exps.weight"] = (cfg.hidden_size, 2 * cfg.moe_intermediate_size,
                                            cfg.num_experts)
        s[p + "ffn_down_exps.weight"] = (cfg.moe_intermediate_size, cfg.hidden_size,
                                         cfg.num_experts)
        s[p + "ffn_down_exps.scale"] = (cfg.num_experts,)
    return s


def gguf_metadata(cfg: Gemma4TextConfig) -> dict:
    """The `gemma4.*` keys the real header carries, including the per-layer arrays."""
    md = {
        "general.architecture": "gemma4",
        "general.name": "tiny gemma4 test",
        "gemma4.block_count": cfg.num_hidden_layers,
        "gemma4.context_length": cfg.max_position_embeddings,
        "gemma4.embedding_length": cfg.hidden_size,
        "gemma4.feed_forward_length": cfg.intermediate_size,
        "gemma4.attention.head_count": cfg.num_attention_heads,
        "gemma4.attention.head_count_kv": [cfg.kv_heads_for(i) for i in
                                           range(cfg.num_hidden_layers)],
        "gemma4.rope.freq_base": cfg.rope_full.rope_theta,
        "gemma4.rope.freq_base_swa": cfg.rope_sliding.rope_theta,
        "gemma4.attention.layer_norm_rms_epsilon": cfg.rms_norm_eps,
        "gemma4.attention.key_length": cfg.global_head_dim,
        "gemma4.attention.value_length": cfg.global_head_dim,
        "gemma4.attention.key_length_swa": cfg.head_dim,
        "gemma4.attention.value_length_swa": cfg.head_dim,
        "gemma4.final_logit_softcapping": cfg.final_logit_softcapping,
        "gemma4.attention.sliding_window": cfg.sliding_window,
        "gemma4.attention.shared_kv_layers": cfg.num_kv_shared_layers,
        "gemma4.embedding_length_per_layer": cfg.hidden_size_per_layer_input,
        "gemma4.attention.sliding_window_pattern": [not cfg.is_full(i) for i in
                                                    range(cfg.num_hidden_layers)],
        "gemma4.rope.dimension_count": cfg.global_head_dim,
        "gemma4.rope.dimension_count_swa": cfg.head_dim,
        "gemma4.vocab_size": cfg.vocab_size,
    }
    # A dense Gemma 4 (gemma-4-12B) writes no expert keys; mirror that so the fixture exercises the
    # same header shape a real dense file has.
    if cfg.enable_moe_block:
        md["gemma4.expert_count"] = cfg.num_experts
        md["gemma4.expert_used_count"] = cfg.top_k_experts
        md["gemma4.expert_feed_forward_length"] = cfg.moe_intermediate_size
    md.update(tiny_tokenizer_metadata(cfg))
    return md


def tiny_tokenizer_metadata(cfg: Gemma4TextConfig) -> dict:
    """A `tokenizer.ggml.*` block matching `vocab_size`, so the fixture exercises the tokenizer
    path too and a launcher can actually start against it.  The real file's vocabulary is 262144
    pieces; this is the same keys at the tiny model's size."""
    names = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<turn|>", "<|channel>", "<|think|>"]
    pieces = [f"t{i}" for i in range(cfg.vocab_size - len(names))]
    tokens = names + pieces
    if len(tokens) != cfg.vocab_size:
        raise ValueError(f"fixture vocab {len(tokens)} != vocab_size {cfg.vocab_size}")
    types = [3, 1, 3, 3, 3, 3, 3, 3] + [1] * len(pieces)     # 3 = CONTROL, 1 = NORMAL
    return {
        "tokenizer.ggml.model": "gemma4",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
                                  -1000.0, -1000.0] + [-1.0] * len(pieces),
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 5,                    # <turn|>, as the real file says
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.add_space_prefix": False,
        "tokenizer.ggml.add_bos_token": True,
        "general.sampling.temp": 1.0,
        "general.sampling.top_p": 0.95,
        "general.sampling.top_k": 64,
    }


def rope_freq_values(cfg: Gemma4TextConfig) -> list[float]:
    """What the real file writes: 1.0 on the rotated channels, 1e30 on the NoPE ones."""
    n = cfg.global_head_dim // 2
    rotated = int(cfg.rope_full.partial_rotary_factor * cfg.global_head_dim // 2)
    return [1.0] * rotated + [1e30] * (n - rotated)


def write_tiny_gguf(path: pathlib.Path, cfg: Gemma4TextConfig, *, seed: int = 0,
                    drop: set[str] | None = None, break_shape: str | None = None,
                    quant: dict[str, str] | None = None) -> None:
    """A complete, loadable Gemma 4 GGUF with random weights.  `drop` omits tensors (validation),
    `break_shape` writes one with a wrong shape (the shape check), `quant` picks an encoding per
    tensor name (default F32)."""
    rng = np.random.default_rng(seed)
    drop = drop or set()
    quant = quant or {}
    md = gguf_metadata(cfg)
    md["rope_freqs_note"] = "rope_freqs is a tensor, not metadata"
    md.pop("rope_freqs_note")
    tensors = []

    def put(name: str, shape: tuple[int, ...]):
        if name in drop:
            return
        if break_shape == name:
            shape = tuple(shape[:-1]) + (shape[-1] + 1,)
        ty = quant.get(name, "F32")
        if name.endswith("layer_output_scale.weight"):
            arr = np.array([float(rng.uniform(0.05, 1.0))], np.float32)
        elif name.endswith("_norm.weight") or name.endswith(".scale"):
            arr = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.5
        else:
            arr = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        tensors.append(Q.make_tensor(name, arr, ty))

    put("token_embd.weight", (cfg.hidden_size, cfg.vocab_size))
    put("output_norm.weight", (cfg.hidden_size,))
    put("rope_freqs.weight", (cfg.global_head_dim // 2,))
    # rope_freqs is a factor table, not random weights
    for i, t in enumerate(tensors):
        if t.name == "rope_freqs.weight":
            tensors[i] = Q.make_tensor(t.name, np.array(rope_freq_values(cfg), np.float32), "F32")
    for layer in range(cfg.num_hidden_layers):
        for name, shape in _tensor_shapes(cfg, layer).items():
            put(f"blk.{layer}.{name}", shape)
    Q.write_gguf(path, md, tensors)


def tiny_model(cfg: Gemma4TextConfig, seed: int = 0, xp=None) -> Gemma4ForCausalLM:
    """The same tiny model assembled in memory (no file), for the pure-math tests."""
    xp = xp or NumpyOps()
    rng = np.random.default_rng(seed)
    layers = []
    for layer in range(cfg.num_hidden_layers):
        spec = cfg.layer_spec(layer)
        w = {
            "attn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_attn_norm": np.ones(cfg.hidden_size, np.float32),
            "ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_1": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "pre_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "q": rng.standard_normal((cfg.hidden_size, cfg.q_proj_out(layer))).astype(np.float32),
            "q_norm": np.ones(cfg.head_dim_for(layer), np.float32),
            "k": rng.standard_normal((cfg.hidden_size, cfg.kv_proj_out(layer))).astype(np.float32),
            "k_norm": np.ones(cfg.head_dim_for(layer), np.float32),
            "o": rng.standard_normal((cfg.q_proj_out(layer), cfg.hidden_size)).astype(np.float32),
            "gate": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "up": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "down": rng.standard_normal((cfg.intermediate_size, cfg.hidden_size)).astype(np.float32),
            "router_w": rng.standard_normal((cfg.hidden_size, cfg.num_experts)).astype(np.float32),
            "router_scale": (rng.standard_normal((cfg.hidden_size,)) * 4 + 32).astype(np.float32),
            "expert_scale": (rng.standard_normal((cfg.num_experts,)) * 0.02 + 1.0).astype(np.float32),
            "gate_up_exps": rng.standard_normal(
                (cfg.hidden_size, 2 * cfg.moe_intermediate_size, cfg.num_experts)).astype(np.float32),
            "down_exps": rng.standard_normal(
                (cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts)).astype(np.float32),
            "layer_scale": float(rng.uniform(0.05, 1.0)),
        }
        if cfg.has_v_proj(layer):
            w["v"] = rng.standard_normal(
                (cfg.hidden_size, cfg.kv_proj_out(layer))).astype(np.float32)
        w = {k: (xp.asarray(v, xp.float32) if hasattr(v, "shape") or isinstance(v, (list, tuple))
                 else v) for k, v in w.items()}
        router = Gemma4Router(w, cfg, xp=xp)
        experts = StackedExperts(w, cfg.moe_intermediate_size, xp=xp)
        layers.append(Gemma4DecoderLayer(cfg, spec, w, xp=xp, router=router, experts=experts))
    embed = (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) * 0.05).astype(np.float32)
    model = Gemma4Model(cfg, layers, xp.asarray(embed, xp.float32),
                        xp.ones((cfg.hidden_size,), xp.float32) if hasattr(xp, "ones")
                        else np.ones(cfg.hidden_size, np.float32), xp=xp)
    return Gemma4ForCausalLM(Gemma4Config(text=cfg), model, xp.asarray(embed, xp.float32), xp=xp)


# ------------------------------------------------------------------ scalar dequant reference
# Transcribed from ggml-quants.c's dequantize_row_q4_K / q5_1 / q8_0, one element at a time, with
# no numpy vectorisation.  If the fast version and this one disagree, one of them is wrong; the
# fast version is the one under test.
def _ref_scale_min(scales: bytes) -> tuple[list[int], list[int]]:
    sc, mn = [0] * 8, [0] * 8
    for j in range(8):
        if j < 4:
            sc[j] = scales[j] & 63
            mn[j] = scales[j + 4] & 63
        else:
            sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
            mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mn


def ref_dequant_q4_K(raw: bytes) -> list[float]:
    out: list[float] = []
    for i in range(len(raw) // 144):
        b = raw[i * 144:(i + 1) * 144]
        d = float(np.frombuffer(b, dtype="<f2", count=1, offset=0)[0])
        dmin = float(np.frombuffer(b, dtype="<f2", count=1, offset=2)[0])
        sc, mn = _ref_scale_min(b[4:16])
        q = b[16:144]
        is_ = 0
        for j in range(0, 256, 64):
            d1, m1 = d * sc[is_], dmin * mn[is_]
            d2, m2 = d * sc[is_ + 1], dmin * mn[is_ + 1]
            for l in range(32):
                out.append(d1 * (q[l] & 0xF) - m1)
            for l in range(32):
                out.append(d2 * (q[l] >> 4) - m2)
            q = q[32:]
            is_ += 2
    return out


def ref_dequant_q5_1(raw: bytes) -> list[float]:
    out: list[float] = []
    for i in range(len(raw) // 24):
        b = raw[i * 24:(i + 1) * 24]
        d = float(np.frombuffer(b, dtype="<f2", count=1, offset=0)[0])
        m = float(np.frombuffer(b, dtype="<f2", count=1, offset=2)[0])
        qh = int.from_bytes(b[4:8], "little")
        qs = b[8:24]
        blk = [0.0] * 32
        for j in range(16):
            # ggml writes y[j] and y[j + qk/2], NOT two adjacent values, and reads the 5th bit of
            # the second half from qh bit (j + 16) - `(qh >> (j+12)) & 0x10` is bit j+16.
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            blk[j] = d * ((qs[j] & 0xF) | xh0) + m
            blk[j + 16] = d * ((qs[j] >> 4) | xh1) + m
        out.extend(blk)
    return out


def ref_dequant_q8_0(raw: bytes) -> list[float]:
    out: list[float] = []
    for i in range(len(raw) // 34):
        b = raw[i * 34:(i + 1) * 34]
        d = float(np.frombuffer(b, dtype="<f2", count=1, offset=0)[0])
        for j in range(32):
            out.append(b[2 + j] * d if b[2 + j] < 128 else (b[2 + j] - 256) * d)
    return out


class TestDequant(unittest.TestCase):
    def test_q4_K_matches_the_scalar_reference(self):
        rng = np.random.default_rng(11)
        w = rng.standard_normal(256 * 3).astype(np.float32)
        raw = Q.quantize_q4_k(w)
        self.assertEqual(len(raw), 3 * 144)
        ref = ref_dequant_q4_K(raw)
        got = G.dequant("Q4_K", raw)
        self.assertEqual(len(ref), got.size)
        np.testing.assert_allclose(got, np.asarray(ref, np.float32), rtol=0, atol=0)
        # and the quantisation error itself stays small
        self.assertLess(float(np.max(np.abs(got - w))), 0.35)

    def test_q5_1_matches_the_scalar_reference(self):
        rng = np.random.default_rng(12)
        w = rng.standard_normal(32 * 5).astype(np.float32)
        raw = Q.quantize_q5_1(w)
        self.assertEqual(len(raw), 5 * 24)
        ref = ref_dequant_q5_1(raw)
        got = G.dequant("Q5_1", raw)
        np.testing.assert_allclose(got, np.asarray(ref, np.float32), rtol=0, atol=0)
        self.assertLess(float(np.max(np.abs(got - w))), 0.12)

    def test_q8_0_matches_the_scalar_reference(self):
        rng = np.random.default_rng(13)
        w = rng.standard_normal(32 * 4).astype(np.float32)
        raw = Q.quantize_q8_0(w)
        ref = ref_dequant_q8_0(raw)
        got = G.dequant("Q8_0", raw)
        np.testing.assert_allclose(got, np.asarray(ref, np.float32), rtol=0, atol=1e-6)

    def test_bf16_and_f16_roundtrip(self):
        w = np.array([1.0, -2.5, 0.125, 7.0], np.float32)
        np.testing.assert_allclose(G.dequant("BF16", Q.make_tensor("t", w, "BF16").data), w)
        np.testing.assert_allclose(G.dequant("F16", Q.make_tensor("t", w, "F16").data), w)

    def test_unsupported_encoding_names_itself(self):
        with self.assertRaises(G.UnsupportedQuant) as cm:
            G.dequant("Q2_K", b"\0" * 84)
        self.assertIn("Q2_K", str(cm.exception))


# ------------------------------------------------------------------ config
class TestConfig(unittest.TestCase):
    def test_layer_types_are_the_source_of_truth(self):
        cfg = tiny_text_config()
        self.assertEqual(cfg.layer_kind(0), LayerKind.SLIDING)
        self.assertEqual(cfg.layer_kind(1), LayerKind.FULL)
        self.assertEqual(cfg.n_full_layers(), 1)
        self.assertEqual(cfg.n_sliding_layers(), 1)

    def test_real_26b_layer_pattern(self):
        kinds = [LayerKind.FULL if (i + 1) % 6 == 0 else LayerKind.SLIDING for i in range(30)]
        cfg = Gemma4TextConfig(num_hidden_layers=30, layer_types=tuple(kinds))
        self.assertEqual(cfg.n_full_layers(), 5)
        self.assertTrue(cfg.is_full(5) and cfg.is_full(29))
        self.assertFalse(cfg.is_full(0))

    def test_two_ropes(self):
        cfg = tiny_text_config()
        self.assertEqual(cfg.rope_for(0).rope_theta, 10000.0)
        self.assertEqual(cfg.rope_for(1).rope_theta, 1_000_000.0)
        self.assertEqual(cfg.rope_for(1).rope_type, "proportional")

    def test_proportional_rope_is_partial_with_nope(self):
        # head_dim 512, partial 0.25 -> 64 real frequencies, 192 zeros (NoPE), 256 total
        rope = RopeParams("proportional", 1_000_000.0, 0.25)
        inv = rope.inv_freq(512)
        self.assertEqual(len(inv), 256)
        self.assertEqual(sum(1 for f in inv if f == 0.0), 192)
        self.assertAlmostEqual(inv[0], 1.0)
        self.assertAlmostEqual(inv[1], 1_000_000 ** (-2 / 512))

    def test_default_rope_rotates_the_whole_head(self):
        inv = RopeParams("default", 10000.0).inv_freq(256)
        self.assertEqual(len(inv), 128)
        self.assertTrue(all(f > 0 for f in inv))

    def test_rope_freq_factors_map_the_nope_channels_to_zero(self):
        rope = RopeParams("proportional", 1_000_000.0, 0.25)
        base = rope.inv_freq(512)
        factors = [1.0] * 64 + [1e30] * 192
        got = apply_rope_factors(base, factors)
        self.assertEqual(sum(1 for f in got if f == 0.0), 192)
        self.assertIsNone(check_rope_freqs(factors, base))
        self.assertIsNotNone(check_rope_freqs([2.0] + [1.0] * 255, base))
        with self.assertRaises(ConfigError):
            apply_rope_factors(base, [1.0] * 10)

    def test_per_layer_head_width(self):
        cfg = Gemma4TextConfig(head_dim=256, global_head_dim=512, num_attention_heads=16,
                               num_key_value_heads=8, num_global_key_value_heads=2,
                               kv_heads_per_layer=(8, 2), layer_types=(LayerKind.SLIDING,
                                                                       LayerKind.FULL),
                               num_hidden_layers=2)
        self.assertEqual(cfg.q_proj_out(0), 16 * 256)
        self.assertEqual(cfg.q_proj_out(1), 16 * 512)
        self.assertEqual(cfg.kv_proj_out(0), 8 * 256)
        self.assertEqual(cfg.kv_proj_out(1), 2 * 512)
        self.assertTrue(cfg.has_v_proj(0))
        self.assertFalse(cfg.has_v_proj(1))          # K==V on full layers

    def test_kv_heads_array_wins_over_the_scalars(self):
        cfg = Gemma4TextConfig(num_hidden_layers=3, kv_heads_per_layer=(8, 8, 2),
                               layer_types=(LayerKind.SLIDING, LayerKind.SLIDING, LayerKind.FULL))
        self.assertEqual([cfg.kv_heads_for(i) for i in range(3)], [8, 8, 2])

    def test_from_dict_parses_the_real_shape(self):
        d = {"model_type": "gemma4", "text_config": {
            "hidden_size": 2816, "num_hidden_layers": 30, "vocab_size": 262144,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
            "global_head_dim": 512, "num_global_key_value_heads": 2, "num_experts": 128,
            "top_k_experts": 8, "moe_intermediate_size": 704, "intermediate_size": 2112,
            "sliding_window": 1024, "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                                   "partial_rotary_factor": 0.25}},
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}}
        cfg = Gemma4Config.from_dict(d)
        self.assertEqual(cfg.text.num_experts, 128)
        self.assertEqual(cfg.text.final_logit_softcapping, 30.0)
        self.assertEqual(cfg.text.rope_full.partial_rotary_factor, 0.25)
        self.assertEqual(cfg.layer_kind(5), LayerKind.FULL)
        self.assertAlmostEqual(cfg.text.effective_embed_scale(), np.sqrt(2816))

    def test_unsupported_rope_is_an_error(self):
        with self.assertRaises(ConfigError):
            RopeParams.from_dict({"rope_type": "something_new"})

    def test_attention_scale_is_one_not_inverse_sqrt(self):
        # the reference sets self.scaling = 1.0 because Q and K are RMS-normalised
        self.assertEqual(Gemma4TextConfig().attention_scale, 1.0)


class TestDenseGemma4(unittest.TestCase):
    """A dense Gemma 4 (gemma-4-12B: `enable_moe_block: false`, no router, no expert stack) must
    load, validate and generate. The MoE-only path used to skip the layer's MLP entirely."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = tiny_text_config(enable_moe_block=False)
        self.path = pathlib.Path(self.tmp.name) / "dense.gguf"

    def test_a_dense_config_json_loads(self):
        # the real 12B config.json writes eos_token_id as the LIST [1, 106]
        import json
        d = {"model_type": "gemma4_unified",
             "eos_token_id": [1, 106], "pad_token_id": 0, "boi_token_id": 255999,
             "text_config": {"hidden_size": 3840, "num_hidden_layers": 48,
                             "enable_moe_block": False, "num_experts": None,
                             "top_k_experts": None, "moe_intermediate_size": None,
                             "global_head_dim": 512, "head_dim": 256,
                             "num_key_value_heads": 8, "num_global_key_value_heads": 1}}
        p = pathlib.Path(self.tmp.name) / "config.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        cfg = Gemma4Config.from_json(p)
        self.assertEqual(cfg.eos_token_ids, (1, 106))
        self.assertEqual(cfg.eos_token_id, 1)
        self.assertFalse(cfg.text.enable_moe_block)
        self.assertEqual(cfg.text.n_full_layers(), 8)
        self.assertEqual(cfg.text.kv_heads_for(5), 1)
        self.assertFalse(cfg.text.has_v_proj(5))     # full layers fold V into K
        self.assertTrue(cfg.text.has_v_proj(0))

    def test_a_dense_gguf_validates(self):
        write_tiny_gguf(self.path, self.cfg, seed=31)
        with G.Gemma4GGUF.open(self.path) as g:
            self.assertEqual(G.validate(g, Gemma4Config(text=self.cfg)), [])
            cfg = g.config()
        self.assertFalse(cfg.text.enable_moe_block)

    def test_a_dense_model_generates(self):
        write_tiny_gguf(self.path, self.cfg, seed=32)
        eng = Gemma4Engine(path=self.path, backend_name="numpy", max_context=64)
        out = list(eng.generate([2, 5, 11], max_new=6, sampling={"temperature": 0.0},
                                cancel=threading.Event()))
        self.assertEqual(len(out), 6)
        self.assertEqual(eng.last["finish"], "length")
        self.assertEqual(eng.last["generated"], 6)
        self.assertEqual(eng.eos_ids, {1, 106})

    def test_dense_estimate_ignores_the_expert_stack(self):
        from serve.gemma4.backends import estimate_bytes
        moe = estimate_bytes(Gemma4Config(text=tiny_text_config()))
        dense = estimate_bytes(Gemma4Config(text=self.cfg))
        self.assertLess(dense, moe)


class TestMtpSpec(unittest.TestCase):
    def test_a_drafter_is_recognised(self):
        md = {"general.architecture": "gemma4-assistant",
              "gemma4-assistant.block_count": 4,
              "gemma4-assistant.embedding_length": 1024,
              "gemma4-assistant.nextn_predict_layers": 4,
              "gemma4-assistant.attention.shared_kv_layers": 4,
              "gemma4-assistant.embedding_length_out": 2816,
              "gemma4-assistant.feed_forward_length": 8192}
        spec = MtpSpec.from_metadata(md)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.block_count, 4)
        self.assertEqual(spec.embedding_length_out, 2816)
        self.assertIsNone(MtpSpec.from_metadata({"general.architecture": "gemma4"}))


# ------------------------------------------------------------------ numerics
class TestNumerics(unittest.TestCase):
    def test_rms_norm_applies_the_weight_directly(self):
        x = np.array([[3.0, 4.0]], np.float32)
        w = np.ones(2, np.float32)
        out = rms_norm(x, w, 0.0, NumpyOps())
        self.assertTrue(np.allclose(out, x / np.sqrt(12.5), atol=1e-5))
        # a weight of 2 doubles it; the Gemma 1/2/3 (1+w) form would triple it
        out2 = rms_norm(x, np.full(2, 2.0, np.float32), 0.0, NumpyOps())
        self.assertTrue(np.allclose(out2, 2 * x / np.sqrt(12.5), atol=1e-5))

    def test_rms_norm_without_scale(self):
        x = np.array([[3.0, 4.0]], np.float32)
        out = rms_norm(x, None, 0.0, NumpyOps())
        self.assertTrue(np.allclose(out, x / np.sqrt(12.5), atol=1e-5))

    def test_gelu_tanh_matches_known_values(self):
        xp = NumpyOps()
        self.assertAlmostEqual(float(gelu_pytorch_tanh(np.array([0.0]), xp)[0]), 0.0, places=6)
        self.assertAlmostEqual(float(gelu_pytorch_tanh(np.array([2.0]), xp)[0]), 1.9545975, places=4)

    def test_rope_tables_shape_and_nope(self):
        xp = NumpyOps()
        pos = np.arange(4)
        inv = RopeParams("proportional", 1e6, 0.25).inv_freq(8)     # 1 real, 3 zero
        self.assertEqual(sum(1 for f in inv if f == 0.0), 3)
        cos, sin = rope_tables(pos, inv, xp)
        self.assertEqual(cos.shape, (4, 8))
        # emb = cat(freqs, freqs), so the NoPE entries appear twice: cols 1..3 and 5..7
        for lo, hi in ((1, 4), (5, 8)):
            self.assertTrue(np.allclose(cos[:, lo:hi], 1.0))
            self.assertTrue(np.allclose(sin[:, lo:hi], 0.0))
        self.assertFalse(np.allclose(cos[:, 0], 1.0))     # the rotated channel does move

    def test_apply_rotary_leaves_unrotated_channels(self):
        xp = NumpyOps()
        x = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
        cos = np.ones((1, 1, 8), np.float32)
        sin = np.zeros((1, 1, 8), np.float32)
        out = apply_rotary(x, cos, sin, xp)
        self.assertTrue(np.array_equal(out, x))


# ------------------------------------------------------------------ blocks
class TestBlocks(unittest.TestCase):
    def test_attention_output_shape_and_causality(self):
        cfg = tiny_text_config()
        rng = np.random.default_rng(1)
        xp = NumpyOps()
        w = {
            "q": rng.standard_normal((cfg.hidden_size, cfg.q_proj_out(0))).astype(np.float32),
            "q_norm": np.ones(cfg.head_dim, np.float32),
            "k": rng.standard_normal((cfg.hidden_size, cfg.kv_proj_out(0))).astype(np.float32),
            "k_norm": np.ones(cfg.head_dim, np.float32),
            "v": rng.standard_normal((cfg.hidden_size, cfg.kv_proj_out(0))).astype(np.float32),
            "o": rng.standard_normal((cfg.q_proj_out(0), cfg.hidden_size)).astype(np.float32),
        }
        att = Gemma4Attention(cfg.layer_spec(0), w, eps=1e-6, xp=xp)
        x = rng.standard_normal((3, cfg.hidden_size)).astype(np.float32)
        cache = KVCache(window=cfg.sliding_window, xp=xp)
        out = att.forward(x, np.arange(3), cache, window=cfg.sliding_window)
        self.assertEqual(out.shape, (3, cfg.hidden_size))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(len(cache), 3)

    def test_sliding_cache_trims_to_the_window_and_keeps_positions(self):
        cfg = tiny_text_config(sliding_window=4)
        xp = NumpyOps()
        cache = KVCache(window=4, xp=xp)
        for i in range(10):
            cache.append(np.zeros((1, 1, 8), np.float32), np.zeros((1, 1, 8), np.float32),
                         np.array([i]))
        self.assertEqual(len(cache), 4)
        # the surviving rows are the LAST four positions, and the cache knows which they are
        self.assertEqual(cache.pos.tolist(), [6, 7, 8, 9])

    def test_sliding_mask_uses_positions_not_row_indices(self):
        """A prompt longer than the window trims rows, so row 0 of the cache is not token 0.  An
        index-based mask would then forbid the wrong keys; a position-based one does not."""
        from serve.gemma4.model import _causal_window_mask
        xp = NumpyOps()
        scores = np.zeros((1, 3, 4), np.float32)          # 3 queries, 4 cached keys
        q_pos = np.array([10, 11, 12])
        k_pos = np.array([9, 10, 11, 12])                 # a trimmed cache: starts at 9
        masked = _causal_window_mask(scores, q_pos, k_pos, window=4, xp=xp)
        # query at 10 may see 9,10 (within 4, not future) but not 11,12
        self.assertTrue(np.isfinite(masked[0, 0, 0]))
        self.assertTrue(np.isfinite(masked[0, 0, 1]))
        self.assertTrue(np.isinf(masked[0, 0, 2]) and masked[0, 0, 2] < 0)
        # query at 12 may see all four (12-9 = 3 < 4)
        self.assertTrue(np.all(np.isfinite(masked[0, 2, :])))

    def test_moe_routes_topk_renormalises_and_scales(self):
        cfg = tiny_text_config()
        rng = np.random.default_rng(2)
        xp = NumpyOps()
        w = {
            "router_w": rng.standard_normal((cfg.hidden_size, cfg.num_experts)).astype(np.float32),
            "router_scale": (rng.standard_normal((cfg.hidden_size,)) * 4 + 32).astype(np.float32),
            "expert_scale": (rng.standard_normal((cfg.num_experts,)) * 0.05 + 1.0).astype(np.float32),
            "gate": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "up": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "down": rng.standard_normal((cfg.intermediate_size, cfg.hidden_size)).astype(np.float32),
            "ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_1": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "pre_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "gate_up_exps": rng.standard_normal(
                (cfg.hidden_size, 2 * cfg.moe_intermediate_size, cfg.num_experts)).astype(np.float32),
            "down_exps": rng.standard_normal(
                (cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts)).astype(np.float32),
        }
        router = Gemma4Router(w, cfg, xp=xp)
        moe = Gemma4MoE(cfg, w, router, StackedExperts(w, cfg.moe_intermediate_size, xp=xp), xp=xp)
        x = rng.standard_normal((5, cfg.hidden_size)).astype(np.float32)
        ids, weights = router.route(x)
        self.assertEqual(ids.shape, (5, cfg.top_k_experts))
        # after top-k renormalisation the weights sum to 1; the per-expert scale then breaks that
        # sum in exactly the way the reference does, so divide it back out and check
        unscaled = np.asarray(weights) / w["expert_scale"][np.asarray(ids)]
        self.assertTrue(np.allclose(unscaled.sum(axis=-1), 1.0, atol=1e-5))
        for row in ids:
            self.assertEqual(len(set(row.tolist())), cfg.top_k_experts)   # no expert twice
        out = moe.forward(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    # --------------------------------------------------- routed experts: batched GEMM vs the loop
    def _moe_weights(self, cfg, seed=5, fused=True):
        rng = np.random.default_rng(seed)
        w = {
            "router_w": rng.standard_normal((cfg.hidden_size, cfg.num_experts)).astype(np.float32),
            "router_scale": (rng.standard_normal((cfg.hidden_size,)) * 4 + 32).astype(np.float32),
            "expert_scale": (rng.standard_normal((cfg.num_experts,)) * 0.05 + 1.0).astype(np.float32),
            "gate": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "up": rng.standard_normal((cfg.hidden_size, cfg.intermediate_size)).astype(np.float32),
            "down": rng.standard_normal((cfg.intermediate_size, cfg.hidden_size)).astype(np.float32),
            "ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_1": np.ones(cfg.hidden_size, np.float32),
            "post_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "pre_ffn_norm_2": np.ones(cfg.hidden_size, np.float32),
            "down_exps": rng.standard_normal(
                (cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts)).astype(np.float32),
        }
        if fused:
            w["gate_up_exps"] = rng.standard_normal(
                (cfg.hidden_size, 2 * cfg.moe_intermediate_size, cfg.num_experts)).astype(np.float32)
        else:
            # a file with separate gate/up tensors: the batch has to be concatenated from two gathers
            w["gate_exps"] = rng.standard_normal(
                (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts)).astype(np.float32)
            w["up_exps"] = rng.standard_normal(
                (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts)).astype(np.float32)
        return w

    @staticmethod
    def _routing(cfg, per_expert: list[int], seed=11):
        """A routing with exactly `per_expert[e]` tokens on expert e, as (idx, vals) of shape
        (n_tokens, top_k), ascending by expert id within a token like `Gemma4Router.route` returns.

        Written out rather than taken from the router so the test knows which groups the MoE should
        form: experts with the same token count can share one batched matmul, and a routing that
        happened to be lumpy would make the assertions about batching depend on luck.  Slots are
        handed to the tokens with the most room left, which is what keeps a token from ever getting
        the same expert twice."""
        k = cfg.top_k_experts
        total = sum(per_expert)
        assert total % k == 0, "the counts must fill whole tokens' top-k slots"
        n_tokens = total // k
        assert max(per_expert) <= n_tokens, "an expert cannot be picked more times than there are tokens"
        assert len(per_expert) <= cfg.num_experts
        cap = [k] * n_tokens
        picks: list[list[int]] = [[] for _ in range(n_tokens)]
        for e, cnt in enumerate(per_expert):
            order = sorted(range(n_tokens), key=lambda t: (-cap[t], t))[:cnt]
            assert cap[order[-1]] > 0, "no room left for that expert"
            for t in order:
                cap[t] -= 1
                picks[t].append(e)
        assert all(c == 0 for c in cap)
        idx = np.array([sorted(p) for p in picks], np.int64)
        assert idx.shape == (n_tokens, k) and all(len(set(r)) == k for r in idx.tolist())
        vals = np.random.default_rng(seed).uniform(0.2, 1.0, (n_tokens, k)).astype(np.float32)
        return idx, vals

    def _routed_both_ways(self, cfg, xp, *, per_expert, seed=5, fused=True):
        """The routed branch of one MoE twice on the same routing: once expert-by-expert (with
        `batched_min_rows` so high that no group is ever batched) and once with every group batched.
        Returns (loop_out, batched_out, calls_loop, calls_batched)."""
        w = {k: (xp.asarray(v, xp.float32) if isinstance(v, np.ndarray) else v)
             for k, v in self._moe_weights(cfg, seed=seed, fused=fused).items()}
        router = Gemma4Router(w, cfg, xp=xp)
        idx_np, vals_np = self._routing(cfg, per_expert)
        counts = {"one": 0, "many": 0}

        class Counting(StackedExperts):
            def gate_up(self, e):
                counts["one"] += 1
                return super().gate_up(e)

            def gate_up_range(self, lo, hi):
                counts["many"] += 1          # one batched call, however many experts it covers
                return super().gate_up_range(lo, hi)

        runs = []
        for min_rows in (10 ** 9, 1):
            counts["one"] = counts["many"] = 0
            moe = Gemma4MoE(cfg, w, router, Counting(w, cfg.moe_intermediate_size, xp=xp), xp=xp)
            moe.batched_min_rows = min_rows
            x = xp.asarray(np.random.default_rng(11).standard_normal(
                (idx_np.shape[0], cfg.hidden_size)).astype(np.float32), xp.float32)
            idx, vals = xp.asarray(idx_np, xp.int64), xp.asarray(vals_np, xp.float32)
            out = np.asarray(moe._routed(x, idx, vals), np.float32).copy()
            runs.append((out, counts["one"], counts["many"]))
        return runs

    def test_routed_batched_matches_the_per_expert_loop(self):
        """The batched matmul must reproduce the per-expert loop, on a routing where several experts
        share a token count (equal counts are what let them share one batched matmul)."""
        cfg = tiny_text_config()                      # 4 experts, top_k 2
        xp = NumpyOps()
        # three experts with 12 tokens each (one batch) and one with 6 (alone in its size class);
        # all-equal counts; a decode step (one token each); and a lumpy routing with no ties at all
        for per_expert in ([12, 12, 12, 6], [8, 8, 8, 8], [1, 1, 1, 1], [5, 3, 2, 2]):
            (slow, one_a, many_a), (fast, one_b, many_b) = self._routed_both_ways(cfg, xp,
                                                                                 per_expert=per_expert)
            self.assertEqual(slow.shape, fast.shape)
            self.assertTrue(np.all(np.isfinite(fast)), f"{per_expert}")
            scale = max(1e-6, float(np.abs(slow).max()))
            diff = float(np.abs(slow - fast).max())
            self.assertLess(diff, 2e-5 * scale, f"{per_expert}: max|diff| {diff} vs scale {scale}")
            self.assertGreater(one_a, 0)               # the loop really did go expert by expert
            self.assertEqual(many_a, 0)                # ...and never batched
        # at a width where a group forms, the batch path must actually be taken and must cut calls
        (_slow, one_a, _ma), (_fast, one_b, many_b) = self._routed_both_ways(
            cfg, xp, per_expert=[12, 12, 12, 6])
        self.assertEqual(many_b, 1, "the three 12-token experts should share exactly one batch")
        self.assertEqual(one_b, 1, "the 6-token expert is alone in its size class and stays in the loop")

    def test_routed_batched_with_separate_gate_and_up_tensors(self):
        """A GGUF that ships gate_exps/up_exps apart (not the fused gate_up_exps) must batch too:
        there the batch is two views and a concat, not one view."""
        cfg = tiny_text_config()
        (slow, _, _), (fast, _one_b, many_b) = self._routed_both_ways(
            cfg, NumpyOps(), per_expert=[10, 10, 10, 10], seed=9, fused=False)
        self.assertEqual(many_b, 1, "no expert group was batched")
        scale = max(1e-6, float(np.abs(slow).max()))
        self.assertLess(float(np.abs(slow - fast).max()), 2e-5 * scale)

    def test_batching_never_copies_the_weight_stack(self):
        """The batch must read the experts as a VIEW.  Copying them is the whole reason a batched
        MoE is slower than the loop: one layer's stack is gigabytes, and the copy measured 1311 ms
        against the 100 ms of matmul it would feed.  A source that cannot hand out a view says so
        with `batched_views = False` and is never batched."""
        cfg = tiny_text_config()
        xp = NumpyOps()
        w = {k: xp.asarray(v, xp.float32) if isinstance(v, np.ndarray) else v
             for k, v in self._moe_weights(cfg, seed=4).items()}
        router = Gemma4Router(w, cfg, xp=xp)
        idx_np, vals_np = self._routing(cfg, [12, 12, 12, 12])
        x = xp.asarray(np.random.default_rng(6).standard_normal(
            (idx_np.shape[0], cfg.hidden_size)).astype(np.float32), xp.float32)
        idx, vals = xp.asarray(idx_np, xp.int64), xp.asarray(vals_np, xp.float32)

        class NoViews(StackedExperts):
            batched_views = False

        for cls, expect_range in ((StackedExperts, 1), (NoViews, 0)):
            seen = {"range": 0, "one": 0}

            class Counting(cls):
                def gate_up_range(self, lo, hi):
                    seen["range"] += 1
                    return super().gate_up_range(lo, hi)

                def gate_up(self, e):
                    seen["one"] += 1
                    return super().gate_up(e)

            moe = Gemma4MoE(cfg, w, router, Counting(w, cfg.moe_intermediate_size, xp=xp), xp=xp)
            moe.batched_min_rows = 1
            out = np.asarray(moe._routed(x, idx, vals), np.float32)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertEqual(seen["range"], expect_range, cls.__name__)
            # a view, never a materialised copy of the expert axis
            if expect_range:
                view = StackedExperts(w, cfg.moe_intermediate_size, xp=xp).gate_up_range(0, 4)
                self.assertEqual(view.shape, (4, cfg.hidden_size, 2 * cfg.moe_intermediate_size))
                self.assertEqual(seen["one"], 0, "a batched group also walked the experts one by one")

    @unittest.skipUnless("torch" in available_ops(), "torch is not installed")
    def test_routed_batched_matches_on_torch(self):
        """The batched expert matmul must agree with the per-expert loop on torch as well, and agree
        with numpy: three products inside one kernel is a different summation order, so the parity is
        asserted with a tolerance rather than equality.  torch is also the backend where the batch
        actually pays, so the strided view must not be made contiguous on the way in."""
        import torch
        from serve.gemma4.ops import TorchOps
        cfg = tiny_text_config()
        per_expert = [12, 12, 12, 6]
        np_fast = None
        for xp in (NumpyOps(), TorchOps(device="cpu", dtype=torch.float32)):
            (slow, _one_a, _many_a), (fast, _one_b, many_b) = self._routed_both_ways(
                cfg, xp, per_expert=per_expert)
            scale = max(1e-6, float(np.abs(slow).max()))
            self.assertLess(float(np.abs(slow - fast).max()), 2e-5 * scale, xp.name)
            self.assertEqual(many_b, 1, f"{xp.name}: the three 12-token experts share one batch")
            if xp.name == "numpy":
                np_fast = fast
            else:
                # float32: a batched matmul and a per-expert one sum the same products in a different
                # order, so the two backends are allowed to part company by a few parts in 1e5
                self.assertLess(float(np.abs(np_fast - fast).max()), 1e-4 * max(1e-6, float(np.abs(np_fast).max())),
                                "numpy and torch disagree on the routed branch")

    @unittest.skipUnless("torch" in available_ops(), "torch is not installed")
    def test_expert_range_is_a_view_on_torch(self):
        from serve.gemma4.ops import TorchOps
        import torch
        xp = TorchOps(device="cpu", dtype=torch.float32)
        stacked = xp.zeros((4, 6, 3), xp.float32)
        view = xp.expert_range(stacked, 0, 3)
        self.assertEqual(tuple(view.shape), (3, 4, 6))
        self.assertEqual(view.data_ptr(), stacked.data_ptr(), "expert_range copied instead of viewing")
        self.assertTrue(torch.equal(view[2, 1, 3], stacked[1, 3, 2]))

    def test_streamed_source_stays_on_the_per_expert_loop(self):
        """A source that reads one expert from the file at a time cannot hand out a view of a
        stacked tensor, so it must keep the per-expert path: batching it would mean reading and
        materialising experts the token batch may not even need."""
        from serve.gemma4.model import ExpertSource
        cfg = tiny_text_config()
        xp = NumpyOps()
        w = self._moe_weights(cfg, seed=7)
        stacked = StackedExperts(w, cfg.moe_intermediate_size, xp=xp)

        class OneAtATime(ExpertSource):
            def __init__(self):
                self.xp = xp
                self.seen = []

            def gate_up(self, e):
                self.seen.append(e)
                return stacked.gate_up(e)

            def down(self, e):
                return stacked.down(e)

        self.assertFalse(OneAtATime().batched_views, "a streamed source must not claim view batching")
        src = OneAtATime()
        router = Gemma4Router(w, cfg, xp=xp)
        moe = Gemma4MoE(cfg, w, router, src, xp=xp)
        moe.batched_min_rows = 1
        idx_np, vals_np = self._routing(cfg, [16, 16, 16, 16])
        x = xp.asarray(np.random.default_rng(3).standard_normal((32, cfg.hidden_size)).astype(np.float32),
                       xp.float32)
        out = np.asarray(moe._routed(x, xp.asarray(idx_np, xp.int64), xp.asarray(vals_np, xp.float32)),
                         np.float32)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(sorted(src.seen), [0, 1, 2, 3], "each expert should be read exactly once")
        ref = Gemma4MoE(cfg, w, router, StackedExperts(w, cfg.moe_intermediate_size, xp=xp), xp=xp)
        ref.batched_min_rows = 1
        other = np.asarray(ref._routed(x, xp.asarray(idx_np, xp.int64),
                                       xp.asarray(vals_np, xp.float32)), np.float32)
        self.assertLess(float(np.abs(out - other).max()), 2e-5 * max(1e-6, float(np.abs(other).max())))

    def test_layer_scale_multiplies_the_output(self):
        cfg = tiny_text_config()
        xp = NumpyOps()
        model = tiny_model(cfg, seed=3)
        spec = cfg.layer_spec(0)
        w = model.model.layers[0].w
        base = float(np.abs(w["layer_scale"]))
        self.assertGreater(base, 0.0)
        self.assertIsInstance(base, float)


# ------------------------------------------------------------------ gguf load + forward
class TestGgufAndForward(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_text_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "tiny.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def test_validate_passes_on_a_good_file(self):
        write_tiny_gguf(self.path, self.cfg)
        with G.Gemma4GGUF.open(self.path) as g:
            self.assertEqual(G.validate(g, Gemma4Config(text=self.cfg)), [])

    def test_validate_reports_a_missing_tensor(self):
        write_tiny_gguf(self.path, self.cfg, drop={"blk.1.ffn_gate_up_exps.weight"})
        with G.Gemma4GGUF.open(self.path) as g:
            problems = G.validate(g, Gemma4Config(text=self.cfg))
        self.assertTrue(any("gate_up_exps" in p for p in problems))

    def test_validate_reports_a_wrong_shape(self):
        write_tiny_gguf(self.path, self.cfg, break_shape="blk.0.attn_q.weight")
        with G.Gemma4GGUF.open(self.path) as g:
            problems = G.validate(g, Gemma4Config(text=self.cfg))
        self.assertTrue(any("attn_q.weight: shape" in p for p in problems))

    def test_config_comes_from_the_gguf_metadata(self):
        write_tiny_gguf(self.path, self.cfg)
        with G.Gemma4GGUF.open(self.path) as g:
            cfg = g.config()
        self.assertEqual(cfg.text.num_hidden_layers, self.cfg.num_hidden_layers)
        self.assertEqual(cfg.text.num_experts, self.cfg.num_experts)
        self.assertEqual(cfg.text.top_k_experts, self.cfg.top_k_experts)
        self.assertEqual(cfg.text.global_head_dim, self.cfg.global_head_dim)
        self.assertEqual(cfg.text.head_dim, self.cfg.head_dim)
        self.assertEqual(cfg.text.rope_sliding.rope_theta, self.cfg.rope_sliding.rope_theta)
        self.assertEqual(cfg.text.rope_full.rope_theta, self.cfg.rope_full.rope_theta)
        self.assertEqual(cfg.layer_kind(1), LayerKind.FULL)
        self.assertEqual(cfg.text.kv_heads_per_layer,
                         tuple(self.cfg.kv_heads_for(i) for i in range(2)))

    def test_read_tensor_roundtrips_f32(self):
        write_tiny_gguf(self.path, self.cfg, seed=3)
        with G.Gemma4GGUF.open(self.path) as g:
            t = g.read_tensor("token_embd.weight")
            self.assertEqual(t.shape, (self.cfg.hidden_size, self.cfg.vocab_size))
            self.assertTrue(np.all(np.isfinite(t)))

    def test_quantised_file_loads_and_validates(self):
        write_tiny_gguf(self.path, self.cfg, seed=5, quant={
            "token_embd.weight": "Q8_0",
            "blk.0.ffn_gate_up_exps.weight": "Q4_K",
            "blk.1.ffn_gate_up_exps.weight": "Q4_K",
            "blk.0.ffn_down_exps.weight": "Q5_1",
            "blk.1.ffn_down_exps.weight": "Q5_1",
            "blk.0.attn_q.weight": "BF16",
        })
        with G.Gemma4GGUF.open(self.path) as g:
            cfg = g.config()
            self.assertEqual(G.validate(g, cfg), [])
            self.assertEqual(g.types()["Q4_K"], 2)
            gu = g.read_tensor("blk.0.ffn_gate_up_exps.weight")
            self.assertEqual(gu.shape, (self.cfg.hidden_size, 2 * self.cfg.moe_intermediate_size,
                                        self.cfg.num_experts))
            self.assertTrue(np.all(np.isfinite(gu)))

    def test_read_expert_equals_a_full_read_slice(self):
        # F32 has a block size of 1, so expert slicing works at any width
        write_tiny_gguf(self.path, self.cfg, seed=6)
        with G.Gemma4GGUF.open(self.path) as g:
            for name in ("blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"):
                full = g.read_tensor(name)
                for e in range(self.cfg.num_experts):
                    np.testing.assert_array_equal(g.read_expert(name, e), full[:, :, e])

    def test_read_expert_on_a_quantised_block_aligned_model(self):
        """Expert slicing needs the block size to divide dim 0.  The real model satisfies this
        (gate_up_exps is Q4_K with d0=2816=11*256, down_exps is Q5_1 with d0=704=22*32), so the
        fixture uses the same divisibility at a smaller scale."""
        cfg = tiny_text_config(hidden_size=256, moe_intermediate_size=32, intermediate_size=64,
                               num_experts=4, vocab_size=64)
        p = pathlib.Path(self.tmp.name) / "aligned.gguf"
        write_tiny_gguf(p, cfg, seed=8, quant={
            "blk.0.ffn_gate_up_exps.weight": "Q4_K",
            "blk.0.ffn_down_exps.weight": "Q5_1",
            "blk.1.ffn_gate_up_exps.weight": "Q4_K",
            "blk.1.ffn_down_exps.weight": "Q5_1",
        })
        with G.Gemma4GGUF.open(p) as g:
            self.assertEqual(G.validate(g, Gemma4Config(text=cfg)), [])
            for name in ("blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight",
                         "blk.1.ffn_gate_up_exps.weight", "blk.1.ffn_down_exps.weight"):
                full = g.read_tensor(name)
                for e in range(cfg.num_experts):
                    np.testing.assert_array_equal(g.read_expert(name, e), full[:, :, e])

    def test_read_expert_refuses_an_unslicible_layout(self):
        write_tiny_gguf(self.path, self.cfg, seed=9, quant={
            "blk.0.ffn_gate_up_exps.weight": "Q4_K"})
        with G.Gemma4GGUF.open(self.path) as g:
            with self.assertRaises(G.UnsupportedQuant) as cm:
                g.read_expert("blk.0.ffn_gate_up_exps.weight", 0)
        self.assertIn("do not divide", str(cm.exception))

    def test_split_shards_load_like_one_file(self):
        """Shard 1 holds the metadata and the directory; the bytes live in whichever shard has them."""
        cfg = self.cfg
        rng = np.random.default_rng(7)
        md = gguf_metadata(cfg)
        tensors = [Q.make_tensor("token_embd.weight",
                                 (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) * 0.05
                                  ).astype(np.float32), "F32"),
                   Q.make_tensor("output_norm.weight", np.ones(cfg.hidden_size, np.float32), "F32"),
                   Q.make_tensor("rope_freqs.weight",
                                 np.array(rope_freq_values(cfg), np.float32), "F32")]
        for layer in range(cfg.num_hidden_layers):
            for name, shape in _tensor_shapes(cfg, layer).items():
                tensors.append(Q.make_tensor(f"blk.{layer}.{name}",
                                             (rng.standard_normal(shape) * 0.05).astype(np.float32),
                                             "F32"))
        half = len(tensors) // 2
        base = pathlib.Path(self.tmp.name) / "split"
        Q.write_gguf(base.parent / "split-00001-of-00002.gguf",
                     dict(md, **{"split.no": 0, "split.count": 2,
                                 "split.tensors.count": len(tensors)}), tensors[:half])
        Q.write_gguf(base.parent / "split-00002-of-00002.gguf",
                     {"general.architecture": "gemma4", "split.no": 1, "split.count": 2},
                     tensors[half:])
        with G.Gemma4GGUF.open(base.parent / "split-00001-of-00002.gguf") as g:
            self.assertTrue(g.is_split)
            self.assertEqual(len(list(g.names())), len(tensors))
            self.assertEqual(G.validate(g, Gemma4Config(text=cfg)), [])
            a = g.read_tensor(f"blk.{cfg.num_hidden_layers - 1}.attn_q.weight")
            self.assertEqual(a.shape, (cfg.hidden_size, cfg.q_proj_out(cfg.num_hidden_layers - 1)))
            self.assertTrue(np.all(np.isfinite(a)))

    def test_full_forward_produces_finite_softcapped_logits(self):
        write_tiny_gguf(self.path, self.cfg, seed=4)
        backend = NumpyBackend(path=self.path, config=Gemma4Config(text=self.cfg)).load()
        logits = backend.forward([1, 2, 3], start_pos=0)
        self.assertEqual(logits.shape, (3, self.cfg.vocab_size))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertLessEqual(float(np.abs(logits).max()),
                             self.cfg.final_logit_softcapping + 1e-3)

    def test_incremental_matches_full(self):
        """Token-by-token decode must match one full forward: the KV cache is correct."""
        model = tiny_model(self.cfg, seed=5)
        ids = [3, 9, 17, 2]
        full = np.asarray(model.forward(ids, start_pos=0))[-1]
        model.reset_cache()
        last = None
        for i, tok in enumerate(ids):
            last = np.asarray(model.forward([tok], start_pos=i))[-1]
        np.testing.assert_allclose(full, last, atol=1e-3)

    def test_incremental_matches_full_past_the_sliding_window(self):
        """The sequence is three times the sliding window, so the cache gets trimmed mid-run.
        This is the case an index-based mask gets wrong: after a trim, row 0 of the cache is not
        token 0 of the sequence."""
        cfg = tiny_text_config(sliding_window=4)
        model = tiny_model(cfg, seed=31)
        ids = list(range(1, 13))                       # 12 tokens, window 4
        full = np.asarray(model.forward(ids, start_pos=0))[-1]
        model.reset_cache()
        last = None
        for i, tok in enumerate(ids):
            last = np.asarray(model.forward([tok], start_pos=i))[-1]
        np.testing.assert_allclose(full, last, atol=1e-3)
        # the sliding layer really did drop rows; the full layer kept everything
        self.assertEqual(len(model.model.cache[0]), cfg.sliding_window)
        self.assertEqual(len(model.model.cache[1]), len(ids))

    def test_backend_supports_reports_a_drafter(self):
        p = pathlib.Path(self.tmp.name) / "mtp.gguf"
        Q.write_gguf(p, {"general.architecture": "gemma4-assistant",
                         "gemma4-assistant.block_count": 4,
                         "gemma4-assistant.embedding_length": 1024,
                         "gemma4-assistant.nextn_predict_layers": 4}, [])
        ok, why = backend_supports(p)
        self.assertFalse(ok)
        self.assertIn("MTP drafter", why)


# ------------------------------------------------------------------ tokenizer
class TestTokenizer(unittest.TestCase):
    def _tok(self):
        tokens = ["<unk>", "<bos>", "<eos>", "a", "b", "ab", "c"]
        scores = [0.0, 0.0, 0.0, -1.0, -1.0, -0.5, -1.0]
        types = [3, 3, 3, 1, 1, 1, 1]        # 3 = CONTROL, 1 = NORMAL
        return GgufTokenizer(tokens, scores, types, bos_id=1, eos_id=2, add_space_prefix=False)

    def test_viterbi_picks_the_longer_piece(self):
        self.assertEqual(self._tok().encode("ab"), [5])

    def test_parse_special_matches_control_tokens(self):
        self.assertEqual(self._tok().encode("<bos>ab", parse_special=True), [1, 5])

    def test_decode_roundtrip_and_token_bytes(self):
        tok = self._tok()
        self.assertEqual(tok.token_bytes(5), b"ab")
        self.assertEqual(tok.decode([5, 6]), "abc")

    def test_word_marker_becomes_space(self):
        tok = GgufTokenizer(["<unk>", "\u2581hello"], [0.0, -1.0], [3, 1], add_space_prefix=False)
        self.assertEqual(tok.token_bytes(1), b" hello")


# ------------------------------------------------------------------ engine protocol
class TestEngine(unittest.TestCase):
    def test_greedy_generate_yields_tokens_and_sets_last(self):
        cfg = tiny_text_config()
        eng = Gemma4Engine(backend=NumpyBackend.from_model(tiny_model(cfg, seed=6)),
                           max_context=64, eos_ids=set())
        self.assertTrue(eng.alive())
        out = list(eng.generate([1, 2, 3], 4, {"temperature": 0.0}, threading.Event()))
        self.assertEqual(len(out), 4)
        self.assertTrue(all(isinstance(t, int) for t in out))
        self.assertEqual(eng.last["generated"], 4)
        self.assertEqual(eng.last["prompt_tokens"], 3)
        self.assertIn("finish", eng.last)

    def test_cancel_stops_generation(self):
        cfg = tiny_text_config()
        eng = Gemma4Engine(backend=NumpyBackend.from_model(tiny_model(cfg, seed=7)),
                           max_context=64, eos_ids=set())
        cancel = threading.Event()
        got = []
        for t in eng.generate([1, 2], 100, {"temperature": 0.0}, cancel):
            got.append(t)
            if len(got) == 2:
                cancel.set()
        self.assertLess(len(got), 100)
        self.assertEqual(eng.last["finish"], "cancel")

    def test_eos_stops(self):
        cfg = tiny_text_config()
        eng = Gemma4Engine(backend=NumpyBackend.from_model(tiny_model(cfg, seed=8)),
                           max_context=64, eos_ids=set(range(cfg.vocab_size)))
        out = list(eng.generate([1], 10, {"temperature": 0.0}, threading.Event()))
        self.assertEqual(out, [])
        self.assertEqual(eng.last["finish"], "stop")

    def test_unload_and_restart(self):
        cfg = tiny_text_config()
        eng = Gemma4Engine(backend=NumpyBackend.from_model(tiny_model(cfg, seed=9)),
                           max_context=64, eos_ids=set())
        eng.unload()
        self.assertFalse(eng.alive())
        eng.restart()
        self.assertTrue(eng.alive())

    def test_default_sampling_reads_the_gguf(self):
        md = {"general.sampling.temp": 1.0, "general.sampling.top_p": 0.95,
              "general.sampling.top_k": 64}
        self.assertEqual(default_sampling(md), {"temperature": 1.0, "top_p": 0.95, "top_k": 64})


# ------------------------------------------------------------------ sampler
class TestSampler(unittest.TestCase):
    def test_greedy_is_argmax(self):
        logits = np.array([0.1, 5.0, 0.2], np.float32)
        self.assertEqual(_sample(logits, 0.0, 0, 0.0, np.random.default_rng(0)), 1)

    def test_top_k_excludes_low(self):
        logits = np.array([0.0, 0.0, 0.0, 10.0], np.float32)
        picks = {_sample(logits, 1.0, 1, 0.0, np.random.default_rng(0)) for _ in range(20)}
        self.assertEqual(picks, {3})


# ------------------------------------------------------------------ template
class TestTemplate(unittest.TestCase):
    def test_renders_a_user_turn(self):
        tpl = Gemma4ChatTemplate(bos_token="")
        out = tpl.render([{"role": "user", "content": "hello"}])
        self.assertIn("<|turn>user", out)
        self.assertIn("hello", out)

    def test_system_and_generation_prompt(self):
        tpl = Gemma4ChatTemplate(bos_token="")
        out = tpl.render([{"role": "system", "content": "be nice"},
                          {"role": "user", "content": "hi"}])
        self.assertIn("<|turn>system", out)
        self.assertIn("be nice", out)
        self.assertIn("<|turn>model", out)

    def test_thinking_off_emits_an_empty_thought_block(self):
        """Gemma 4 always emits the thought channel; with thinking off it is empty.  A launcher
        that forgot the marker would train the model on a prompt shape it never saw."""
        tpl = Gemma4ChatTemplate(bos_token="", enable_thinking=False)
        out = tpl.render([{"role": "user", "content": "hi"}])
        self.assertIn("<|channel>thought", out)

    def test_from_gguf_uses_the_template_the_file_carries(self):
        """A GGUF's `tokenizer.chat_template` is the release's own; the bundled jinja is a
        fallback, not the authority."""
        cfg = tiny_text_config()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = pathlib.Path(tmp.name) / "t.gguf"
        write_tiny_gguf(p, cfg)
        marker = "<|turn>model\nCUSTOM-MARKER\n"
        from serve.gemma4 import quant as _q
        md = dict(gguf_metadata(cfg))
        md["tokenizer.chat_template"] = ("{{- bos_token -}}<|turn>user\\n"
                                        "{{ messages[0]['content'] }}" + marker)
        _q.write_gguf(p, md, [])
        tpl = Gemma4ChatTemplate.from_gguf(p, bos_token="")
        out = tpl.render([{"role": "user", "content": "hello"}])
        self.assertIn("CUSTOM-MARKER", out)
        # and a file with no template falls back to the bundled one
        p2 = pathlib.Path(tmp.name) / "u.gguf"
        _q.write_gguf(p2, gguf_metadata(cfg), [])
        tpl2 = Gemma4ChatTemplate.from_gguf(p2, bos_token="")
        self.assertIn("<|turn>model", tpl2.render([{"role": "user", "content": "hello"}]))


# ------------------------------------------------------------------ registry / ops
class TestRegistry(unittest.TestCase):
    def test_numpy_and_torch_are_registered(self):
        self.assertIn("numpy", available_backends())
        self.assertIn("torch", available_backends())
        self.assertIsInstance(get_backend("numpy", path=None), NumpyBackend)

    def test_unknown_backend_names_the_alternatives(self):
        from serve.gemma4.backends import BackendError
        with self.assertRaises(BackendError):
            get_backend("does-not-exist")

    def test_ops_surface_is_complete(self):
        """NumpyOps and TorchOps must expose the same methods, or a backend silently misses one."""
        from serve.gemma4.ops import NumpyOps as N, Ops
        import inspect
        abstract = {n for n, f in inspect.getmembers(Ops, inspect.isfunction)
                    if getattr(f, "__isabstractmethod__", False)}
        have = {n for n, f in inspect.getmembers(N, inspect.isfunction)}
        self.assertEqual(abstract - have, set())
        self.assertIn("torch", available_ops())


@unittest.skipUnless("torch" in available_ops(), "torch is not installed")
class TestTorchParity(unittest.TestCase):
    """The same `model.py` on two array backends must give the same logits."""

    def test_numpy_and_torch_logits_match(self):
        import torch
        from serve.gemma4.ops import TorchOps
        cfg = tiny_text_config()
        a = tiny_model(cfg, seed=21, xp=NumpyOps())
        b = tiny_model(cfg, seed=21, xp=TorchOps(device="cpu", dtype=torch.float32))
        ids = [2, 5, 11]
        la = np.asarray(a.forward(ids, start_pos=0))
        lb = np.asarray(b.forward(ids, start_pos=0).detach().cpu().numpy())
        np.testing.assert_allclose(la, lb, rtol=1e-4, atol=1e-4)

    def test_torch_backend_loads_a_real_gguf(self):
        import torch
        cfg = tiny_text_config()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = pathlib.Path(tmp.name) / "tiny.gguf"
        write_tiny_gguf(p, cfg, seed=22)
        be = TorchBackend(path=p, config=Gemma4Config(text=cfg), device="cpu",
                          dtype="float32", expert_stream=0).load()
        logits = be.forward([1, 2, 3], start_pos=0)
        self.assertEqual(tuple(logits.shape), (3, cfg.vocab_size))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertLessEqual(float(logits.abs().max()), cfg.final_logit_softcapping + 1e-3)

    def test_torch_backend_streams_experts(self):
        """The streamed profile must produce the same logits as the stacked one."""
        import torch
        cfg = tiny_text_config()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = pathlib.Path(tmp.name) / "tiny.gguf"
        write_tiny_gguf(p, cfg, seed=23)
        stacked = TorchBackend(path=p, config=Gemma4Config(text=cfg), device="cpu",
                               dtype="float32", expert_mode="stack").load()
        streamed = TorchBackend(path=p, config=Gemma4Config(text=cfg), device="cpu",
                                dtype="float32", expert_mode="stream", expert_stream=2).load()
        la = np.asarray(stacked.forward([4, 8, 15], 0).detach().cpu().numpy())
        lb = np.asarray(streamed.forward([4, 8, 15], 0).detach().cpu().numpy())
        np.testing.assert_allclose(la, lb, rtol=1e-4, atol=1e-4)

    def test_torch_backend_block_and_scratch_match_stack(self):
        """The two batching file-backed profiles must give the same logits as the stacked one, on
        torch as well as numpy: they are the modes a 24 GB card actually runs."""
        import torch
        cfg = tiny_text_config()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = pathlib.Path(tmp.name) / "tiny.gguf"
        write_tiny_gguf(p, cfg, seed=24)
        kw = dict(config=Gemma4Config(text=cfg), device="cpu", dtype="float32")
        stacked = TorchBackend(path=p, expert_mode="stack", **kw).load()
        la = np.asarray(stacked.forward([4, 8, 15], 0).detach().cpu().numpy())
        for mode, extra in (("block", dict(expert_block=2, expert_blocks=1)),
                            ("scratch", dict(expert_group=2))):
            be = TorchBackend(path=p, expert_mode=mode, **extra, **kw).load()
            lb = np.asarray(be.forward([4, 8, 15], 0).detach().cpu().numpy())
            np.testing.assert_allclose(la, lb, rtol=1e-4, atol=1e-4, err_msg=mode)


class TestExpertModes(unittest.TestCase):
    """The three profiles that leave the routed experts in the GGUF, against the stacked reference.

    The numbers in the docstrings were measured on gemma-4-26B-A4B Q4_0 (30 layers, hidden 2816,
    moe_intermediate_size 704, 128 experts, top_k 8) on torch CPU: one expert is 22.7 MiB dequantised
    (15.1 gate_up + 7.6 down), reading and dequantising one costs 14.9 ms, and stacking a group of 8
    into one tensor costs 12.5 ms.  What these tests pin down is the shape of the deal: a block is
    paid once and reused, a scratch buffer is paid per group, and neither may ever copy the whole
    layer's stack (measured at 1311 ms)."""

    def setUp(self):
        self.cfg = tiny_text_config()                 # 4 experts, top_k 2
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "tiny.gguf"
        write_tiny_gguf(self.path, self.cfg, seed=31)
        self.g = G.Gemma4GGUF.open(self.path)
        self.addCleanup(self.g.close)
        self.xp = NumpyOps()

    def _moe(self, experts, per_expert, seed=12, min_rows=8):
        """One real MoE layer, its non-expert weights from the fixture, over `experts`.

        `_routing` is the helper `TestBlocks` uses to build a routing with exact per-expert counts,
        so the assertions below are about the grouping the source allows, not about luck."""
        from serve.gemma4.backends import load_layer_weights
        w = load_layer_weights(self.g, Gemma4Config(text=self.cfg), 0, self.xp, stack_experts=False)
        router = Gemma4Router(w, self.cfg, xp=self.xp)
        moe = Gemma4MoE(self.cfg, w, router, experts, xp=self.xp)
        moe.batched_min_rows = min_rows
        idx_np, vals_np = TestBlocks._routing(self.cfg, per_expert, seed=seed)
        x = self.xp.asarray(np.random.default_rng(seed + 1).standard_normal(
            (idx_np.shape[0], self.cfg.hidden_size)).astype(np.float32), self.xp.float32)
        return moe, x, self.xp.asarray(idx_np, self.xp.int64), self.xp.asarray(vals_np, self.xp.float32)

    def _routed(self, experts, per_expert, **kw):
        moe, x, idx, vals = self._moe(experts, per_expert, **kw)
        return np.asarray(moe._routed(x, idx, vals), np.float32)

    def _stacked(self, per_expert, **kw):
        from serve.gemma4.backends import load_layer_weights
        w = load_layer_weights(self.g, Gemma4Config(text=self.cfg), 0, self.xp, stack_experts=True)
        src = StackedExperts(w, self.cfg.moe_intermediate_size, xp=self.xp)
        return self._routed(src, per_expert, **kw)

    def test_block_mode_batches_inside_each_block(self):
        """With blocks of 2, four experts carrying 12 tokens each must become TWO batches of 2, never
        one batch of 4: a range crossing a block boundary is not one view of one resident block."""
        from serve.gemma4.backends import BlockedExperts
        seen = []

        class Counting(BlockedExperts):
            def gate_up_range(self, lo, hi):
                seen.append((lo, hi))
                return super().gate_up_range(lo, hi)

        out = self._routed(Counting(self.g, 0, Gemma4Config(text=self.cfg), self.xp,
                                    block=2, keep=1), [12, 12, 12, 12])
        self.assertEqual(seen, [(0, 2), (2, 4)], "the groups crossed a block boundary")
        ref = self._stacked([12, 12, 12, 12])
        self.assertLess(float(np.abs(out - ref).max()), 2e-5 * max(1e-6, float(np.abs(ref).max())))

    def test_a_wider_block_merges_the_group(self):
        """The same routing with blocks of 4 is one batch: the block size is what caps the group."""
        from serve.gemma4.backends import BlockedExperts
        seen = []

        class Counting(BlockedExperts):
            def gate_up_range(self, lo, hi):
                seen.append((lo, hi))
                return super().gate_up_range(lo, hi)

        self._routed(Counting(self.g, 0, Gemma4Config(text=self.cfg), self.xp,
                              block=4, keep=1), [12, 12, 12, 12])
        self.assertEqual(seen, [(0, 4)])

    def test_block_mode_never_builds_a_block_for_one_expert(self):
        """A decode step picks 8 experts out of 128.  Building a block per pick would read 16 experts
        to use one, so the single-expert path must read exactly the expert it was asked for."""
        from serve.gemma4.backends import BlockedExperts
        src = BlockedExperts(self.g, 0, Gemma4Config(text=self.cfg), self.xp, block=2, keep=1)
        out = self._routed(src, [1, 1, 1, 1])
        self.assertEqual(src.built, 0, "a decode step built expert blocks")
        ref = self._stacked([1, 1, 1, 1])
        self.assertLess(float(np.abs(out - ref).max()), 2e-5 * max(1e-6, float(np.abs(ref).max())))

    def test_block_mode_refuses_a_range_across_a_block(self):
        from serve.gemma4.backends import BlockedExperts
        src = BlockedExperts(self.g, 0, Gemma4Config(text=self.cfg), self.xp, block=2, keep=1)
        with self.assertRaises(ValueError):
            src.gate_up_range(1, 4)

    def test_scratch_refills_once_per_group(self):
        """gate_up_range and down_range are called in sequence for the same group; the second must not
        dequantise the group again, and must not overwrite what the first handed back."""
        from serve.gemma4.backends import ScratchExperts
        src = ScratchExperts(self.g, 0, Gemma4Config(text=self.cfg), self.xp, group=4)
        out = self._routed(src, [12, 12, 12, 12])
        self.assertEqual(src.fills, 1, "the scratch buffer was refilled inside one group")
        ref = self._stacked([12, 12, 12, 12])
        self.assertLess(float(np.abs(out - ref).max()), 2e-5 * max(1e-6, float(np.abs(ref).max())))

    def test_scratch_caps_the_group_and_refills_between_groups(self):
        from serve.gemma4.backends import ScratchExperts
        seen = []

        class Counting(ScratchExperts):
            def gate_up_range(self, lo, hi):
                seen.append((lo, hi))
                return super().gate_up_range(lo, hi)

        src = Counting(self.g, 0, Gemma4Config(text=self.cfg), self.xp, group=2)
        out = self._routed(src, [12, 12, 12, 12])
        self.assertEqual(seen, [(0, 2), (2, 4)])
        self.assertEqual(src.fills, 2, "two different groups, one refill each")
        ref = self._stacked([12, 12, 12, 12])
        self.assertLess(float(np.abs(out - ref).max()), 2e-5 * max(1e-6, float(np.abs(ref).max())))

    def test_every_mode_matches_the_stacked_reference(self):
        """The whole point: where the experts live may not change what the model computes.  Routings
        with one big group, with no ties, and with lumpy sizes, so both paths are exercised."""
        from serve.gemma4.backends import BlockedExperts, ScratchExperts, StreamedExperts
        cfg = Gemma4Config(text=self.cfg)
        for per_expert in ([12, 12, 12, 6], [8, 8, 8, 8], [1, 1, 1, 1], [5, 3, 2, 2]):
            ref = self._stacked(per_expert)
            scale = max(1e-6, float(np.abs(ref).max()))
            for label, src in (
                    ("stream", StreamedExperts(self.g, 0, cfg, self.xp, keep=2)),
                    ("block-2", BlockedExperts(self.g, 0, cfg, self.xp, block=2, keep=1)),
                    ("block-4", BlockedExperts(self.g, 0, cfg, self.xp, block=4, keep=1)),
                    ("scratch-2", ScratchExperts(self.g, 0, cfg, self.xp, group=2)),
                    ("scratch-4", ScratchExperts(self.g, 0, cfg, self.xp, group=4))):
                out = self._routed(src, per_expert)
                self.assertTrue(np.all(np.isfinite(out)), f"{label} {per_expert}")
                self.assertLess(float(np.abs(out - ref).max()), 2e-5 * scale,
                                f"{label} {per_expert}")

    def test_stream_mode_with_one_cached_expert_survives_a_pair_of_calls(self):
        """gate_up and down are called back to back for the same expert.  An LRU that evicts between
        the two calls would re-read the expert and, at keep=0, never hold the pair."""
        from serve.gemma4.backends import StreamedExperts
        src = StreamedExperts(self.g, 0, Gemma4Config(text=self.cfg), self.xp, keep=1)
        out = self._routed(src, [12, 12, 12, 12])
        ref = self._stacked([12, 12, 12, 12])
        self.assertLess(float(np.abs(out - ref).max()), 2e-5 * max(1e-6, float(np.abs(ref).max())))

    def test_only_stack_and_block_and_scratch_claim_view_batching(self):
        from serve.gemma4.backends import (BlockedExperts, ScratchExperts, StreamedExperts)
        cfg = Gemma4Config(text=self.cfg)
        stream = StreamedExperts(self.g, 0, cfg, self.xp, keep=2)
        block = BlockedExperts(self.g, 0, cfg, self.xp, block=2, keep=1)
        scratch = ScratchExperts(self.g, 0, cfg, self.xp, group=2)
        self.assertFalse(stream.batched_views)
        self.assertTrue(block.batched_views and block.aligned_groups)
        self.assertTrue(scratch.batched_views and not scratch.aligned_groups)
        self.assertEqual(block.max_group, 2)
        self.assertEqual(scratch.max_group, 2)
        self.assertIsNone(StackedExperts({}, self.cfg.moe_intermediate_size).max_group)

    def test_expert_resident_bytes_is_the_shapes_times_the_kept_experts(self):
        """The preflight guard is only as good as this arithmetic: one expert of the tiny fixture is
        3*hidden*inter*4 bytes, and each mode keeps a different number of them per layer."""
        from serve.gemma4.backends import (default_pick_keep, estimate_bytes,
                                           expert_resident_bytes)
        t = tiny_text_config(num_experts=16)            # wide enough that the caches are not the lot
        cfg = Gemma4Config(text=t)
        per_expert = 3 * t.hidden_size * t.moe_intermediate_size * 4
        keep = default_pick_keep(cfg)                   # the per-expert LRU every file mode keeps
        self.assertEqual(keep, 2 * t.top_k_experts)
        self.assertLess(keep, t.num_experts)
        self.assertEqual(expert_resident_bytes(cfg, "stack"),
                         per_expert * t.num_experts * t.num_hidden_layers)
        self.assertEqual(expert_resident_bytes(cfg, "block", expert_block=2, expert_blocks=1),
                         per_expert * (keep + 2) * t.num_hidden_layers)
        self.assertEqual(expert_resident_bytes(cfg, "scratch", expert_group=2),
                         per_expert * (keep + 2) * t.num_hidden_layers)
        self.assertEqual(expert_resident_bytes(cfg, "stream", expert_stream=2),
                         per_expert * 2 * t.num_hidden_layers)
        # nothing may claim more than the model actually has
        for mode, kw in (("block", dict(expert_block=2, expert_blocks=99)),
                         ("scratch", dict(expert_group=99)),
                         ("stream", dict(expert_stream=99))):
            self.assertEqual(expert_resident_bytes(cfg, mode, **kw),
                             expert_resident_bytes(cfg, "stack"), mode)
        # the real geometry (30 layers, top_k 8): one expert is 22.6875 MiB in float32, and every
        # file-backed mode also keeps 2*top_k = 16 of them in the per-expert LRU
        real = tiny_text_config(hidden_size=2816, moe_intermediate_size=704, num_experts=128,
                                top_k_experts=8, num_hidden_layers=30)
        rc = Gemma4Config(text=real)
        self.assertEqual(default_pick_keep(rc), 16)
        self.assertAlmostEqual(expert_resident_bytes(rc, "stack") / 2**30, 85.1, delta=0.2)
        self.assertAlmostEqual(expert_resident_bytes(rc, "block", expert_block=8,
                                                     expert_blocks=1) / 2**30,
                               22.6875 * 24 * 30 / 2**10, delta=0.1)
        # a 24 GB card, float16: the non-expert weights plus the experts must fit
        non_expert = (estimate_bytes(rc) - expert_resident_bytes(rc, "stack")) * 0.5
        fits = [mode for mode, kw in (("block", dict(expert_block=8, expert_blocks=1)),
                                      ("block", dict(expert_block=16, expert_blocks=1)),
                                      ("block", dict(expert_block=24, expert_blocks=1)),
                                      ("scratch", dict(expert_group=8)),
                                      ("stack", {}))
                if non_expert + expert_resident_bytes(rc, mode, **kw) * 0.5 < 24 * 2**30]
        self.assertIn("block", fits, "no block size fits a 24 GB card in float16")
        self.assertNotIn("stack", fits, "the full expert stack must not look like it fits")
        with self.assertRaises(Exception):
            expert_resident_bytes(cfg, "nope")

    def test_auto_expert_profile_picks_by_regime_not_by_label(self):
        """`auto` has to choose the regime the measurements support, and the choice is binary.

        Measured on gemma-4-26B-A4B Q4_0 (torch CPU float32, one process per config,
        `bench/results/2026-10-03-expert-residency/`): every profile with a layer's 128 experts
        resident runs 95-167 ms per MoE layer at 128 tokens and every profile below that runs
        2.1-3.3 s.  Below residency `block` is the WORST option at equal memory (3252 ms against
        stream's 2109 ms for 48 experts per layer), so a budget that cannot reach residency must get
        `stream`, and one that can must get `stack` - not `block`, which costs the same bytes."""
        from serve.gemma4.backends import auto_expert_profile, default_pick_keep
        defaults = inspect.signature(TorchBackend.__init__).parameters
        self.assertEqual(defaults["expert_mode"].default, "auto")
        self.assertEqual(inspect.signature(NumpyBackend.__init__).parameters["expert_mode"].default,
                         "auto")

        real = tiny_text_config(hidden_size=2816, moe_intermediate_size=704, num_experts=128,
                                top_k_experts=8, num_hidden_layers=30)
        rc = Gemma4Config(text=real)
        experts_f16 = expert_resident_bytes(rc, "stack") * 0.5
        non_expert_f16 = (estimate_bytes(rc) - expert_resident_bytes(rc, "stack")) * 0.5

        # a 24 GB card: residency is out of reach, so stream, sized to what is left
        kw = auto_expert_profile(rc, 24 * 2**30, dtype_ratio=0.5)
        self.assertEqual(kw["expert_mode"], "stream")
        self.assertLess(kw["expert_stream"], real.num_experts)
        self.assertGreaterEqual(kw["expert_stream"], default_pick_keep(rc),
                                "it must at least keep the per-expert LRU every mode needs")
        spent = non_expert_f16 + expert_resident_bytes(rc, "stream",
                                                       expert_stream=kw["expert_stream"]) * 0.5
        self.assertLess(spent, 24 * 2**30, "the profile it picks has to actually fit")

        # a machine that can hold everything: stack, because block costs the same bytes and measured
        # slower (95 ms against 110-167 ms)
        kw = auto_expert_profile(rc, int(experts_f16 + non_expert_f16 + 8 * 2**30), dtype_ratio=0.5)
        self.assertEqual(kw["expert_mode"], "stack")

        # a dense model has no experts to place
        dense = Gemma4Config(text=tiny_text_config(enable_moe_block=False))
        self.assertEqual(auto_expert_profile(dense, 1), {"expert_mode": "stack"})

    def test_partial_residency_is_reported_at_load(self):
        """A profile that re-reads experts every forward is ~20x slower, and the user has to be told
        they bought it rather than left to infer it from tokens/s.

        The fixture has 4 experts and `default_pick_keep` is 2*top_k = 4, so every file-backed mode
        would already be fully resident through its LRU and stay quiet; widen the expert count so
        there is a partial regime to report."""
        wide = tiny_text_config(num_experts=8, top_k_experts=2)     # pick_keep 4, so 8 is reachable
        rc = Gemma4Config(text=wide)
        be = TorchBackend(config=rc, device="cpu", dtype="float32",
                          expert_mode="block", expert_block=2, expert_blocks=1)
        self.assertLess(be._resident_experts(), wide.num_experts)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            be._warn_partial_residency()
        self.assertIn("experts per layer", buf.getvalue())
        # a profile that does hold the whole layer must stay quiet
        be2 = TorchBackend(config=rc, device="cpu", dtype="float32",
                           expert_mode="block", expert_block=wide.num_experts, expert_blocks=1)
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            be2._warn_partial_residency()
        self.assertEqual(buf2.getvalue(), "", "a fully resident layer should not warn")
        # and stream is judged on its own keep, not on the block defaults it ignores
        be3 = TorchBackend(config=rc, device="cpu", dtype="float32",
                           expert_mode="stream", expert_stream=wide.num_experts)
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            be3._warn_partial_residency()
        self.assertEqual(buf3.getvalue(), "", "stream keep=8 holds all 8 experts")

    def test_a_full_cuda_device_refuses_instead_of_tracing(self):
        """`torch.cuda.mem_get_info` itself raises when the card is already full, which is the one
        situation the preflight exists for.

        Reproduced on a 4090 held by another process (23955 of 24564 MiB): the query throws
        `AcceleratorError: CUDA error: out of memory`, so the guard died in the query and the user got
        a torch stack trace instead of an explanation.  A card that cannot answer the question has
        answered it."""
        from serve.gemma4.backends import BackendError

        class FakeFull:
            class cuda:
                @staticmethod
                def mem_get_info():
                    raise RuntimeError("CUDA error: out of memory\n"
                                       "Search for `cudaErrorMemoryAllocation' at some long URL")

        class FakeRoomy:
            class cuda:
                @staticmethod
                def mem_get_info():
                    return (1024 * 1024, 24 * 1024 ** 3)

        class FakeBroken:
            class cuda:
                @staticmethod
                def mem_get_info():
                    raise RuntimeError("some driver bug")

        class FakeDevice:
            type = "cuda"

        class FakeXp:
            device = FakeDevice()

        be = TorchBackend(config=Gemma4Config(text=self.cfg), device="cuda", dtype="float16",
                          expert_mode="stream", expert_stream=2)
        be.xp = FakeXp()
        with self.assertRaises(BackendError) as cm:
            be._preflight(FakeFull)
        msg = str(cm.exception)
        self.assertIn("would not say how much is free", msg)
        self.assertIn("--device cpu", msg)
        self.assertNotIn("cudaErrorMemoryAllocation", msg, "the multi-line torch text leaked")

        # a card that CAN answer but has too little still gets the sized refusal. The tiny fixture
        # needs a few KiB, so it would pass any budget; ask about the real geometry instead.
        big = Gemma4Config(text=tiny_text_config(hidden_size=2816, moe_intermediate_size=704,
                                                 num_experts=128, top_k_experts=8,
                                                 num_hidden_layers=30))
        be2 = TorchBackend(config=big, device="cuda", dtype="float16", expert_mode="stack")
        be2.xp = FakeXp()
        with self.assertRaises(BackendError) as cm2:
            be2._preflight(FakeRoomy)
        msg2 = str(cm2.exception)
        self.assertIn("and only", msg2)
        self.assertIn("--expert-mode auto", msg2,
                      "a stack refusal must not suggest block knobs that do nothing")
        self.assertNotIn("lower --expert-block", msg2)

        # a query that fails for any OTHER reason is a bug, and must not be dressed up as a full card
        with self.assertRaises(RuntimeError) as cm3:
            be2._preflight(FakeBroken)
        self.assertNotIsInstance(cm3.exception, BackendError)

    def test_store_row_writes_in_place(self):
        """The scratch buffer must be refilled, not reallocated: 22.7 MiB per expert per group."""
        buf = self.xp.zeros((2, self.cfg.hidden_size, 2 * self.cfg.moe_intermediate_size),
                            self.xp.float32)
        src = np.full((self.cfg.hidden_size, 2 * self.cfg.moe_intermediate_size), 3.0, np.float32)
        self.xp.store_row(buf, 1, src)
        self.assertEqual(float(buf[1].max()), 3.0)
        self.assertEqual(float(buf[0].max()), 0.0, "it wrote the wrong expert")

    @unittest.skipUnless("torch" in available_ops(), "torch is not installed")
    def test_store_row_keeps_the_same_storage_on_torch(self):
        import torch
        from serve.gemma4.ops import TorchOps
        xp = TorchOps(device="cpu", dtype=torch.float32)
        buf = xp.zeros((2, 4, 6), xp.float32)
        ptr = buf.data_ptr()
        xp.store_row(buf, 0, torch.ones(4, 6))
        self.assertEqual(buf.data_ptr(), ptr, "store_row reallocated the buffer")
        self.assertTrue(torch.equal(buf[0], torch.ones(4, 6)))

    def test_a_stacked_range_is_a_view_not_a_copy(self):
        """The whole reason block and scratch exist: the group's weights must be read where they
        already are.  A copy of one layer's stack measured 1311 ms against the 100 ms of matmul."""
        from serve.gemma4.backends import BlockedExperts, ScratchExperts
        cfg = Gemma4Config(text=self.cfg)
        for src in (BlockedExperts(self.g, 0, cfg, self.xp, block=2, keep=1),
                    ScratchExperts(self.g, 0, cfg, self.xp, group=2)):
            view = src.gate_up_range(0, 2)
            self.assertEqual(view.shape, (2, self.cfg.hidden_size,
                                          2 * self.cfg.moe_intermediate_size),
                             f"{type(src).__name__}: wrong batch layout")
            again = src.gate_up_range(0, 2)
            self.assertTrue(np.shares_memory(view, again),
                            f"{type(src).__name__}: the range was materialised, not viewed")


class TestLauncherSampling(unittest.TestCase):
    """The GGUF's `general.sampling.*` must reach the Service, or every request that omits a
    temperature decodes greedily instead of at the model's tuned 1.0 / 0.95 / 64."""

    def test_gguf_sampling_reads_the_header(self):
        from serve.gemma4.server import gguf_sampling
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = pathlib.Path(tmp.name) / "tiny.gguf"
        write_tiny_gguf(p, tiny_text_config(), seed=41)
        got = gguf_sampling(p)
        self.assertEqual(set(got), {"temperature", "top_p", "top_k"})
        # the header stores floats as float32, so compare approximately
        self.assertAlmostEqual(got["temperature"], 1.0, places=6)
        self.assertAlmostEqual(got["top_p"], 0.95, places=6)
        self.assertEqual(got["top_k"], 64)

    def test_gguf_sampling_on_a_missing_file_is_empty(self):
        from serve.gemma4.server import gguf_sampling
        self.assertEqual(gguf_sampling("/nope/nothing.gguf"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

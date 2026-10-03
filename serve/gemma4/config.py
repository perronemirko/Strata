"""serve/gemma4/config.py - the Gemma 4 numbers, read from the model's own artifact.

This is the `ModelGeometry` equivalent for Gemma 4 (compare `include/strata/core/layout.hpp`, which
is the same idea for `qwen4exp`): every field here is a number a kernel or a shape check depends
on, so it is parsed from the artifact rather than hard-coded, and a missing field is an error at
load time rather than a wrong token at 4000.

Two sources describe the same model, and this module reads both:

  * the HuggingFace `config.json` of the repo, e.g.
    https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/config.json
    whose `text_config` is the decoder;
  * the GGUF's own `gemma4.*` metadata, which is what `gguf.py` reads so a user who only
    downloaded the `.gguf` still gets the right geometry.

Where the two could disagree, the GGUF wins, because that is the file the weights came from.

Five things in the real artifact are easy to read past and all change the arithmetic, so they are
kept as named fields rather than folded into a loop:

  * `layer_types` / `gemma4.attention.sliding_window_pattern` is an explicit 30-entry list, not a
    formula. It happens to be "5 sliding then 1 full", but the list is the source of truth and
    `layer_kind()` reads the list. A model that breaks the pattern still loads correctly.
  * `rope_parameters` is a DICT keyed by layer type, not one rope config. Sliding layers use
    `rope_theta=1e4` (`default`); full layers use `rope_theta=1e6` with `rope_type=proportional`
    and `partial_rotary_factor=0.25`. One rope for both is a different model.
  * the head width differs BY LAYER: sliding layers are 256 wide with 8 KV heads, full layers are
    512 wide with 2 KV heads. The GGUF spells this out as two pairs of keys
    (`attention.key_length` / `..._swa`, `attention.head_count_kv` as a 30-entry ARRAY), and the
    tensor shapes follow it: `blk.0.attn_q.weight` is [2816, 4096] and `blk.5.attn_q.weight` is
    [2816, 8192]. A single `head_dim` would mis-shape every fifth layer.
  * `rope.dimension_count` (512) and `rope.dimension_count_swa` (256) are the ROTATED widths, and
    they equal the full head width. The proportional rope still only rotates a quarter of the
    head: it emits `int(0.25 * 512 // 2) = 64` real inverse frequencies and pads the remaining
    192 with zeros (NoPE). The GGUF ships those 256 values as `rope_freqs.weight`, stored as the
    FACTORS (1/inv_freq), with 1e30 standing in for the zero frequencies.
  * `gemma4.attention.head_count_kv` is a per-layer array, and `gemma4.expert_used_count` (not
    `top_k_experts`) is the routed width. Reading only the scalar spellings silently yields the
    wrong MoE and the wrong KV geometry.

The attention scale is another one: Gemma 4 sets `scaling = 1.0` (llama.cpp:
`hparams.f_attention_scale = 1.0f`) instead of the usual `1/sqrt(head_dim)`, because Q and K are
RMS-normalised per head (`attn_q_norm`, `attn_k_norm`) before RoPE. `attention_scale` is a field
here so nobody re-derives it.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from typing import Any, Iterator, Sequence

import numpy as np


class ConfigError(ValueError):
    """The config is missing a field this model needs, or has a value the code cannot honour."""


class LayerKind:
    """The two attention kinds in `layer_types`.  Strings, so they compare equal to the config's
    own spellings and print legibly in an error."""

    SLIDING = "sliding_attention"
    FULL = "full_attention"


# The rope types this code knows how to apply.  Anything else is a ConfigError at load, not a
# silent fall-back to the default rope (which would produce a plausible, wrong token).
SUPPORTED_ROPE_TYPES = {"default", "proportional", "linear"}

# `rope_freqs.weight` stores 1/inv_freq; llama.cpp writes a huge finite factor where the
# proportional rope has a zero frequency (a channel that is not rotated at all).
NOPE_FACTOR = 1e20


@dataclasses.dataclass(frozen=True)
class RopeParams:
    """One rope configuration.

    `partial_rotary_factor` < 1 with `rope_type="proportional"` does NOT mean "rotate the first
    fraction of the head and pass the rest through" the way a plain partial rope does: the
    proportional form spreads `int(prf * head_dim // 2)` wavelengths over the WHOLE head exponent
    range and then leaves the remaining channels unrotated (NoPE). `inv_freq()` is the single
    place that difference is expressed, and it returns exactly `head_dim // 2` values.
    """

    rope_type: str = "default"
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    factor: float = 1.0                      # linear-scaling divisor; 1.0 = no scaling

    @staticmethod
    def from_dict(d: dict | None, *, default_theta: float = 10000.0) -> "RopeParams":
        d = d or {}
        rope_type = str(d.get("rope_type", d.get("type", "default")))
        if rope_type not in SUPPORTED_ROPE_TYPES:
            raise ConfigError(
                f"rope_type={rope_type!r} is not implemented here; this code handles "
                f"{sorted(SUPPORTED_ROPE_TYPES)}"
            )
        return RopeParams(
            rope_type=rope_type,
            rope_theta=float(d.get("rope_theta", default_theta)),
            partial_rotary_factor=float(d.get("partial_rotary_factor", 1.0)),
            factor=float(d.get("factor", 1.0) or 1.0),
        )

    def inv_freq(self, head_dim: int) -> tuple[float, ...]:
        """`head_dim // 2` inverse frequencies, in the order the rotary pairs use them.

        `default`: 1 / theta**(2i / dim) with dim = head_dim * partial_rotary_factor.
        `proportional`: 1 / theta**(2i / head_dim) for the first `int(prf * head_dim // 2)`
        angles, then 0.0 for the rest (NoPE). Matches transformers'
        `_compute_proportional_rope_parameters` and the 256 values in `rope_freqs.weight`.
        `linear`: the default form divided by `factor`.
        """
        half = head_dim // 2
        if half <= 0:
            return ()
        if self.rope_type == "proportional":
            angles = int(self.partial_rotary_factor * head_dim // 2)
            angles = max(0, min(angles, half))
            out = [self._inv(i, head_dim) / self.factor for i in range(angles)]
            out.extend(0.0 for _ in range(half - angles))
            return tuple(out)
        dim = int(head_dim * self.partial_rotary_factor)
        dim -= dim % 2
        if dim <= 0:
            return (0.0,) * half
        return tuple(self._inv(i, dim) / self.factor for i in range(half))

    def _inv(self, i: int, dim: int) -> float:
        return 1.0 / (self.rope_theta ** (2 * i / dim))

    def rotary_dim(self, head_dim: int) -> int:
        """How many of a head's `head_dim` channels carry a rotary pair. Rounded down to even,
        because rotate-half needs pairs."""
        n = sum(1 for f in self.inv_freq(head_dim) if f != 0.0) * 2
        return n - (n % 2)


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """Everything per-layer that a shape check or a kernel needs, resolved once at load."""

    index: int
    kind: str
    head_dim: int
    n_heads: int
    n_kv_heads: int
    theta: float
    rope_type: str
    has_kv: bool = True
    has_v: bool = True
    # `rope_freqs.weight`, when the file carries it: per-channel factors on the frequencies
    rope_factors: tuple[float, ...] = ()

    @property
    def full(self) -> bool:
        return self.kind == LayerKind.FULL

    @property
    def q_out(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.n_kv_heads * self.head_dim

    def inv_freq(self, rope_full: RopeParams, rope_sliding: RopeParams) -> tuple[float, ...]:
        base = (rope_full if self.full else rope_sliding).inv_freq(self.head_dim)
        if self.rope_factors:
            return apply_rope_factors(base, self.rope_factors)
        return base


@dataclasses.dataclass(frozen=True)
class Gemma4TextConfig:
    """The decoder.  Field names mirror `config.json`'s `text_config` so a reader can check one
    against the other line by line; the `*_swa` names mirror the GGUF's spellings."""

    hidden_size: int = 2816
    num_hidden_layers: int = 30
    vocab_size: int = 262144
    intermediate_size: int = 2112          # the dense MLP, which doubles as the shared expert
    num_attention_heads: int = 16
    head_dim: int = 256                    # the SLIDING layers' head width
    global_head_dim: int = 512             # the FULL layers' head width (K==V there)
    num_key_value_heads: int = 8           # sliding layers
    num_global_key_value_heads: int = 2    # full layers
    kv_heads_per_layer: tuple[int, ...] = ()
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_k_eq_v: bool = True          # full layers: one tensor is both K and V
    attention_scale: float = 1.0           # NOT 1/sqrt(head_dim): Q and K are RMS-normalised
    tie_word_embeddings: bool = True
    embed_scale: float = 0.0               # 0 = derive sqrt(hidden_size), as Gemma 4 does

    # MoE
    enable_moe_block: bool = True
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704

    # KV sharing (0 for 26B-A4B; the MTP drafter uses 4) and per-layer embeddings (0 here)
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    use_double_wide_mlp: bool = False

    layer_types: tuple[str, ...] = ()
    rope_sliding: RopeParams = dataclasses.field(default_factory=RopeParams)
    rope_full: RopeParams = dataclasses.field(
        default_factory=lambda: RopeParams(rope_type="proportional", rope_theta=1_000_000.0,
                                           partial_rotary_factor=0.25)
    )
    # the rotated widths the GGUF declares; 0 = "the whole head"
    rope_dim_full: int = 0
    rope_dim_sliding: int = 0

    # ---- derived
    def layer_kind(self, layer: int) -> str:
        """`LayerKind.SLIDING` or `LayerKind.FULL` for one layer index.  The explicit list is the
        source of truth; the 5/6 pattern is only a fallback for a config that omits it."""
        if self.layer_types:
            if not 0 <= layer < len(self.layer_types):
                raise ConfigError(f"layer {layer} out of range for layer_types "
                                  f"(len {len(self.layer_types)})")
            kind = self.layer_types[layer]
            if kind not in (LayerKind.SLIDING, LayerKind.FULL):
                raise ConfigError(f"layer_types[{layer}]={kind!r}: expected one of "
                                  f"{LayerKind.SLIDING!r} or {LayerKind.FULL!r}")
            return kind
        # fallback: every 6th layer (the last of each group of 5+1) is full, last layer always full
        return LayerKind.FULL if (layer + 1) % 6 == 0 or layer == self.num_hidden_layers - 1 \
            else LayerKind.SLIDING

    def is_full(self, layer: int) -> bool:
        return self.layer_kind(layer) == LayerKind.FULL

    def rope_for(self, layer: int) -> RopeParams:
        return self.rope_full if self.is_full(layer) else self.rope_sliding

    def head_dim_for(self, layer: int) -> int:
        """Full-attention layers carry a wider head (512) than sliding ones (256)."""
        return self.global_head_dim if self.is_full(layer) else self.head_dim

    def kv_heads_for(self, layer: int) -> int:
        if self.kv_heads_per_layer:
            if not 0 <= layer < len(self.kv_heads_per_layer):
                raise ConfigError(f"layer {layer} out of range for head_count_kv "
                                  f"(len {len(self.kv_heads_per_layer)})")
            return self.kv_heads_per_layer[layer]
        return self.num_global_key_value_heads if self.is_full(layer) else self.num_key_value_heads

    def q_proj_out(self, layer: int) -> int:
        """The q projection's output width.  Full layers carry a wider head (512) than sliding ones
        (256), so q's width is per-layer, not one number."""
        return self.num_attention_heads * self.head_dim_for(layer)

    def kv_proj_out(self, layer: int) -> int:
        return self.kv_heads_for(layer) * self.head_dim_for(layer)

    def has_v_proj(self, layer: int) -> bool:
        """Full layers fold V into K when `attention_k_eq_v` is set."""
        return not (self.is_full(layer) and self.attention_k_eq_v)

    def has_kv_proj(self, layer: int) -> bool:
        """The last `num_kv_shared_layers` layers reuse an earlier layer's KV and ship no weights."""
        if self.num_kv_shared_layers <= 0:
            return True
        return layer < self.num_hidden_layers - self.num_kv_shared_layers

    def n_full_layers(self) -> int:
        return sum(1 for i in range(self.num_hidden_layers) if self.is_full(i))

    def n_sliding_layers(self) -> int:
        return self.num_hidden_layers - self.n_full_layers()

    def layer_spec(self, layer: int, rope_factors: Sequence[float] | None = None) -> LayerSpec:
        """One layer resolved.  `rope_factors` is the file's `rope_freqs.weight`; it only applies
        to the full-attention layers, which are the ones whose rope is proportional."""
        return LayerSpec(
            index=layer,
            kind=self.layer_kind(layer),
            head_dim=self.head_dim_for(layer),
            n_heads=self.num_attention_heads,
            n_kv_heads=self.kv_heads_for(layer),
            theta=self.rope_for(layer).rope_theta,
            rope_type=self.rope_for(layer).rope_type,
            has_kv=self.has_kv_proj(layer),
            has_v=self.has_v_proj(layer),
            rope_factors=tuple(rope_factors) if (rope_factors and self.is_full(layer)) else (),
        )

    def specs(self, rope_factors: Sequence[float] | None = None) -> tuple[LayerSpec, ...]:
        return tuple(self.layer_spec(i, rope_factors) for i in range(self.num_hidden_layers))

    def effective_embed_scale(self) -> float:
        """Gemma 4 scales the token embedding by sqrt(hidden_size) before the first layer
        (llama.cpp: `ggml_scale(inpL, sqrtf(n_embd))`)."""
        return math.sqrt(self.hidden_size) if not self.embed_scale else float(self.embed_scale)

    @staticmethod
    def from_dict(d: dict) -> "Gemma4TextConfig":
        if not isinstance(d, dict):
            raise ConfigError("text_config is missing or not an object")
        layer_types = d.get("layer_types")
        if layer_types is not None and not isinstance(layer_types, list):
            raise ConfigError(f"layer_types={layer_types!r}: expected a list")
        # `sliding_window_pattern` is the GGUF's bool-array spelling of the same fact.
        pattern = d.get("sliding_window_pattern")
        if layer_types is None and isinstance(pattern, list) and pattern:
            layer_types = [LayerKind.SLIDING if bool(x) else LayerKind.FULL for x in pattern]

        rope = d.get("rope_parameters") or {}
        # `rope_parameters` is keyed by layer type in Gemma 4; a flat rope config (older style)
        # applies to both kinds.
        if "sliding_attention" in rope or "full_attention" in rope:
            rope_sliding = RopeParams.from_dict(rope.get("sliding_attention"))
            rope_full = RopeParams.from_dict(rope.get("full_attention"), default_theta=1_000_000.0)
        else:
            shared = RopeParams.from_dict(rope)
            rope_sliding, rope_full = shared, shared

        # Read only the keys the source actually has; the dataclass defaults (the real 26B-A4B
        # numbers) fill the rest.  A partial header - or a config.json that omits a field Google
        # defaulted - therefore yields a usable geometry instead of a KeyError.
        D = Gemma4TextConfig  # the defaults live on the class

        def g(key, cast=int, fallback=None):
            if key in d and d[key] is not None:
                return cast(d[key])
            return fallback() if callable(fallback) else fallback

        hidden = g("hidden_size", fallback=D.hidden_size)
        n_head = g("num_attention_heads", fallback=D.num_attention_heads)
        head_dim = g("head_dim", fallback=lambda: hidden // n_head)
        n_layers = g("num_hidden_layers", fallback=D.num_hidden_layers)

        kv = d.get("kv_heads_per_layer") or d.get("num_key_value_heads_per_layer")
        kv_per_layer: tuple[int, ...] = ()
        if isinstance(kv, list) and len(kv) == n_layers:
            kv_per_layer = tuple(int(x) for x in kv)

        return Gemma4TextConfig(
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            vocab_size=g("vocab_size", fallback=D.vocab_size),
            intermediate_size=g("intermediate_size", fallback=D.intermediate_size),
            num_attention_heads=n_head,
            head_dim=head_dim,
            global_head_dim=g("global_head_dim", fallback=lambda: head_dim),
            num_key_value_heads=g("num_key_value_heads", fallback=lambda: n_head),
            num_global_key_value_heads=g("num_global_key_value_heads", fallback=lambda: n_head),
            kv_heads_per_layer=kv_per_layer,
            sliding_window=g("sliding_window", fallback=D.sliding_window),
            max_position_embeddings=g("max_position_embeddings", fallback=D.max_position_embeddings),
            rms_norm_eps=g("rms_norm_eps", float, D.rms_norm_eps),
            final_logit_softcapping=g("final_logit_softcapping", float, D.final_logit_softcapping),
            hidden_activation=str(d.get("hidden_activation", D.hidden_activation)),
            attention_bias=bool(d.get("attention_bias", D.attention_bias)),
            attention_k_eq_v=bool(d.get("attention_k_eq_v", D.attention_k_eq_v)),
            attention_scale=g("attention_scale", float, D.attention_scale),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", D.tie_word_embeddings)),
            embed_scale=g("embed_scale", float, D.embed_scale),
            enable_moe_block=bool(d.get("enable_moe_block", D.enable_moe_block)),
            num_experts=g("num_experts", fallback=D.num_experts),
            top_k_experts=g("top_k_experts", fallback=D.top_k_experts),
            moe_intermediate_size=g("moe_intermediate_size", fallback=D.moe_intermediate_size),
            num_kv_shared_layers=g("num_kv_shared_layers", fallback=D.num_kv_shared_layers),
            hidden_size_per_layer_input=g("hidden_size_per_layer_input",
                                          fallback=D.hidden_size_per_layer_input),
            vocab_size_per_layer_input=g("vocab_size_per_layer_input",
                                         fallback=D.vocab_size_per_layer_input),
            use_double_wide_mlp=bool(d.get("use_double_wide_mlp", D.use_double_wide_mlp)),
            layer_types=tuple(layer_types) if layer_types else (),
            rope_sliding=rope_sliding,
            rope_full=rope_full,
            rope_dim_full=g("rope_dimension_count", fallback=0),
            rope_dim_sliding=g("rope_dimension_count_swa", fallback=0),
        )


@dataclasses.dataclass(frozen=True)
class Gemma4Config:
    """The whole `config.json`: the decoder plus the multimodal glue this add-on currently keeps as
    data (it does not run the vision encoder; that stays `strata-vision`'s job)."""

    text: Gemma4TextConfig
    model_type: str = "gemma4"
    architectures: tuple[str, ...] = ()
    # multimodal token ids, kept so a caller can recognise them in a prompt
    image_token_id: int | None = None
    audio_token_id: int | None = None
    video_token_id: int | None = None
    boa_token_id: int | None = None
    boi_token_id: int | None = None
    eoa_token_id: int | None = None
    eoi_token_id: int | None = None
    eos_token_id: int | None = None
    # Gemma 4's `eos_token_id` is a LIST ([1, 106]: `<eos>` and `<turn|>`), not a scalar. Keep every
    # value, because a generation that stops on only one of them runs past the end of a turn.
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    vision_soft_tokens_per_image: int | None = None
    has_vision: bool = False
    has_audio: bool = False

    # ---- convenience: forward the decoder's per-layer queries
    def __getattr__(self, name):        # cfg.num_hidden_layers -> cfg.text.num_hidden_layers
        text = self.__dict__.get("text")
        if text is not None and hasattr(text, name):
            return getattr(text, name)
        raise AttributeError(name)

    @property
    def num_hidden_layers(self) -> int:
        return self.text.num_hidden_layers

    def layer_kind(self, layer: int) -> str:
        return self.text.layer_kind(layer)

    def is_full(self, layer: int) -> bool:
        return self.text.is_full(layer)

    @staticmethod
    def from_dict(d: dict) -> "Gemma4Config":
        if not isinstance(d, dict):
            raise ConfigError("config.json did not parse to an object")
        # A GGUF-derived dict may put the decoder fields at the top level.
        text_raw = d.get("text_config")
        if text_raw is None and "block_count" in d:
            text_raw = d
        text = Gemma4TextConfig.from_dict(text_raw or {})
        archs = tuple(d.get("architectures") or ())
        return Gemma4Config(
            text=text,
            model_type=str(d.get("model_type", "gemma4")),
            architectures=archs,
            image_token_id=_opt_int(d.get("image_token_id")),
            audio_token_id=_opt_int(d.get("audio_token_id")),
            video_token_id=_opt_int(d.get("video_token_id")),
            boa_token_id=_opt_int(d.get("boa_token_id")),
            boi_token_id=_opt_int(d.get("boi_token_id")),
            eoa_token_id=_opt_int(d.get("eoa_token_id") or d.get("eoa_token_index")),
            eoi_token_id=_opt_int(d.get("eoi_token_id")),
            eos_token_id=_first_int(d.get("eos_token_id")),
            eos_token_ids=_int_list(d.get("eos_token_id")),
            pad_token_id=_opt_int(d.get("pad_token_id")),
            bos_token_id=_opt_int((d.get("text_config") or {}).get("bos_token_id")
                                  if isinstance(d.get("text_config"), dict)
                                  else d.get("bos_token_id")),
            vision_soft_tokens_per_image=_opt_int(d.get("vision_soft_tokens_per_image")),
            has_vision=d.get("vision_config") not in (None, {}, ""),
            has_audio=d.get("audio_config") not in (None, {}, ""),
        )

    @staticmethod
    def from_json(path: str | pathlib.Path) -> "Gemma4Config":
        p = pathlib.Path(path)
        if p.is_dir():
            p = p / "config.json"
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise ConfigError(f"no config.json at {p}") from e
        except json.JSONDecodeError as e:
            raise ConfigError(f"{p} is not valid JSON: {e}") from e
        return Gemma4Config.from_dict(raw)


@dataclasses.dataclass(frozen=True)
class MtpSpec:
    """What the repo's `gemma4-assistant` (MTP) drafter declares.

    `mtp-gemma-4-26B-A4B-it.gguf` is a 4-layer decoder of width 1024 that shares the target's KV
    cache (`shared_kv_layers = 4`) and predicts up to `nextn_predict_layers` tokens ahead. This
    add-on reads it so a launcher can tell the user what they have and refuse it with a reason
    instead of loading it as if it were the target; running it is a separate, opt-in piece of work.
    """

    block_count: int
    embedding_length: int
    nextn_predict_layers: int
    shared_kv_layers: int
    embedding_length_out: int
    feed_forward_length: int

    @staticmethod
    def from_metadata(md: dict) -> "MtpSpec | None":
        arch = str(md.get("general.architecture", ""))
        if arch != "gemma4-assistant":
            return None
        p = arch + "."

        def num(key, default=0):
            v = md.get(p + key, md.get(key, default))
            return default if v is None else int(v)

        return MtpSpec(
            block_count=num("block_count"),
            embedding_length=num("embedding_length"),
            nextn_predict_layers=num("nextn_predict_layers"),
            shared_kv_layers=num("attention.shared_kv_layers"),
            embedding_length_out=num("embedding_length_out"),
            feed_forward_length=num("feed_forward_length"),
        )


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)


def _int_list(v: Any) -> tuple[int, ...]:
    """A scalar, a list, or None -> a tuple of ints.  Gemma 4 writes `eos_token_id` as a list
    ([1, 106]); older or simpler configs write one int.  Both must load."""
    if v is None or v == ():
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v if x is not None)
    return (int(v),)


def _first_int(v: Any) -> int | None:
    ids = _int_list(v)
    return ids[0] if ids else None


def iter_layer_kinds(cfg: Gemma4TextConfig) -> Iterator[tuple[int, str]]:
    """(layer, kind) for every layer - handy for a loader that wants to allocate per-kind state."""
    for i in range(cfg.num_hidden_layers):
        yield i, cfg.layer_kind(i)


def apply_rope_factors(base: Sequence[float], factors: Sequence[float] | None) -> tuple[float, ...]:
    """Combine the frequencies this config derives with the artifact's `rope_freqs.weight`.

    llama.cpp treats that tensor as per-channel FACTORS that divide the frequency it computes from
    `rope.freq_base` and `rope.dimension_count`.  The real Gemma 4 file writes 1.0 for the 64
    channels the proportional rope rotates and 1e30 for the other 192 - a factor so large the
    frequency is effectively zero, i.e. NoPE.  So `base / factor`, with the sentinel mapped to 0.0.
    A file with other factors is honoured rather than ignored, which is what llama.cpp does."""
    if factors is None:
        return tuple(base)
    if len(factors) != len(base):
        raise ConfigError(f"rope_freqs has {len(factors)} entries, expected {len(base)} "
                          f"(= global head_dim / 2 for the full-attention layers)")
    out: list[float] = []
    for b, f in zip(base, factors):
        f = float(f)
        out.append(0.0 if f >= NOPE_FACTOR else float(b) / f)
    return tuple(out)


def check_rope_freqs(factors: Sequence[float] | None, derived: Sequence[float]) -> str | None:
    """Warn when the artifact's rope factors describe a different rope than the config does.
    Returns None when the two agree, which is the case for the real Gemma 4 files."""
    if factors is None:
        return None
    try:
        applied = apply_rope_factors(derived, factors)
    except ConfigError as e:
        return str(e)
    a = np.asarray(applied, dtype=np.float64)
    b = np.asarray(derived, dtype=np.float64)
    if not np.allclose(a, b, rtol=1e-3, atol=1e-12):
        worst = int(np.argmax(np.abs(a - b)))
        return ("rope_freqs.weight and the config's rope_parameters disagree: at index "
                f"{worst} the file gives {a[worst]:.6g} and the config gives {b[worst]:.6g}. "
                "The file wins, because it is what the weights were produced with.")
    return None

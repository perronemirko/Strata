"""serve/gemma4/gguf.py - the GGUF side of Gemma 4: names, shapes, validation, dequantisation.

Reuses `tools/gguf_reader.py` (the repo's own header-only GGUF v3 reader) rather than adding a
dependency on `gguf-py`, which cannot even open this artifact family (its enum has no member for
type 42 - see that file's docstring).  This module adds what the reader deliberately leaves out:
reading a tensor's bytes, turning them into float32, and following a SPLIT model.

The tensor names and shapes here are not guesses.  They were read out of
`unsloth/gemma-4-26B-A4B-it-GGUF` - the `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` header (658 tensors)
and the two `BF16/*-of-00002.gguf` shards - and they differ from what a Gemma 3 or a Llama loader
expects in ways that each break a load:

    ffn_gate_up_exps.weight   [hidden, 2*moe_ffn, experts]  gate and up are ONE fused tensor
    ffn_down_exps.weight      [moe_ffn, hidden, experts]
    ffn_down_exps.scale       [experts]                     router's per-expert output scale
    ffn_gate_inp.scale        [hidden]                      router's per-channel `scale`
    attn_q_norm/attn_k_norm   [head_dim]                    QK-norm, every layer
    layer_output_scale.weight [1]                           the layer's output multiplier
    rope_freqs.weight         [global_head_dim/2]           proportional-rope factors, shared
    post_attention_norm, post_ffw_norm, post_ffw_norm_1/2, pre_ffw_norm_2

There is no `ffn_*_shexp`: the "shared expert" is the ordinary dense MLP
(`ffn_gate`/`ffn_up`/`ffn_down`), which llama.cpp loads exactly that way ("for expert layers, we use
normal FFN as shared expert").

Memory order, stated once because it is where every GGUF bug lives
------------------------------------------------------------------
GGUF stores a tensor's dimensions with dim 0 varying FASTEST.  A PyTorch linear weight
`W[out, in]` is therefore written with GGUF shape `[in, out]`, and reading it back as
`flat.reshape(in, out, order="F")` gives exactly the matrix you multiply `x` by on the right
(`y = x @ W_gguf`).  `order="F"` is not a detail: numpy's default C-order reshape of the same
bytes returns the TRANSPOSE, which keeps every shape correct and makes every weight wrong.  An
embedding `E[vocab, hidden]` comes back as `[hidden, vocab]`, so a token lookup is `E[:, tok]`.
`read_tensor` returns the array in GGUF order and does NOT transpose; the model code indexes the
axis the shape says to.  Keeping the choice in one sentence here is the point: a silent transpose
in the reader and a transpose in the model would cancel to a wrong model.

Dequantisation support
----------------------
The reference numpy backend decodes F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K,
IQ4_NL and MXFP4 - which covers every quant Unsloth ships for this model (`UD-Q4_K_M` is
Q4_K + Q5_1 + Q8_0 + F32; `MXFP4_MOE` is MXFP4 + F32).  A file in an encoding this file does not
implement loads its header and validates its shapes, but `read_tensor` raises `UnsupportedQuant`
naming the tensor and its type and points at the fast backend - it never returns zeros or a guess.

The block layouts come from `third_party/ggml/ggml-common.h` (the structs) and are written to match
`ggml-quants.c`'s `dequantize_row_*` exactly.  `test_gemma4.py` checks each one against a scalar
reference transcribed from the same C, so a vectorisation bug fails a test instead of quietly
shifting a weight by one nibble.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import sys
from typing import Iterable

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gguf_reader import BLOCK_GEOMETRY, GGUFFile, TensorInfo  # noqa: E402  (repo's own reader)

from .config import (  # noqa: E402
    ConfigError, Gemma4Config, Gemma4TextConfig, LayerKind, MtpSpec, RopeParams,
)


class UnsupportedQuant(ValueError):
    """The file is fine; this reference reader just cannot decode that encoding yet."""


class MissingTensor(KeyError):
    """The pack does not have a tensor the architecture requires."""


# Encodings the reference reader can turn into float32.  Everything else -> UnsupportedQuant.
_DEQUANTABLE = frozenset({"F32", "F16", "BF16", "Q8_0", "Q4_0", "Q5_0", "Q5_1",
                          "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "MXFP4"})

# A GGUF shard is `<base>-NNNNN-of-MMMMM.gguf` (llama.cpp's spelling, and the one Unsloth ships:
# `gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf`).
_SHARD_RE = re.compile(r"^(?P<base>.+?)-(?P<idx>\d{5,})-of-(?P<total>\d{5,})\.gguf$")


def shard_group(path: str | pathlib.Path) -> tuple[str, int] | None:
    """(base_name, total) when `path` is a shard of a split model, else None."""
    p = pathlib.Path(path)
    m = _SHARD_RE.match(p.name)
    if not m:
        return None
    return m.group("base"), int(m.group("total"))


# ------------------------------------------------------------------ logical tensor names
# One logical tensor per role, each with the spellings a real file may use.  A loader asks by role
# and gets whichever name the file carries, so the HF-style names Unsloth writes and the
# llama.cpp-internal names both work.
LAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "attn_norm": ("attn_norm.weight",),
    "wq": ("attn_q.weight",),
    "wqkv": ("attn_qkv.weight",),
    "wk": ("attn_k.weight",),
    "wv": ("attn_v.weight",),
    "wo": ("attn_output.weight",),
    "q_norm": ("attn_q_norm.weight",),
    "k_norm": ("attn_k_norm.weight",),
    "post_attn_norm": ("post_attention_norm.weight", "attn_post_norm.weight"),
    "ffn_norm": ("ffn_norm.weight",),
    "gate": ("ffn_gate.weight",),
    "up": ("ffn_up.weight",),
    "down": ("ffn_down.weight",),
    "post_ffn_norm": ("post_ffw_norm.weight", "ffn_post_norm.weight"),
    "post_ffn_norm_1": ("post_ffw_norm_1.weight", "ffn_post_norm_1.weight"),
    "post_ffn_norm_2": ("post_ffw_norm_2.weight", "ffn_post_norm_2.weight"),
    "pre_ffn_norm_2": ("pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight"),
    "layer_scale": ("layer_output_scale.weight",),
    "router_w": ("ffn_gate_inp.weight",),
    "router_scale": ("ffn_gate_inp.scale",),
    "expert_scale": ("ffn_down_exps.scale", "ffn_up_exps.scale", "ffn_gate_exps.scale"),
    "gate_up_exps": ("ffn_gate_up_exps.weight",),
    "gate_exps": ("ffn_gate_exps.weight",),
    "up_exps": ("ffn_up_exps.weight",),
    "down_exps": ("ffn_down_exps.weight",),
}

MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "embed": ("token_embd.weight",),
    "output": ("output.weight",),
    "output_norm": ("output_norm.weight",),
    "rope_freqs": ("rope_freqs.weight",),
}


class Gemma4GGUF:
    """A Gemma 4 GGUF, single-file or split: its header, its tensor index, and float32 reads.

    A split model is opened by finding every `*-NNNNN-of-MMMMM.gguf` next to `path`.  Shard 1
    carries the metadata and the full tensor directory; each tensor's bytes live in whichever
    shard lists it, with offsets relative to that shard's own aligned data start."""

    def __init__(self, path: str | pathlib.Path, shards: list[pathlib.Path] | None = None):
        self.path = pathlib.Path(path)
        self.g = GGUFFile(self.path)
        self.metadata = self.g.metadata
        self._index: dict[str, TensorInfo] = {t.name: t for t in self.g.tensors}
        self._align = self.g.alignment
        self._files: dict[int, object] = {}
        self._data_starts: dict[int, int] = {0: self.g.data_start}
        self.shards: list[pathlib.Path] = [self.path] + [
            pathlib.Path(s) for s in (shards or []) if pathlib.Path(s) != self.path]
        self._shard_of: dict[str, int] = {t.name: 0 for t in self.g.tensors}

    # ---- opening / discovery
    @staticmethod
    def open(path: str | pathlib.Path) -> "Gemma4GGUF":
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no GGUF at {p}")
        grp = shard_group(p)
        if grp is None:
            return Gemma4GGUF(p)
        base, total = grp
        siblings = sorted(p.parent.glob(f"{base}-*-of-*.gguf"))
        g = Gemma4GGUF(p, shards=[s for s in siblings if s != p])
        if not g.is_split:
            raise FileNotFoundError(
                f"{p.name} says it is 1 of {total} but no sibling shard "
                f"({base}-NNNNN-of-*.gguf) is next to it")
        g._map_split_tensors()
        return g

    def _map_split_tensors(self) -> None:
        """Assign every tensor in the directory to the shard that actually stores its bytes."""
        for i, sp in enumerate(self.shards[1:], start=1):
            gi = GGUFFile(sp)
            self._data_starts[i] = gi.data_start
            for t in gi.tensors:
                self._index[t.name] = t
                self._shard_of[t.name] = i

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    @property
    def is_split(self) -> bool:
        return len(self.shards) > 1

    def has(self, name: str) -> bool:
        return name in self._index

    def names(self) -> Iterable[str]:
        return self._index.keys()

    def info(self, name: str) -> TensorInfo:
        if name not in self._index:
            raise MissingTensor(name)
        return self._index[name]

    def find(self, aliases: tuple[str, ...]) -> str | None:
        """First name in `aliases` this file has."""
        for a in aliases:
            if a in self._index:
                return a
        return None

    def find_layer(self, layer: int, logical: str) -> str | None:
        for alias in LAYER_ALIASES.get(logical, (logical,)):
            name = f"blk.{layer}.{alias}"
            if name in self._index:
                return name
        return None

    def types(self) -> dict[str, int]:
        """{encoding: how many tensors use it} - what a loader prints so a user can see the quant."""
        out: dict[str, int] = {}
        for t in self._index.values():
            out[t.type_name] = out.get(t.type_name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ---- config
    def config(self, hf_config: str | pathlib.Path | None = None) -> Gemma4Config:
        """The geometry.  If an HF `config.json` is available it is the starting point; the GGUF's
        own `gemma4.*` metadata then overrides every field it carries, because those describe the
        weights actually on disk."""
        if hf_config is not None:
            cfg = Gemma4Config.from_json(hf_config)
            return _merge_gguf_over_hf(cfg, self.metadata)
        return Gemma4Config.from_dict(_minimal_dict_from_gguf(self.metadata))

    def mtp(self) -> MtpSpec | None:
        """Non-None when this file is a `gemma4-assistant` MTP drafter rather than a target."""
        return MtpSpec.from_metadata(self.metadata)

    # ---- reading
    def _file(self, shard: int):
        fh = self._files.get(shard)
        if fh is None:
            fh = self.shards[shard].open("rb")
            self._files[shard] = fh
        return fh

    def close(self) -> None:
        for fh in self._files.values():
            if fh is not None:
                fh.close()
        self._files.clear()

    def __enter__(self) -> "Gemma4GGUF":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _tensor_bytes(self, t: TensorInfo) -> bytes:
        shard = self._shard_of.get(t.name, 0)
        off = _align_up(self._data_starts[shard] + t.offset, self._align)
        n = _raw_nbytes(t)
        fh = self._file(shard)
        fh.seek(off)
        raw = fh.read(n)
        if len(raw) != n:
            raise IOError(f"{t.name}: wanted {n} bytes at offset {off} of "
                          f"{self.shards[shard].name}, got {len(raw)}")
        return raw

    def read_tensor(self, name: str, dtype=np.float32) -> np.ndarray:
        """One tensor as an ndarray in GGUF order (reshaped to the GGUF shape).  `dtype` is the
        COMPUTE dtype (float32 default); the stored encoding is decoded to it."""
        t = self.info(name)
        if t.type_name not in _DEQUANTABLE:
            raise UnsupportedQuant(
                f"{t.name} is {t.type_name}; the reference numpy backend decodes "
                f"{sorted(_DEQUANTABLE)}. Run this file on the fast backend (serve/gemma4/backends.py)."
            )
        raw = self._tensor_bytes(t)
        flat = dequant(t.type_name, raw)
        if flat.size != t.elements:
            raise ValueError(f"{t.name}: decoded {flat.size} elements, header says {t.elements}")
        # order='F' because GGUF's flat order has dim 0 VARYING FASTEST, and numpy's default
        # reshape is C-order (last axis fastest).  A plain reshape would hand back the transpose
        # of every weight - a wrong model that still has every right shape.
        return flat.reshape(tuple(t.shape), order="F").astype(dtype, copy=False)

    def read_optional(self, name: str | None, dtype=np.float32) -> np.ndarray | None:
        return None if name is None or name not in self._index else self.read_tensor(name, dtype)

    def read_expert(self, name: str, expert: int, dtype=np.float32) -> np.ndarray:
        """One expert's slice of a stacked [d0, d1, n_experts] tensor, WITHOUT reading the others.

        GGUF stores dim 0 fastest and the quantisation blocks run along it, so expert `e` occupies
        a contiguous byte range: `e * d1 * row_bytes` for `d1` rows of `d0` elements.  That is what
        makes expert streaming possible - a Q4_K expert stack is 285 MB per layer and a token needs
        8 of the 128.  The encoding must have a block size that divides d0, which is exactly true
        for this model (gate_up_exps is Q4_K with d0=2816=11*256; down_exps is Q5_1/Q8_0 with
        d0=704=22*32)."""
        t = self.info(name)
        if len(t.shape) != 3:
            raise ValueError(f"{name}: read_expert needs a 3-D stacked tensor, shape is {t.shape}")
        d0, d1, n_exp = (int(x) for x in t.shape)
        if not 0 <= expert < n_exp:
            raise ValueError(f"{name}: expert {expert} out of range ({n_exp})")
        geom = BLOCK_GEOMETRY.get(t.type_name)
        if geom is None:
            raise UnsupportedQuant(f"{name} ({t.type_name}): no block geometry in the reader")
        bs, bb = geom
        if d0 % bs:
            raise UnsupportedQuant(f"{name}: {t.type_name} blocks of {bs} do not divide d0={d0}; "
                                   "cannot slice by expert")
        row_bytes = d0 // bs * bb
        shard = self._shard_of.get(name, 0)
        off = _align_up(self._data_starts[shard] + t.offset, self._align) + expert * d1 * row_bytes
        fh = self._file(shard)
        fh.seek(off)
        raw = fh.read(d1 * row_bytes)
        if len(raw) != d1 * row_bytes:
            raise IOError(f"{name}[{expert}]: wanted {d1 * row_bytes} bytes at {off}, "
                          f"got {len(raw)}")
        flat = dequant(t.type_name, raw)
        # the bytes are d1 rows of d0 elements, each row in GGUF order -> (d1, d0) -> transpose
        return flat.reshape((d1, d0)).T.astype(dtype, copy=False)     # (d0, d1), matmul-ready


# ------------------------------------------------------------------ byte geometry / dequant
def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _raw_nbytes(t: TensorInfo) -> int:
    """Stored size of a tensor, straight from the reader's block geometry table."""
    if t.type_name == "F32":
        return t.elements * 4
    if t.type_name in ("F16", "BF16"):
        return t.elements * 2
    if t.type_name == "F64":
        return t.elements * 8
    exp = t.expected_bytes()
    if exp is None:
        raise UnsupportedQuant(f"{t.name} ({t.type_name}): no byte geometry in the reader")
    return exp


def _scales(raw: np.ndarray, lo: int, size: int) -> np.ndarray:
    """Re-slice a (n, block) uint8 view and read the fp16 field at `lo` as float32."""
    return raw[:, lo:lo + 2].copy().view(np.float16).astype(np.float32).reshape(-1)


def dequant(type_name: str, raw: bytes) -> np.ndarray:
    """Decode one tensor's bytes to a flat float32 array.  Public because the torch backend and
    any future packer share exactly this contract."""
    fn = _DEQUANT.get(type_name)
    if fn is None:
        raise UnsupportedQuant(f"{type_name}: not implemented (implemented: {sorted(_DEQUANT)})")
    return fn(raw)


def _dq_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4")


def _dq_f16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def _dq_bf16(raw: bytes) -> np.ndarray:
    # bf16 = the top 16 bits of a float32: widen by shifting into the high half.
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u << 16).view(np.float32)


def _dq_q8_0(raw: bytes) -> np.ndarray:
    """block_q8_0 (34 B): fp16 d then 32 int8; value = d * q."""
    n = len(raw) // 34
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 34)
    d = _scales(b, 0, 2)
    q = b[:, 2:].astype(np.int8).astype(np.float32)      # qs is int8, not uint8
    return (q * d[:, None]).reshape(-1)


def _dq_q4_0(raw: bytes) -> np.ndarray:
    """block_q4_0 (18 B): fp16 d, 16 bytes of 4-bit; value = d*(q - 8), low nibble first."""
    n = len(raw) // 18
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 18)
    d = _scales(b, 0, 2)
    qs = b[:, 2:]
    out = np.empty((n, 32), np.float32)
    out[:, :16] = (qs & 0x0F).astype(np.float32) - 8.0
    out[:, 16:] = (qs >> 4).astype(np.float32) - 8.0
    return (out * d[:, None]).reshape(-1)


def _u32_le(b: np.ndarray, lo: int) -> np.ndarray:
    """The 4 little-endian bytes at `lo` of each row as one uint32 per row (ggml's memcpy of qh)."""
    return b[:, lo:lo + 4].copy().view(np.uint32).reshape(-1)


def _dq_q5_0(raw: bytes) -> np.ndarray:
    """block_q5_0 (22 B): fp16 d, 4 bytes qh, 16 bytes qs; value = d*(q - 16), q in [0, 31]."""
    n = len(raw) // 22
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 22)
    d = _scales(b, 0, 2)
    qh = _u32_le(b, 2)[:, None]
    qs = b[:, 6:]
    j = np.arange(16, dtype=np.uint32)[None, :]
    lo = (qs & 0x0F) | (((qh >> j) << 4) & 0x10)
    hi = (qs >> 4) | ((qh >> (j + 12)) & 0x10)
    out = np.empty((n, 32), np.float32)
    out[:, :16] = lo
    out[:, 16:] = hi
    return ((out - 16.0) * d[:, None]).reshape(-1)


def _dq_q5_1(raw: bytes) -> np.ndarray:
    """block_q5_1 (24 B): fp16 d, fp16 m, 4 bytes qh, 16 bytes qs; value = d*q + m."""
    n = len(raw) // 24
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 24)
    d = _scales(b, 0, 2)
    m = _scales(b, 2, 2)
    qh = _u32_le(b, 4)[:, None]
    qs = b[:, 8:]
    j = np.arange(16, dtype=np.uint32)[None, :]
    lo = (qs & 0x0F) | (((qh >> j) << 4) & 0x10)
    hi = (qs >> 4) | ((qh >> (j + 12)) & 0x10)
    out = np.empty((n, 32), np.float32)
    out[:, :16] = lo
    out[:, 16:] = hi
    return (out * d[:, None] + m[:, None]).reshape(-1)


def _scale_min_k4(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml's `get_scale_min_k4`: 8 6-bit scales and 8 6-bit mins packed into 12 bytes (the K-quant
    super-block has 8 sub-blocks of 32).  `s` is (n, 12) uint8 -> two (n, 8) arrays, sub-block order."""
    sc = np.empty((s.shape[0], 8), np.uint8)
    mn = np.empty((s.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    for k in range(4, 8):
        sc[:, k] = (s[:, k + 4] & 0x0F) | ((s[:, k - 4] >> 6) << 4)
        mn[:, k] = (s[:, k + 4] >> 4) | ((s[:, k] >> 6) << 4)
    return sc, mn


def _dq_q4_K(raw: bytes) -> np.ndarray:
    """block_q4_K (144 B): fp16 d, fp16 dmin, 12 packed scale/min bytes, 128 bytes of 4-bit codes.
    8 sub-blocks of 32; value = (d*sc)*q - (dmin*m)."""
    n = len(raw) // 144
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 144)
    d = _scales(b, 0, 2)
    dmin = _scales(b, 2, 2)
    sc, mn = _scale_min_k4(b[:, 4:16])
    qs = b[:, 16:]                                     # 128 bytes, 2 codes each
    out = np.empty((n, 256), np.float32)
    for g in range(4):                                 # each 64-element group: 32 low, 32 high
        out[:, g * 64:g * 64 + 32] = (qs[:, g * 32:g * 32 + 32] & 0x0F)
        out[:, g * 64 + 32:g * 64 + 64] = (qs[:, g * 32:g * 32 + 32] >> 4)
    scale = np.repeat(d[:, None] * sc.astype(np.float32), 32, axis=1)
    mins = np.repeat(dmin[:, None] * mn.astype(np.float32), 32, axis=1)
    return (out * scale - mins).reshape(-1)


def _dq_q5_K(raw: bytes) -> np.ndarray:
    """block_q5_K (176 B): as Q4_K, plus 32 bytes of 5th bits that every 64-element group reuses
    with a different mask (ggml's running `u1`/`u2`)."""
    n = len(raw) // 176
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 176)
    d = _scales(b, 0, 2)
    dmin = _scales(b, 2, 2)
    sc, mn = _scale_min_k4(b[:, 4:16])
    qh = b[:, 16:48]                                   # 32 bytes, indexed l = 0..31
    qs = b[:, 48:]                                     # 128 bytes, 2 codes each
    out = np.empty((n, 256), np.float32)
    for g in range(4):
        u1 = np.uint8(1 << (2 * g))                    # 5th bit for the low-nibble half
        u2 = np.uint8(1 << (2 * g + 1))                # ... and for the high-nibble half
        grp = qs[:, g * 32:g * 32 + 32]
        out[:, g * 64:g * 64 + 32] = (grp & 0x0F) + np.where(qh & u1, 16, 0)
        out[:, g * 64 + 32:g * 64 + 64] = (grp >> 4) + np.where(qh & u2, 16, 0)
    scale = np.repeat(d[:, None] * sc.astype(np.float32), 32, axis=1)
    mins = np.repeat(dmin[:, None] * mn.astype(np.float32), 32, axis=1)
    return (out * scale - mins).reshape(-1)


def _dq_q6_K(raw: bytes) -> np.ndarray:
    """block_q6_K (210 B): 128 B ql, 64 B qh (upper 2 bits), 16 int8 scales, fp16 d.
    value = d * sc * (q - 32).  Two 128-element halves, each with its own 8 scales."""
    n = len(raw) // 210
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 210)
    ql = b[:, 0:128]                                     # ql[QK_K/2]
    qh = b[:, 128:192]                                   # qh[QK_K/4] = 64 B, 32 per half
    sc = b[:, 192:208].astype(np.int8).astype(np.int32)  # 16 SIGNED scales
    d = _scales(b, 208, 2)
    out = np.empty((n, 256), np.int32)
    sidx = np.empty(256, np.int32)
    for h in range(2):                                   # two 128-element halves
        base = h * 128
        qlh = ql[:, h * 64:(h + 1) * 64]
        qhh = qh[:, h * 32:(h + 1) * 32]
        for l in range(32):
            is_ = l // 16
            hi = qhh[:, l].astype(np.int32)
            out[:, base + l] = ((qlh[:, l] & 0x0F) | ((hi & 3) << 4)) - 32
            out[:, base + l + 32] = ((qlh[:, l + 32] & 0x0F) | (((hi >> 2) & 3) << 4)) - 32
            out[:, base + l + 64] = ((qlh[:, l] >> 4) | (((hi >> 4) & 3) << 4)) - 32
            out[:, base + l + 96] = ((qlh[:, l + 32] >> 4) | (((hi >> 6) & 3) << 4)) - 32
            sidx[base + l] = h * 8 + is_ + 0
            sidx[base + l + 32] = h * 8 + is_ + 2
            sidx[base + l + 64] = h * 8 + is_ + 4
            sidx[base + l + 96] = h * 8 + is_ + 6
    scf = sc[:, sidx].astype(np.float32)
    return (out.astype(np.float32) * scf * d[:, None]).reshape(-1)


# kvalues_iq4nl, from third_party/ggml/ggml-common.h
_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                  np.float32)


def _dq_iq4_nl(raw: bytes) -> np.ndarray:
    """block_iq4_nl (18 B): fp16 d + 16 bytes of 4-bit indices into the 16-value codebook."""
    n = len(raw) // 18
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 18)
    d = _scales(b, 0, 2)
    qs = b[:, 2:]
    out = np.empty((n, 32), np.float32)
    out[:, :16] = _IQ4NL[qs & 0x0F]
    out[:, 16:] = _IQ4NL[qs >> 4]
    return (out * d[:, None]).reshape(-1)


# kvalues_fp4 (e2m1, doubled), from third_party/ggml/ggml-common.h
_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def _dq_mxfp4(raw: bytes) -> np.ndarray:
    """block_mxfp4 (17 B): one E8M0 exponent byte + 16 bytes of e2m1 codes, 32 elements.
    d = 0.5 * 2**(e - 127) (ggml's E8M0_TO_FP32_HALF, which pairs with the doubled codebook)."""
    n = len(raw) // 17
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, 17)
    e = b[:, 0].astype(np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - np.uint32(1)) << np.uint32(23))
    d = bits.view(np.float32)
    qs = b[:, 1:]
    out = np.empty((n, 32), np.float32)
    out[:, :16] = _MXFP4[qs & 0x0F]
    out[:, 16:] = _MXFP4[qs >> 4]
    return (out * d[:, None]).reshape(-1)


_DEQUANT = {
    "F32": _dq_f32, "F16": _dq_f16, "BF16": _dq_bf16, "Q8_0": _dq_q8_0,
    "Q4_0": _dq_q4_0, "Q5_0": _dq_q5_0, "Q5_1": _dq_q5_1,
    "Q4_K": _dq_q4_K, "Q5_K": _dq_q5_K, "Q6_K": _dq_q6_K,
    "IQ4_NL": _dq_iq4_nl, "MXFP4": _dq_mxfp4,
}


# ------------------------------------------------------------------ config from GGUF metadata
# The key spellings below are the ones the real Unsloth header uses (verified against the
# Q4_K_M and BF16 files), with the older scalar spellings kept as fallbacks.
def _meta(md: dict, *keys: str, default=None):
    for k in keys:
        if k in md:
            return md[k]
    return default


def _minimal_dict_from_gguf(md: dict) -> dict:
    """A `config.json`-shaped dict built from `gemma4.*` metadata, enough for `Gemma4Config`.
    Unknown fields keep the dataclass defaults, which are the real 26B-A4B numbers, so a partial
    header still yields a usable geometry instead of zeros."""
    arch = str(md.get("general.architecture", "gemma4"))

    def p(*suf, default=None):
        return _meta(md, *(f"{arch}.{s}" for s in suf), default=default)

    n_layers = p("block_count", "num_hidden_layers")
    hidden = p("embedding_length", "hidden_size")
    n_head = p("attention.head_count", "num_attention_heads")
    n_kv = p("attention.head_count_kv", "num_key_value_heads", default=2)
    vocab = p("vocab_size")
    inter = p("feed_forward_length", "intermediate_size")
    n_exp = p("expert_count", "num_experts")
    topk = p("expert_used_count", "expert_used", "top_k_experts")
    moe_inter = p("expert_feed_forward_length", "moe_intermediate_size")
    win = p("attention.sliding_window", "sliding_window")
    theta_full = p("rope.freq_base", "rope_theta")
    theta_swa = p("rope.freq_base_swa")
    eps = p("attention.layer_norm_rms_epsilon")
    softcap = p("final_logit_softcapping")
    kv_shared = p("attention.shared_kv_layers")
    ple = p("embedding_length_per_layer")

    # A dense Gemma 4 (gemma-4-12B) writes no expert keys at all. Left unset, the dataclass default
    # (True) would make `validate` demand a router and an expert stack the file does not have.
    moe = p("enable_moe_block")
    if moe is None:
        moe = any(v is not None for v in (n_exp, topk, moe_inter))

    text: dict = {"enable_moe_block": bool(moe)}
    for key, val in (("num_hidden_layers", n_layers), ("hidden_size", hidden),
                     ("num_attention_heads", n_head), ("vocab_size", vocab),
                     ("intermediate_size", inter), ("num_experts", n_exp),
                     ("top_k_experts", topk), ("moe_intermediate_size", moe_inter),
                     ("sliding_window", win), ("num_kv_shared_layers", kv_shared),
                     ("hidden_size_per_layer_input", ple)):
        if val is not None:
            text[key] = val
    if eps is not None:
        text["rms_norm_eps"] = float(eps)
    if softcap is not None:
        text["final_logit_softcapping"] = float(softcap)

    # Head widths: `key_length` is the FULL layers' head, `key_length_swa` the sliding ones.
    hd_full = p("attention.key_length", "attention.value_length")
    hd_swa = p("attention.key_length_swa", "attention.value_length_swa")
    if hd_full is not None:
        text["global_head_dim"] = int(hd_full)
    if hd_swa is not None:
        text["head_dim"] = int(hd_swa)
    elif hd_full is not None:
        text["head_dim"] = int(hd_full)

    # layer kinds first: `head_count_kv` is a per-layer ARRAY in this family and is often the only
    # place the full/sliding split is recorded alongside the KV widths.
    pat = p("attention.sliding_window_pattern", "layer_types")
    kinds: list[str] | None = None
    if isinstance(pat, list) and pat:
        if isinstance(pat[0], bool):
            kinds = [LayerKind.SLIDING if x else LayerKind.FULL for x in pat]
        else:
            kinds = [str(x) for x in pat]
        text["layer_types"] = kinds
    if kinds is None and n_layers:
        kinds = [LayerKind.FULL if (i + 1) % 6 == 0 else LayerKind.SLIDING
                 for i in range(int(n_layers))]

    if isinstance(n_kv, list):
        text["kv_heads_per_layer"] = [int(x) for x in n_kv]
        if n_layers and len(n_kv) == int(n_layers) and kinds is not None:
            full = [k for k, kind in zip(n_kv, kinds) if kind == LayerKind.FULL]
            swa = [k for k, kind in zip(n_kv, kinds) if kind == LayerKind.SLIDING]
            if full:
                text["num_global_key_value_heads"] = int(full[0])
            if swa:
                text["num_key_value_heads"] = int(swa[0])
    elif n_kv is not None:
        text["num_key_value_heads"] = int(n_kv)
        text["num_global_key_value_heads"] = int(n_kv)

    rope: dict = {}
    if theta_swa is not None:
        rope["sliding_attention"] = {"rope_type": "default", "rope_theta": float(theta_swa)}
    if theta_full is not None:
        rope["full_attention"] = {"rope_type": "proportional", "rope_theta": float(theta_full),
                                  "partial_rotary_factor": 0.25}
    if rope:
        text["rope_parameters"] = rope
    rd, rd_swa = p("rope.dimension_count"), p("rope.dimension_count_swa")
    if rd is not None:
        text["rope_dimension_count"] = int(rd)
    if rd_swa is not None:
        text["rope_dimension_count_swa"] = int(rd_swa)
    return {"model_type": "gemma4", "text_config": text}


def _merge_gguf_over_hf(cfg: Gemma4Config, md: dict) -> Gemma4Config:
    """An HF config plus the GGUF's own numbers: the GGUF wins where it has a value, because those
    describe the weights on disk (a converter that changed head widths would otherwise lie)."""
    over = _minimal_dict_from_gguf(md).get("text_config") or {}
    base = {f.name: getattr(cfg.text, f.name) for f in dataclasses.fields(Gemma4TextConfig)}
    for k, v in over.items():
        if v is not None:
            base[k] = v
    return dataclasses.replace(cfg, text=Gemma4TextConfig(**base))


# ------------------------------------------------------------------ validation
def required_layer_roles(cfg: Gemma4TextConfig, layer: int) -> list[str]:
    """The logical per-layer tensors this layer needs, by role (see `LAYER_ALIASES`)."""
    roles = ["attn_norm", "wq", "q_norm", "wo", "post_attn_norm",
             "ffn_norm", "gate", "up", "down", "post_ffn_norm"]
    if cfg.has_kv_proj(layer):
        roles += ["wk", "k_norm"]
        if cfg.has_v_proj(layer):
            roles.append("wv")
    if cfg.enable_moe_block:
        roles += ["router_w", "router_scale", "pre_ffn_norm_2", "post_ffn_norm_1",
                  "post_ffn_norm_2", "down_exps", "expert_scale"]
    return roles


def _has_expert_pair(gguf: "Gemma4GGUF", layer: int) -> bool:
    """The experts are either one fused gate/up tensor or two separate ones; either is fine."""
    return (gguf.find_layer(layer, "gate_up_exps") is not None
            or (gguf.find_layer(layer, "gate_exps") is not None
                and gguf.find_layer(layer, "up_exps") is not None))


def validate(gguf: Gemma4GGUF, cfg: Gemma4Config, *, check_shapes: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty = the file matches the config).  Collects
    every problem rather than stopping at the first, so one run tells the user the whole story."""
    problems: list[str] = []
    arch = gguf.architecture
    spec = gguf.mtp()
    if spec is not None:
        problems.append(
            f"this is an MTP drafter (arch {arch!r}, {spec.block_count} layers, width "
            f"{spec.embedding_length}, {spec.nextn_predict_layers} predicted tokens), not the "
            f"26B-A4B target - pass the target as --model")
        return problems
    if arch and arch not in ("gemma4", "gemma4_text"):
        problems.append(f"general.architecture={arch!r}, expected 'gemma4'")
    for role, names in (("embed", MODEL_ALIASES["embed"]), ("output_norm", MODEL_ALIASES["output_norm"])):
        if gguf.find(names) is None:
            problems.append(f"missing tensor {'/'.join(names)}")
    if not cfg.text.tie_word_embeddings and gguf.find(MODEL_ALIASES["output"]) is None:
        problems.append("untied embeddings but no output.weight")
    for layer in range(cfg.text.num_hidden_layers):
        for role in required_layer_roles(cfg.text, layer):
            if gguf.find_layer(layer, role) is None:
                problems.append(f"layer {layer}: missing {role} "
                                f"(looked for {'/'.join(LAYER_ALIASES.get(role, (role,)))})")
        if cfg.text.enable_moe_block and not _has_expert_pair(gguf, layer):
            problems.append(f"layer {layer}: missing the routed experts "
                            f"(looked for ffn_gate_up_exps.weight, or ffn_gate_exps + ffn_up_exps)")
    if check_shapes:
        problems += _check_shapes(gguf, cfg)
    return problems


def _check_shapes(gguf: Gemma4GGUF, cfg: Gemma4Config) -> list[str]:
    """The shapes the kernels depend on, asserted at load time (the `check_layer` idea from
    `include/strata/core/layout.hpp`, for this architecture)."""
    out: list[str] = []
    t = cfg.text

    def want(name: str | None, shape: tuple[int, ...]):
        if name is None or name not in gguf._index:
            return
        got = tuple(gguf.info(name).shape)
        if got != shape:
            out.append(f"{name}: shape {got}, expected {shape}")

    want(gguf.find(MODEL_ALIASES["embed"]), (t.hidden_size, t.vocab_size))   # GGUF order
    want(gguf.find(MODEL_ALIASES["output_norm"]), (t.hidden_size,))
    want(gguf.find(MODEL_ALIASES["rope_freqs"]), (t.global_head_dim // 2,))
    for layer in range(t.num_hidden_layers):
        want(gguf.find_layer(layer, "wq"), (t.hidden_size, t.q_proj_out(layer)))
        want(gguf.find_layer(layer, "q_norm"), (t.head_dim_for(layer),))
        if t.has_kv_proj(layer):
            want(gguf.find_layer(layer, "wk"), (t.hidden_size, t.kv_proj_out(layer)))
            want(gguf.find_layer(layer, "k_norm"), (t.head_dim_for(layer),))
            if t.has_v_proj(layer):
                want(gguf.find_layer(layer, "wv"), (t.hidden_size, t.kv_proj_out(layer)))
        want(gguf.find_layer(layer, "wo"), (t.q_proj_out(layer), t.hidden_size))
        want(gguf.find_layer(layer, "gate"), (t.hidden_size, t.intermediate_size))
        want(gguf.find_layer(layer, "up"), (t.hidden_size, t.intermediate_size))
        want(gguf.find_layer(layer, "down"), (t.intermediate_size, t.hidden_size))
        if t.enable_moe_block:
            want(gguf.find_layer(layer, "router_w"), (t.hidden_size, t.num_experts))
            want(gguf.find_layer(layer, "router_scale"), (t.hidden_size,))
            want(gguf.find_layer(layer, "down_exps"),
                 (t.moe_intermediate_size, t.hidden_size, t.num_experts))
            want(gguf.find_layer(layer, "expert_scale"), (t.num_experts,))
            fused = gguf.find_layer(layer, "gate_up_exps")
            if fused:
                want(fused, (t.hidden_size, 2 * t.moe_intermediate_size, t.num_experts))
            else:
                want(gguf.find_layer(layer, "gate_exps"),
                     (t.hidden_size, t.moe_intermediate_size, t.num_experts))
                want(gguf.find_layer(layer, "up_exps"),
                     (t.hidden_size, t.moe_intermediate_size, t.num_experts))
    return out


def describe(gguf: Gemma4GGUF) -> str:
    """One line a launcher prints: what this file says it is."""
    md = gguf.metadata
    n = len(gguf._index)
    t = f"{n} tensors in {len(gguf.shards)} shards" if gguf.is_split else f"{n} tensors"
    return (f"{md.get('general.name', gguf.path.name)}: arch={gguf.architecture or '?'}, {t}, "
            f"layers={md.get('gemma4.block_count', '?')}, quant={gguf.types()}")

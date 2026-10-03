"""serve/gemma4/quant.py - write a Gemma 4 GGUF, in the encodings the loader can read.

Two jobs, both so the add-on can be tested against a file that looks like the real one:

  * `quantize_*`: reference encoders for Q8_0, Q4_K and Q5_1, the three that a `UD-Q4_K_M` release
    actually uses for the big tensors.  They are the exact inverse of `gguf.dequant` - not in
    bit-for-bit round trip (quantisation loses information by design) but in LAYOUT, which is the
    thing a loader can get wrong silently.
  * `write_gguf`: a minimal GGUF v3 writer that emits those encodings.  `tools/gguf_writer.py`
    already exists and writes F32 and Q2_0; extending it would touch a file the qwen4exp tests
    depend on, so this is a separate writer for this architecture instead.  It writes the same
    header layout and is read back by the same `tools/gguf_reader.py`.

Why bother writing quantised fixtures at all: a test that only ever loads F32 tensors never
exercises the block geometry, the nibble order or the 6-bit scale packing, and those are exactly
where a GGUF reader breaks.  `test_gemma4.py` writes a tiny Q4_K/Q5_1/Q8_0 model and requires the
dequantised values to match what the encoder was given, within the error the format allows.
"""
from __future__ import annotations

import pathlib
import struct
from typing import Iterable, Sequence

import numpy as np

GGUF_MAGIC = 0x46554747
DEFAULT_ALIGNMENT = 32

# ggml_type ids for the encodings this module writes (values match tools/gguf_reader.py)
TYPE_IDS = {"F32": 0, "F16": 1, "Q4_0": 2, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
            "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ4_NL": 20, "BF16": 30, "MXFP4": 39}

_META_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
             "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12}


# ------------------------------------------------------------------ encoders
def quantize_q8_0(w: np.ndarray) -> bytes:
    """(n,) -> 34-byte blocks: fp16 scale, 32 int8.  n must be a multiple of 32."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 32:
        raise ValueError(f"Q8_0 needs a multiple of 32 elements, got {w.size}")
    blocks = w.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax == 0, np.float32(0), amax / 127.0)
    q = np.where(d[:, None] == 0, 0, np.rint(blocks / d[:, None])).astype(np.int8)
    out = bytearray()
    dh = d.astype(np.float16).view(np.uint16)
    for i in range(blocks.shape[0]):
        out += struct.pack("<H", int(dh[i]))
        out += q[i].tobytes()
    return bytes(out)


def _pack_scale_min(sc: np.ndarray, mn: np.ndarray) -> bytes:
    """The 12-byte `scales[]` field of a K-quant block: 8 6-bit scales, 8 6-bit mins.
    Inverse of `gguf._scale_min_k4`, written the way `quantize_row_q4_K_impl` does it."""
    s = np.zeros(12, np.uint8)
    for j in range(4):
        s[j] = sc[j] & 0x3F
        s[j + 4] = mn[j] & 0x3F
    for j in range(4, 8):
        s[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4)
        s[j - 4] |= (sc[j] >> 4) << 6
        s[j + 0] |= (mn[j] >> 4) << 6
    return s.tobytes()


def quantize_q4_k(w: np.ndarray) -> bytes:
    """(n,) -> 144-byte blocks: fp16 d, fp16 dmin, 12 packed scale/min bytes, 128 bytes of codes.

    A simple min/max encoder per 32-element sub-block (not ggml's search): good enough to test the
    reader's layout, and it produces a file a real llama.cpp could open.  n % 256 == 0."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 256:
        raise ValueError(f"Q4_K needs a multiple of 256 elements, got {w.size}")
    out = bytearray()
    for base in range(0, w.size, 256):
        blk = w[base:base + 256].reshape(8, 32)
        lo = blk.min(axis=1)
        hi = blk.max(axis=1)
        # `scales[j]` is the STEP of sub-block j (codes span 0..15), not its range; the 8 steps are
        # then themselves quantised to 6 bits against a super-block scale d = max(step)/63.  That
        # two-level scaling is what `make_qkx3_quants(32, 15, ...)` + `make_qp_quants(8, 63, ...)`
        # do in ggml, and getting it backwards yields codes that all land on 0 or 1.
        scales = np.maximum((hi - lo) / 15.0, 1e-8)
        mins = -lo
        d = max(float(np.max(scales)) / 63.0, 1e-12)
        dmin = max(float(np.max(mins)) / 63.0, 1e-12)
        sc = np.clip(np.rint(scales / d), 0, 63).astype(np.uint8)
        mn = np.clip(np.rint(mins / dmin), 0, 63).astype(np.uint8)
        # ggml's convention is x ~= d*sc*q - dmin*mn, so the code is the (x - min) offset, which is
        # what makes the dequant's `- m1` land back on x.
        q = np.clip(np.rint((blk - lo[:, None]) / np.maximum(sc[:, None].astype(np.float32) * d,
                             1e-12)), 0, 15).astype(np.uint8)
        # ggml's nibble order: for each 64-element group, the FIRST 32 bytes' low nibbles are the
        # group's first 32 elements and the same 32 bytes' high nibbles are its last 32.  So
        # byte g*32 + l packs sub-block 2g's element l with sub-block 2g+1's element l.
        qs = np.empty(128, np.uint8)
        for g in range(4):
            qs[g * 32:g * 32 + 32] = q[2 * g] | (q[2 * g + 1] << 4)
        out += struct.pack("<e", np.float16(d))
        out += struct.pack("<e", np.float16(dmin))
        out += _pack_scale_min(sc, mn)
        out += qs.tobytes()
    return bytes(out)


def quantize_q5_1(w: np.ndarray) -> bytes:
    """(n,) -> 24-byte blocks: fp16 d, fp16 m, 4 bytes qh (5th bit), 16 bytes qs. n % 32 == 0."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 32:
        raise ValueError(f"Q5_1 needs a multiple of 32 elements, got {w.size}")
    out = bytearray()
    for base in range(0, w.size, 32):
        blk = w[base:base + 32]
        lo, hi = float(blk.min()), float(blk.max())
        d = max((hi - lo) / 31.0, 1e-12)
        m = lo
        q = np.clip(np.rint((blk - m) / d), 0, 31).astype(np.uint8)
        low = q & 0x0F
        high = (q >> 4) & 1
        qh = 0
        # ggml's layout: qh bit j is the 5th bit of element j, and bit (j + 16) is the 5th bit of
        # element (16 + j).  The dequantiser reads the second half as `(qh >> (j+12)) & 0x10`,
        # which is bit j+16 - easy to misread as j+12 and get a file that loads with half its
        # elements wrong.
        for j in range(16):
            qh |= int(high[j]) << j
            qh |= int(high[16 + j]) << (j + 16)
        qs = low[:16] | (low[16:] << 4)
        out += struct.pack("<e", np.float16(d))
        out += struct.pack("<e", np.float16(m))
        out += struct.pack("<I", qh)
        out += qs.tobytes()
    return bytes(out)


ENCODERS = {"Q8_0": quantize_q8_0, "Q4_K": quantize_q4_k, "Q5_1": quantize_q5_1}


# ------------------------------------------------------------------ writer
class Tensor:
    def __init__(self, name: str, shape: Sequence[int], type_name: str, data: bytes):
        self.name, self.shape, self.type_name, self.data = name, list(shape), type_name, data


def make_tensor(name: str, arr: np.ndarray, type_name: str = "F32") -> Tensor:
    """One tensor, quantised along dim 0 (the axis GGUF's blocks run along).

    `arr` is given in GGUF order, i.e. shape == the GGUF shape with dim 0 fastest-varying.  For a
    stacked expert tensor [d0, d1, n_experts] the encoder runs over d0 in chunks, which is what
    `Gemma4GGUF.read_expert` relies on."""
    a = np.asarray(arr, dtype=np.float32)
    if type_name == "F32":
        return Tensor(name, a.shape, "F32", np.ascontiguousarray(a, dtype="<f4").tobytes())
    if type_name == "BF16":
        f = a.astype(np.float32).view(np.uint32)
        return Tensor(name, a.shape, "BF16", (f >> 16).astype("<u2").tobytes())
    if type_name == "F16":
        return Tensor(name, a.shape, "F16", np.ascontiguousarray(a, dtype="<f2").tobytes())
    enc = ENCODERS.get(type_name)
    if enc is None:
        raise ValueError(f"cannot write {type_name}: no encoder here (have {sorted(ENCODERS)})")
    # GGUF order: dim 0 varies fastest, so the flat stream is a.reshape(order='F')
    flat = a.reshape(-1, order="F")
    return Tensor(name, a.shape, type_name, enc(flat))


def write_gguf(path: str | pathlib.Path, metadata: dict, tensors: Iterable[Tensor],
               alignment: int = DEFAULT_ALIGNMENT) -> pathlib.Path:
    """A GGUF v3 file: `metadata` is {key: value} (Python types inferred), `tensors` are `Tensor`s."""
    path = pathlib.Path(path)
    body = bytearray()
    body += struct.pack("<I", GGUF_MAGIC)
    body += struct.pack("<I", 3)
    tlist = list(tensors)
    body += struct.pack("<Q", len(tlist))
    body += struct.pack("<Q", len(metadata))
    for k, v in metadata.items():
        body += _kv_bytes(k, v)

    offset = 0
    infos = bytearray()
    for t in tlist:
        infos += struct.pack("<Q", len(t.name)) + t.name.encode("utf-8")
        infos += struct.pack("<I", len(t.shape))
        infos += struct.pack(f"<{len(t.shape)}Q", *t.shape)
        infos += struct.pack("<I", TYPE_IDS[t.type_name])
        infos += struct.pack("<Q", offset)
        offset += len(t.data)
        offset = (offset + alignment - 1) // alignment * alignment
    body += infos

    body += b"\0" * ((-len(body)) % alignment)
    for t in tlist:
        body += t.data
        body += b"\0" * ((-len(t.data)) % alignment)
    path.write_bytes(bytes(body))
    return path


def _kv_bytes(key: str, value) -> bytes:
    type_name = _infer(value)
    out = bytearray(struct.pack("<Q", len(key)) + key.encode("utf-8"))
    if type_name.startswith("array:"):
        elem = type_name.split(":", 1)[1]
        out += struct.pack("<I", _META_IDS["array"])
        out += struct.pack("<I", _META_IDS[elem])
        out += struct.pack("<Q", len(value))
        for v in value:
            out += _scalar(elem, v)
        return bytes(out)
    out += struct.pack("<I", _META_IDS[type_name])
    out += _scalar(type_name, value)
    return bytes(out)


def _infer(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "i32" if -2**31 <= value < 2**31 else "i64"
    if isinstance(value, float):
        return "f32"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)) and value:
        return "array:" + _infer(value[0])
    raise TypeError(f"cannot infer a GGUF type for {value!r}")


def _scalar(type_name: str, v) -> bytes:
    if type_name == "string":
        b = v.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
           "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[type_name]
    return struct.pack(fmt, v)

"""serve/gemma4/tokenizer.py - Gemma 4's tokenizer, rebuilt from the GGUF's own metadata.

Why not a library
-----------------
The GGUF repo ships no `tokenizer.json`; llama.cpp stores the SentencePiece model inside the GGUF
as metadata (`tokenizer.ggml.tokens`, `.scores`, `.token_type`, ...).  Strata already reads GGUF
metadata with its own header-only reader (`tools/gguf_reader.py`), so the tokenizer is rebuilt from
those arrays here rather than adding a SentencePiece dependency the rest of Strata does not carry.

The interface Strata needs
--------------------------
`serve/server.py`'s `Detokenizer` and `Service` call three things on a tokenizer:

    encode(text, parse_special=False) -> list[int]
    decode(ids, errors="replace")     -> str
    token_bytes(id)                   -> bytes      (present => incremental UTF-8 decode)

`GgufTokenizer` implements exactly that, so it drops into `Service` unchanged.  `token_bytes`
returns one token's raw bytes with the SentencePiece word marker `▁` already turned into a space,
which is what makes the server's incremental detokeniser produce the same text as a whole decode.

SentencePiece unigram, in brief
-------------------------------
The model is a set of pieces, each with a log-probability.  Encoding a string finds the
highest-probability segmentation into pieces (Viterbi over the byte sequence).  Control pieces
(`<bos>`, `<eos>`, the chat markers) are matched first and never split.  Byte-fallback pieces
(`<0xNN>`) let out-of-vocabulary characters be spelled byte by byte.

ASSUMPTIONS (marked in code): the NMT pre-tokeniser is the reduced form - Unicode NFKC, collapse
extra whitespace, map spaces to `▁`.  Full SentencePiece also does a specific Unicode normalisation
table; if a token ever disagrees with llama.cpp, that normalisation is the first place to look, and
it is isolated in `_normalize`.
"""
from __future__ import annotations

import pathlib
import sys
import unicodedata
from typing import Iterable

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gguf_reader import GGUFFile  # noqa: E402

WORD_MARKER = "\u2581"          # ▁, SentencePiece's word-boundary piece
_BYTE_PIECE_FMT = "<0x{:02X}>"  # a single byte as a piece, e.g. <0xE2>

# tokenizer.ggml.token_type values (llama.cpp's enum)
_TT_NORMAL, _TT_UNKNOWN, _TT_CONTROL, _TT_USER, _TT_BYTE = 1, 2, 3, 4, 6


class TokenizerError(ValueError):
    """The GGUF does not carry a tokenizer this code can rebuild."""


class GgufTokenizer:
    """A SentencePiece unigram tokenizer read from a Gemma 4 GGUF's metadata."""

    def __init__(self, tokens: list[str], scores: list[float] | None, token_type: list[int] | None,
                 *, bos_id: int | None = None, eos_id: int | None = None,
                 add_bos: bool = False, byte_fallback: bool = True,
                 add_space_prefix: bool = True, remove_extra_ws: bool = True):
        self.tokens = list(tokens)
        self.n = len(self.tokens)
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.scores = list(scores) if scores is not None else [0.0] * self.n
        self.token_type = list(token_type) if token_type is not None else [_TT_NORMAL] * self.n
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.add_bos = bool(add_bos)
        self.byte_fallback = bool(byte_fallback)
        self.add_space_prefix = bool(add_space_prefix)
        self.remove_extra_ws = bool(remove_extra_ws)

        # control/user pieces are matched whole (never split); normal pieces feed the Viterbi.
        self.specials: list[str] = [t for t, tt in zip(self.tokens, self.token_type)
                                    if tt in (_TT_CONTROL, _TT_USER)]
        self.special_set = set(self.specials)
        # longest-first so a longer control marker wins over a shorter one that is its prefix
        self.specials.sort(key=len, reverse=True)
        # byte pieces -> their byte value, for decode and byte-fallback encode
        self.byte_piece: dict[str, int] = {}
        for t in self.tokens:
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                try:
                    self.byte_piece[t] = int(t[3:5], 16)
                except ValueError:
                    pass
        self._build_trie()

    # ---- construction from a GGUF
    @staticmethod
    def from_gguf(path: str | pathlib.Path) -> "GgufTokenizer":
        g = GGUFFile(pathlib.Path(path))
        md = g.metadata
        tokens = md.get("tokenizer.ggml.tokens")
        if not isinstance(tokens, list) or not tokens:
            raise TokenizerError("the GGUF has no tokenizer.ggml.tokens")
        scores = md.get("tokenizer.ggml.scores")
        ttype = md.get("tokenizer.ggml.token_type")
        return GgufTokenizer(
            tokens,
            [float(s) for s in scores] if isinstance(scores, list) else None,
            [int(t) for t in ttype] if isinstance(ttype, list) else None,
            bos_id=_maybe_int(md.get("tokenizer.ggml.bos_token_id")),
            eos_id=_maybe_int(md.get("tokenizer.ggml.eos_token_id")),
            add_bos=bool(md.get("tokenizer.ggml.add_bos_token", False)),
            byte_fallback=bool(md.get("tokenizer.ggml.byte_fallback", True)),
            add_space_prefix=bool(md.get("tokenizer.ggml.add_space_prefix", True)),
            remove_extra_ws=bool(md.get("tokenizer.ggml.remove_extra_whitespaces", True)),
        )

    # ---- the piece set, as a byte trie for Viterbi
    def _build_trie(self) -> None:
        """Index every non-control piece by its bytes, for the Viterbi segmentation.  A dict keyed
        by the piece's byte string is enough: the encoder tries every length up to the longest
        piece at each position, so no tree is needed and the lookup is one hash."""
        self._piece_id_at: dict[bytes, int] = {}
        for i, piece in enumerate(self.tokens):
            if self.token_type[i] == _TT_CONTROL:
                continue                       # controls are matched separately, not by Viterbi
            self._piece_id_at[piece.encode("utf-8")] = i
        self._max_piece_len = max((len(b) for b in self._piece_id_at), default=0)

    # ---- encode
    def encode(self, text: str, parse_special: bool = False) -> list[int]:
        if parse_special:
            return self._encode_with_specials(text)
        return self._encode_plain(self._normalize(text))

    def _encode_with_specials(self, text: str) -> list[int]:
        """Split on control markers (longest match first), Viterbi the plain runs between them."""
        out: list[int] = []
        buf = ""
        i = 0
        while i < len(text):
            hit = None
            for s in self.specials:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit is not None:
                if buf:
                    out += self._encode_plain(self._normalize(buf))
                    buf = ""
                out.append(self.id_of[hit])
                i += len(hit)
            else:
                buf += text[i]
                i += 1
        if buf:
            out += self._encode_plain(self._normalize(buf))
        return out

    def _encode_plain(self, text: str) -> list[int]:
        if self.add_space_prefix and text and not text.startswith(WORD_MARKER):
            text = WORD_MARKER + text
        data = text.encode("utf-8")
        ids = self._viterbi(data)
        if ids is None and self.byte_fallback:
            ids = self._byte_fallback(data)
        return ids

    def _viterbi(self, data: bytes) -> list[int] | None:
        """Highest-score segmentation of `data` into pieces.  Returns None if some byte cannot be
        covered (the caller may then byte-fallback)."""
        n = len(data)
        best = [None] * (n + 1)                 # (score, prev_index, piece_id)
        best[0] = (0.0, -1, -1)
        for j in range(n):
            if best[j] is None:
                continue
            base = best[j][0]
            for L in range(1, self._max_piece_len + 1):
                if j + L > n:
                    break
                pid = self._piece_id_at.get(data[j:j + L])
                if pid is None:
                    continue
                cand = base + self.scores[pid]
                if best[j + L] is None or cand > best[j + L][0]:
                    best[j + L] = (cand, j, pid)
        if best[n] is None:
            return None
        out: list[int] = []
        j = n
        while j > 0:
            out.append(best[j][2])
            j = best[j][1]
        out.reverse()
        return out

    def _byte_fallback(self, data: bytes) -> list[int]:
        """Cover every byte with a `<0xNN>` piece (or <unk> if the model has no byte pieces)."""
        unk = next((i for i, tt in enumerate(self.token_type) if tt == _TT_UNKNOWN), None)
        out: list[int] = []
        for b in data:
            pid = self.id_of.get(_BYTE_PIECE_FMT.format(b))
            if pid is None:
                if unk is None:
                    raise TokenizerError(f"cannot encode byte {b:#02x}: no byte piece and no <unk>")
                out.append(unk)
            else:
                out.append(pid)
        return out

    # ---- decode
    def token_bytes(self, tid: int) -> bytes:
        """One token's raw bytes, `▁` already mapped to a space, byte pieces to their byte.  This is
        what the server's incremental detokeniser feeds to a UTF-8 decoder."""
        piece = self.tokens[tid]
        tt = self.token_type[tid]
        if tt == _TT_CONTROL:
            return b"" if piece in ("<unk>", "<pad>") else piece.encode("utf-8")
        if piece in self.byte_piece:
            return bytes([self.byte_piece[piece]])
        return piece.replace(WORD_MARKER, " ").encode("utf-8")

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        raw = bytearray()
        for t in ids:
            raw += self.token_bytes(int(t))
        text = raw.decode("utf-8", errors=errors)
        # a leading space is the word marker of the first token; SentencePiece drops it
        return text[1:] if text.startswith(" ") else text

    # ---- normalisation (the ASSUMPTION lives here)
    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        if self.remove_extra_ws:
            text = " ".join(text.split())
        return text.replace(" ", WORD_MARKER)

    # ---- facts the server reads
    @property
    def eos_token_ids(self) -> set[int]:
        ids = set()
        if self.eos_id is not None:
            ids.add(int(self.eos_id))
        return ids


def _maybe_int(v) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None

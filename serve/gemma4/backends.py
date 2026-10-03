"""serve/gemma4/backends.py - what actually runs the forward pass, behind one interface.

The engine (`engine.py`) does not know or care how a forward pass happens; it asks a `Backend` for
the next token's logits.  That seam is where a faster implementation goes, exactly as
`serve/server.py` keeps `StrataEngine` and `MockEngine` behind one `Engine` protocol.

Registry
--------
Backends register by name; `get_backend(name, **kw)` builds one.  `numpy` is always present;
`torch` registers itself when torch imports.  A future backend adds itself with
`@register("cuda-graph")` in its own module and is selected with `--backend cuda-graph`; nothing
here changes and the numpy path keeps working if the fast one is absent.

What is here
------------
    NumpyBackend   the reference.  float32, everything in RAM, experts stacked.  Correct and
                   readable; it will run a real BF16/Q8_0 file on a machine with enough RAM, slowly.
                   Nobody should quote a tokens/s figure for it.
    TorchBackend   the same `model.py` over `TorchOps`.  Weights live on one device (GPU when there
                   is one).  `--expert-stream` keeps the routed experts in the GGUF and reads only
                   the 8-of-128 a token actually needs, which is what makes the 26B fit next to a
                   24 GB card's other work.

Both build the model through `build_model()`, so the two cannot disagree about which tensor means
what - there is one loader.
"""
from __future__ import annotations

import abc
import collections
import pathlib
from typing import Any, Sequence

import numpy as np

from .config import Gemma4Config, check_rope_freqs
from .model import (ExpertSource, Gemma4DecoderLayer, Gemma4ForCausalLM, Gemma4Model, Gemma4Router,
                    StackedExperts)
from .ops import NumpyOps, Ops, TorchOps


class BackendError(RuntimeError):
    """A backend could not be built or run (missing weights, unsupported quant, out of RAM)."""


class Backend(abc.ABC):
    """One loaded model that can produce logits for a run of token ids."""

    name = "abstract"

    @abc.abstractmethod
    def load(self) -> "Backend":
        """Read the weights and build the graph.  Idempotent."""

    @abc.abstractmethod
    def forward(self, ids: Sequence[int], start_pos: int) -> np.ndarray:
        """(n_tok, vocab) logits.  Returning every position (not just the last) lets a caller also
        teacher-force.  Appends to the KV cache."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget the KV cache: a new sequence starts from position 0."""

    def close(self) -> None:
        """Release memory.  Optional."""

    @property
    def loaded(self) -> bool:
        return getattr(self, "_loaded", False)


# ------------------------------------------------------------------ registry
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: makes `get_backend(name, ...)` able to build this backend."""
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs) -> Backend:
    if name not in _REGISTRY:
        raise BackendError(f"unknown backend {name!r}; available: {available_backends()}")
    return _REGISTRY[name](**kwargs)


# ------------------------------------------------------------------ the one loader
def _pick(g, layer: int, logical: str) -> str | None:
    return g.find_layer(layer, logical)


def load_layer_weights(g, cfg: Gemma4Config, layer: int, xp: Ops, *,
                       stack_experts: bool = True) -> dict:
    """One decoder layer's weights, in the roles `model.py` expects, on `xp`'s device.

    `stack_experts=False` leaves the expert tensors out of the dict; the caller supplies a streamed
    `ExpertSource` instead.  That is the whole difference between the two memory profiles."""
    t = cfg.text
    w: dict[str, Any] = {}

    def put(role: str, logical: str, required: bool = True):
        name = _pick(g, layer, logical)
        if name is None:
            if required:
                raise BackendError(f"layer {layer}: the file has no {logical} tensor")
            w[role] = None
            return
        w[role] = xp.asarray(g.read_tensor(name), _dtype(xp))

    w["attn_norm"] = xp.asarray(g.read_tensor(_pick(g, layer, "attn_norm")), _dtype(xp))
    put("q", "wq")
    put("q_norm", "q_norm")
    put("o", "wo")
    put("post_attn_norm", "post_attn_norm", required=False)
    if t.has_kv_proj(layer):
        put("k", "wk")
        put("k_norm", "k_norm")
        if t.has_v_proj(layer):
            put("v", "wv")
    put("ffn_norm", "ffn_norm")
    put("gate", "gate")
    put("up", "up")
    put("down", "down")
    put("post_ffn_norm", "post_ffn_norm")
    if t.enable_moe_block:
        put("router_w", "router_w")
        put("router_scale", "router_scale")
        put("expert_scale", "expert_scale")
        put("pre_ffn_norm_2", "pre_ffn_norm_2")
        put("post_ffn_norm_1", "post_ffn_norm_1")
        put("post_ffn_norm_2", "post_ffn_norm_2")
        if stack_experts:
            fused = _pick(g, layer, "gate_up_exps")
            if fused is not None:
                w["gate_up_exps"] = xp.asarray(g.read_tensor(fused), _dtype(xp))
            else:
                w["gate_exps"] = xp.asarray(g.read_tensor(_pick(g, layer, "gate_exps")), _dtype(xp))
                w["up_exps"] = xp.asarray(g.read_tensor(_pick(g, layer, "up_exps")), _dtype(xp))
            w["down_exps"] = xp.asarray(g.read_tensor(_pick(g, layer, "down_exps")), _dtype(xp))
    ls = _pick(g, layer, "layer_scale")
    w["layer_scale"] = float(g.read_tensor(ls).reshape(-1)[0]) if ls else None
    return w


def _dtype(xp: Ops):
    return getattr(xp, "compute", None) or np.float32


class _FileExperts(ExpertSource):
    """Reading code shared by the three profiles that leave the routed experts in the GGUF.

    This is what decides whether the 26B fits: the routed experts are ~22 B of the 25 B parameters,
    and a token uses 8 of 128 per layer.  `read_expert` slices the file directly, so a miss costs one
    seek plus one dequant, not a re-read of the layer - measured at 14.9 ms per expert (gate_up and
    down together, warm page cache, torch CPU) on gemma-4-26B-A4B Q4_0, 22.7 MiB dequantised.

    That 14.9 ms is the number every file-backed profile has to beat, and it is why all three keep a
    small LRU of single experts (`pick_keep`) for the per-expert path: a decode step touches 8 experts
    per layer and does ~0.6 ms of matmul with each, so re-reading one costs twenty times the work it
    feeds.  The LRU is what `--expert-stream` sizes today.

    Subclasses that also hold stacked tensors (blocks, a scratch buffer) reuse them here when the
    expert asked for happens to be inside one."""

    def __init__(self, g, layer: int, cfg: Gemma4Config, xp: Ops, pick_keep: int | None = None):
        self.g = g
        self.layer = layer
        self.cfg = cfg
        self.xp = xp
        t = cfg.text
        self.n_experts = int(t.num_experts)
        self.hidden = int(t.hidden_size)
        self._inter = int(t.moe_intermediate_size)
        self._gu_name = _pick(g, layer, "gate_up_exps")
        self._down_name = _pick(g, layer, "down_exps")
        self._gate_name = _pick(g, layer, "gate_exps") if self._gu_name is None else None
        self._up_name = _pick(g, layer, "up_exps") if self._gu_name is None else None
        self.pick_keep = default_pick_keep(cfg) if pick_keep is None else int(pick_keep)
        self.cache: collections.OrderedDict[int, tuple[Any, Any]] = collections.OrderedDict()
        self.reads = 0
        self.hits = 0

    def _slice(self, name: str, e: int):
        self.reads += 1
        return self.xp.asarray(self.g.read_expert(name, e), _dtype(self.xp))

    def _read_pair(self, e: int):
        """One expert straight from the file: (gate_up (hidden, 2*inter), down (inter, hidden))."""
        dn = self._slice(self._down_name, e)
        if self._gu_name is not None:
            return self._slice(self._gu_name, e), dn
        return (self.xp.concat_axis1(self._slice(self._gate_name, e),
                                     self._slice(self._up_name, e)), dn)

    def _alloc(self, n: int):
        """Empty stacked buffers for `n` experts, EXPERT AXIS FIRST: gate_up [n, hidden, 2*inter],
        down [n, inter, hidden].

        The file stores the expert axis last, and so does `StackedExperts`, but a buffer that is
        refilled expert by expert wants it first: writing one expert into `dst[:, :, i]` writes into a
        strided slice and measured 3.7 s over 128 experts against 1.9 s for the reads themselves
        (gemma-4-26B-A4B Q4_0, torch CPU).  Expert axis first also hands `matmul3` its [B, k, n]
        without a permute."""
        dt = _dtype(self.xp)
        return (self.xp.zeros((n, self.hidden, 2 * self._inter), dt),
                self.xp.zeros((n, self._inter, self.hidden), dt))

    def _fill(self, gu_buf, dn_buf, lo: int, hi: int) -> None:
        """Dequantise experts [lo, hi) into buffers that are already allocated."""
        for i, e in enumerate(range(lo, hi)):
            g_mat, d_mat = self._read_pair(e)
            self.xp.store_row(gu_buf, i, g_mat)
            self.xp.store_row(dn_buf, i, d_mat)

    def _cached(self, e: int):
        """The pair for expert `e` from the per-expert LRU, reading it if it is not there."""
        ent = self.cache.get(e)
        if ent is None:
            ent = self._read_pair(e)
            self.cache[e] = ent
            while len(self.cache) > max(1, self.pick_keep):
                self.cache.popitem(last=False)
        else:
            self.hits += 1
            self.cache.move_to_end(e)
        return ent

    def _pick(self, e: int):
        """The pair for expert `e`, preferring a stacked tensor a subclass already holds."""
        return self._cached(e)

    def gate_up(self, e: int):
        return self._pick(e)[0]

    def down(self, e: int):
        return self._pick(e)[1]

    def close(self) -> None:
        self.cache.clear()


class StreamedExperts(_FileExperts):
    """An LRU of `keep` single experts, never batched.

    It is deliberately NOT batchable (`batched_views = False`): there is no stacked tensor on device
    to take a view of, and building one costs more than the matmul it would feed - stacking one
    layer's 128 experts measured 1311 ms against the ~100 ms of matmul at 128 tokens.  Batching here
    would also read experts the batch may not need.  `BlockedExperts` and `ScratchExperts` are the
    profiles that do batch."""

    batched_views = False

    def __init__(self, g, layer: int, cfg: Gemma4Config, xp: Ops, keep: int = 24):
        super().__init__(g, layer, cfg, xp, pick_keep=keep)


class BlockedExperts(_FileExperts):
    """Holds the experts in aligned blocks of `block`, stacked, and batches inside one block.

    The block is the unit of both memory and batching: `aligned_groups` tells the MoE a group may
    never cross a block boundary, so `gate_up_range` is always a view of a block that is already
    resident - no copy at matmul time.  Cost is `keep` blocks per layer: one expert of
    gemma-4-26B-A4B is 22.7 MiB in float32, so block=24, keep=1 is 544 MiB per layer, 16.0 GiB for
    the model in float32 and 8.0 GiB in float16.

    The single-expert path does NOT build a block - a decode step picks 8 experts out of 128, and
    building a block per pick would read 24 experts to use one.  It takes the expert from a resident
    block when one already covers it, and otherwise falls back to the per-expert LRU."""

    batched_views = True
    aligned_groups = True

    def __init__(self, g, layer: int, cfg: Gemma4Config, xp: Ops, block: int = 24, keep: int = 1,
                 pick_keep: int | None = None):
        super().__init__(g, layer, cfg, xp, pick_keep=pick_keep)
        self.block = max(1, int(block))
        self.keep = max(1, int(keep))
        self.max_group = self.block
        self.blocks: collections.OrderedDict[int, tuple[Any, Any]] = collections.OrderedDict()
        self.built = 0

    def _block(self, b: int):
        ent = self.blocks.get(b)
        if ent is None:
            lo = b * self.block
            hi = min(lo + self.block, self.n_experts)
            gu_buf, dn_buf = self._alloc(hi - lo)
            self._fill(gu_buf, dn_buf, lo, hi)
            ent = (gu_buf, dn_buf)
            self.blocks[b] = ent
            self.built += 1
            while len(self.blocks) > self.keep:
                self.blocks.popitem(last=False)
        else:
            self.hits += 1
            self.blocks.move_to_end(b)
        return ent

    def _pick(self, e: int):
        b = e // self.block
        ent = self.blocks.get(b)
        if ent is None:
            return self._cached(e)
        self.hits += 1
        i = e - b * self.block
        return ent[0][i], ent[1][i]

    def _block_and_offsets(self, lo: int, hi: int):
        b = lo // self.block
        if hi > (b + 1) * self.block:
            raise ValueError(f"BlockedExperts: range [{lo}, {hi}) crosses a block of {self.block}")
        return self._block(b), lo - b * self.block, hi - b * self.block

    def gate_up_range(self, lo, hi):
        (gu_buf, _), o_lo, o_hi = self._block_and_offsets(lo, hi)
        return gu_buf[o_lo:o_hi]

    def down_range(self, lo, hi):
        (_, dn_buf), o_lo, o_hi = self._block_and_offsets(lo, hi)
        return dn_buf[o_lo:o_hi]

    def close(self) -> None:
        self.blocks.clear()
        super().close()


class ScratchExperts(_FileExperts):
    """One reusable buffer pair per layer, refilled with each group of experts.

    Nothing is kept between groups beyond the per-expert LRU, so the footprint is `group` experts per
    layer on top of that LRU: group=8 is 182 MiB per layer in float32, 2.7 GiB in float16.  The price
    is a copy the other profiles do not pay - measured at 12.5 ms to stack a group of 8 (182 MiB) on
    torch CPU, against the 12 ms the batched matmul saves at 128 tokens.  It is paid once per group
    per layer, so it only pays off when a group carries enough rows.

    `gate_up_range` and `down_range` are called in sequence for the same group, and the second must
    not overwrite what the first handed back: the buffer is refilled only when the range changes."""

    batched_views = True
    aligned_groups = False

    def __init__(self, g, layer: int, cfg: Gemma4Config, xp: Ops, group: int = 8,
                 pick_keep: int | None = None):
        super().__init__(g, layer, cfg, xp, pick_keep=pick_keep)
        self.group = max(2, int(group))
        self.max_group = self.group
        self._buf_gu = None
        self._buf_dn = None
        self._span: tuple[int, int] | None = None
        self.fills = 0

    def _ensure(self, lo: int, hi: int) -> None:
        if hi - lo > self.group:
            raise ValueError(f"ScratchExperts: range [{lo}, {hi}) is wider than the buffer "
                             f"of {self.group}")
        if self._buf_gu is None:
            self._buf_gu, self._buf_dn = self._alloc(self.group)
        if self._span != (lo, hi):
            self._fill(self._buf_gu, self._buf_dn, lo, hi)
            self._span = (lo, hi)
            self.fills += 1
        else:
            self.hits += 1

    def gate_up_range(self, lo, hi):
        self._ensure(lo, hi)
        return self._buf_gu[:hi - lo]

    def down_range(self, lo, hi):
        self._ensure(lo, hi)
        return self._buf_dn[:hi - lo]

    def close(self) -> None:
        self._buf_gu = self._buf_dn = None
        self._span = None
        super().close()


EXPERT_MODES = ("stack", "stream", "block", "scratch")
#: `auto` is not a profile of its own: it asks `auto_expert_profile` to pick one of the four for the
#: memory actually available, which is the only decision the measurements in `TorchBackend` support.
EXPERT_MODE_CHOICES = EXPERT_MODES + ("auto",)


def auto_expert_profile(cfg: Gemma4Config, budget_bytes: int, *, dtype_ratio: float = 0.5,
                        reserve_bytes: int = 2 * 2**30) -> dict:
    """The expert profile a given budget can actually keep resident, as `build_model` kwargs.

    The measurements behind the choice (`bench/results/2026-10-03-expert-residency/`) are unambiguous:
    a layer whose experts are ALL resident runs a MoE forward in 95-167 ms at 128 tokens, and any
    profile below that runs 2.1-3.3 s.  So there are only two regimes, and the budget decides which:

      * all the experts fit -> `stack`.  Note `block` with `expert_block * expert_blocks >= n_experts`
        costs exactly the same bytes (it holds every expert too, just built lazily), so residency is
        not a choice between stack and block - it is a choice between residency and not.  `stack`
        measured 95 ms at 128 tokens against block's 110-167 ms.
      * they do not -> `stream`, sized to whatever fits.  It measured the fastest of the partial
        profiles at equal memory (2109 ms against block's 3252 ms for 48 experts per layer), because
        block pays a stacked copy per forward that the batching never pays back.

    `budget_bytes` is in the dtype being loaded, so `dtype_ratio` scales the float32 estimates, and
    `reserve_bytes` stays unspent for the KV cache and activations - a profile that fits the weights
    exactly and then cannot allocate a cache is worse than one that runs 20x slower."""
    t = cfg.text
    if not t.enable_moe_block:
        return {"expert_mode": "stack"}
    n_exp = int(t.num_experts)
    non_expert = estimate_bytes(cfg) - expert_resident_bytes(cfg, "stack")
    experts_all = expert_resident_bytes(cfg, "stack")
    avail = int(budget_bytes) - int(non_expert * dtype_ratio) - int(reserve_bytes)
    if experts_all * dtype_ratio <= avail:
        return {"expert_mode": "stack"}
    n_layers = int(t.num_hidden_layers)
    per_expert = experts_all // (n_exp * n_layers)
    # `keep` is per LAYER and every layer has its own LRU, so the budget buys keep * n_layers experts;
    # dividing by one expert's bytes alone would size it n_layers times too big.
    per_keep = max(1, int(per_expert * dtype_ratio)) * n_layers
    keep = avail // per_keep
    keep = max(default_pick_keep(cfg), min(keep, n_exp - 1))
    return {"expert_mode": "stream", "expert_stream": keep}


def build_model(g, cfg: Gemma4Config, xp: Ops, *, expert_mode: str = "stack",
                expert_stream: int = 24, expert_block: int = 8, expert_blocks: int = 4,
                expert_group: int = 8) -> Gemma4ForCausalLM:
    """Assemble a `Gemma4ForCausalLM` from an open GGUF.  One loader for every backend, so the
    backends cannot disagree about which tensor means what.

    `expert_mode` picks where the routed experts live; see `EXPERT_MODES`.  `stack` puts them in the
    weight dict (the reference, and the only one that can batch every consecutive run); the other
    three leave them in the file and differ in what they keep on device:

        stream   an LRU of `expert_stream` single experts, never batched
        block    `expert_blocks` aligned blocks of `expert_block` experts, batched inside a block
        scratch  one reusable buffer of `expert_group` experts, refilled per group
    """
    if expert_mode not in EXPERT_MODES:
        raise BackendError(f"unknown expert mode {expert_mode!r}; available: {list(EXPERT_MODES)}")
    from . import gguf as _gguf
    problems = _gguf.validate(g, cfg)
    if problems:
        raise BackendError("this GGUF does not match the Gemma 4 architecture:\n  " +
                           "\n  ".join(problems[:20]) + ("\n  ..." if len(problems) > 20 else ""))
    t = cfg.text
    rf = g.read_optional(_pick_model(g, "rope_freqs"))
    factors = np.asarray(rf).reshape(-1).tolist() if rf is not None else None
    if factors is not None:
        warn = check_rope_freqs(factors, t.rope_full.inv_freq(t.global_head_dim))
        if warn:
            print(f"[gemma4] warning: {warn}", flush=True)
    layers = []
    for layer in range(t.num_hidden_layers):
        spec = t.layer_spec(layer, factors)
        w = load_layer_weights(g, cfg, layer, xp, stack_experts=(expert_mode == "stack"))
        router = Gemma4Router(w, t, xp=xp) if t.enable_moe_block and "router_w" in w else None
        experts = None
        if t.enable_moe_block:
            if expert_mode == "stack":
                experts = StackedExperts(w, t.moe_intermediate_size, xp=xp)
            elif expert_mode == "stream":
                # keep >= 1: gate_up and down are called back to back for the same expert, and an
                # LRU of zero would drop the pair between the two calls
                experts = StreamedExperts(g, layer, cfg, xp, keep=max(1, expert_stream))
            elif expert_mode == "block":
                experts = BlockedExperts(g, layer, cfg, xp, block=expert_block, keep=expert_blocks)
            else:
                experts = ScratchExperts(g, layer, cfg, xp, group=expert_group)
        layers.append(Gemma4DecoderLayer(t, spec, w, xp=xp, router=router, experts=experts))
    embed = xp.asarray(g.read_tensor(_pick_model(g, "embed")), _dtype(xp))
    final_norm = xp.asarray(g.read_tensor(_pick_model(g, "output_norm")), _dtype(xp))
    out_name = _pick_model(g, "output")
    lm_head = xp.asarray(g.read_tensor(out_name), _dtype(xp)) if out_name else embed
    model = Gemma4Model(t, layers, embed, final_norm, xp=xp)
    return Gemma4ForCausalLM(cfg, model, lm_head, xp=xp)


def _pick_model(g, logical: str) -> str | None:
    from .gguf import MODEL_ALIASES
    return g.find(MODEL_ALIASES[logical])


def estimate_bytes(cfg: Gemma4Config) -> int:
    """Approximate float32 footprint of the weights, from the config's shapes (no file read).

    A dense Gemma 4 (`enable_moe_block` off, e.g. gemma-4-12B) ships no router and no expert stack;
    counting them would overstate the need by an order of magnitude and make the RAM guard refuse a
    model that fits."""
    t = cfg.text
    per_layer = (
        t.hidden_size * t.q_proj_out(0)                       # q (upper bound; full layers wider)
        + 2 * t.hidden_size * t.kv_proj_out(0)                # k, v
        + t.q_proj_out(0) * t.hidden_size                     # o
        + 3 * t.hidden_size * t.intermediate_size             # the dense MLP (shared expert)
        + 8 * t.hidden_size                                   # the six-plus norms
    )
    if t.enable_moe_block:
        per_layer += (t.hidden_size * t.num_experts           # router
                      + 3 * t.hidden_size * t.moe_intermediate_size * t.num_experts)  # routed experts
    emb = t.hidden_size * t.vocab_size * (1 if t.tie_word_embeddings else 2)
    return (per_layer * t.num_hidden_layers + emb) * 4        # float32


def default_pick_keep(cfg: Gemma4Config) -> int:
    """Single experts a file-backed mode keeps for the per-expert path.

    Two decode steps' worth: a step picks `top_k` per layer, and re-reading one costs 14.9 ms against
    the 0.6 ms of matmul it feeds (gemma-4-26B-A4B Q4_0, torch CPU), so this cache is not optional."""
    return max(2, 2 * int(cfg.text.top_k_experts))


def expert_resident_bytes(cfg: Gemma4Config, mode: str, *, expert_stream: int = 24,
                          expert_block: int = 8, expert_blocks: int = 4,
                          expert_group: int = 8) -> int:
    """What the routed experts cost, in float32 bytes, for one expert mode.

    `stack` holds all of them: 2.84 GiB per layer on gemma-4-26B-A4B, 85.1 GiB for the model, which
    is why it is not the profile for a 24 GB card.  The other three keep a bounded amount per layer -
    one expert is 22.7 MiB in float32 (15.1 gate_up + 7.6 down) - and every one of them also keeps the
    per-expert LRU, which is what makes a decode step readable rather than 3.6 s long."""
    t = cfg.text
    if not t.enable_moe_block:
        return 0
    n_layers = int(t.num_hidden_layers)
    n_exp = int(t.num_experts)
    per_expert = 3 * int(t.hidden_size) * int(t.moe_intermediate_size) * 4
    all_experts = per_expert * n_exp * n_layers
    if mode == "stack":
        return all_experts
    if mode == "stream":
        kept = min(int(expert_stream), n_exp)
    elif mode == "block":
        kept = min(default_pick_keep(cfg) + int(expert_block) * int(expert_blocks), n_exp)
    elif mode == "scratch":
        kept = min(default_pick_keep(cfg) + int(expert_group), n_exp)
    else:
        raise BackendError(f"unknown expert mode {mode!r}; available: {list(EXPERT_MODES)}")
    return per_expert * kept * n_layers


def free_ram_bytes() -> int | None:
    """Available RAM, or None when it cannot be determined (then the load just tries)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ the reference backend
@register("numpy")
class NumpyBackend(Backend):
    """Runs `model.py` in float32.  Build it from a GGUF (`path=...`) for a real model, or from an
    already-assembled `Gemma4ForCausalLM` (`model=...`) for a test."""

    def __init__(self, path: str | pathlib.Path | None = None, *, config: Gemma4Config | None = None,
                 model: Gemma4ForCausalLM | None = None, max_ram_gb: float = 0.0,
                 expert_mode: str = "auto", expert_stream: int = 24, expert_block: int = 8,
                 expert_blocks: int = 4, expert_group: int = 8):
        self.path = pathlib.Path(path) if path else None
        self.config = config
        self._model = model
        self.max_ram_gb = float(max_ram_gb)      # 0 = no guard (a test, or the user knows best)
        self.expert_mode = expert_mode
        self.expert_stream = int(expert_stream)
        self.expert_block = int(expert_block)
        self.expert_blocks = int(expert_blocks)
        self.expert_group = int(expert_group)
        self._loaded = model is not None
        self.xp = NumpyOps()

    def _resolve_expert_mode(self) -> None:
        """`auto` for a backend that only has system RAM: float32, so the ratio is 1.0."""
        if self.expert_mode != "auto":
            return
        budget = free_ram_bytes()
        if budget is None:
            self.expert_mode, self.expert_stream = "stream", default_pick_keep(self.config)
            return
        kw = auto_expert_profile(self.config, budget, dtype_ratio=1.0)
        self.expert_mode = kw["expert_mode"]
        self.expert_stream = kw.get("expert_stream", self.expert_stream)
        sized = ", ".join(f"{k}={v}" for k, v in kw.items() if k != "expert_mode")
        print(f"[gemma4] expert mode auto -> {self.expert_mode}"
              f"{f' ({sized})' if sized else ''}, from {budget / 2**30:.1f} GiB free RAM", flush=True)

    @staticmethod
    def from_model(model: Gemma4ForCausalLM) -> "NumpyBackend":
        """Wrap an assembled model (used by the unit tests, which build a tiny one in memory)."""
        return NumpyBackend(model=model, config=model.config)

    def load(self) -> "NumpyBackend":
        if self._loaded:
            return self
        if self._model is not None:
            self._loaded = True
            return self
        if self.path is None:
            raise BackendError("NumpyBackend needs a path or a model")
        from . import gguf as _gguf
        g = _gguf.Gemma4GGUF.open(self.path)
        self.config = self.config or g.config()
        self._resolve_expert_mode()
        need = (estimate_bytes(self.config)
                - expert_resident_bytes(self.config, "stack")
                + expert_resident_bytes(self.config, self.expert_mode,
                                        expert_stream=self.expert_stream,
                                        expert_block=self.expert_block,
                                        expert_blocks=self.expert_blocks,
                                        expert_group=self.expert_group))
        free = free_ram_bytes()
        if need and free is not None and free < need + self.max_ram_gb * 1e9:
            g.close()
            raise BackendError(
                f"this backend would load ~{need / 1e9:.0f} GB of float32 weights and the machine "
                f"has ~{free / 1e9:.0f} GB free. Use a smaller quant, --expert-mode block or "
                f"scratch, or the torch backend on a GPU.")
        try:
            self._model = build_model(g, self.config, self.xp, expert_mode=self.expert_mode,
                                      expert_stream=self.expert_stream,
                                      expert_block=self.expert_block,
                                      expert_blocks=self.expert_blocks,
                                      expert_group=self.expert_group)
        except _gguf.UnsupportedQuant as e:
            g.close()
            raise BackendError(str(e)) from e
        if self.expert_mode == "stack":
            g.close()                     # every weight is already in RAM
        else:
            self._gguf = g                # the expert source still reads through it
        self._loaded = True
        return self

    def forward(self, ids: Sequence[int], start_pos: int) -> np.ndarray:
        if not self._loaded:
            self.load()
        return self._model.forward(ids, start_pos)

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_cache()

    def close(self) -> None:
        # Keep a model that was handed in (a test, an in-memory build): `load()` must be able to
        # bring it back, as `Service.ensure_loaded` expects after an unload.  A file-backed model
        # is dropped, because reloading it from disk is what `load()` does.
        if self._model is not None and self.path is not None:
            self._model.close()
            self._model = None
        if getattr(self, "_gguf", None) is not None:      # a file-backed expert source still reads it
            self._gguf.close()
            self._gguf = None
        self._loaded = False


# ------------------------------------------------------------------ the fast backend
@register("torch")
class TorchBackend(Backend):
    """The same `model.py`, on torch.  `device='cuda'` when there is a GPU.

    Where the routed experts live is `expert_mode` (see `EXPERT_MODES`); everything else - embeddings,
    attention, the dense MLP, the router - is always resident:

      * `stack`: every weight is dequantised to `dtype` and lives on the device.  A Q4_K_M 26B is
        ~17 GB on disk and ~50 GB in float32, so this wants a big card or `dtype='bfloat16'`.
      * `stream`: an LRU of `expert_stream` single experts.  Nothing is ever batched.
      * `block` (the default): `expert_blocks` aligned blocks of `expert_block` experts per layer,
        stacked, and the MoE batches inside a block.
      * `scratch`: one reusable buffer of `expert_group` experts per layer, refilled per group, and
        the only mode whose footprint does not grow with the block size.

    What the modes actually cost, measured on gemma-4-26B-A4B Q4_0 layer 0, torch CPU float32, one
    process per config so no earlier allocation could distort the next timing
    (`bench/results/2026-10-03-expert-residency/one-mode.txt`, best of up to 5 repeats, the routing
    touching every expert of the layer once):

        stack                    95 ms at 128 tok, 147 ms at 512 tok   128/128 experts resident
        block 16 keep=8        100 ms                132 ms           128/128
        stream keep=128        107 ms                158 ms           128/128
        block 8 keep=16        110 ms                167 ms           128/128
        stream keep=48        2109 ms               2216 ms            48/128
        stream keep=24        2151 ms               2191 ms            24/128
        scratch group=16      2763 ms               2835 ms            32/128
        scratch group=32      2823 ms               2823 ms            48/128
        block 16 keep=2       3178 ms               3185 ms            48/128
        block 8 keep=4        3252 ms               3320 ms            48/128
        block 8 keep=1        3287 ms               3337 ms            24/128

    Two things the table says, and both change what the default should be.

    1. The split is between resident and not, not between the modes.  The four resident profiles land
       in 95-167 ms and the seven partial ones in 2.1-3.3 s: a ~20x gap, and the label on the row does
       not decide which side of it it falls on.  Reading and dequantising one expert is 14.9 ms
       (22.7 MiB), so a forward touching all 128 pays ~1.9 s of reads however the matmuls are grouped.
       The batching is worth 1.1-1.2x on top of residency - real, and it moves `block 16 keep=8` past
       `stack` (132 ms against 147 ms at 512 tokens), but it does not close the gap.  Residency is not
       a warm-up detail either: over four turns (512-token prefill + 64 decode steps each, skewed
       routing) `block 8 keep=4` still read 3522 times across the last three turns - 3.2 s per prefill,
       5.4 s per decode - against 171 ms and 383 ms with the whole layer resident (`cross-forward.txt`).
       The per-expert LRU cannot rescue a partial footprint, because a prefill touches every expert.

    2. Below residency, `block` is the WORST of the three, at equal memory.  Holding 48 experts per
       layer costs `stream keep=48` 2109 ms, `scratch group=32` 2823 ms and `block 8 keep=4` 3252 ms
       (`equal-memory.txt`): block is 1.55x slower than stream for the same bytes.  The reason is in
       `_fill` - a block copies every expert into a stacked buffer (`store_row`, measured at 3.7 s over
       a layer's 128 experts against 1.9 s for the reads), and with `keep` small that copy is repeated
       every forward (`built=32`) while the batching it pays for never happens, because the groups the
       router produces are spread across more blocks than are resident.

    So the default is chosen by what fits, and `block` is only the right answer when it fits ENTIRELY:
    `expert_block * expert_blocks >= num_experts`.  One expert is 22.7 MiB float32 / 11.4 MiB float16,
    and the model's 30 layers hold 128 each, so all of them is 85.1 GiB float32 / 42.5 GiB float16 on
    top of 8.7 GiB / 4.4 GiB for everything else.  A 24 GB card cannot reach residency in any dtype, so
    `auto_expert_profile` picks `stream` there and sizes `--expert-stream` from the free bytes;
    `_warn_partial_residency` says out loud which regime the user bought."""

    def __init__(self, path: str | pathlib.Path | None = None, *, config: Gemma4Config | None = None,
                 device: str = "auto", dtype: str = "float16", expert_mode: str = "auto",
                 expert_stream: int = 24, expert_block: int = 8, expert_blocks: int = 4,
                 expert_group: int = 8, model: Gemma4ForCausalLM | None = None):
        self.path = pathlib.Path(path) if path else None
        self.config = config
        self.device = device
        self.dtype_name = dtype
        self.expert_mode = expert_mode
        self.expert_stream = int(expert_stream)
        self.expert_block = int(expert_block)
        self.expert_blocks = int(expert_blocks)
        self.expert_group = int(expert_group)
        self._model = model
        self._loaded = model is not None
        self.xp = None

    def _ops(self) -> TorchOps:
        import torch
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev == "cuda" and not torch.cuda.is_available():
            raise BackendError("--device cuda but torch has no CUDA device; use --device cpu")
        dt = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}.get(self.dtype_name)
        if dt is None:
            raise BackendError(f"unknown dtype {self.dtype_name!r}; "
                               f"use float32, float16 or bfloat16")
        return TorchOps(device=dev, dtype=dt)

    def _dtype_ratio(self) -> float:
        return {"float32": 1.0, "float16": 0.5, "bfloat16": 0.5}[self.dtype_name]

    def _free_device_bytes(self) -> int | None:
        """What the device the weights go on can actually give us, or None when it cannot be asked.

        A CUDA device that is already full makes `mem_get_info` itself raise
        `torch.AcceleratorError: CUDA error: out of memory` - torch cannot even ask. That is not "I do
        not know", it is "there is nothing", and reporting it as unknown would have `auto` pick a
        profile and then die in the allocator. So an OOM from the query means 0 free; only a query that
        fails for another reason (no driver, unsupported device) is unknown."""
        if self.xp is None:
            return None
        if self.xp.device.type == "cpu":
            return free_ram_bytes()
        try:
            import torch
            return int(torch.cuda.mem_get_info()[0])
        except Exception as e:
            if "out of memory" in str(e).lower():
                return 0
            return None

    def _resolve_expert_mode(self) -> None:
        """Turn `expert_mode='auto'` into one of the four, sized from the memory actually free.

        Called before the preflight, because the profile decides how much is needed: picking `block`
        where only `stream` fits is how a load dies with an allocator error instead of running."""
        if self.expert_mode != "auto":
            return
        budget = self._free_device_bytes()
        if budget is None:
            self.expert_mode, self.expert_stream = "stream", default_pick_keep(self.config)
            print(f"[gemma4] expert mode auto: cannot measure free memory, using "
                  f"stream keep={self.expert_stream}", flush=True)
            return
        kw = auto_expert_profile(self.config, budget, dtype_ratio=self._dtype_ratio())
        self.expert_mode = kw["expert_mode"]
        self.expert_stream = kw.get("expert_stream", self.expert_stream)
        self.expert_block = kw.get("expert_block", self.expert_block)
        self.expert_blocks = kw.get("expert_blocks", self.expert_blocks)
        sized = ", ".join(f"{k}={v}" for k, v in kw.items() if k != "expert_mode")
        print(f"[gemma4] expert mode auto -> {self.expert_mode}"
              f"{f' ({sized})' if sized else ''}, from {budget / 2**30:.1f} GiB free on "
              f"{self.xp.device}", flush=True)

    def _resident_bytes(self) -> int:
        """Float32 bytes the weights would occupy in this profile: everything resident plus whatever
        the expert mode keeps on device."""
        return (estimate_bytes(self.config)
                - expert_resident_bytes(self.config, "stack")
                + expert_resident_bytes(self.config, self.expert_mode,
                                        expert_stream=self.expert_stream,
                                        expert_block=self.expert_block,
                                        expert_blocks=self.expert_blocks,
                                        expert_group=self.expert_group))

    def _preflight(self, torch) -> None:
        """Say how much the device would need, before the CUDA allocator says it in a stack trace."""
        if getattr(self.xp, "device", None) is None or self.xp.device.type == "cpu":
            return
        ratio = self._dtype_ratio()
        need = self._resident_bytes() * ratio
        try:
            free, cap = torch.cuda.mem_get_info()
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise            # a real bug, not a full card; do not dress it up as one
            # A device this full cannot even answer the question, which is itself the answer. torch's
            # own text runs to several lines (an error code plus a docs URL); keep the first.
            why = str(e).strip().splitlines()[0]
            raise BackendError(
                f"{self.xp.device} would not say how much is free ({why}). Something already holds "
                f"the whole card - this profile needs ~{need / 1e9:.1f} GB. Free the device, use "
                f"--device cpu, or stop the process that owns the card.") from e
        if need > free:
            floor = self._resident_bytes_floor() * ratio
            if floor < free:
                hint = f"--expert-mode auto would need ~{floor / 1e9:.1f} GB"
            else:
                hint = "no expert mode fits this device as configured"
            # only a block profile has block knobs to turn; telling a `stack` user to lower
            # --expert-block sends them somewhere that does nothing
            knobs = ("--expert-mode auto" if self.expert_mode == "stack"
                     else "lower --expert-block or --expert-blocks")
            raise BackendError(
                f"this profile needs ~{need / 1e9:.1f} GB on {self.xp.device} and only "
                f"{free / 1e9:.1f} GB is free (of {cap / 1e9:.1f} GB). {hint}: {knobs}, "
                f"use --dtype float16, or free the device.")

    def _resident_bytes_floor(self) -> int:
        """The same estimate with the smallest expert footprint of the three file-backed modes.  Used
        only to make the refusal say what would fit.  It is `stream` at the auto-sized keep, not
        `scratch`: scratch also carries the per-expert LRU, so it is never the smaller of the two."""
        smallest = min(expert_resident_bytes(self.config, m, expert_group=self.expert_group,
                                             expert_block=self.expert_block,
                                             expert_blocks=self.expert_blocks,
                                             expert_stream=self.expert_stream)
                       for m in ("stream", "block", "scratch"))
        return estimate_bytes(self.config) - expert_resident_bytes(self.config, "stack") + smallest

    def load(self) -> "TorchBackend":
        if self._loaded:
            return self
        if self.path is None:
            raise BackendError("TorchBackend needs a path")
        from . import gguf as _gguf
        import torch
        self.xp = self._ops()
        g = _gguf.Gemma4GGUF.open(self.path)
        self.config = self.config or g.config()
        self._resolve_expert_mode()
        self._preflight(torch)
        try:
            self._model = build_model(g, self.config, self.xp, expert_mode=self.expert_mode,
                                      expert_stream=self.expert_stream,
                                      expert_block=self.expert_block,
                                      expert_blocks=self.expert_blocks,
                                      expert_group=self.expert_group)
        except _gguf.UnsupportedQuant as e:
            g.close()
            raise BackendError(str(e)) from e
        self._gguf = g            # a file-backed expert source still reads through it
        self._loaded = True
        self._warn_partial_residency()
        return self

    def _resident_experts(self) -> int:
        """How many of a layer's experts this profile keeps, from the same arithmetic the preflight
        uses, so the note cannot disagree with the number that sized the profile.

        `expert_resident_bytes` counts experts across ALL layers, so the divisor is one expert's
        bytes times the layer count - dividing by the expert size alone would report 12 of 8."""
        t = self.config.text
        n_exp, n_layers = int(t.num_experts), int(t.num_hidden_layers)
        per_expert = expert_resident_bytes(self.config, "stack") // (n_exp * n_layers)
        kept = expert_resident_bytes(self.config, self.expert_mode,
                                     expert_stream=self.expert_stream,
                                     expert_block=self.expert_block,
                                     expert_blocks=self.expert_blocks,
                                     expert_group=self.expert_group)
        return kept // max(1, per_expert * n_layers)

    def _warn_partial_residency(self) -> None:
        """Tell the user when the profile cannot keep a layer's experts resident, because that is the
        difference between ~150 ms and ~3 s per MoE layer, and no flag in this file closes it.

        Measured on gemma-4-26B-A4B Q4_0, torch CPU float32: every profile with all 128 experts of a
        layer resident runs 95-167 ms at 128 tokens, every profile below that runs 2.1-3.3 s, and an
        LRU does not help across turns (see `TorchBackend`'s docstring).  So a partial footprint is not
        a slower fast path - it is a different, ~20x worse regime, and the user should know which one
        they bought."""
        if self.expert_mode == "stack" or self.config is None:
            return
        t = self.config.text
        if not t.enable_moe_block:
            return
        resident = self._resident_experts()
        if resident >= int(t.num_experts):
            return
        ratio = self._dtype_ratio()
        all_experts = expert_resident_bytes(self.config, "stack") * ratio / 2**30
        print(f"[gemma4] note: {self.expert_mode} keeps {resident} of {t.num_experts} experts per "
              f"layer, so every forward re-reads the rest. Measured on this model that costs ~20x "
              f"(~3 s vs ~0.15 s per layer at 128 tokens), and no amount of raising "
              f"--expert-block/--expert-blocks changes it: a block holding all {t.num_experts} experts "
              f"is the same {all_experts:.1f} GiB of experts in {self.dtype_name} as --expert-mode "
              f"stack, which measured faster. Raise --dtype float16, use a bigger card, or accept the "
              f"~20x.", flush=True)

    def forward(self, ids: Sequence[int], start_pos: int):
        if not self._loaded:
            self.load()
        return self._model.forward(ids, start_pos)

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_cache()

    def close(self) -> None:
        if self._model is not None and self.path is not None:
            self._model.close()
            self._model = None
            if getattr(self, "_gguf", None) is not None:
                self._gguf.close()
                self._gguf = None
        self._loaded = False


def backend_supports(path: str | pathlib.Path) -> tuple[bool, str]:
    """(can_run, why) for a file, without loading it - what `--model` checks before starting."""
    from . import gguf as _gguf
    try:
        g = _gguf.Gemma4GGUF.open(path)
    except Exception as e:
        return False, str(e)
    try:
        if g.mtp() is not None:
            return False, "this is an MTP drafter (gemma4-assistant), not a runnable target"
        cfg = g.config()
        bad = [n for n, t in g._index.items() if t.type_name not in _gguf._DEQUANTABLE]
        if bad:
            encs = sorted({g._index[n].type_name for n in bad})
            return False, (f"{len(bad)} tensors use {encs}, which this build cannot decode; "
                           f"supported: {sorted(_gguf._DEQUANTABLE)}")
        probs = _gguf.validate(g, cfg)
        if probs:
            return False, "; ".join(probs[:3])
        return True, _gguf.describe(g)
    finally:
        g.close()

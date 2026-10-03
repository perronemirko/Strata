"""serve/gemma4/model.py - the Gemma 4 26B A4B architecture, as classes over one array vocabulary.

This is the specification made executable.  Every step is transcribed from two independent
implementations of the same model and cross-checked against the artifact:

  * `transformers/src/transformers/models/gemma4/modeling_gemma4.py` (Google's reference), and
  * `llama.cpp/src/models/gemma4.cpp` (the graph that runs the GGUFs this package loads).

Where the two agree, the code follows them.  Where a fact can only come from the file - tensor
names, the `rope_freqs.weight` sentinel - it comes from the file, and the comment says so.

The pieces, top to bottom
-------------------------
    rms_norm                pre-norm; Gemma 4 multiplies by the weight, NOT by (1 + weight)
    rope_tables / apply_rotary   per-layer theta, proportional rope with NoPE channels
    Gemma4Attention         GQA + QK-norm + scaleless V-norm; sliding OR full; K==V on full layers
    Gemma4Router            scaleless RMSNorm -> *scale/sqrt(hidden) -> softmax -> top-k -> *per-expert
    Gemma4MoE               dense MLP as the shared expert + top-8 of 128 routed experts
    Gemma4DecoderLayer      the norm sandwich the reference actually uses (six norms per layer)
    Gemma4Model             scaled embedding -> N layers -> final norm, with a per-layer KV cache
    Gemma4ForCausalLM       + tied lm_head and final_logit_softcapping

One arithmetic path, two array backends
---------------------------------------
All array work goes through the `Ops` adapter in `ops.py`, so the SAME code runs on numpy (the
reference) and on torch (the fast path, on GPU).  That is deliberate: a second transcription of the
architecture for the fast path is how a backend drifts from the one that was tested.  The parity
test runs this file twice, with a different `Ops`, and requires the logits to match.

Things that are easy to get wrong, and where they are pinned down
----------------------------------------------------------------
  * RMSNorm is `x/rms * w`.  Gemma 1/2/3 use `(1 + w)`; Gemma 4's reference does not
    (`normed_output * self.weight.float()`), and the file proves it: `output_norm.weight` has mean
    29.3 and `blk.0.attn_norm.weight` mean 4.5 - multipliers, not offsets from 1.
  * Q and K are RMS-normalised PER HEAD (`attn_q_norm`, `attn_k_norm`, dim = head_dim) before RoPE,
    and V is normalised with NO weight (`with_scale=False` in the reference, bare `ggml_rms_norm`
    in llama.cpp).  Because of that the attention scale is 1.0, not 1/sqrt(head_dim)
    (`hparams.f_attention_scale = 1.0f`).
  * V is the raw `v_proj` output (or K's, on full layers) normalised WITHOUT rope.  Reusing the
    roped, k-normed K as V is a plausible-looking mistake that changes every full layer.
  * the MoE has no "shared expert" tensor: the ordinary dense MLP IS the shared expert, and the
    router reads the layer's INPUT (the post-attention residual), not the MLP's output.  The routed
    experts read a separately normed copy (`pre_ffw_norm_2`); each branch is normed again
    (`post_ffw_norm_1`, `post_ffw_norm_2`), the two are summed, and the sum is normed once more.
  * every layer's output is multiplied by `layer_output_scale.weight` (one float; 0.0703 in layer
    0).  Missing it rescales the residual stream at every layer.
  * the token embedding is multiplied by sqrt(hidden_size) before layer 0.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from .config import Gemma4Config, Gemma4TextConfig, LayerSpec, RopeParams
from .ops import NumpyOps, Ops


# ------------------------------------------------------------------ numerics
def rms_norm(x, weight, eps: float, xp: Ops):
    """Root-mean-square norm over the last axis, then a per-channel gain.

    `weight=None` is the scaleless form the router and V use (`Gemma4RMSNorm(with_scale=False)`)."""
    var = xp.mean_last(x * x)
    x = x / xp.sqrt(var + eps)
    if weight is None:
        return x
    return x * weight


def gelu_pytorch_tanh(x, xp: Ops):
    """The exact tanh approximation PyTorch uses for hidden_activation="gelu_pytorch_tanh"."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * xp.cube(x))
    return 0.5 * x * (1.0 + xp.tanh(inner))


def softmax(x, xp: Ops):
    m = xp.max_last(x, keepdims=True)
    e = xp.exp(x - m)
    return e / xp.sum_last(e, keepdims=True)


def _rotate_half(x, xp: Ops):
    half = x.shape[-1] // 2
    a = xp.slice_last(x, 0, half)
    b = xp.slice_last(x, half, 2 * half)
    return xp.concat_last(xp.neg(b), a)


def rope_inv_freq(rope: RopeParams, head_dim: int, factors=None) -> tuple[float, ...]:
    """The `head_dim // 2` inverse frequencies one layer uses, after the artifact's factors.

    `factors` is `rope_freqs.weight`: 1.0 on rotated channels and 1e30 on the NoPE ones, so
    dividing by it zeroes those channels (see `config.apply_rope_factors`)."""
    from .config import apply_rope_factors
    return apply_rope_factors(rope.inv_freq(head_dim), factors)


def rope_tables(pos, inv_freq, xp: Ops, dtype=None):
    """(cos, sin) of shape (n_pos, head_dim), built the way the reference does it:
    `emb = cat(freqs, freqs)` with `freqs = pos @ inv_freq`.  A zero inverse frequency gives
    cos=1, sin=0, so that channel passes through unrotated.

    The angles are computed in float64 and returned in `dtype`: at position 262143 a float32 angle
    has ~0.01 rad of slack, which is enough to change which token wins.  Casting at the end (not
    computing in float32) is what keeps a 256K context honest, and it keeps the result the same
    dtype as the activations - torch will not einsum a float64 against a float32."""
    inv = xp.asarray(inv_freq, xp.float64)
    p = xp.asarray(pos, xp.float64)
    ang = xp.outer(p, inv)
    emb = xp.concat_last(ang, ang)
    cos, sin = xp.cos(emb), xp.sin(emb)
    if dtype is not None:
        cos, sin = xp.astype(cos, dtype), xp.astype(sin, dtype)
    return cos, sin


def apply_rotary(x, cos, sin, xp: Ops):
    """Rotate every pair in the last axis.  `x` is (n, n_heads, head_dim); cos/sin are
    (n, head_dim) and the caller has already added the head axis."""
    return (x * cos) + _rotate_half(x, xp) * sin


# ------------------------------------------------------------------ attention
@dataclasses.dataclass
class KVCache:
    """Per-layer key/value history for one sequence, with the POSITION of every row.

    A sliding layer keeps only the last `window` rows: it can never attend further back, so the
    rest is dead memory.  A full layer keeps everything.

    The positions are kept because the mask has to be built from them, not from row indices.  A
    prompt longer than the window trims rows, and then "row i of the cache" is no longer "token i
    of the sequence" - an index-based mask would let a query attend to a key outside its window and
    forbid one inside it.  Comparing positions is correct for any prompt length."""

    k: Any = None                                  # (n, kv_heads, head_dim)
    v: Any = None
    pos: Any = None                                # (n,) absolute position of each row
    window: int | None = None
    xp: Ops = dataclasses.field(default_factory=NumpyOps, repr=False)

    def __len__(self) -> int:
        return 0 if self.k is None else int(self.k.shape[0])

    def append(self, k, v, positions) -> None:
        xp = self.xp
        self.k = xp.concat_axis0(self.k, k)
        self.v = xp.concat_axis0(self.v, v)
        self.pos = xp.concat_axis0(self.pos, positions)
        if self.window is None:
            return
        # A sliding layer keeps `window` rows, PLUS the rows this call just added minus one: during
        # a multi-token prefill the earliest query in the batch is `n-1` positions behind the
        # newest, and it still has to see its own window.  Trimming to exactly `window` here would
        # delete rows the batch's own queries need, leaving a query with nothing to attend to and
        # a softmax over all -inf, i.e. NaN hidden states from layer 0 onward.
        keep = self.window + int(k.shape[0]) - 1
        if self.k.shape[0] > keep:
            self.k = self.k[-keep:]
            self.v = self.v[-keep:]
            self.pos = self.pos[-keep:]

    def reset(self) -> None:
        self.k = self.v = self.pos = None

    def rows(self):
        return self.k, self.v, self.pos


class Gemma4Attention:
    """Grouped-query attention with QK-norm.  One instance per layer."""

    def __init__(self, spec: LayerSpec, w: dict, eps: float, scale: float = 1.0,
                 inv_freq: Sequence[float] | None = None, xp: Ops = None):
        self.spec = spec
        self.w = w
        self.eps = eps
        self.scale = scale
        self.xp = xp or NumpyOps()
        self.head_dim = spec.head_dim
        self.n_heads = spec.n_heads
        self.n_kv = spec.n_kv_heads
        self.inv_freq = tuple(inv_freq) if inv_freq is not None else rope_inv_freq(
            RopeParams("proportional" if spec.full else "default", spec.theta), spec.head_dim)

    def _repeat_kv(self, x, n_hist):
        """(n, kv_heads, hd) -> (n, n_heads, hd) by repeating each kv head n_heads/n_kv times."""
        xp = self.xp
        if self.n_heads == self.n_kv:
            return x
        return xp.repeat_axis1(x, self.n_heads // self.n_kv)

    def forward(self, x, pos, cache: KVCache, window: int | None = None):
        """x is (n_tok, hidden); pos is (n_tok,) absolute positions.  Returns (n_tok, hidden)."""
        xp = self.xp
        n = x.shape[0]
        q = x @ self.w["q"]                                   # (n, n_heads*head_dim)
        q = rms_norm(xp.reshape(q, (n, self.n_heads, self.head_dim)),
                     self.w.get("q_norm"), self.eps, xp)
        if "k" in self.w:
            k = x @ self.w["k"]
            k = rms_norm(xp.reshape(k, (n, self.n_kv, self.head_dim)),
                         self.w.get("k_norm"), self.eps, xp)
            v_src = x @ self.w["v"] if "v" in self.w else x @ self.w["k"]
            # V: normalised with NO weight and NOT roped
            v = rms_norm(xp.reshape(v_src, (n, self.n_kv, self.head_dim)), None, self.eps, xp)
        else:                                                # KV-shared layer: reuse the cache
            k = v = None

        cos, sin = rope_tables(pos, self.inv_freq, xp, dtype=xp.dtype_of(q))
        cos, sin = cos[:, None, :], sin[:, None, :]
        q = apply_rotary(q, cos, sin, xp)
        if k is not None:
            k = apply_rotary(k, cos, sin, xp)
            cache.append(k, v, pos)
        k_all, v_all, k_pos = cache.rows()

        qh = xp.transpose(q, (1, 0, 2))                       # (n_heads, n, hd)
        kh = xp.transpose(self._repeat_kv(k_all, k_all.shape[0]), (1, 0, 2))
        vh = xp.transpose(self._repeat_kv(v_all, v_all.shape[0]), (1, 0, 2))
        scores = xp.einsum("hnd,hmd->hnm", qh, kh) * self.scale
        scores = _causal_window_mask(scores, pos, k_pos, window, xp)
        attn = softmax(scores, xp)
        out = xp.einsum("hnm,hmd->hnd", attn, vh)
        out = xp.transpose(out, (1, 0, 2))
        out = xp.reshape(out, (n, self.n_heads * self.head_dim))
        return out @ self.w["o"]


def _causal_window_mask(scores, q_pos, k_pos, window: int | None, xp: Ops):
    """Additive mask, built from ABSOLUTE positions: a query may attend to a key only when the key
    is not in the future and (for a sliding layer) within `window` of it.

    Positions, not row indices, because a sliding layer's cache has been trimmed: row 0 of the
    cache is not token 0 of the sequence."""
    q = q_pos[:, None]
    k = k_pos[None, :]
    allowed = k <= q
    if window is not None:
        allowed = allowed & ((q - k) < window)
    neg = xp.full((1,), float("-inf"), xp.dtype_of(scores))
    return xp.where(allowed, scores, neg)


# ------------------------------------------------------------------ MoE
class Gemma4Router:
    """The router the reference uses: a scaleless RMSNorm, a per-channel scale, softmax over all
    experts, top-k renormalised, then multiplied by a per-expert scale.

    Note what it reads: the layer's INPUT (the post-attention residual), not the MLP output.  And
    what it does not have: no bias, and no logit for the shared expert."""

    def __init__(self, w: dict, cfg: Gemma4TextConfig, xp: Ops = None):
        self.w = w
        self.cfg = cfg
        self.xp = xp or NumpyOps()
        self.n_experts = cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.eps = cfg.rms_norm_eps
        self.scalar_root = cfg.hidden_size ** -0.5

    def route(self, x):
        """(n, hidden) -> (ids (n, top_k), weights (n, top_k)), both ascending by expert id."""
        xp = self.xp
        h = rms_norm(x, None, self.eps, xp)                    # with_scale=False
        h = h * (self.w["router_scale"] * self.scalar_root)
        logits = h @ self.w["router_w"]                        # (n, n_experts)
        probs = softmax(logits, xp)
        vals, idx = xp.topk_last(probs, self.top_k)
        vals = vals / xp.clip_min(xp.sum_last(vals, keepdims=True), 1e-9)
        vals = vals * xp.index_cols(self.w["expert_scale"], idx)   # per_expert_scale
        return idx, vals


class ExpertSource:
    """Where one expert's two matrices come from.

    `StackedExperts` slices arrays already in memory.  A streaming source (see `backends.py`) reads
    the GGUF blocks for just the experts this batch needs, which is what lets a 22-billion-parameter
    expert stack run on a card that cannot hold it.

    `batched_views` says whether this source can hand a RANGE of experts to a batched matmul as a
    view of one stacked tensor.  Only then is the batch worth taking: the alternative is copying the
    weights, and one layer's expert stack is 2.8 GiB in float32 on gemma-4-26B-A4B - measured at
    1311 ms against the 100 ms of matmul it would feed.  A source that reads the file per expert
    (`StreamedExperts`) leaves it False and keeps the per-expert loop.

    `max_group` is the largest group of consecutive experts a batch may cover, and `aligned_groups`
    says whether a group must also stay inside one aligned block of that size.  A source that keeps
    its experts in fixed blocks needs both (a range that crossed a block boundary would not be one
    view); a source with one reusable scratch buffer only needs the cap.  `max_group = None` means the
    MoE may batch every consecutive run it finds."""

    xp: Ops = NumpyOps()
    batched_views = False
    max_group: int | None = None
    aligned_groups = False

    def gate_up(self, expert: int):
        """-> (hidden, 2*moe_intermediate), gate first then up."""
        raise NotImplementedError

    def down(self, expert: int):
        """-> (moe_intermediate, hidden)."""
        raise NotImplementedError

    def gate_up_range(self, lo: int, hi: int):
        """-> (hi-lo, hidden, 2*moe_intermediate) for experts [lo, hi), as a view."""
        raise NotImplementedError

    def down_range(self, lo: int, hi: int):
        """-> (hi-lo, moe_intermediate, hidden) for experts [lo, hi), as a view."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class StackedExperts(ExpertSource):
    """All experts already in memory as [in, out, n_experts] arrays (the reference profile)."""

    def __init__(self, w: dict, inter: int, xp: Ops = None):
        self.w = w
        self.inter = inter
        self.xp = xp or NumpyOps()

    def gate_up(self, e):
        xp = self.xp
        gu = self.w.get("gate_up_exps")
        if gu is not None:
            return xp.slice_expert(gu, e)
        g = xp.slice_expert(self.w["gate_exps"], e)
        u = xp.slice_expert(self.w["up_exps"], e)
        return xp.concat_axis1(g, u)

    def down(self, e):
        return self.xp.slice_expert(self.w["down_exps"], e)

    batched_views = True

    def gate_up_range(self, lo, hi):
        """A view of the stacked tensor: no copy, no concat, whatever the group's size."""
        xp = self.xp
        gu = self.w.get("gate_up_exps")
        if gu is not None:
            return xp.expert_range(gu, lo, hi)
        return xp.concat_last(xp.expert_range(self.w["gate_exps"], lo, hi),
                              xp.expert_range(self.w["up_exps"], lo, hi))

    def down_range(self, lo, hi):
        return self.xp.expert_range(self.w["down_exps"], lo, hi)


class Gemma4MoE:
    """Dense MLP (the shared expert) + top-k routed experts, combined exactly as the reference does.

    `forward` returns the COMBINED feed-forward output for one layer; the caller adds the residual.
    The norms that belong to the two branches live here because the reference puts them here."""

    def __init__(self, cfg: Gemma4TextConfig, w: dict, router: Gemma4Router | None,
                 experts: ExpertSource | None, xp: Ops = None):
        self.cfg = cfg
        self.w = w
        self.xp = xp or NumpyOps()
        self.router = router
        self.experts = experts
        self.inter = cfg.moe_intermediate_size

    def _dense(self, x):
        """The ordinary MLP, which doubles as the shared expert."""
        xp = self.xp
        g = x @ self.w["gate"]
        u = x @ self.w["up"]
        return gelu_pytorch_tanh(g, xp) * u @ self.w["down"]

    #: Batch a group of experts only when each carries at least this many tokens.  Below it the
    #: batched kernel's fixed cost is not paid back; at one token per expert - every decode step -
    #: the per-expert GEMV is what wins.
    batched_min_rows = 8

    def _routed(self, x, idx, vals):
        """The routed branch, grouped by expert.

        Tokens are sorted by expert, so `runs` hands back one entry per chosen expert, in ascending
        expert id and with that expert's rows contiguous in `order`.  A run of CONSECUTIVE ids that
        all carry the same number of tokens becomes one batched matmul (`_expert_batch`): consecutive
        ids means the weights are a contiguous slice of the stacked tensor, so the batch reads them
        as a view.  Anything else keeps the per-expert path (`_expert_rows`).

        Consecutive ids and equal counts are both required, and they are not the same constraint:
        a batched matmul has one shape (so equal rows), and only a contiguous id range is a view
        (so no weight copy).  A routing that splits a layer's experts into several size classes gets
        one batch per class, which is still far fewer launches than one per expert."""
        xp = self.xp
        n, d = x.shape
        k = idx.shape[1]
        out = xp.zeros((n, d), xp.dtype_of(x))
        flat = xp.reshape(idx, (-1,))
        order = xp.argsort_flat(flat)
        sorted_ids = flat[order]
        slots = xp.arange(0, n * k)
        tok = slots // k
        runs = [r for r in xp.runs(sorted_ids) if r[1] > r[0]]
        can_batch = self.experts.batched_views
        cap = self.experts.max_group
        aligned = self.experts.aligned_groups

        def fits(k, size):
            """Whether run `k` may join a group that starts at run `i` and holds `size` runs."""
            if cap is None:
                return True
            if aligned:
                return runs[k][2] // cap == runs[i][2] // cap
            return size < cap

        i = 0
        while i < len(runs):
            count = runs[i][1] - runs[i][0]
            j = i + 1
            while (j < len(runs) and runs[j][1] - runs[j][0] == count
                   and runs[j][2] == runs[j - 1][2] + 1 and fits(j, j - i)):
                j += 1
            if can_batch and j - i >= 2 and count >= self.batched_min_rows:
                self._expert_batch(out, x, order, tok, vals, runs[i:j])
            else:
                for lo, hi, e in runs[i:j]:
                    self._expert_rows(out, x, order, tok, vals, lo, hi, e)
            i = j
        return out

    def _expert_rows(self, out, x, order, tok, vals, lo, hi, e):
        """One expert, three plain matmuls over the rows routed to it."""
        xp = self.xp
        sel = order[lo:hi]
        rows = tok[sel]
        xe = x[rows]
        gu = self.experts.gate_up(e)
        act = gelu_pytorch_tanh(xe @ xp.slice_last(gu, 0, self.inter), xp) \
            * (xe @ xp.slice_last(gu, self.inter, 2 * self.inter))
        ye = act @ self.experts.down(e)
        ye = ye * vals.reshape(-1)[sel][:, None]
        xp.add_rows(out, rows, ye)

    def _expert_batch(self, out, x, order, tok, vals, group):
        """`group` consecutive experts with `count` tokens each, as three batched matmuls.

        The group's rows are already contiguous in `order` (the runs are adjacent), so the activations
        reshape straight into [G, count, hidden] and the weights stay a view of the stacked tensor:
        the three products run G times inside one kernel instead of one launch per expert."""
        xp = self.xp
        count = group[0][1] - group[0][0]
        e0, e1 = group[0][2], group[-1][2] + 1
        g = e1 - e0
        sel = order[group[0][0]:group[-1][1]]
        rows = tok[sel]
        xe = xp.reshape(x[rows], (g, count, x.shape[1]))
        gu = self.experts.gate_up_range(e0, e1)
        act = gelu_pytorch_tanh(xp.matmul3(xe, xp.slice_last(gu, 0, self.inter)), xp) \
            * xp.matmul3(xe, xp.slice_last(gu, self.inter, 2 * self.inter))
        ye = xp.matmul3(act, self.experts.down_range(e0, e1))
        ye = ye * xp.reshape(vals.reshape(-1)[sel], (g, count, 1))
        xp.add_rows(out, rows, xp.reshape(ye, (g * count, x.shape[1])))

    def forward(self, x):
        """x is (n_tok, hidden) - the post-attention residual.  Returns (n_tok, hidden)."""
        xp = self.xp
        c = self.cfg.rms_norm_eps
        h = rms_norm(x, self.w["ffn_norm"], c, xp)
        mlp = self._dense(h)
        if self.router is None or self.experts is None:
            return rms_norm(mlp, self.w["post_ffn_norm"], c, xp)
        mlp_n = rms_norm(mlp, self.w["post_ffn_norm_1"], c, xp)
        idx, vals = self.router.route(x)                       # the router reads the layer INPUT
        h2 = rms_norm(x, self.w["pre_ffn_norm_2"], c, xp)
        h2 = self._routed(h2, idx, vals)
        h2 = rms_norm(h2, self.w["post_ffn_norm_2"], c, xp)
        comb = mlp_n + h2
        return rms_norm(comb, self.w["post_ffn_norm"], c, xp)


# ------------------------------------------------------------------ layer / model
class Gemma4DecoderLayer:
    """norm -> attention -> norm -> residual -> (MLP + MoE) -> norm -> residual -> *layer_scale."""

    def __init__(self, cfg: Gemma4TextConfig, spec: LayerSpec, w: dict, xp: Ops = None,
                 router: Gemma4Router | None = None, experts: ExpertSource | None = None):
        self.cfg = cfg
        self.spec = spec
        self.w = w
        self.xp = xp or NumpyOps()
        self.window = None if spec.full else cfg.sliding_window
        self.attn = Gemma4Attention(
            spec, w, eps=cfg.rms_norm_eps, scale=cfg.attention_scale,
            inv_freq=spec.inv_freq(cfg.rope_full, cfg.rope_sliding), xp=self.xp)
        # Always build the feed-forward block: with `enable_moe_block` off (a dense Gemma 4, e.g.
        # gemma-4-12B) the dense MLP *is* the layer's feed-forward, and `Gemma4MoE.forward` takes
        # the router-free path when router/experts are None. Skipping it here would drop the MLP
        # entirely and produce garbage tokens.
        self.moe = Gemma4MoE(cfg, w, router, experts, xp=self.xp)

    def forward(self, x, pos, cache: KVCache):
        xp = self.xp
        c = self.cfg.rms_norm_eps
        residual = x
        h = rms_norm(x, self.w["attn_norm"], c, xp)
        h = self.attn.forward(h, pos, cache, window=self.window)
        h = rms_norm(h, self.w.get("post_attn_norm"), c, xp)
        x = residual + h
        if self.moe is not None:
            x = x + self.moe.forward(x)
        if self.w.get("layer_scale") is not None:
            x = x * self.w["layer_scale"]
        return x


class Gemma4Model:
    """Scaled embedding -> decoder layers -> final norm.  Owns one KVCache per layer."""

    def __init__(self, cfg: Gemma4TextConfig, layers: Sequence[Gemma4DecoderLayer],
                 embed, final_norm, xp: Ops = None):
        self.cfg = cfg
        self.layers = list(layers)
        self.embed = embed                 # (hidden, vocab), GGUF order
        self.final_norm = final_norm
        self.xp = xp or NumpyOps()
        self.embed_scale = cfg.effective_embed_scale()
        self.reset_cache()

    def reset_cache(self) -> None:
        self.cache = [KVCache(window=(None if l.spec.full else self.cfg.sliding_window), xp=self.xp)
                      for l in self.layers]

    def embed_tokens(self, ids):
        xp = self.xp
        ids = xp.asarray(ids, xp.int64)
        e = xp.take_rows(self.embed, ids, axis=1)   # stored [hidden, vocab]
        e = xp.transpose(e, (1, 0))
        return e * self.embed_scale

    def forward(self, ids, start_pos: int = 0):
        """Runs the decoder over `ids` (positions start at `start_pos`) and returns the post-norm
        hidden states (n_tok, hidden).  Appends to the KV cache; call `reset_cache` per sequence."""
        xp = self.xp
        ids = xp.asarray(ids, xp.int64)
        if len(ids.shape) == 0:
            ids = xp.reshape(ids, (1,))
        x = self.embed_tokens(ids)
        n = x.shape[0]
        pos = xp.arange(start_pos, start_pos + n)
        for layer, cache in zip(self.layers, self.cache):
            x = layer.forward(x, pos, cache)
        return rms_norm(x, self.final_norm, self.cfg.rms_norm_eps, xp)


class Gemma4ForCausalLM:
    """The full model: `Gemma4Model` plus the (tied) output head and the final logit softcap."""

    def __init__(self, config: Gemma4Config, model: Gemma4Model, lm_head, xp: Ops = None):
        self.config = config
        self.model = model
        self.lm_head = lm_head             # (hidden, vocab) when tied, else the output.weight
        self.softcap = config.text.final_logit_softcapping
        self.xp = xp or NumpyOps()

    def logits(self, hidden):
        """(n, hidden) -> (n, vocab).  Gemma 4 soft-caps the final logits: cap*tanh(logits/cap),
        which bounds them and is easy to forget (without it the distribution is sharper)."""
        xp = self.xp
        z = hidden @ self.lm_head
        if self.softcap and self.softcap > 0:
            z = self.softcap * xp.tanh(z / self.softcap)
        return z

    def forward(self, ids, start_pos: int = 0):
        return self.logits(self.model.forward(ids, start_pos))

    def reset_cache(self) -> None:
        self.model.reset_cache()

    def close(self) -> None:
        for layer in self.model.layers:
            if layer.moe is not None and layer.moe.experts is not None:
                layer.moe.experts.close()

"""serve/gemma4/ops.py - the small array vocabulary the model is written against.

`model.py` does arithmetic through one object with a fixed set of methods, so the SAME code runs on
numpy (the reference, always available) and on torch (the fast path, GPU).  Writing `x @ w` and
`xp.mean(...)` directly would have meant two transcriptions of the architecture, and two
transcriptions is how a fast backend drifts from the one that was tested.

The adapter exists because the two libraries disagree on details that are invisible until they are
wrong: `axis=` vs `dim=`, `repeat` vs `repeat_interleave`, `clip` vs `clamp`, `concatenate` vs
`cat`, and whether a 3-D transpose takes a tuple or two ints.  Each method below is where one of
those differences is settled, once.

`NumpyOps` and `TorchOps` expose the identical surface.  `test_gemma4.py` asserts that surface
method-for-method, so adding a third backend (a GGML graph, a JAX port) means implementing this
list and nothing else.
"""
from __future__ import annotations

import abc
from typing import Any, Sequence

import numpy as np


class Ops(abc.ABC):
    """One array library, behind the calls `model.py` makes."""

    name = "abstract"

    # ---- construction
    @abc.abstractmethod
    def asarray(self, values, dtype: Any = None): ...

    @abc.abstractmethod
    def zeros(self, shape: Sequence[int], dtype: Any): ...

    @abc.abstractmethod
    def ones(self, shape: Sequence[int], dtype: Any): ...

    @abc.abstractmethod
    def arange(self, start: int, stop: int): ...

    @abc.abstractmethod
    def full(self, shape: Sequence[int], value: float, dtype: Any): ...

    # ---- elementwise / reductions
    @abc.abstractmethod
    def mean_last(self, x): ...

    @abc.abstractmethod
    def sum_last(self, x, keepdims: bool = True): ...

    @abc.abstractmethod
    def max_last(self, x, keepdims: bool = True): ...

    @abc.abstractmethod
    def exp(self, x): ...

    @abc.abstractmethod
    def sqrt(self, x): ...

    @abc.abstractmethod
    def tanh(self, x): ...

    @abc.abstractmethod
    def cos(self, x): ...

    @abc.abstractmethod
    def sin(self, x): ...

    @abc.abstractmethod
    def cube(self, x): ...

    @abc.abstractmethod
    def clip_min(self, x, lo: float): ...

    @abc.abstractmethod
    def where(self, cond, a, b): ...

    # ---- shape / indexing
    @abc.abstractmethod
    def reshape(self, x, shape: Sequence[int]): ...

    @abc.abstractmethod
    def transpose(self, x, axes: Sequence[int]): ...

    @abc.abstractmethod
    def concat_last(self, a, b): ...

    @abc.abstractmethod
    def concat_axis0(self, a, b): ...

    @abc.abstractmethod
    def repeat_axis1(self, x, times: int): ...

    @abc.abstractmethod
    def argsort_flat(self, x): ...

    @abc.abstractmethod
    def topk_last(self, x, k: int):
        """(values, indices) of the k largest entries of the last axis."""

    @abc.abstractmethod
    def take_rows(self, table, rows, axis: int): ...

    @abc.abstractmethod
    def gather1d(self, table, idx): ...

    @abc.abstractmethod
    def add_rows(self, out, rows, values): ...

    @abc.abstractmethod
    def einsum(self, spec: str, a, b): ...

    @abc.abstractmethod
    def slice_last(self, x, start: int, stop: int): ...

    @abc.abstractmethod
    def slice_axis1(self, x, start: int, stop: int): ...

    @abc.abstractmethod
    def slice_expert(self, x, e: int):
        """`x` is a stacked [in, out, n_experts] tensor -> the [in, out] matrix for expert `e`."""

    @abc.abstractmethod
    def concat_axis1(self, a, b): ...

    @abc.abstractmethod
    def neg(self, x): ...

    @abc.abstractmethod
    def outer(self, a, b):
        """(n,) x (d,) -> (n, d)."""

    @abc.abstractmethod
    def index_cols(self, table, idx):
        """`table` is 1-D, `idx` any int tensor -> elementwise gather with idx's shape."""

    @abc.abstractmethod
    def runs(self, sorted_ids) -> list[tuple[int, int, int]]:
        """[(start, stop, value)] for each run of equal values in a sorted 1-D array.  The MoE uses
        it to visit each chosen expert exactly once per token batch."""

    # ---- batched expert GEMMs
    @abc.abstractmethod
    def stack_first(self, mats):
        """[m0, m1, ...] of equal shape -> [len(mats), *shape]."""

    @abc.abstractmethod
    def store_row(self, dst, i: int, src):
        """`dst[i] = src`, in place, `dst` already allocated and expert-axis-FIRST ([n, rows, cols]).
        A buffer refilled per group must not reallocate, and the destination row must be contiguous:
        writing into a strided slice of an expert-axis-last tensor measured 3.7 s over 128 experts,
        against 1.9 s for the reads themselves (gemma-4-26B-A4B Q4_0, torch CPU)."""

    @abc.abstractmethod
    def expert_range(self, x, lo: int, hi: int):
        """`x` is a stacked [in, out, n_experts] tensor -> [hi-lo, in, out] for experts [lo, hi).

        This must be a VIEW, not a copy.  One layer's expert stack is 2.8 GiB in float32 (1.9 GiB of
        gate/up plus 0.95 GiB of down on gemma-4-26B-A4B), and materialising a slice of it measured
        1311 ms against the 100 ms of matmul it would feed - a copy turns a 1.4x win into a 20x loss."""

    @abc.abstractmethod
    def matmul3(self, a, b):
        """Batched matmul: [B, m, k] @ [B, k, n] -> [B, m, n].  `a` and `b` may be strided views."""

    # ---- dtypes / device
    @abc.abstractmethod
    def astype(self, x, dtype): ...

    @abc.abstractmethod
    def dtype_of(self, x): ...

    @abc.abstractmethod
    def shape_of(self, x): ...

    def move(self, x):                       # to this backend's device
        return x


# ------------------------------------------------------------------ numpy
class NumpyOps(Ops):
    name = "numpy"

    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    int32 = np.int32

    def asarray(self, values, dtype=None):
        return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)

    def zeros(self, shape, dtype):
        return np.zeros(tuple(shape), dtype=dtype)

    def ones(self, shape, dtype):
        return np.ones(tuple(shape), dtype=dtype)

    def arange(self, start, stop):
        return np.arange(start, stop)

    def full(self, shape, value, dtype):
        return np.full(tuple(shape), value, dtype=dtype)

    def mean_last(self, x):
        return np.mean(x, axis=-1, keepdims=True)

    def sum_last(self, x, keepdims=True):
        return np.sum(x, axis=-1, keepdims=keepdims)

    def max_last(self, x, keepdims=True):
        return np.max(x, axis=-1, keepdims=keepdims)

    def exp(self, x): return np.exp(x)
    def sqrt(self, x): return np.sqrt(x)
    def tanh(self, x): return np.tanh(x)
    def cos(self, x): return np.cos(x)
    def sin(self, x): return np.sin(x)
    def cube(self, x): return np.power(x, 3)

    def clip_min(self, x, lo):
        return np.clip(x, lo, None)

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def reshape(self, x, shape):
        return x.reshape(tuple(shape))

    def transpose(self, x, axes):
        return np.transpose(x, tuple(axes))

    def concat_last(self, a, b):
        return np.concatenate([a, b], axis=-1)

    def concat_axis0(self, a, b):
        if a is None:
            return b
        return np.concatenate([a, b], axis=0)

    def repeat_axis1(self, x, times):
        return np.repeat(x, times, axis=1)

    def argsort_flat(self, x):
        return np.argsort(x, kind="stable")

    def slice_expert(self, x, e):
        return x[:, :, e]

    def concat_axis1(self, a, b):
        return np.concatenate([a, b], axis=1)

    def topk_last(self, x, k):
        idx = np.argsort(x, axis=-1, kind="stable")[..., -k:]
        return np.take_along_axis(x, idx, axis=-1), idx

    def take_rows(self, table, rows, axis):
        # an embedding is stored [hidden, vocab], so a token lookup indexes the LAST axis
        return np.take(table, rows, axis=axis)

    def gather1d(self, table, idx):
        return table[idx]

    def add_rows(self, out, rows, values):
        np.add.at(out, rows, values)
        return out

    def einsum(self, spec, a, b):
        return np.einsum(spec, a, b)

    def slice_last(self, x, start, stop):
        return x[..., start:stop]

    def slice_axis1(self, x, start, stop):
        return x[:, start:stop]

    def neg(self, x):
        return -x

    def outer(self, a, b):
        return np.outer(a, b)

    def index_cols(self, table, idx):
        return table[idx]

    def runs(self, sorted_ids):
        ids = np.asarray(sorted_ids)
        if ids.size == 0:
            return []
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        stops = np.concatenate((cuts, [ids.size]))
        return [(int(s), int(e), int(ids[s])) for s, e in zip(starts, stops)]

    def take_experts(self, x, ids):
        # x[:, :, ids] is [in, out, len(ids)] with the experts on the LAST axis, which is the strided
        # axis for a batched matmul; transposing to [len(ids), in, out] makes the batch axis outermost
        # and contiguous, which is what BLAS' strided-batch path wants.
        return np.ascontiguousarray(np.transpose(x[:, :, ids], (2, 0, 1)))

    def stack_first(self, mats):
        return np.stack(list(mats), axis=0)

    def store_row(self, dst, i, src):
        dst[i] = src
        return dst

    def expert_range(self, x, lo, hi):
        return np.transpose(x[:, :, lo:hi], (2, 0, 1))

    def matmul3(self, a, b):
        return np.matmul(a, b)

    def astype(self, x, dtype):
        return x.astype(dtype, copy=False)

    def dtype_of(self, x):
        return x.dtype

    def shape_of(self, x):
        return x.shape


# ------------------------------------------------------------------ torch
class TorchOps(Ops):
    """torch, on one device.  `device` is a torch device string ('cuda', 'cpu', 'cuda:0')."""

    name = "torch"

    def __init__(self, device: str = "cpu", dtype=None):
        import torch                                   # the caller has already checked for it
        self.torch = torch
        self.device = torch.device(device)
        self.float32 = torch.float32
        self.float64 = torch.float64
        self.int64 = torch.int64
        self.int32 = torch.int32
        self.compute = dtype or torch.float32

    def ones(self, shape, dtype):
        return self.torch.ones(tuple(shape), dtype=dtype, device=self.device)

    def asarray(self, values, dtype=None):
        t = self.torch
        if isinstance(values, t.Tensor):
            return values.to(dtype=dtype or self.compute, device=self.device)
        a = np.asarray(values)
        # A dequantised tensor is often a view over read-only bytes (np.frombuffer).  torch cannot
        # share that memory, and `as_tensor` on it raises or warns; copy when the source is not
        # writable so the device tensor is always ours.
        if not a.flags.writeable:
            a = a.copy()
        return t.from_numpy(a).to(dtype=dtype or self.compute, device=self.device)

    def zeros(self, shape, dtype):
        return self.torch.zeros(tuple(shape), dtype=dtype, device=self.device)

    def arange(self, start, stop):
        return self.torch.arange(start, stop, device=self.device)

    def full(self, shape, value, dtype):
        return self.torch.full(tuple(shape), value, dtype=dtype, device=self.device)

    def mean_last(self, x):
        return x.mean(dim=-1, keepdim=True)

    def sum_last(self, x, keepdims=True):
        return x.sum(dim=-1, keepdim=keepdims)

    def max_last(self, x, keepdims=True):
        return x.amax(dim=-1, keepdim=keepdims)

    def exp(self, x): return x.exp()
    def sqrt(self, x): return x.sqrt()
    def tanh(self, x): return x.tanh()
    def cos(self, x): return x.cos()
    def sin(self, x): return x.sin()
    def cube(self, x): return x.pow(3)

    def clip_min(self, x, lo):
        return x.clamp(min=lo)

    def where(self, cond, a, b):
        return self.torch.where(cond, a, b)

    def reshape(self, x, shape):
        return x.reshape(*shape)

    def transpose(self, x, axes):
        return x.permute(*axes)

    def concat_last(self, a, b):
        return self.torch.cat([a, b], dim=-1)

    def concat_axis0(self, a, b):
        if a is None:
            return b
        return self.torch.cat([a, b], dim=0)

    def repeat_axis1(self, x, times):
        return x.repeat_interleave(times, dim=1)

    def argsort_flat(self, x):
        return x.argsort(stable=True)

    def slice_expert(self, x, e):
        return x[:, :, e]

    def concat_axis1(self, a, b):
        return self.torch.cat([a, b], dim=1)

    def topk_last(self, x, k):
        # flipped to ascending, so both backends hand the MoE the same column order
        vals, idx = self.torch.topk(x, k, dim=-1)
        return vals.flip(-1), idx.flip(-1)

    def take_rows(self, table, rows, axis):
        return table.index_select(axis, rows.to(table.device))

    def gather1d(self, table, idx):
        return table[idx.to(table.device)]

    def add_rows(self, out, rows, values):
        out.index_add_(0, rows.to(out.device), values.to(out.dtype))
        return out

    def einsum(self, spec, a, b):
        return self.torch.einsum(spec, a, b)

    def slice_last(self, x, start, stop):
        return x[..., start:stop]

    def slice_axis1(self, x, start, stop):
        return x[:, start:stop]

    def neg(self, x):
        return -x

    def outer(self, a, b):
        return self.torch.outer(a, b)

    def index_cols(self, table, idx):
        return table[idx]

    def runs(self, sorted_ids):
        ids = sorted_ids
        if ids.numel() == 0:
            return []
        t = self.torch
        cuts = (ids[1:] != ids[:-1]).nonzero().flatten() + 1
        starts = t.cat([t.zeros(1, dtype=ids.dtype, device=ids.device), cuts])
        stops = t.cat([cuts, t.tensor([ids.numel()], dtype=ids.dtype, device=ids.device)])
        vals = ids[starts]
        return list(zip(starts.tolist(), stops.tolist(), vals.tolist()))

    def take_experts(self, x, ids):
        t = self.torch
        if not isinstance(ids, t.Tensor):
            ids = t.as_tensor(np.asarray(ids), device=x.device)
        return x.index_select(2, ids.to(x.device)).permute(2, 0, 1).contiguous()

    def stack_first(self, mats):
        return self.torch.stack(list(mats), dim=0)

    def store_row(self, dst, i, src):
        dst[i].copy_(src)
        return dst

    def expert_range(self, x, lo, hi):
        return x[:, :, lo:hi].permute(2, 0, 1)

    def matmul3(self, a, b):
        return self.torch.matmul(a, b)

    def astype(self, x, dtype):
        return x.to(dtype=dtype)

    def dtype_of(self, x):
        return x.dtype

    def shape_of(self, x):
        return tuple(x.shape)

    def move(self, x):
        return x.to(self.device)


def available_ops() -> list[str]:
    out = ["numpy"]
    try:
        import torch  # noqa: F401
        out.append("torch")
    except Exception:
        pass
    return out


def make_ops(name: str, device: str = "cpu", dtype=None) -> Ops:
    if name == "numpy":
        return NumpyOps()
    if name == "torch":
        return TorchOps(device=device, dtype=dtype)
    raise ValueError(f"unknown ops backend {name!r}; available: {available_ops()}")

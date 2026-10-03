# Where the routed experts have to live

Measured on `gemma-4-26B-A4B-it-QAT-Q4_0.gguf`, layer 0 only, torch CPU float32, 20 torch threads.
The GPU was busy with the user's own Strata server (23.9 of 24.5 GiB), so nothing here is a GPU number.
Every config ran in its own process: an earlier run of the combined bench reported `stack` at 307 ms
for 128 tokens while an isolated probe of the same matmuls on the same tensors did 88 ms, because the
process had already allocated several gigabytes of expert caches. Numbers that depend on what ran
before them are not numbers.

## How to reproduce

```sh
python bench_one_mode.py block block=8 keep=4      # one config, fresh interpreter
python bench_xforward.py stream keep=24 --zipf 0.9 # four turns, prefill + decode
python probe_block.py block=8 keep=16 n=128        # every repeat printed, plus RSS
```

`one-mode.txt`, `cross-forward.txt` and `equal-memory.txt` are the outputs quoted below.

## The one-forward table (`one-mode.txt`)

Best of up to 5 repeats. The routing spreads `n * top_k` slots over all 128 experts, so every expert
of the layer is touched exactly once - which is the honest worst case for a prefill and the reason the
resident/partial split is so sharp.

| profile | experts resident per layer | 1 tok | 128 tok | 512 tok |
|---|---|---|---|---|
| stack | 128/128 | 5.5 ms | 95 ms | 147 ms |
| block 16 keep=8 | 128/128 | 5.3 ms | 100 ms | 132 ms |
| stream keep=128 | 128/128 | 5.4 ms | 107 ms | 158 ms |
| block 8 keep=16 | 128/128 | 5.5 ms | 110 ms | 167 ms |
| stream keep=48 | 48/128 | 5.3 ms | 2109 ms | 2216 ms |
| stream keep=24 | 24/128 | 5.2 ms | 2151 ms | 2191 ms |
| scratch group=16 | 32/128 | 5.1 ms | 2763 ms | 2835 ms |
| scratch group=32 | 48/128 | 5.4 ms | 2823 ms | 2823 ms |
| block 16 keep=2 | 48/128 | 5.3 ms | 3178 ms | 3185 ms |
| block 8 keep=4 | 48/128 | 5.3 ms | 3252 ms | 3320 ms |
| block 8 keep=1 | 24/128 | 5.5 ms | 3287 ms | 3337 ms |

Two findings.

**The split is resident versus not, not mode versus mode.** The four resident profiles land in
95-167 ms, the seven partial ones in 2.1-3.3 s: a ~20x gap, and the label on the row does not decide
which side of it it falls on. Reading and dequantising one expert is 14.9 ms (22.7 MiB), so a forward
that touches all 128 pays ~1.9 s of reads however the matmuls are grouped.

**Batching is real but small, and it only rides on top of residency.** `block 16 keep=8` beats `stack`
at 512 tokens (132 ms against 147 ms, 1.11x) because one block is one view and the MoE batches inside
it. That is the whole size of the win. It does nothing for a profile that is not resident.

## At equal memory, block is the worst option (`equal-memory.txt`)

Holding 48 experts per layer costs:

| profile | experts resident | 128 tok |
|---|---|---|
| stream keep=48 | 48/128 | 2109 ms |
| scratch group=32 | 48/128 | 2823 ms |
| block 8 keep=4 | 48/128 | 3252 ms |

block is **1.55x slower than stream for the same bytes**. The reason is in `_FileExperts._fill`: a
block copies every expert it reads into a stacked buffer (`store_row`, measured at 3.7 s over a
layer's 128 experts against 1.9 s for the reads themselves). With `keep` small that copy is repeated
every forward - `built=32` per forward at `block 8 keep=4` - while the batching it pays for never
happens, because the groups the router produces span more blocks than are resident.

## Residency is not a warm-up detail (`cross-forward.txt`)

Four turns of 512-token prefill + 64 decode steps, skewed routing (zipf s=0.9), one layer. This is
the measurement a single-forward bench cannot make: an LRU can be empty inside one forward and still
be free across a turn. It is not.

| profile | warm prefill | warm decode | reads over the last 3 turns |
|---|---|---|---|
| stack | 156 ms | 412 ms | 0 |
| stream keep=128 | 160 ms | 364 ms | 256 (first turn only) |
| block 8 keep=16 | 171 ms | 383 ms | 256 (first turn only) |
| block 8 keep=4 | 3205 ms | 5360 ms | 3522 |
| block 8 keep=2 | 3244 ms | 5602 ms | 3672 |
| block 24 keep=1 | 3272 ms | 5877 ms | 3736 |
| scratch group=8 | 2935 ms | 5968 ms | 3784 |
| stream keep=24 | 11481 ms | 19440 ms | 3284 |

`block 8 keep=4` still read 3522 times across the last three turns. The re-read is per forward, not a
warm-up cost, because a prefill touches every expert of the layer and an LRU of 48 cannot hold them.

`stream keep=24` is the outlier at 11.5 s / 19.4 s: with 20 torch threads and other benches running,
its 3284 reads plus 3478 LRU churns thrashed. Treat that row as a floor, not a precise number; the
one-forward table puts the same profile at 2.1 s.

## What this decided

`--expert-mode` now defaults to `auto`, which picks by regime (`backends.auto_expert_profile`):

* the experts all fit -> `stack`. `block` with `expert_block * expert_blocks >= num_experts` costs
  exactly the same bytes, so residency is not a choice between stack and block; stack measured faster.
* they do not -> `stream`, sized from the free bytes, because it measured fastest below residency.

On a 24 GB card in float16 that is `stream keep=47` (19.9 GiB total with the 4.4 GiB of non-expert
weights). Full residency needs 42.5 GiB of experts in float16, which no 24 GB card has, so the honest
answer is that a 24 GB card runs ~20x off the fast path whatever the flags say - and the server now
prints which regime it bought instead of implying the default is fast.

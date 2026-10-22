# Strata — Specification verified against the code (2026-10-22)

Documentto useable as a specification by another LLM. Base: the paper `docs/paper/Strata-Paper.pdf` (September 2026) plus the code in the repository root `RiStarta/Strata`.

Legend for every claim:

- **[V]** verified by reading the code directly (file and line quoted).
- **[D]** the current code diverges from the paper: the value in the code is authoritative.
- **[N]** not verified in the code; the figure comes from the paper or `DETAILS.md`, treat as estimate.
- **[U]** file not read line by line (`verify.cpp` 4290 righe, `expert_source.cpp` 6388, `layer.cpp` 1638, 164 file of kernel); the logic is reconstructed from the comments and the paper.

Paths are relative to the root of the repository `RiStarta/Strata`.

---

## 0. What I read and what I did not

Read directly: [`geometry.hpp`](include/strata/core/geometry.hpp), [`expert_cache.hpp`](include/strata/core/expert_cache.hpp), [`pinned.hpp`](include/strata/core/pinned.hpp), [`mtp.hpp`](include/strata/core/mtp.hpp), [`foresight_swap.hpp`](include/strata/core/foresight_swap.hpp), [`conversation_cache.hpp`](include/strata/core/conversation_cache.hpp), [`draft_policy.cpp`](src/spec/draft_policy.cpp) + [`draft_policy.hpp`](include/strata/spec/draft_policy.hpp), [`controller.cpp`](src/spec/controller.cpp) + [`controller.hpp`](include/strata/spec/controller.hpp), [`suffix_drafter.hpp`](include/strata/spec/suffix_drafter.hpp), [`spec_prob.hpp`](include/strata/core/spec_prob.hpp), [`ple_reader.hpp/cpp`](src/ngram/ple_reader.cpp), [`arch_defaults.cpp`](src/core/arch_defaults.cpp), the doorbell of [`layer.cpp`](src/core/layer.cpp:430), the protocol and the defaults of [`generate.cpp`](src/program/generate.cpp), the server [`serve/server.py`](serve/server.py).

Not read line by line: `verify.cpp`, `expert_source.cpp`, the 164 kernel files. Where their internal logic is needed I mark **[U]**.

---

## 1. Idea central [V]

MoE 125B, ~6B active per token, engine written for one model family and for PC consumer. Three levels of memory, and inside each layer GPU, CPU and DMA work in parallel:

| Level | Content | Why |
|---|---|---|
| VRAM | dense part (mixer, router, shared experts, gated residual, output head), layer MTP, KV, recurrent state, expert cache | touched at every token |
| RAM pinned | all the 24.576 experts (48 × 512) | the CPU computes them in place; the GPU reads them via DMA |
| SSD | n-gram table (PLE) | few rows per token, addressable in advance |

The three-way split of the missed experts (cache hit / PCIe-DMA / CPU) is not an optimization but a necessity: the header of [`expert_cache.hpp`](include/strata/core/expert_cache.hpp:16) measures `h_layer` held-out = **0.0456**, only 4.6% of (layer, token) pairs have all ten experts resident, so a per-layer grouped kernel is impossible **[V]**.

---

## 2. Geometry of the target model — [`geometry.hpp`](include/strata/core/geometry.hpp) [V]

All the values of the document are exact; the struct is `strata::core::ModelGeometry`, and `kCompiledGeometry` is the shape the kernels are compiled for ([line 71](include/strata/core/geometry.hpp:71)); every kernel that pins a dimension `static_assert`s against it. A file of another shape is refused at load by `geometry_unsupported`, naming the metadata key that differs **[V]**.

| Parameter | Value | GGUF key |
|---|---|---|
| n_embd / n_layers / n_vocab | 2560 / 48 / 248320 | embedding_length / block_count / tokenizer.ggml.tokens |
| QSA interval | 4 → `layer % 4 == 3` (12 layers) | full_attention_interval |
| GDN | state 128, K heads 16, V heads 48, conv 4, conv channels 10240, value dim 6144 | ssm.* |
| QSA | 24 Q heads, 2 KV, head_dim 256, RoPE 64 | attention.* |
| Indexer | 4 heads, key_dim 128, one pooled key every 4 cells, top_k 2048 | attention.indexer.* |
| Hyper-connections | 4 flows, low-rank 320 (residual width 4×2560) | hyper_connection.* |
| MoE | 512 experts, 10 active, FFN 640, shared expert 640 | expert_count / expert_used_count / expert_feed_forward_length |

**[D] Evolution of the code relative to the paper:** the geometry is no longer a set of bare constants but a struct read from the metadata, with fields added for the families: `family` (`ModelFamily::Qwen4Exp`, Qwen3.6 = `qwen35moe`), `n_mtp_layers`, predicates `has_indexer()`, `has_hyper_connections()`, `n_qsa_layers()` derived ([lines 57-67](include/strata/core/geometry.hpp:57)). The separate engine for other families remains in `src/qwen35/`, so the practical statement "single-model, the kernels do not parametrize" is still true **[V]**, but the container for a port already exists in the code and is the right starting point.

---

## 3. Census of the bytes [N] except where noted

| Part | Size | Read per token |
|---|---|---|
| Routed experts | 34.0 GB | 663.6 MB (2%) |
| Mixer, attention, shared, router | 1.8 GB | all |
| Gated residual | 1.3 GB | all |
| Output head | 0.44 GB | all |
| PLE table | 28.8 GB (paper) | ≤ 23 KB |
| MTP layer | 0.8 GB from BF16 checkpoint | at every draft |

**[V]** the figure per token is in the code: the header of [`expert_cache.hpp`](include/strata/core/expert_cache.hpp:3) says **663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms** on a token of ~53 ms, and `test_expert_pool` reports the pipeline hides **1.055 ms of 19.035** — the CPU work is 96% exposed because the residual chain is strictly serial. This is the real number around which everything is designed.

**[D]** the size of the PLE table in the code is **320.001.536 rows × 90 bytes = 26.8 GiB** (IQ4_NL), or 160 B/row = 51.2 GB (FP8 E4M3) ([ple_reader.hpp:3](include/strata/ngram/ple_reader.hpp:3)). The 28.8 GB of the paper do not match the code; use 26.8 GiB.

**[V]** the MTP layer: all 512 routed experts in VRAM = **708 MB**, and `load()` takes ~**0.9 GB** ([mtp.hpp:19](include/strata/core/mtp.hpp:19), [mtp.hpp:45](include/strata/core/mtp.hpp:45)).

---

## 4. Decode step, layer by layer

Unit of work = **verify window**: last accepted token + drafts, processed together through the 48 layers.

**[D] Window size.** The paper says up to 3 drafts (window 4). The code goes to **8**: `kMaxT = 8` ([draft_policy.hpp:24](include/strata/spec/draft_policy.hpp:24)), `K_MAX = 8` ([controller.hpp:23](include/strata/spec/controller.hpp:23)), `kShape[9]` for windows up to 8 ([draft_policy.cpp:12](src/spec/draft_policy.cpp:12)), `--spec` capped at 8 (`kVerifyMaxT`, [generate.cpp:3180](src/program/generate.cpp:3180)).

Per-layer flow:

- GPU: gated-read → mixer (GDN or QSA) → gated-write → router, which writes the 10 expert ids per token into the **doorbell** **[V]**: [`layer.cpp:430`](src/core/layer.cpp:430) `doorbell_publish`, host memory `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer` ([layer.cpp:1315](src/core/layer.cpp:1315)), with a `seq` counter and a `flag`.
- CPU: spins on the doorbell without calling the driver **[U]**; for each requested expert: if resident (residency table) → GPU group; else a fixed share → DMA copy RAM→VRAM; the rest → CPU in place **[U]**.
- GPU: shared expert + cached experts + those arrived by DMA; CPU writes results into mapped memory; GPU sums → next layer **[U]**.

**[V]** the share of misses that goes to DMA is exactly as the document says: `pcie_frac` default −1 means **0.2 direct for the Q2_0 pack, 0.55 DMA for native packs** ([generate.cpp:680](src/program/generate.cpp:680)).

**[D]** today there is `PcieAuto`: on single-GPU serve (CUDA + HIP/gfx1100) without `--pcie-frac`, the share is refined during the first decode windows starting from the base ([generate.cpp:681](src/program/generate.cpp:681), [3508-3544](src/program/generate.cpp:3508)), and there is a start-up PCIe probe (best of several bursts) that adjusts it ([generate.cpp:3539](src/program/generate.cpp:3539)). Modes `--pcie-mode auto|dma|kernel|direct`.

**[V]** one CUDA graph per window size, so the host never blocks on synchronous calls: confirmed by the structure of [`verify.cpp`](src/core/verify.cpp) (includes `host_graph_launch.hpp`, "the window graph", `STRATA_ONE_TOKEN_COMMIT`, "the first launch of each window graph") — the capture itself is inside `verify.cpp` **[U]**.

Timing figures (0.6 ms/layer, ~34 ms per 4-token round with the breakdown GPU 14.6 / wait 13.4 / drafts 2.0 / other 4.3, and the proof that splitting the window in two groups is ~7% slower) are **[N]** from the paper.

Cost model of the step, [`controller.cpp:8`](src/spec/controller.cpp:8) **[V]**, exactly:

```
step_ms(k) = dense_ms·dense_ratio[k+1]
           + (1 − hit_rate) · cpu_all_miss_ms · U(n) · (1 + extra_use_cost·(n/U(n) − 1))
           + sync_ms + k · draft_ms
```

with `dense_ratio` and `distinct_ratio` = U(n)/U(1) tables in [`controller.hpp:27,30`](include/strata/spec/controller.hpp:27) **[V]**.

---

## 5. Expert cache — [`expert_cache.hpp`](include/strata/core/expert_cache.hpp) [V]

Data structures **[V]**:

- arena of slots in VRAM + table `residency[layer × 512] → slot | -1` (`kNotResident = -1`, [line 38](include/strata/core/expert_cache.hpp:38)); bounds-checked `slot_of` ([152](include/strata/core/expert_cache.hpp:152)).
- `open_sized(slot_bytes, ...)`: slots sized in bytes per layer ([96-98](include/strata/core/expert_cache.hpp:96)); stride `(blob + guard + 255)/256*256` ([134](include/strata/core/expert_cache.hpp:134)).
- optional guard after each slot for the prompt MMQ that reads past the `down` matrix: `set_slot_guard`, `STRATA_MMQ_DIRECT` ([125-132](include/strata/core/expert_cache.hpp:125)).
- `--vram-elastic` (#533): arena in a CUDA VMM range with physical segments, so `shrink`/`grow` give back and retake VRAM to other programs while every slot keeps its address (the captured graphs hold addresses into it) ([117-146](include/strata/core/expert_cache.hpp:117)); CUDA only, a HIP build refuses it.
- `verify_slot()` reads the slot back and compares byte for byte; the comment states that a table right on indices and wrong on bytes produces a plausible but wrong token, "the failure this project has paid for most often" ([212-215](include/strata/core/expert_cache.hpp:212)).
- `admit()` never evicts: eviction is a measured question (LFU-decay vs LRU sweep) ([155-158](include/strata/core/expert_cache.hpp:155)).
- `replace(layer, old, new)` publishes the new expert only after its copy has landed, so nobody waits for the copy ([160-165](include/strata/core/expert_cache.hpp:160)).

Initial fill **[V]**: profile file `profile.bin` / STRP format = `STRP`, then `version, n_layers, n_expert, slots, n_ranked`, then `n_ranked` pairs `(layer, expert)` uint16×2 ordered by descending routing frequency, then a `n_layers × n_expert` int32 frequency table ([40-55](include/strata/core/expert_cache.hpp:40)). Generated by `tools/make_profile.py`. `--expert-profile-save` (#477) writes back what the adaptive tier learned (`rank_learned_profile`, [57-73](include/strata/core/expert_cache.hpp:57)), atomically via `<path>.tmp` + rename.

Measured numbers in the header **[V as measured by the project]**: at 4.105 slots, ten-fold leave-one-out `h_expert = 0.6447 [0.6110, 0.6750]`; the 0.6573 published in the README is in-sample; a single-corpus measure gave 0.4720; the compulsory-miss policy ("first experts seen") measured **0.4864**; 6414 MiB free vs 5413 MiB needed. The document's "≈50% with ~4.500 slots" is consistent with these **[N]**.

Adaptive tier **[V]**: `--adapt-every 4`, `--adapt-swaps 96` ([generate.cpp:686,716](src/program/generate.cpp:686)) — exactly the document's "every 4 rounds, up to 96 experts". Additional parameters the document lacks: `--adapt-decay 0.7` (counts decay after each adaptation) and `--adapt-min-gain 1.5` (an expert is swapped in only if its decayed count beats the one it replaces by more than this) ([688,689](src/program/generate.cpp:688)); `--adapt-async` opt-in: the rounds advance between windows on a helper thread, not bit-exact run to run; #463/#481 govern when a window takes the previous swaps.

**[V]** `per_layer_admission` (R4.2g): default off; without it the first N slots get consumed by ~26 layers of position 0; measured `--expert-cache 256` → 2.97% vs 21.4% at 8 slots/layer and 70.4% at 64 ([167-186](include/strata/core/expert_cache.hpp:167)).

**[V]** foresight swap ([foresight_swap.hpp:1](include/strata/core/foresight_swap.hpp:1)): opt-in `STRATA_FS_SLOTS`, extra VRAM slots per layer and card refilled on their own copy stream with experts that just missed; a missed expert becomes a plain GPU group if its slot copy has landed; safety by finished-window count (`STRATA_FS_DEPTH` default 4), `STRATA_FS_BUDGET` 32 copies/card/window, `STRATA_FS_ADMIT` 1; with `STRATA_FS_AHEAD=1` (default when on) the router look-ahead (`RouterLookahead`, one layer ahead) feeds the slots.

**[V]** order of magnitude: `device_free_bytes()` sizes the cache from real free VRAM; on UMA systems (DGX Spark, `cudaDevAttrIntegrated`) the figure is MemAvailable − 6 GiB (`STRATA_UMA_HEADROOM_GIB`) ([75-79](include/strata/core/expert_cache.hpp:75)).

---

## 6. Speculative decoding

Draft sources **[V where indicated]**:

- **MTP**: one layer, the model's own; validated at steps 1-3 accepting 0.89/0.86/0.85 of greedy drafts ([mtp.hpp:3-7](include/strata/core/mtp.hpp:3)). Not in the GGUF: loaded from a runtime directory produced by `tools/mtp_rt.py` from the BF16 checkpoint; large projections run as Q8_0 through MMVQ (option `--mtp-q4` → Q4_0); attention is **dense** over every cell seen, identical to the sparse selection below 2.051 cells; `mtp_window = 32768` default ([generate.cpp:677](src/program/generate.cpp:677)). Draft head over a vocabulary subset: `rt/draft_vocab.bin`, the "en" subset is **40.525 ids** ([mtp.cpp:640](src/core/mtp.cpp:640), `DETAILS.md:134` ~110 MiB of VRAM saved) **[V]**; `--mtp-draft-vocab` lets you change it.
- **Prompt lookup**: suffix drafter, no weights, no GPU ([suffix_drafter.hpp:1](include/strata/spec/suffix_drafter.hpp:1)): trigram index, open addressing, `WAYS = 4` most-recent end positions, bounded (~20 B per history token), `min_match = 3`, `max_match = 32` (the engine builds it with 64, [generate.cpp:11032](src/program/generate.cpp:11032)).
- **[D]** default `--suffix-draft 3`, not 5 ([generate.cpp:769](src/program/generate.cpp:769)); on by default when `--spec ≥ 2`: it sets `mtp_max_t = spec` and raises `spec = min(spec+2, 8)` ([3170-3180](src/program/generate.cpp:3170)) — the MTP keeps its windows, a lookup window may be up to 2 tokens longer.
- **Chain**: `--lookup-chain` opt-in, `propose_after(extra, ...)` continues where the MTP drafts end ([suffix_drafter.hpp:31](include/strata/spec/suffix_drafter.hpp:31)); policy in [`draft_policy.cpp:117`](src/spec/draft_policy.cpp:117) with chain priors {0.45, 0.65, 0.8, 0.9}.

**[D]** the "≥ 50% confidence" gate of the paper is now a parameter: `--spec-min-p` default **0.0 = always spec−1 drafts** ([generate.cpp:697](src/program/generate.cpp:697)); when > 0 a draft extends the window only while its probability under the drafter is ≥ it ([14203](src/program/generate.cpp:14203), [14728](src/program/generate.cpp:14728)).

Policy adaptive, [`draft_policy.cpp`](src/spec/draft_policy.cpp) **[V — every constant matches the document]**: EMA of measured round times `kCostAlpha = 0.1`; shape prior `kShape` for unmeasured sizes; lookup success tracked by match-length buckets `<6 / <12 / <24 / ≥24` ([30-32](src/spec/draft_policy.cpp:30)) with priors `0.75/0.88/0.93/0.96` weighted as `kPriorN = 4` observations, decay `0.97`; chooses the `t` that maximizes expected tokens/ms against the MTP base with a minimum margin; sizes not measured for `kStaleRounds = 300` rounds are re-probed, and unmeasured sizes get `kProbes = 3` trials when a confident lookup would need them.

Exactness: the verify pass over the 48 layers decides every emitted token; output identical token-for-token to greedy decode even with forced-wrong drafts and rollback ([mtp.hpp:13-14](include/strata/core/mtp.hpp:13), "the verify window decides every emitted token, never the draft"); the test hooks exist: `--spec-oracle`, `--spec-corrupt N` (every N-th draft replaced by a wrong token) ([generate.cpp:693,695](src/program/generate.cpp:693)). The equality property itself is enforced by tests **[U]**.

**[D] Sampling — the document's open question is resolved: it exists.** [`spec_prob.hpp`](include/strata/core/spec_prob.hpp) implements probabilistic draft acceptance (Leviathan/Chen rejection sampling) for the MTP drafts of a sampled request, opt-in `STRATA_SPEC_PROB=1` (off by default): accept with `min(1, p(d)/q(d))`, residual `norm(max(0, p−q))`, three disjoint Philox counter spaces per row (`pos | 1<<62`, `1<<63`, drafter `1<<61`); "no q means point mass, never q = 0" (llama.cpp #29924). Plus `STRATA_SPEC_COUPLED=1`: coupled draft sampling, the drafter samples with the target's chain and its Philox draw ([mtp.hpp:111-125](include/strata/core/mtp.hpp:111)). The server accepts sampling keys in the GEN line ([generate.cpp:12139-12196](src/program/generate.cpp:12139)).

Performance figures (2.4-3.2 tokens/pass, 1.6-1.8× gain, 47-57 → 82-92 tok/s) **[N]** paper.

---

## 7. Prefill

**[V]** chunk: the auto scan walks `{32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256}` ([generate.cpp:7560](src/program/generate.cpp:7560)); a bare `auto` always reaches **8192**; `auto:16384`/`auto:32768` or `STRATA_PREFILL_AUTO_MAX` are opt-in (#282) ([7491-7564](src/program/generate.cpp:7491)). The paper's 2.048 is obsolete.

**[V]** mechanism: per layer the experts not in the cache are copied from the pinned arena while the GPU computes the previous ones, then dequantized to FP16 and multiplied with cuBLAS; buffers are borrowed from the expert cache and returned; the streaming ring is sized in bytes (#340, #583) ([prefill.hpp:91,126,129](include/strata/prefill/prefill.hpp:91)).

**[D]** CPU share for short prompts: `STRATA_PREFILL_CPU_SHARE` is on by default for **chunks below 1024 tokens** ([prefill.hpp:246](include/strata/prefill/prefill.hpp:246)); a `--prefill-pipe` plan (layer split, opt-in) takes no CPU share ([prefill.hpp:120](include/strata/prefill/prefill.hpp:120)).

Speed figures (539 → 496 tok/s flat from 4K to 262K, ~7 GB/s of the 26 of PCIe used) **[N]** paper.

---

## 8. PLE n-gram table on SSD

**[V]** ~51B parameters addressed by the last three token ids: 320.001.536 rows; **16 rows per token on 16 different pages**, all depending on the token itself, so the only moment to hide them is the embedding + layer 0: split API `issue()` before layer 0, `collect()` before layer 1 ([ple_reader.hpp:3-20](include/strata/ngram/ple_reader.hpp:3)).

**[V]** row cache: set-associative **8 ways**, **round-robin** replacement inside a set, bounded, sized a priori ([ple_reader.cpp:32-67](src/ngram/ple_reader.cpp:32)); measured on the frozen corpus: 1M rows (~95 MB) would serve up to ~82% of reads; within one long prompt 20-34% of rows recur ([ple_reader.hpp:3-13](include/strata/ngram/ple_reader.hpp:3)).

**[D]** I/O: modes are `direct | mmap | ram` ([generate.cpp:806](src/program/generate.cpp:806)). The probe threshold is exactly **15.000 rows/s** (~60 MB/s of 4 KiB reads; a healthy NVMe > 100k) ([generate.cpp:4010](src/program/generate.cpp:4010)) but the behaviour differs from the document: if `--ple-io` was **not** given explicitly the engine **switches automatically to mmap** ([4019-4023](src/program/generate.cpp:4019)); an explicit `--ple-io direct` is kept and only warned about ([4034](src/program/generate.cpp:4034)). `ram` locks the table (working-set lock, [3987](src/program/generate.cpp:3987)).

Formats and per-row errors (Q4_0, Q5_0, Q5_1, Q8_0 0.53%, IQ4_NL 7.6%, BF16, KL ~0.012): **[N]** — the code I read only fixes IQ4_NL 90 B and FP8 E4M3 160 B.

---

## 9. KV, recurrent state, conversation cache

**[V]** GDN has a fixed-size recurrent state; only the 12 QSA layers have KV (2 heads + indexer with pooled keys every 4 cells) — from the geometry and `QsaState` in [`session.hpp`](include/strata/core/session.hpp).

**[D]** KV quantization: today the default is **fp16**; `--kv int8` (codes + fp16 scale every 64 values, half the VRAM) is opt-in "until gate G-C accepts int8", plus `--kv q4_0` (after a Hadamard rotation) and `--kv k8v4` (INT8 K + rotated Q4_0 V, 816 B/cell) ([generate.cpp:482-484,814-819](src/program/generate.cpp:482)). The paper's "8-bit beyond 32K" is not the current default.

**[V]** KV streaming exists: `--kv-resident N` = resident cells per QSA layer, 0 = all in VRAM ([generate.cpp:483,2464](src/program/generate.cpp:483)), with `qsa_kv_resident_min` and an auto-enable path ([5154](src/program/generate.cpp:5154)); the policy that decides the resident fraction at 64K/262K is inside `verify.cpp`/`qsa` **[U]**. The cost figures at 262K (KV+buffers ~4 GB, experts 4.500→1.600, 95→56 tok/s) **[N]**.

**[V]** conversation cache ([conversation_cache.hpp](include/strata/core/conversation_cache.hpp)): a checkpoint holds token ids, image keys, GDN state, PLE data, codes and KV pages with scales and indexer rows; reuse requires token equality + image identity + steering mode (`Match best(prompt, images, cvec)`, [251](include/strata/core/conversation_cache.hpp:251)); retention is LRU with checkpoints "pinned" for shared prefixes (`pin=N`, [27-31](include/strata/core/conversation_cache.hpp:27)); disk save/restore `SAVE <path>` / `RESTORE <path>` between requests ([generate.cpp:11899-11909](src/program/generate.cpp:11899)), `SAVED <n> <bytes> <ms>`; `--session-min-free-mib` (default 4096). The paper listed this as future work: today it is implemented **[V]**. There is also a split/merge for layer-split checkpoints ([80-136](include/strata/core/conversation_cache.hpp:80)).

---

## 10. Quantization and CPU kernels [N] except the following

**[V]** the Q2_0 grid {−1,0,1,2} × scale and the i-quant non-repackability are consistent with the code layout (`native pack` vs Q2_0 pack in `expert_source`), but the AVX-512 VNNI kernel (one unpack + one VNNI every 64 weights) and the GB/s figures (42 of 52) are **[U/N]**.

**[V]** the i-quant GPU dequantizers transcribed from llama.cpp and the CPU dot products of ggml are referenced in the code; the new AVX-512 kernel 7× faster for the `down` projections of i-quant Q2_0 is in `src/kernels/cpu/` **[U]**.

**[N]** KL 0.022 vs llama.cpp reference, under the 0.058 that separates CPU and GPU builds.

---

## 11. Pinned memory and platforms — [`pinned.hpp`](include/strata/core/pinned.hpp) [V]

**[V]** arena page-locked, large pages 2 MB via `mmap(MAP_HUGETLB)`/hugetlbfs on Linux, `VirtualAlloc(MEM_LARGE_PAGES)` on Windows; 33.97 GB at 4 KB = 8.3M TLB entries; the engine **always reports the backing it got** (`PageBacking {LargePages, NormalPages, PinnedByCuda}` + `note`) ([1-15,45-99](include/strata/core/pinned.hpp:1)); on Windows large pages need `SeLockMemoryPrivilege`, so the fallback is the common case, not an error path.

**[V]** when the whole arena cannot be registered: registration in **slices** (`slice_bytes`, one slice per layer of a native pack), `registered_bytes` is the pinned prefix so a copy inside one slice can DMA directly ([51-56,60-68](include/strata/core/pinned.hpp:51)); `STRATA_ARENA_PIN_GIB` cap (−2 = "auto": stays below the GPU's shared-memory budget on Windows); "pin by need" registers only the hottest ranges, the holes stay resident via working-set lock ([69-90](include/strata/core/pinned.hpp:69)); deferred registration on a thread (#285, one registration of 33 GiB cost ~2.5 s).

**[D]** the "34-43 GB fail in one range" figure is not in the code **[N]**; the mechanism (slices + working-set lock) is **[V]**.

**[V]** UMA/integrated: on an integrated GPU whose firmware cannot survive a KFD queue eviction (Radeon 780M, gfx1103) the arena is **not** registered (a registered range is a userptr invalidated by page migration, and every invalidation evicts the GPU queues); residency comes from the working-set lock ([32-43](include/strata/core/pinned.hpp:32)). Strix Halo on Windows (gfx1151): registering the 47 GiB arena took all the free VRAM (91 GiB → 0), the expert cache got 0 slots and the engine exited; unregistered it gets 24.576 slots ([arch_defaults.cpp:76-86](src/core/arch_defaults.cpp:76)).

**[V]** new, not in the document: `STRATA_NUMA_SPLIT` — a 2 MiB huge page would put several slices on the node of its toucher, so the complement is placed per-NUMA-node with pinned helper threads ([expert_source.cpp:527-653](src/core/expert_source.cpp:527)).

**[V]** per-architecture tables of env switches ([arch_defaults.cpp](src/core/arch_defaults.cpp)): gfx12/RDNA4 gets only the prompt half (6 switches, reported bitwise-exact); gfx1151/Strix Halo gets 19 switches each with its measured gain and declared bitwise-exact; the integrated-GPU prompt set (780M) is **not** bitwise ("answers differ in the last bits", 1.6×–2× faster prompts); a switch the user sets always wins ([117-121](src/core/arch_defaults.cpp:117)).

---

## 12. Protocol server ↔ engine [V]

The server Python starts `strata --serve` and speaks on stdin/stdout, one line per message. Verified in [`generate.cpp`](src/program/generate.cpp) and [`serve/server.py`](serve/server.py):

| Direction | Line |
|---|---|
| engine → server | `READY <ctx> stop` at start; `T <id>`; `PP <done> <total> <ms> <tok/s>` (prompt progress, also heartbeat); `RESUME <n>`; `DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> <drafts accepted> <drafts offered> <reused> ... <read>`; `BDONE <slot> ...`; `ERR ...`; `INFO ...` (capabilities dump); `SAVED <n> <bytes> <ms>` |
| server → engine | `GEN <max_new> [keys] <id,id,...>`; `GENI <max_new> <embeddings file> <ids>` with `img_keys=<hex>,...`; `BGEN/BGENI <slot> ...`; `BT <slot> <id>`; `BSTOP <slot>`; `STOP`; `QUIT`; `SAVE <file>` / `RESTORE <file>`; `VRAM [mib]` (#533 hot resize between requests) |

Sampling keys optional between `max_new` and the ids: `temperature=`, `top_p=`, `top_k=`, `min_p=`, `spec_min_p=`, penalties ([12139-12196](src/program/generate.cpp:12139)).

Server layer **[V]**: OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` (stream and non-stream), `/v1/models`, `/props`, `/slots`, `/health`, `/mcp`; 400 if the prompt does not fit, never truncated; reduction of `max_tokens` to the room left (`fit_max_tokens`); watchdog on engine silence (two: generous + quick for a frozen engine), SSE heartbeat every 10 s; vision through `strata-vision` (mmproj), images converted (WebP/TIFF/AVIF decoded); MCP tools offered only to requests with `"strata_mcp": true`.

---

## 13. What transfers to another model — unchanged from the document, plus corrections

Reusable as-is (any large MoE with top-k routing and few active experts) **[V as code structure]**:

1. three-level split + `residency[layer × n_expert] → slot | -1`.
2. hybrid layer: router → doorbell (pinned mapped memory, seq/flag), host splits misses in three groups, one graph per window size.
3. adaptive cache: initial profile STRP, periodic swap with publish-after-copy, byte-sized slots, guard, optional VMM resize.
4. speculation: measured cost EMA + shape prior, choose window by tokens/ms with margin, re-probe of stale sizes, exact verify.
5. prefill: streaming of experts, GPU compute, byte ring.
6. checkpoints of state and reusable KV, line protocol, byte-for-byte slot verification.

Specific to Qwen3.8-Flash-Next, to rewrite **[V]**: all the compile-time geometry constants and their kernels; GDN+QSA mixers with indexer top-k 2048 and 4-flow hyper-connections; the PLE table and its reader (if the target has none, drop the SSD level); the MTP loaded from BF16 checkpoint and the reduced draft head (40.525 ids); chat template, tokenizer, weight format.

**[D] correction to the document:** the code already provides the seam for a second family — `ModelGeometry` read from metadata + `ModelFamily` + `n_mtp_layers` + `has_indexer()/has_hyper_connections()` predicates, and a separate engine directory per family (`src/qwen35/`, `sycl/`, `metal/`, `gptoss_ext/`, and this workspace's `dsv4_ext/`). A port should follow that pattern: new engine directory, new compiled geometry, reuse of the structural files.

Recommended implementation plan (unchanged from the document): reference implementation → byte census → expert source + CPU kernel measured in GB/s → hybrid layer + residency + measured hit rate (leave-one-out, not in-sample) → adaptive cache → speculation → validation (token equality with/without speculation, with forced-wrong drafts; KL vs reference) → only then optimize (byte slots, DMA share, arch variants).

Reference numbers to check a replication (RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Q2_0): 95 tok/s at 4K, ~34 ms/round, hit ~0.72, 2.4-3.2 tok/pass, 539 tok/s prefill, ~4.500 experts in VRAM — all **[N]** paper, single machine, ±20% on other cards.

---

## 14. Summary of divergences found (code vs paper)

| # | Paper/document | Code today | Where |
|---|---|---|---|
| 1 | verify window ≤ 4 (3 drafts) | up to **8** (`kMaxT=K_MAX=8`) | draft_policy/controller/generate |
| 2 | lookup up to 5 tokens | default `--suffix-draft 3`, window +2 capped at 8 | generate.cpp:769,3170 |
| 3 | MTP gate fixed ≥ 50% | `--spec-min-p` default **0** (off) | generate.cpp:697 |
| 4 | PLE table 28.8 GB | **26.8 GiB** (320.001.536×90 B) | ple_reader.hpp:3 |
| 5 | slow-IO probe only warns | auto-switch to `mmap` unless `--ple-io` explicit | generate.cpp:4019 |
| 6 | KV 8-bit beyond 32K | default **fp16**; int8/q4_0/k8v4 opt-in | generate.cpp:482 |
| 7 | DMA share fixed 55/20% | same defaults, plus **PcieAuto** refinement + start probe | generate.cpp:680,3508 |
| 8 | decode only greedy | probabilistic acceptance exists, opt-in `STRATA_SPEC_PROB=1` | spec_prob.hpp |
| 9 | geometry single-model | struct read from metadata + `ModelFamily` (qwen35moe, n_mtp_layers) | geometry.hpp |
| 10 | (not in doc) | `--prefill-pipe`, `STRATA_NUMA_SPLIT`, `--vram-elastic` #533, `--expert-profile-save` #477, foresight swap knobs | various |

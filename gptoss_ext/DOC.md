# GPT‑OSS Inference Engine – Markdown Summary  

> **What this repository does**  
> A **stand‑alone C++ inference engine** for the **GPT‑OSS‑1200B** (and similar) model.  
> It supports both **CPU‑only** and **CUDA‑accelerated** execution, a **Mixture‑of‑Experts (MoE)** architecture, **KV‑cache** with sliding windows, **device‑resident pre‑fill**, **prefetch / background loading of experts**, and a **light‑weight JSON / safetensor loader**.  
> The engine is used by the Python “Strata” server (`serve/server.py`) via the line‑protocol implemented in `serve_loop.hpp`.

---

## Table of Contents

1. [High‑level Architecture](#high-level-architecture)  
2. [Key Concepts](#key-concepts)  
   - [Dense Transformer Layers](#dense-transformer-layers)  
   - [Mixture‑of‑Experts (MoE)](#mixture‑of‑experts-moe)  
   - [KV‑Cache (host & device)](#kv‑cache-host‑vs‑device)  
   - [Prefill & Decode Paths](#prefill‑vs‑decode)  
3. [Core Source Files](#core-source-files)  
4. [Data Flow – From Model Load to Token Generation](#data-flow)  
5. [Runtime Options (`RunOpts`)](#runtime-options)  
6. [Profiling & Statistics](#profiling‑statistics)  
7. [Expert‑Slot Management (LRU / Pinning / Promotion)](#expert‑slot‑management)  
8. [KV‑Snapshot / Reuse‑Prefix](#kv‑snapshot‑reuse‑prefix)  
9. [Build & Execution](#build‑execution)  
10. [File‑by‑File Quick Reference](#file‑reference)  
11. [Future‑Work / Extensibility](#future‑work)  

---

## High‑level Architecture

```
+-------------------+          +-------------------+          +-------------------+
|   Model (C++)     |  <--->   |   GPU Backend     |  <--->   |   CUDA Device     |
|   (model.cpp)     |          |   (gpu.hpp/.cpp) |          |   (GPU kernels)   |
+-------------------+          +-------------------+          +-------------------+

   ^                     ^                     ^
   |                     |                     |
   |                     |                     |
   |                     |                     |
   |                     |                     |
   v                     v                     v
+------------------------------------------------------------+
|  Host‑side utilities                                         |
|   - safetensors loader (safetensors.hpp / StFiles)          |
|   - JSON parser (json.hpp)                                   |
|   - CPU matvec / MXFP4 (weights.hpp)                         |
|   - Ops (rmsnorm, rope, swiglu, routing) (ops.hpp)          |
+------------------------------------------------------------+

```

* **`Model`** (`model.cpp` + `model.hpp`) – the public API: `load`, `prefill`, `forward`, KV snapshot handling, profiling, etc.  
* **GPU Backend** – two interchangeable implementations:  
  * **CPU‑emulated** (`src/gpu.cpp`) – all “device” calls become ordinary `malloc`/`memcpy`. Used for testing and when CUDA isn’t available.  
  * **CUDA** (`src/gpu_cuda.cu`) – real device kernels for BF16 matmul, MXFP4 MoE, attention, KV quantisation, etc.  
* **Device‑resident pre‑fill** – a specialised CUDA path (`src/prefill_dev.cu`) that keeps the entire pre‑fill chunk on the GPU, reducing PCIe round‑trips dramatically.  

---

## Key Concepts  

### Dense Transformer Layers  

| Symbol | Meaning |
|--------|---------|
| **`q/k/v/o`** | Linear projections (BF16) for query, key, value, and output. |
| **`router`** | Linear projection that produces the expert‑selection logits. |
| **`norm`** | RMS‑norm (ε configurable). |
| **`heads` / `kv_heads`** | Number of attention heads and KV‑heads (grouped). |
| **`head_dim = D`** | Must be a multiple of **32** for CUDA kernels. |
| **`window`** | For **banded** layers: size of the sliding KV window (`sliding == true`). |
| **`sliding` flag** (per‑layer) | Determines whether the layer uses a ring buffer (`true`) or a full‑dense KV (`false`). |

### Mixture‑of‑Experts (MoE)

* **`n_exp = E`** experts per layer, each with two blocks:
  * **Gate‑up** (`gate_up_proj`) – MXFP4 (FP4 blocks + per‑row 8‑bit scales) of size **2 × I × H**.  
  * **Down** (`down_proj`) – MXFP4 of size **H × I**.  
* **`topk = K`** (default 4) experts are selected per token via `route()`.  
* **Execution Paths**  
  * **GPU slot** – if an expert resides in a VRAM *slot*, `expert_run_async` launches four kernels (gate‑up matmul, SwigLU, down matmul, copy‑back).  
  * **CPU fallback** – if no slot is free (or `GPU_MISS=1`), the expert runs on the host (`expert_cpu`).  
  * **Prefetch** – the next layer’s predicted top‑k experts are copied to a free slot in the background (`prefetch_next`/`prefetch_land`).  

### KV‑Cache (host vs. device)

| Cache | Location | Layout | Notes |
|-------|----------|--------|-------|
| **Host KV** | `Layer::K` / `Layer::V` (std::vector\<float\>) | **FP32**, ring buffer for sliding layers. | Used by every decode token, checkpointing, and by the CPU fallback. |
| **Device KV** | `Layer::dKc` / `Layer::dVc` (GPU memory) | **Packed** (E4M3 / FP16 / FP32) – format chosen via `--key-format` / `--value-format`. | Only dense layers (non‑sliding) can keep a copy; accessed by `attn_decode`. |
| **Device‑resident pre‑fill KV** | Managed by `pdev::` (prefill_dev) | Same packed format as above, but lives only for the current pre‑fill chunk. | After a chunk finishes, rows are copied back to the host cache. |

### Prefill vs. Decode  

| Phase | Where data lives | Main GPU kernels | PCIe traffic |
|-------|------------------|------------------|--------------|
| **Prefill (batched)** | Whole chunk (`X`) + per‑layer activations (`a_X`, `a_Hn`, …) | `run_layer` (norm → q/k/v → attention → router) **once per chunk**; `run_experts` (one kernel per expert) | **Two** transfers per chunk: `X` upload & `X` download (plus KV upload for the chunk). |
| **Decode (token‑by‑token)** | Single token (`x`) on host; dense‑layer KV may be on device (`dev_kv_pos`). | `attn_in`/`attn_out` (host) *or* `attn_decode` (device) for dense layers; `expert_run_async` for MoE. | **Potential** PCIe upload of an expert on a miss; optional OOM fallback. |

---

## Core Source Files  

| File | Category | What it contains |
|------|----------|-------------------|
| `model.hpp / model.cpp` | **Engine core** | Public `Model` class, loading, forward, pre‑fill, KV snapshot, expert‑profile handling, profiling, slot management. |
| `gpu.hpp` | **GPU abstraction** | API used by the engine (`init`, `alloc`, `matmat_bf16`, `expert_run_async`, etc.). Two implementations: `gpu.cpp` (CPU stub) and `gpu_cuda.cu` (real CUDA). |
| `prefill_dev.hpp` | **Device‑resident pre‑fill** | API (`available`, `init`, `run_layer`, `run_experts`, `kv_prefetch`, …). Two implementations: stub (`prefill_dev_stub.cpp`) and CUDA (`prefill_dev.cu`). |
| `weights.hpp` | **CPU matvec / MXFP4** | BF16 / FP16 → FP32 conversion, dense matvec (`matvec`, `matmat`), MXFP4 matvec (`matvec_mx`). AVX2 / AVX‑512 kernels selectable via `set_cpu_isa`. |
| `ops.hpp` | **Mathematical primitives** | RMS‑norm, YaRN rope table, rotary, routing (top‑k soft‑max), SwigLU. |
| `config.hpp` | **Model configuration** | `GptConfig` struct (hidden size, layers, heads, MoE params, sliding‑layer flags, etc.) and JSON parser helpers. |
| `json.hpp` | **Minimal JSON reader** | Hand‑rolled parser used for `config.json`, `*.index.json`, etc. |
| `safetensors.hpp` | **Multi‑shard safetensor loader** | Memory‑maps one or more `.safetensors` files, exposes `StTensor` descriptors (dtype, shape, offsets). |
| `serve_loop.hpp` | **Strata line‑protocol server** | Reads commands from stdin, writes responses, drives the `Model`. |
| `sampler.hpp` | **Token sampling** | Greedy / temperature / top‑k / top‑p / min‑p / repetition penalties, RNG state per request. |
| `kv_quant.hpp` | **KV packing / unpacking** | Templates for converting FP32 ↔ packed formats (E4M3, FP16, FP32) used inside CUDA kernels. |
| `gguf_loader.hpp` | **Experimental GGUF loader** (half‑precision support) | Structures for Q4\_K\_L block and async de‑quantisation on GPU (`__half`). |
| `config.cpp`, `weights.cpp`, `ops.cpp`, `gpu.cpp`, `gpu_cuda.cu`, `prefill_dev.cu`, `serve_loop.cpp`, `sampler.cpp` (not shown) | **Implementation files** | Real code for the APIs above. |

---

## Data Flow – From Model Load to Token Generation  

1. **Load** (`Model::load`)  
   * Parse `config.json` → `GptConfig`.  
   * Discover all `*.safetensors` shards → `StFiles`.  
   * Map tensors with `mmap`.  
   * Allocate **dense** weights on GPU (if `--cuda`).  
   * Allocate per‑layer **KV ring buffers** on host.  
   * Reserve **scratch** for batched pre‑fill (`reserve_scratch`).  
   * Initialise **GPU** (`gpu::init`) → create streams, events, check kernel support (`attn_supported`).  
   * Optionally start **background expert preload** thread (fills the most‑used slots).  

2. **Prompt pre‑fill** (`Model::prefill`)  
   * Split prompt into **chunks** (`chunk_max`).  
   * If `--prefill-dev` is enabled and enough VRAM exists → **device‑resident** path (`prefill_pdev`).  
   * Otherwise fall back to **host‑only batched** (`prefill_gpu`).  
   * For each chunk & each layer:  
     * **RMS‑norm** → `q/k/v/o/router` projection (dense BF16 matmul).  
     * **RoPE** applied in‑place on the GPU.  
     * **Attention** (`attn_rows` on device or `attn_decode` for dense layers).  
     * **Router** → top‑k selection (`route`).  
     * **MoE** – group tokens by expert, call `moe_prefill_layer` (one kernel per expert).  
   * After the last layer, the final token’s logits are produced (`lm_head`).  

3. **Reuse of KV cache** (`Model::reuse_prefix`)  
   * Checks how many tokens of a new request are already present in `cached`.  
   * If the required look‑back exceeds any sliding layer’s `window + kv_rollback`, the cache is **invalidated** (`cached.clear()`).  

4. **Token‑by‑token decode** (`Model::forward`)  
   * For each layer:  
     * `attn_in` → dense projections + KV write (host ring).  
     * `attn_out` → softmax + O + residual addition.  
     * `route_tok` → router + top‑k logits.  
     * `experts_decode` → launch **up to 4** GPU expert contexts in parallel (`gpu::expert_run_async`).  
       * Missing slot → async PCIe upload (`upload_expert`) + optional CPU fallback (`GPTOSS_CPU_MISS`).  
   * Update **device‑dense KV** (`dd_sync`) if `--attn-decode` is enabled.  
   * After the final layer: RMS‑norm + `lm_head` → logits.  

5. **KV Snapshot (checkpoint)** (`save_kv` / `load_kv`)  
   * Serialises the **host KV** (only rows currently stored in the ring) plus the token list (`cached`).  
   * Used by the server to *resume* a conversation without recomputing the whole prompt.  

6. **Expert Usage Profile** (`save_profile` / `load_profile` / `decay_profile`)  
   * Counts how many times each `(layer, expert)` pair is selected.  
   * On shutdown / after each request the profile is written (`*.profile`).  
   * On start the most‑used experts are **pinned** in VRAM (`--pin-pct`).  
   * Optional decay (`--profile-decay`) halves all counters when the total exceeds a threshold, giving recent sessions more weight.  

---

## Runtime Options (`RunOpts`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--ctx` | 8192 | Size of the **dense‑layer KV cache** (positions stored in VRAM). |
| `--threads` | 0 | Number of CPU threads (`0` = all cores). |
| `--prefill-chunk` | 0 (auto) | Tokens per batched pre‑fill; 8192 on CUDA, 512 on CPU. |
| `--kv-rollback` | 8192 | Extra positions kept in a sliding‑window layer, enabling a *rewind* into the cache. |
| `--kv-checkpoint` | (empty) | Path to a KV snapshot file that is loaded at start and rewritten after each request. |
| `--cuda` | false | Enable the real CUDA backend (`gpu_cuda.cu`). |
| `--gpu-attn` | true | Use **device‑resident attention** (dense layers keep a packed KV in VRAM). |
| `--gpu-slots` | –1 | Upper bound on the number of expert slots in VRAM. |
| `--pin-pct` | 80 | Percentage of VRAM slots that are *pinned* to the most‑used experts (from the profile). |
| `--preload` / `--preload-wait` | true / false | Background preload of expert slots; `--preload-wait` blocks `Model::load` until it finishes. |
| `--pin-ram-gib` | 0 | GiB of **page‑locked host memory** for expert weights (DMA‑direct copies). |
| `--expert-promote` | 0 | Enable LRU‑to‑pinned promotion when an expert’s hit‑count exceeds this threshold. |
| `--expert-lru-layered` | false | When evicting, prefer a victim from the *same* layer first. |
| `--profile-decay` | 0 | If >0, halves every expert counter after a profile save once the total exceeds this value. |
| `--device-graphs` | true | Use CUDA graphs for the decode‑expert pipeline (reduces kernel launch overhead). |
| `--async-dense` | true | Batch the three dense projections (`q/k/v`) for a token into a single async upload+kernel+copy. |
| `--prefill-dev-pct` | 50 | Fraction of remaining VRAM (after dense weights) dedicated to the **device‑resident pre‑fill** buffers. |
| `--no-concentration` | false | Disable the YaRN concentration factor (default = `0.1*ln(factor)+1`). |
| `--glu-halves` | false | Layout of gate‑up rows (`interleaved` vs `first‑half/second‑half`). |
| `--rope-pairs` | false | Apply RoPE on adjacent pairs instead of the half‑split layout. |
| `--verbose` | true | Print informative messages during load / warm‑up. |

> **Tip:** All flags are exposed through `RunOpts` in `model.hpp`; they are parsed in the driver (`main.cpp`) and passed to `Model::load`.

---

## Profiling & Statistics  

* Enable with `export GPTOSS_PROFILE=1`.  
* **Per‑token timings** (ms) are accumulated in `Impl`:
  * `pr_attn` – attention (softmax, matmul).  
  * `pr_route` – routing (RMS‑norm + matvec + top‑k).  
  * `pr_exp` – expert execution (GPU enqueue + CPU fallback).  
  * `pr_head` – final LM head.  
* **Counters** (printed by `Model::stats()`):  
  * `hits` – expert slots found already in VRAM.  
  * `pcie_served` – expert uploads over PCIe.  
  * `host_served` – experts executed on CPU.  
  * `pin_hits` – hits on *pinned* experts (never evicted).  
  * `oom_fallbacks` – number of times a CUDA OOM forced a host fallback.  
* **Prefetch statistics** (when `--prefetch-n > 0`):  
  * `%` of next‑layer routed experts that were already uploaded (`pf_pred_hit`).  
  * Number of experts uploaded ahead (`pf_uploaded`).  

The `serve_loop` command `DONE` line mirrors these numbers (`hits`, `lookups`, `offloaded`) for external monitoring.

---

## Expert‑Slot Management (LRU / Pinning / Promotion)

| Concept | Data Structure | Behaviour |
|---------|----------------|-----------|
| **Slot** (`struct Slot`) | `void* base`, BF16 blocks/scales, bias pointers, `layer`, `exp`, `tick`, `pinned`, `busy`. | Holds a *single* expert in VRAM. |
| **Lookup Table** (`where`) | `unordered_map<long long, int>` (`layer*E + exp → slot idx`). | O(1) find of a resident expert. |
| **LRU Tick** | `tick` (monotonically increasing per request). | Least‑recently‑used slot is evicted first. |
| **Pinned Slots** | Determined at start by `--pin-pct` (most‑used experts from the profile). | Never evicted; always a *hit*. |
| **Victim Selection** | `pick_victim` → first free slot; if none, `victim_slot` (layer‑local if `--expert-lru-layered`). | Guarantees a slot for a miss unless **all** slots are pinned/busy. |
| **Promotion** (`--expert-promote`) | When an LRU expert’s hit count ≥ `expert_promote`, it swaps with the *least‑used* pinned expert. | Keeps the *most useful* experts pinned without extra PCIe traffic. |
| **Background Preload** | `preload_thread` → copies the hottest experts (profile + runtime usage) to VRAM before any request. | Reduces cold‑miss latency for the first few requests. |

---

## KV‑Snapshot / Reuse‑Prefix  

* **Snapshot** (`snap_stash`)  
  * Triggered when a request ends and the cached prompt length ≥ 64 tokens (or `KV_SLOTS_GIB` limit).  
  * Serialises **only the active portion** of each layer’s KV (from `lo = max(0, n‑cap)` to `n`).  
  * Stores: magic header, model fingerprint, token list, packed K/V rows.  

* **Restore** (`snap_restore`)  
  * Restores the host KV cache from a previously saved snapshot (used by `reuse_prefix`).  
  * Resets `pdev_kv_pos` / `dev_kv_pos` so the next pre‑fill will re‑upload the missing rows.  

* **Reuse‑Prefix** (`Model::reuse_prefix`)  
  * Called by the server when a new request shares a prefix with a previous conversation.  
  * Finds the longest common prefix (`k`).  
  * If any sliding layer would need to overwrite rows that are still in its ring (`k` too far back), the cache is cleared (`return 0`).  

---

## Build & Execution

### Build (Linux)

```bash
# 1️⃣ CPU‑only (always works; also runs the core test suite on tests/tiny)
sh build.sh               # creates build-out/gptoss (+ build-out/test_gptoss_core, which is executed)

# 2️⃣ CUDA 12.8 for the RTX 4090 (sm_89); NVCC=/percorso/nvcc and CUDA_ARCH=sm_89|native override
sh build.sh --cuda        # creates build-out/gptoss_cuda (links src/gpu_cuda.cu + src/prefill_dev.cu)

# The ./gptoss wrapper picks gptoss_cuda when it exists, else the CPU binary.
```

### Runtime — chat mode (the default front door)

```bash
./gptoss --model ./models/12b --prompt "Ciao, come va?"   # one‑shot: answer and exit
./gptoss --model ./models/12b                             # interactive REPL (/exit /reset /system TEXT)
./gptoss --model ./models/12b --no-cuda                   # force the CPU path on a CUDA build
```

* Chat mode tokenizes with the snapshot's own `tokenizer.json` (o200k_harmony, [`src/tokenizer.cpp`](src/tokenizer.cpp)), renders the Harmony template ([`build_chat_prompt`](include/gptoss/tokenizer.hpp:64)) and **streams the final channel** as it is generated (`--show-thinking` also streams the analysis).
* Between REPL turns only the delta is prefilled (`Model::reuse_prefix`); `/reset` clears both conversation and KV cache. Stop ids: `<|return|>` (200002) and `<|call|>` (200012).
* A CUDA build runs chat on the GPU by default; `--no-cuda` forces the CPU path.

### Runtime — token / serve paths

```bash
./build-out/gptoss --model SNAPSHOT --prompt-ids 200006,17360,200008 --n-predict 32   # raw ids in/out
./build-out/gptoss_cuda --model SNAPSHOT --cuda --vram-pct 90 --serve --ctx 32768 \
    --kv-checkpoint /var/tmp/gptoss.kv --eos-ids 200002,200012                        # Strata line‑protocol server
```

* `--serve` starts the **Strata line‑protocol** server (`serve_loop`), driven by Python (`tools/serve_gptoss_strata.py`) or a terminal.

### Environment variables for debugging

| Variable | Effect |
|----------|--------|
| `GPTOSS_PROFILE=1` | Enable per‑token profiling (printed on exit). |
| `GPTOSS_CHECK_PREFILL=1` | Run the GPU pre‑fill, then a CPU pre‑fill on the same prompt and log max absolute differences. |
| `GPTOSS_CHECK_DECODE_ATTN=1` | Compare GPU decode‑attention with the host version token‑by‑token. |
| `GPTOSS_CPU_MISS=1` | On an expert miss, compute it on the CPU *while* the GPU continues with other experts. |
| `GPTOSS_ATTEN_DECODE=0` | Disable the packed‑KV decode path (forces host attention). |

---

## File Reference – Quick Look‑up  

| Header | Primary purpose |
|--------|------------------|
| **`model.hpp / model.cpp`** | Public inference API, KV management, expert profiling, pre‑fill/forward logic. |
| **`gpu.hpp`** | Abstract GPU interface (alloc, copy, matmul, expert kernels). |
| **`gpu_cuda.cu`** | Real CUDA kernels (BF16 matmul, MXFP4, attention, MoE, KV quantisation). |
| **`gpu.cpp`** | CPU‑emulated stub (malloc + memcpy). |
| **`prefill_dev.hpp`** | Device‑resident pre‑fill API. |
| **`prefill_dev.cu`** | CUDA implementation that keeps the whole chunk on the GPU. |
| **`prefill_dev_stub.cpp`** | No‑GPU fallback (all functions no‑op). |
| **`weights.hpp / weights.cpp`** | BF16 / FP16 / FP32 matvec & MXFP4 matvec, SIMD dispatch. |
| **`ops.hpp / ops.cpp`** | RMS‑norm, YaRN rope, rotary, routing, SwigLU. |
| **`config.hpp / config.cpp`** | Model metadata (hidden size, heads, MoE params, sliding flags). |
| **`json.hpp`** | Minimal JSON parser used for `config.json` and other meta‑files. |
| **`safetensors.hpp / safetensors.cpp`** | Multi‑shard safetensors loader (memory‑maps files). |
| **`serve_loop.hpp / serve_loop.cpp`** | Strata line‑protocol server (READY, GEN, T, DONE, STOP, QUIT). |
| **`sampler.hpp / sampler.cpp`** | Token sampling (temperature, top‑k/p, repetition penalties). |
| **`tokenizer.hpp / tokenizer.cpp`** | o200k_harmony BPE tokenizer (PCRE2 pre‑tokenizer), Harmony chat template, channel parsing — the chat mode front end. |
| **`kv_quant.hpp`** | Packing / unpacking helpers for device KV (E4M3, FP16, FP32). |
| **`gguf_loader.hpp`** | Experimental GGUF loader (Q4\_K\_L blocks, async de‑quantisation). |
| **`config.hpp`** | `GptConfig` definition (model hyper‑parameters). |

---

## Future Work / Extensibility  

| Area | What could be added / improved |
|------|--------------------------------|
| **Unified Quantisation** | Add BF8 / NF4 support for both dense weights and KV, with automatic fallback. |
| **Dynamic Expert Loading** | Load expert weights on‑the‑fly from disk (paged‑in) to support *many* more experts than VRAM permits. |
| **TensorRT / cuBLASLT Integration** | Replace custom BF16 matmul with cuBLASLT GEMM for potentially higher throughput. |
| **Multi‑GPU / Sharding** | Distribute layers or expert slots across several GPUs (pipeline parallelism). |
| **Better KV Compression** | Per‑row adaptive scaling (log‑quant) for the device KV cache to reduce bandwidth. |
| **Batch Decoding** | Extend the decode path to support batched generation (multiple concurrent requests). |
| **More Profiling Hooks** | Export per‑layer timings to a JSON/Prometheus endpoint for real‑time monitoring. |
| **Python Bindings** | Provide a thin Cython/pybind11 wrapper so the engine can be called directly from Python without the line‑protocol server. |

---

### TL;DR  

* **`Model`** orchestrates **dense** + **MoE** transformer inference, optionally using **CUDA** for almost every operation.  
* **GPU back‑ends** are interchangeable (CPU stub vs. real CUDA).  
* **Device‑resident pre‑fill** (`prefill_dev`) dramatically reduces PCIe traffic for long prompts.  
* **Expert slots** are managed with an LRU + pinning system; a background preload keeps the hottest experts ready.  
* **KV‑snapshot** + **reuse‑prefix** let you resume long conversations without recomputing the whole prompt.  
* The whole engine is driven by a simple **line‑protocol** (`serve_loop`) that Strata’s Python server already knows.  

You now have a **complete mental map** of the project: where each piece lives, how it interacts, and which files you need to explore for deeper details (especially the missing implementation files `gpu_cuda.cu`, `prefill_dev.cu`, `weights.cpp`, `ops.cpp`, etc.). Happy hacking!
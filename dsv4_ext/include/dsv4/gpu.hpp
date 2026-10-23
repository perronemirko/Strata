/**
 * @file gpu.hpp
 * @brief Device layer: two interchangeable backends behind one API.
 *
 * This module abstracts the compute device with a unified `dsv4::gpu` namespace that has exactly
 * two implementations — only ONE is ever compiled into a binary:
 *
 * | Backend              | File          | Description                                          |
 * |----------------------|---------------|------------------------------------------------------|
 * | CPU-emulated         | [`src/gpu.cpp`](../../src/gpu.cpp) | malloc "VRAM", memcpy H2D/D2H, ggml-identical matvecs. Always buildable. |
 * | CUDA device          | [`src/gpu_cuda.cu`](../../src/gpu_cuda.cu) | Real kernels on NVIDIA GPU. Built with `--cuda` / cmake `-DDSV4_WITH_CUDA=ON`. |
 *
 * @par Why two backends?
 *   The emulated backend lets the entire HIT/MISS residency pipeline, memory planning, and tests
 *   run without a GPU — "VRAM" is host memory, H2D/D2H are memcpy, kernels use the same ggml-identical
 *   row math. The CUDA backend replaces this with real device kernels for production inference.
 *
 * @par MISS affordability
 *   A decoded IQ1_M expert element-by-element on the host costs more per token than the rest of the
 *   forward pass. On the device, `experts_hit()` evaluates resident experts ~100x cheaper, and the
 *   PCIe transfer of packed blocks hides behind other kernels via async staging.
 *
 * @par Capability gating
 *   `type_supported()` / `experts_supported()` decide at load time which tensors get uploaded and
 *   which MoE layers run on device. A `matvec()` or `experts_hit()` returning false wrote NOTHING —
 *   the caller falls back to host with one warning. Adding a ggml type = one trait in `dq_traits.hpp`
 *   plus one case in `matvec()`.
 *
 * @par Prefill batching
 *   A prompt of T tokens evaluates the SAME weight row T times. `matvec()` re-decodes once per token;
 *   `matmul()` decodes each row once and dots against T activation rows — saving the decode cost which
 *   dominates at ~1.8 GB/token for DeepSeek-V4-Flash's MoE path.
 *
 * @par Pinned host memory and doorbells (docs/TODO.md items 1-3, 5, 7)
 *   The expert path moves ~2.3 GB of packed weights per token. Four things keep that honest:
 *     - `pin_host()` gives the model page-locked arenas, so a MISS expert's bytes are copied
 *       exactly ONCE (shard -> arena) and then DMA'ed straight to the card: no staging bounce;
 *     - `Doorbell` + `publish()` let the device write a result into mapped pinned memory and
 *       announce it with a sequence counter, so the host learns it is ready without a synchronous
 *       DtoH copy and without draining the stream;
 *     - `d2h_rows()` reads back a strided device block in ONE call instead of one synchronous
 *       DtoH per token row;
 *     - `graph_begin()` / `graph_end()` capture the decode MoE launch sequence once per shape and
 *       replay it, which is what removes the per-layer launch and sync overhead.
 *   The CPU-emulated backend implements all of them as plain host memory and immediate writes, so
 *   the same model code and the same tests exercise both spellings.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dsv4 {
namespace gpu {

/**
 * @def kMaxHit
 * @brief Compile-time CAPACITY of `ExpPtrs`: the largest top-k this header can hold (16).
 *
 * This is an array bound, NOT the model's routing width. The runtime limit is the model's own
 * `expert_used_count`, which the engine derives from the GGUF: hardcoding 8 here was the latent
 * trap in docs/TODO.md item 6 — a model with `expert_used_count = 10` (Strata's verified geometry)
 * would silently push 2 experts per layer and token to the host. `model.cpp` clamps against this
 * capacity and warns when a model asks for more of it.
 */
constexpr int kMaxHit = 16;

/**
 * @struct ExpPtrs
 * @brief Pointers into the VRAM expert pool for `experts_hit()`.
 *
 * An expert slot is [gate | up | down], so the three row arrays hold device pointers to each matrix.
 * Each expert also has an associated routing weight `w[]` already applied by the caller's combine step.
 *
 * @par Host fallback
 *   If device pointers (gate/up/down) are null, host pointers (hg/hu/hd) are used instead. The backend
 *   stages them into its internal pool via `staging()` and evaluates on-device like a resident expert.
 *   The CPU-emulated backend accepts host pointers directly since its "device" IS host memory.
 */
struct ExpPtrs {
    int n = 0;                                    ///< Number of experts (1..kMaxHit).
    uint8_t* gate[kMaxHit] = {};                  ///< Device pointers to gate matrices.
    uint8_t* up[kMaxHit] = {};                    ///< Device pointers to up matrices.
    uint8_t* down[kMaxHit] = {};                  ///< Device pointers to down matrices.
    const uint8_t* hg[kMaxHit] = {};              ///< Host fallback: gate source (staged if device has staging).
    const uint8_t* hu[kMaxHit] = {};              ///< Host fallback: up source.
    const uint8_t* hd[kMaxHit] = {};              ///< Host fallback: down source.
    float w[kMaxHit] = {};                        ///< Routing weight per expert (already applied by caller).
    /**
     * @brief Device-side copy of w[] [n], or null.
     *
     * Only meaningful inside a captured CUDA graph: a kernel's scalar arguments are frozen at
     * capture time, so a replay would reuse the routing weights of the token that was captured.
     * When d_w is set the fused SwiGLU reads the weights from there instead, and the caller must
     * have uploaded them BEFORE graph_open() - a host-to-device copy of fresh bytes cannot be
     * captured. w[] is still read by backends that ignore this field.
     */
    const float* d_w = nullptr;
    size_t bpe = 0;                               ///< Bytes per expert (same for all three matrices).
    /**
     * @brief True when hg/hu/hd point into page-locked host memory handed out by `pin_host()`.
     *
     * The contract is exact: the block must be contiguous `[gate | up | down]` starting at `hg`,
     * i.e. `hu == hg + row_bytes(type_g,dim)*ff` and `hd == hu + row_bytes(type_u,dim)*ff`.
     * When it holds, `experts_hit()` DMAs `hg` straight to the card and skips the pinned staging
     * buffer entirely — the second of the two host copies the expert path used to pay per token.
     * A backend that ignores the flag is still correct: it just stages as before.
     */
    bool host_pinned = false;
};

/**
 * @brief Register device scratch buffer for staging host experts during `experts_hit()`.
 *
 * When the model encounters a MISS expert on the device, it needs temporary space to stage the
 * expert's gate/up/down pointers before evaluating. This function registers that staging area.
 *
 * The CPU-emulated backend ignores this (its "device" IS host memory) and always returns true.
 *
 * @param device_pool  Pointer to pre-allocated device memory for staging.
 * @param bytes        Size of the staging pool in bytes (must hold max expert size).
 * @return true if staging is available; false if the device cannot accept it.
 */
bool staging(uint8_t* device_pool, size_t bytes);

/**
 * @brief Check if host experts can be staged into the device pool.
 *
 * The CPU-emulated backend always returns true with no pool needed — its "device" memory is
 * host memory, so staging would be a pointless copy. The CUDA backend checks if `staging()` was called.
 * @return true if staging is available.
 */
bool has_staging();

/**
 * @brief Check if this is the CPU-emulated backend.
 * @return true for emulated (cpu), false for real device (cuda).
 */
bool is_emulated();

/**
 * @brief Initialize the GPU device layer.
 *
 * For the emulated backend, reads `DSV4_EMULATED_VRAM_MIB` environment variable (default 8192 MiB)
 * and allocates the "VRAM" pool. For CUDA, initializes the device and queries real VRAM.
 *
 * @param err  Output error message on failure.
 * @return true if initialization succeeded; false otherwise.
 */
bool init(std::string& err);

/**
 * @brief Allocate memory from the "VRAM" pool.
 *
 * The emulated backend allocates from a host memory pool of `DSV4_EMULATED_VRAM_MIB` bytes.
 * Allocations are never freed individually — the entire pool is released when the model unloads.
 *
 * @param n  Number of bytes to allocate.
 * @return Pointer to allocated memory, or nullptr if insufficient space or not initialized.
 */
void* alloc(size_t n);

/**
 * @brief Release a previously allocated pointer (frees from pool).
 *
 * In the emulated backend, this frees the underlying malloc but does NOT decrease `g_used` —
 * the pool is freed entirely when the model unloads, not per-expert.
 *
 * @param p  Pointer returned by a previous `alloc()` call.
 */
void release(void* p);

/**
 * @brief Query VRAM memory information.
 *
 * Both backends report free VRAM *already net of what this model has put on the card*.
 * CUDA uses `cudaMemGetInfo`; emulated backend tracks its own pool counter.
 * This ensures `plan_expert_memory()` is called with zero fixed costs to avoid double-counting.
 *
 * @param free_bytes   Output: bytes currently available (nullptr to skip).
 * @param total_bytes  Output: total VRAM size (nullptr to skip).
 */
void mem_info(size_t* free_bytes, size_t* total_bytes);

/**
 * @brief Check if this backend can compute over weights of the given ggml type.
 *
 * The emulated device answers yes for every type that `src/dequant.cpp` can decode.
 * The CUDA device answers yes only for types it has a dedicated kernel for.
 *
 * The loader uses this to avoid uploading weights no kernel could read, and to keep those
 * matvecs on the host instead of silently producing zeros.
 *
 * @param type  GgmlType enum value.
 * @return true if the backend has a kernel/decoder for this type.
 */
bool type_supported(uint32_t type);

/**
 * @brief Check if all three MoE matrices (gate/up/down) have kernels for their types.
 *
 * This is the one case where giving a layer VRAM slots is worth anything: only when all three
 * matrices can be evaluated on-device does `experts_hit()` become useful. If any type is unsupported,
 * the caller keeps computing that expert on the host.
 *
 * @param type_gate  GgmlType of the gate matrix.
 * @param type_up    GgmlType of the up matrix.
 * @param type_down  GgmlType of the down matrix.
 * @return true if all three types are supported by this backend.
 */
bool experts_supported(uint32_t type_gate, uint32_t type_up, uint32_t type_down);

// ================================================================= memory copy operations

/**
 * @brief Host-to-device memory copy.
 *
 * Emulated: plain memcpy (host to host pool). CUDA: cudaMemcpyHtoD.
 *
 * @param dst    Destination device pointer.
 * @param src    Source host pointer.
 * @param bytes  Number of bytes to copy.
 */
void h2d(void* dst, const void* src, size_t bytes);

/**
 * @brief Queue a host-to-device copy without blocking the host.
 *
 * Only truly async when `src` is page-locked (see `pin_host()`): from pageable memory the driver
 * has to stage and synchronise it itself, which is the trap docs/TODO.md item 1 is about. The
 * model uses this for the small per-token uploads that must NOT drain the stream - the routing
 * weights a captured graph reads from device memory.
 *
 * @param dst    Device destination.
 * @param src    Host source.
 * @param bytes  Number of bytes.
 */
void h2d_async(void* dst, const void* src, size_t bytes);

/**
 * @brief Device-to-host memory copy.
 *
 * Emulated: plain memcpy (host pool to host). CUDA: cudaMemcpyDtoH.
 *
 * @param dst    Destination host pointer.
 * @param src    Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void d2h(void* dst, const void* src, size_t bytes);

/**
 * @brief Copy `n_rows` blocks of `row_bytes` from a strided device block into a strided host block.
 *
 * The prefill readback: a batched matmul wrote token t at `src + t*src_stride`, and the caller
 * wants exactly [0,row_bytes) of each of those rows. Doing it as `n_rows` separate `d2h()` calls
 * costs one stream drain each — ~256 synchronous DtoH per batched matmul, ~470k per chunk
 * (docs/TODO.md item 7). This is one call, and on the CUDA backend one `cudaMemcpyAsync` plus
 * one sync.
 *
 * @param dst          Host destination; row t starts at `dst + t*dst_stride` (in elements).
 * @param src          Device source; row t starts at `src + t*src_stride` (in elements).
 * @param n_rows       Number of rows.
 * @param row_bytes    Meaningful bytes per row.
 * @param elem_size    Size of one element (4 for float): strides are in elements.
 * @param dst_stride   Elements between consecutive host rows (>= row_bytes/elem_size).
 * @param src_stride   Elements between consecutive device rows (>= row_bytes/elem_size).
 */
void d2h_rows(void* dst, const void* src, size_t n_rows, size_t row_bytes,
              size_t elem_size, size_t dst_stride, size_t src_stride);

// ================================================================= pinned host memory
/**
 * @brief Allocate page-locked host memory the device can DMA from/to directly.
 *
 * CUDA: `cudaHostAlloc`. Emulated: plain `malloc` — there "host memory" IS device memory, so
 * pinning would be a lie that changes nothing. The caller must pair every success with
 * `unpin_host()`; a failed `pin_host()` returns nullptr and the caller falls back to its
 * unpinned allocation, which is slower but exact.
 *
 * @param bytes Arena size.
 * @return Pointer to the arena, or nullptr.
 */
void* pin_host(size_t bytes);

/** @brief Release an arena returned by pin_host(). */
void unpin_host(void* p);

/**
 * @brief True when `p` was handed out by `pin_host()` and a real DMA can start from it.
 *
 * The model asks this once per arena, at load time, and records the answer in
 * `ExpPtrs::host_pinned` so the expert path never has to guess. The emulated backend answers
 * true: its copies are host memcpys and there is nothing to page-lock.
 */
bool host_is_pinned(const void* p);

// ================================================================= fused activation (device-resident chain)
/**
 * @brief a = swiglu_clamped(g, u, limit), entirely on the device.
 *
 * The shared-expert and MoE intermediate of one token is [ff] floats; keeping it on the card is
 * what lets a layer's dense chain stay device-resident instead of paying an h2d+d2h per matvec
 * (docs/TODO.md item 4). The emulated backend calls the same `swiglu_clamped()` the host path
 * uses, so the two spellings stay bit-identical.
 *
 * @param d_g    Device gate outputs [n].
 * @param d_u    Device up outputs [n].
 * @param d_a    Device SwiGLU outputs [n].
 * @param n      Elements.
 * @param limit  Clamp limit (>0 enables clamping, <=0 disables).
 * @return true if computed; false if this backend cannot run it (wrote NOTHING).
 */
bool swiglu(const float* d_g, const float* d_u, float* d_a, int64_t n, float limit);

// ================================================================= matrix-vector product (decode path)

/**
 * @brief Matrix-vector product: y = W * x, with on-the-fly dequantization.
 *
 * Each output row `y[r]` is the dot product of one weight row `W[r]` with input vector `x`.
 * Weights are decoded from compressed format on-the-fly (one block at a time) — this is the
 * decode path where each token evaluates experts independently.
 *
 * @param type      GgmlType enum: quantization format of weight matrix.
 * @param W         Pointer to compressed weight data [rows × in].
 * @param rows      Number of output rows (expert FFN width for gate/up, or dim for down).
 * @param in        Input vector dimension.
 * @param d_x       Input activation vector [in].
 * @param d_y       Output buffer [rows], populated with dot products.
 * @return true if computation succeeded; false if type is unsupported (wrote NOTHING).
 */
bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y);

// ================================================================= prefill: batched matrix multiplication over tokens

/**
 * @def kMaxSel
 * @brief Maximum gather indices a single batched matmul may request (8192).
 *
 * Used for the `sel` array in `matmul()` that gathers which token rows each expert evaluates.
 */
constexpr int kMaxSel = 8192;

/**
 * @brief Batched matrix multiplication: one weight matrix, T activation rows.
 *
 * The prefill primitive: a prompt of T tokens evaluates the SAME weight row T times. This function
 * decodes each row ONCE and dots it against T activation rows, saving the decode cost which dominates
 * at ~1.8 GB/token for DeepSeek-V4-Flash's MoE path.
 *
 * @par Layout (TOKEN-MAJOR on both sides)
 *   - `d_X[k*x_stride + i]` is token k's input vector, x_stride >= in
 *   - `d_Y[k*y_stride + r]` is token k's output element r, y_stride >= rows
 *
 * @par Gather mode
 *   `sel` (nullable, HOST memory) gathers: token k reads row `sel[k]` of d_X. This is the MoE case
 *   where an expert is asked for only the handful of tokens that routed to it within a chunk.
 *
 * @param type      GgmlType enum: quantization format of weight matrix.
 * @param W         Pointer to compressed weight data [rows × in].
 * @param rows      Number of output rows (expert FFN width for gate/up, or dim for down).
 * @param in        Input vector dimension.
 * @param d_X       Activation input buffer, TOKEN-MAJOR layout.
 * @param T         Number of tokens in the batch.
 * @param sel       Optional gather indices [T] (HOST memory); null = contiguous (sel[k] = k).
 * @param d_Y       Output buffer [T × rows], TOKEN-MAJOR layout.
 * @param x_stride  Stride between consecutive token vectors in d_X (>= in).
 * @param y_stride  Stride between consecutive output elements per row (>= rows).
 * @return true if computation succeeded; false if type is unsupported (wrote NOTHING).
 */
bool matmul(uint32_t type, const uint8_t* W, int64_t rows, int64_t in,
            const float* d_X, int T, const int32_t* sel, float* d_Y, int64_t x_stride, int64_t y_stride);

/**
 * @struct ExpBatch
 * @brief One expert's data for batched MoE evaluation (`experts_batch()`).
 *
 * Represents a single expert evaluated over a subset of tokens within a prefill chunk.
 * Device pointers (gate/up/down) are used if available; host pointers (hg/hu/hd) are staged
 * through the device pool via `staging()`.
 */
struct ExpBatch {
    uint8_t* gate = nullptr;              ///< Device pointer to gate matrix [ff × dim].
    uint8_t* up = nullptr;                ///< Device pointer to up matrix [ff × dim].
    uint8_t* down = nullptr;              ///< Device pointer to down matrix [dim × ff].
    const uint8_t* hg = nullptr;          ///< Host fallback: gate source.
    const uint8_t* hu = nullptr;          ///< Host fallback: up source.
    const uint8_t* hd = nullptr;          ///< Host fallback: down source.
    const int32_t* tok = nullptr;         ///< Token indices in chunk [n] (HOST memory).
    const float* w = nullptr;             ///< Routing weights [n] (HOST memory, already applied).
    int n = 0;                            ///< Number of tokens routed to this expert.
    /**
     * @brief True when hg/hu/hd are page-locked (see `ExpPtrs::host_pinned`).
     *
     * Same contract and same meaning as in `ExpPtrs`: when set, `experts_batch()` DMAs the block
     * straight from `hg` and never touches its own pinned ring — which is what lets the prefill
     * and the decode path share one staging implementation (docs/TODO.md item 6).
     */
    bool host_pinned = false;
};

/**
 * @brief Batched MoE HIT path: evaluate multiple experts over token subsets in expert order.
 *
 * The full batched forward pass for a prefill chunk's MoE layer, evaluated in expert order
 * (deterministic, no atomics):
 *   For each expert e with n tokens:
 *     g = Wg * x[tok]  (gate matmul)
 *     u = Wu * x[tok]  (up matmul)
 *     a[w] = w_t * swiglu_clamped(g, u, limit)  (fused SwiGLU with routing weight)
 *     d_Y[tok] += Wd * a  (down matmul, accumulated)
 *
 * @par Scratch requirements
 *   - `d_g`, `d_u`, `d_a`: maxN × ff floats each (maxN = largest ExpBatch::n over all experts)
 *   - `d_dn`: maxN × dim floats
 *   - `d_Y`: T × dim floats, zeroed before accumulation
 *
 * @param e            Array of expert batches.
 * @param n_exp        Number of experts in the array.
 * @param type_g       GgmlType of gate matrices.
 * @param type_u       GgmlType of up matrices.
 * @param type_d       GgmlType of down matrices.
 * @param ff           Expert FFN intermediate dimension.
 * @param dim          Model embedding dimension.
 * @param swiglu_limit SwiGLU clamp limit (>0 enables clamping, <=0 disables).
 * @param d_X          Input activations [T × dim], TOKEN-MAJOR.
 * @param T            Chunk size (total tokens in the prefill batch).
 * @param d_g          Scratch: gate outputs [maxN × ff].
 * @param d_u          Scratch: up outputs [maxN × ff].
 * @param d_a          Scratch: SwiGLU outputs [maxN × ff].
 * @param d_dn         Scratch: down matmul outputs [maxN × dim].
 * @param d_Y          Output: accumulated expert results [T × dim], zeroed then accumulated.
 * @return true if computation succeeded; false if any type is unsupported (wrote NOTHING).
 */
bool experts_batch(const ExpBatch* e, int n_exp, uint32_t type_g, uint32_t type_u, uint32_t type_d,
                   int64_t ff, int64_t dim, float swiglu_limit,
                   const float* d_X, int T, float* d_g, float* d_u, float* d_a, float* d_dn, float* d_Y);

/**
 * @brief Evaluate all HIT experts of one MoE layer (resident in VRAM).
 *
 * The fast path for decode: all top-k experts are already resident in VRAM slots. Computes:
 *   g[k] = Wg[k] * x        (gate matvec)
 *   u[k] = Wu[k] * x        (up matvec)
 *   a[k] = w_k * swiglu_clamped(g[k], u[k], limit)  (fused SwiGLU with routing weight)
 *   y  = sum_k Wd[k] * a[k] (down matvec, accumulated)
 *
 * @par Async behavior
 *   Launches are asynchronous with respect to the host (CUDA streams). The caller can compute
 *   MISS experts on CPU in parallel and only pay for `d2h()` when reading back `d_y`.
 *
 * @param p            Expert pointers (`ExpPtrs`): up to kMaxHit resident experts.
 * @param type_g       GgmlType of gate matrices.
 * @param type_u       GgmlType of up matrices.
 * @param type_d       GgmlType of down matrices.
 * @param ff           Expert FFN intermediate dimension.
 * @param dim          Model embedding dimension.
 * @param swiglu_limit SwiGLU clamp limit (>0 enables clamping, <=0 disables).
 * @param d_x          Input activation vector [dim].
 * @param d_g          Scratch: gate outputs [kMaxHit × ff].
 * @param d_u          Scratch: up outputs [kMaxHit × ff].
 * @param d_a          Scratch: SwiGLU outputs [kMaxHit × ff].
 * @param d_y          Output: accumulated expert results [dim], zeroed then accumulated.
 * @return true if computation succeeded; false if any type is unsupported (d_y untouched).
 */
bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y);

// ================================================================= doorbell: device -> host without a sync
/**
 * @struct Doorbell
 * @brief A result the device writes into mapped pinned memory and announces with a counter.
 *
 * The alternative to `d2h()`, which is a synchronous `cudaMemcpy` and therefore the sync point of
 * every launch before it (docs/TODO.md item 3). Here the payload lives in host memory the device
 * can address directly (`cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer`),
 * and the device's last act is a single ordered store of a sequence number. The host learns the
 * result is ready by reading that counter: no driver call, no stream drain, and the host may do
 * its own work (the CPU expert tier) while the GPU queue is still running.
 *
 * @par Contract
 *   `payload` is host-visible, `d_payload` is the device alias of the SAME bytes. `publish()`
 *   copies a device block into the payload and then increments a device-resident counter whose
 *   host alias is `flag_host`; `wait(expected)` returns once that counter has reached the number
 *   of publishes the caller has issued. The counter is incremented BY THE DEVICE, so the sequence
 *   stays right when the publish is inside a captured CUDA graph: replaying a graph publishes
 *   again, and a constant baked at capture time would not say so.
 *
 *   One payload is outstanding at a time: the caller must `wait()` before issuing the next
 *   `publish()` on the same doorbell. The engine does exactly that, because the value it reads
 *   back is what the next layer's arithmetic needs.
 */
struct Doorbell {
    float* payload = nullptr;      ///< Host-visible result [n_floats] (mapped pinned memory).
    float* d_payload = nullptr;    ///< Device alias of the same bytes.
    uint32_t* flag_host = nullptr; ///< Host-visible publish counter.
    uint32_t* flag_dev = nullptr;  ///< Device alias of the counter.
    size_t n_floats = 0;           ///< Payload size, fixed at open().
    void* impl = nullptr;          ///< Backend-owned allocation handle for close().
};

/**
 * @brief Allocate and map a doorbell of n_floats results.
 * @return true if the doorbell exists; false on failure (the caller must use d2h()).
 */
bool doorbell_open(Doorbell& db, size_t n_floats);

/** @brief Release a doorbell. */
void doorbell_close(Doorbell& db);

/**
 * @brief Copy d_src into the payload and announce it, on the current stream.
 *
 * When d_src is the payload's own device alias nothing is copied and only the counter is bumped.
 * The CALLER owns the sequence number: it increments its own counter once per publish and passes
 * that value to wait(). Counting on the device side is what keeps the pair correct when the
 * publish is inside a captured CUDA graph - a replay bumps the counter again, and a number baked
 * at capture time would not have said so.
 * @return true if the announcement was queued; false if this backend has no doorbell.
 */
bool doorbell_publish(const Doorbell& db, const float* d_src, size_t n_floats);

/**
 * @brief Spin until the counter has reached expected. Never calls the driver.
 * @return true when the payload may be read; false if the doorbell is not usable.
 */
bool doorbell_wait(const Doorbell& db, uint32_t expected);

/** @brief The host-visible payload. Only valid after wait() returned true. */
const float* doorbell_read(const Doorbell& db);

// ================================================================= one CUDA graph per window shape
/**
 * @def GRAPH_NONE / GRAPH_REPLAYED / GRAPH_CAPTURE
 * @brief What `graph_open()` asks the caller to do.
 */
constexpr int GRAPH_NONE = 0;       ///< No capture available: run the launches normally.
constexpr int GRAPH_REPLAYED = 1;   ///< A graph for this key exists and was launched: run nothing.
constexpr int GRAPH_CAPTURE = 2;    ///< Capture is open: run the launches, then graph_close().

/**
 * @struct GraphWindow
 * @brief A captured launch sequence, keyed by the model's own shape signature.
 *
 * The decode MoE of one layer is the same handful of launches for every token: same scratch, same
 * expert slots, same dimensions. Capturing it once and replaying it removes the per-launch host
 * cost and the sync points around it (docs/TODO.md item 5, "one graph per window size" in
 * SPEC_STRATA_VERIFIED.md). The key is whatever the caller decides makes a replay valid — the
 * engine uses (layer, expert-slot set, n) — and the caller must keep every pointer inside the
 * window identical across replays.
 *
 * @par What must stay OUTSIDE a window
 *   Anything whose bytes change between tokens: a host-to-device copy of fresh data, and a host
 *   memcpy into a staging buffer. Those are issued before graph_open(); only the kernels that read
 *   them are captured.
 */
struct GraphWindow {
    uint64_t key = 0;      ///< The signature this window was opened with.
    void* impl = nullptr;  ///< Backend-owned capture state.
};

/**
 * @brief Open a window. Returns what the caller must do next.
 *
 * @param gw   Window to open.
 * @param key  Shape signature; identical keys replay the same graph.
 * @return 0 (GRAPH_NONE): this backend cannot capture — run the launches normally, then call
 *         graph_close() (a no-op). 1 (GRAPH_REPLAYED): a graph for this key already exists and
 *         has been launched — do NOT run the launches. 2 (GRAPH_CAPTURE): capture is open — run
 *         the launches (they are recorded, not executed), then call graph_close().
 */
int graph_open(GraphWindow& gw, uint64_t key);

/**
 * @brief Close a window: instantiate and launch a fresh capture, or end a GRAPH_NONE window.
 *
 * @return true when the captured work has really been executed (GRAPH_CAPTURE path) or when the
 *         window was GRAPH_NONE and the caller ran the launches itself. false means the capture
 *         could not be instantiated: NOTHING has run for this window, and the caller must open the
 *         window again and run the launches normally. A failed capture is remembered, so the
 *         engine does not retry it on every token.
 */
bool graph_close(GraphWindow& gw);

/** @brief Drop every captured graph (used when the model resets or the scratch is freed). */
void graph_flush();

}  // namespace gpu
}  // namespace dsv4

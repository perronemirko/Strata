// gptoss/gpu.hpp - device layer, TWO interchangeable backends behind one API (same design as dsv4_ext):
//   src/gpu.cpp       CPU-emulated device (malloc "VRAM", everything executes immediately). Always buildable; what the tests run against.
//   src/gpu_cuda.cu   real CUDA device (sh build.sh --cuda): non-blocking streams, events, pinned staging, CUDA graphs, tensor-core GEMM (sm_80+).
// Both define the same gptoss::gpu symbols; a binary links ONE of them, never both.
// Threading: the Model serialises every call EXCEPT the copy-side functions (h2d_copy_async, event_record_copy, event_sync), which are internally locked
// so a prefetch thread can run them while the main thread uses the compute side.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace gptoss {
namespace gpu {

bool is_emulated();
bool init(int device, std::string& err);
const char* device_name();
void mem_info(size_t* free_bytes, size_t* total_bytes);
void* alloc(size_t n);                       // nullptr on failure (also clears a non-sticky OOM)
void release(void* p);                       // release(nullptr) is a no-op
bool h2d(void* dev, const void* host, size_t n);        // synchronous: complete when it returns
void set_options(bool use_graphs);           // CUDA graphs for the decode expert pipeline (default on)

/// Allocates the scratch buffers UP FRONT (device + pinned host) for batches of up to `dense_x_floats` / `dense_y_floats` floats of dense-matmat input / output,
/// `expert_batch_n` tokens per expert call, and `dense_async_floats` floats of output for ONE async dense batch (the decode's q/k/v, see below), so no
/// cudaMalloc happens later when the expert slots have taken the VRAM. Pass 0 for `dense_async_floats` to keep that path unreserved.
bool reserve_scratch(size_t dense_x_floats, size_t dense_y_floats, int expert_batch_n, int H, int I, size_t dense_async_floats);

/// y[n][rows] = W x[n][cols], n < N, host x/y. W: BF16 [rows][cols] on the device. N >= 32 uses the tensor-core GEMM (sm_80+), smaller N a warp-per-row kernel.
bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy);
/// Same for MXFP4 (blocks [rows][cols/32][16], scales [rows][cols/32]); cols % 32 == 0.
bool matmat_mx(const void* dBlocks, const void* dScales, int rows, int cols, const float* hx, int N, float* hy);

/// One whole expert on the device: y[n] = wt[n] * ( down( swiglu( gate_up(x[n]) + b1 ) ) + b2 ). All pointers are device pointers.
struct ExpertDesc {
    const void *gu_b = nullptr, *gu_s = nullptr, *gu_bias = nullptr;   // gate_up: MXFP4 blocks [2I][H/32][16], scales [2I][H/32], bf16 bias [2I]
    const void *dn_b = nullptr, *dn_s = nullptr, *dn_bias = nullptr;   // down:    MXFP4 blocks [H][I/32][16],  scales [H][I/32],  bf16 bias [H]
    int H = 0, I = 0;
    int halves = 0;                          // gate/up layout of the 2I rows: 0 interleaved (gate = even), 1 first half / second half
    float limit = 7.0f, alpha = 1.702f;
};
constexpr int kExpertCtxBatch = 4;           // contexts 0..3: decode (N == 1, graph-able, each on its own stream); 4: batch (prefill)
/// Enqueues the whole expert (H2D of x and wt, 4 kernels, D2H of y) on the context's stream and returns WITHOUT waiting. Contexts are independent:
/// enqueue up to 4 decode experts back to back, then collect them.
bool expert_run_async(int ctx, const ExpertDesc& d, const float* hx, const float* wt, int N);
bool expert_collect(int ctx, float* hy);     // waits for that context's work and copies y[N][H] out

/// Async dense projections for the DECODE. The q/k/v of one layer read the SAME input x, so they run as ONE x upload,
/// `count` kernels, ONE result copy and ONE wait instead of `count` full round-trips (each of which was an upload, a
/// kernel, a copy back and a cudaStreamSynchronize). The results land in the caller's host buffers at dense_collect.
/// The dense context is independent of the expert contexts; dense_run_async records its own event.
struct DenseOp { const void* W = nullptr; int rows = 0, cols = 0; };   // W: BF16 [rows][cols] already on the device
constexpr int kDenseCtxBatch = 1;                                      // context 0: the decode's q/k/v batch
constexpr int kDenseMaxOps = 4;                                       // ops per batch: q, k, v (+ o if ever batched)
bool dense_run_async(int ctx, const float* hx, int N, const DenseOp* ops, int count);
bool dense_collect(int ctx, float* const* out);                        // waits for that context, then copies each op's [N][rows] block

// ---- copy side (prefetch): its own stream
int event_create();
bool event_sync(int ev);                     // host waits for the event
bool h2d_copy_async(void* dev, const void* host, size_t n);   // on the copy stream; direct DMA if the host range was registered, else through pinned staging
bool event_record_copy(int ev);              // records `ev` after every copy issued so far on the copy stream
bool host_register(const void* p, size_t n); // page-locks a host range (read-only) so copies from it are true DMA; pages inside only

// ---- device-resident activations (batched prefill with ATTENTION on the device). Every pointer here is a DEVICE pointer (gpu::alloc) unless named h*.
// The calls are ordered on the compute stream; d2h waits. A failure sets last_error() and returns false.
bool d2h(void* host, const void* dev, size_t n);                                       // synchronous
/// Same as matmat_bf16 but x and y are already on the device: dy[n][rows] = W dx[n][cols].
bool dmm_bf16(const void* dW, int rows, int cols, const float* dx, int N, float* dy);
/// dy[n] = dg * (dx[n] * 1/sqrt(mean(dx[n]^2) + eps)), dg == nullptr -> gain 1 (the host rmsnorm).
bool rms_rows(const float* dx, const float* dg, float eps, int N, int H, float* dy);
/// dX[n][i] += (dA ? dA[n][i] : 0) + (dbias ? dbias[i] : 0)
bool add_rows(float* dX, const float* dA, const float* dbias, int N, int H);
struct AttnShape {
    int heads = 0, kv_heads = 0, D = 0;      // D % 32 == 0, D <= 128
    int window = 0;                          // banded layers: keys visible = window, current token included (sliding == true)
    bool sliding = false;
    int cap = 0;                             // positions held by the K/V cache of the layer: slot(pos) = sliding ? pos % cap : pos
    bool rope_pairs = false;
};
/// Tokens p0 .. p0+N-1. dq [N][heads*D] and dk [N][kv_heads*D] are the raw projections: adds the biases, applies RoPE (cos/sin tables [pos][D/2] on the device,
/// indexed by ABSOLUTE position) IN PLACE, and writes the rotated K and the biased V of every token into the layer's device cache (dKc, dVc: [cap][kv_heads*D]).
bool qk_rope_kv(float* dq, float* dk, const float* dv, const float* dbq, const float* dbk, const float* dbv, const float* dcos, const float* dsin,
                int N, int p0, const AttnShape& a, float* dKc, float* dVc);
/// Attention of the N tokens over the device cache, sink in the softmax denominator only: dout [N][heads*D].
bool attn_rows(const float* dq, const float* dKc, const float* dVc, const float* dsinks, int N, int p0, const AttnShape& a, float* dout);
/// Can THIS backend run the attention kernels for this shape? The emulated device runs the host math and accepts any shape (it is the executable
/// spec the tests run against); the CUDA kernel owns one dimension per lane, so it needs D % 32 == 0 and D <= 128. The Model asks before
/// reserving anything, so an unsupported shape turns the path off with a printed reason instead of failing mid-prefill.
bool attn_supported(const AttnShape& a);

std::string last_error();
bool last_oom();                       // true if the last failure was an allocation that did not fit (not a poisoned context: the host can shrink and retry)

}  // namespace gpu
}  // namespace gptoss

// Real CUDA device. Build: sh build.sh --cuda   (CUDA_ARCH=sm_89 etc., default "native").
// Decoding contract = src/weights.cpp: BF16 = high half of a float; MXFP4 = 32-element blocks of 16 bytes, element 2k = low nibble, 2k+1 = high nibble,
// value = fp4[nibble] * 2^(scale-127).
//
//  * streams: compute (dense, sync API), one stream per expert context (decode: 4 experts in flight), one copy stream (prefetch)
//  * the expert pipeline (H2D x, gate_up, bias+swiglu, down, bias*weight, D2H y) is ONE enqueue with no host step in between; for N == 1 it is captured
//    once into a CUDA graph. The kernels read their pointers from a small descriptor in device memory, so ONE graph serves every expert of every layer.
//  * tensor-core GEMM (bf16 WMMA, 64x64 tile, MXFP4 / BF16 weights decoded to shared memory) for N >= 32 on sm_80+; warp-per-row kernels otherwise.
#include "gptoss/gpu.hpp"
#include "gptoss/weights.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace gptoss {
namespace gpu {
namespace {

constexpr int TILE = 8;                 // tokens whose accumulators live in registers at once (warp-per-row kernels)
constexpr int WARPS = 4;                // output rows per block (warp-per-row kernels)
constexpr int NCTX = 5;                 // expert contexts: 0..3 decode, 4 batch

std::string g_err;
std::mutex g_err_mu, g_copy_mu;
cudaStream_t g_comp = nullptr, g_copy = nullptr;
int g_cc = 0;                           // compute capability, major*10 + minor
bool g_graphs = true;
__constant__ float c_fp4[16];

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

// What the expert kernels read from device memory (filled from ExpertDesc by an async H2D at the start of every run).
struct DDesc {
    const uint8_t *gu_b, *gu_s;
    const unsigned short* gu_bias;
    const uint8_t *dn_b, *dn_s;
    const unsigned short* dn_bias;
    int H, I, halves, N;
    float limit, alpha;
    const float* wt;
};

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(((unsigned) v) << 16); }

// ---------------------------------------------------------------- warp-per-row bodies (one warp = one output row, TILE tokens at a time)
__device__ void bf16_body(const unsigned short* __restrict__ W, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;                                   // the whole warp leaves together: the shuffles below stay legal
    const unsigned short* w = W + (size_t) row * cols;
    for (int n0 = 0; n0 < N; n0 += TILE) {
        const int nn = min(TILE, N - n0);
        float acc[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;
        for (int c = lane; c < cols; c += 32) {
            const float wv = bf2f(w[c]);
#pragma unroll
            for (int t = 0; t < TILE; ++t)
                if (t < nn) acc[t] += wv * x[(size_t) (n0 + t) * cols + c];
        }
#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t < nn) y[(size_t) (n0 + t) * rows + row] = v;
        }
    }
}

__device__ void mx_body(const uint8_t* __restrict__ B, const uint8_t* __restrict__ S, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int G = cols >> 5;
    const uint8_t* b = B + (size_t) row * G * 16;
    const uint8_t* s = S + (size_t) row * G;
    for (int n0 = 0; n0 < N; n0 += TILE) {
        const int nn = min(TILE, N - n0);
        float acc[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;
        for (int g = lane; g < G; g += 32) {                   // lanes stride over the row's 32-element blocks
            const float sc = ldexpf(1.0f, (int) s[g] - 127);
            const uint4 raw = *reinterpret_cast<const uint4*>(b + (size_t) g * 16);   // 16 bytes; slot/row offsets are 16-aligned
            const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
            float wv[32];
#pragma unroll
            for (int k = 0; k < 16; ++k) { wv[2 * k] = c_fp4[q[k] & 15] * sc; wv[2 * k + 1] = c_fp4[q[k] >> 4] * sc; }
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                if (t < nn) {
                    const float* xx = x + (size_t) (n0 + t) * cols + (size_t) g * 32;
                    float p = 0.0f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) p += wv[j] * xx[j];
                    acc[t] += p;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t < nn) y[(size_t) (n0 + t) * rows + row] = v;
        }
    }
}

// ---------------------------------------------------------------- tensor-core GEMM (sm_80+): Y[N][rows] = X[N][cols] . W[rows][cols]^T
// Block = 128 threads (4 warps) -> a 64 (weight rows) x 64 (tokens) tile. Per 32-wide K step: 64 threads decode one weight row each into shared bf16, all
// threads convert the x tile to bf16; each warp then multiplies its 16 rows x 4 token tiles with bf16 WMMA 16x16x16 (fp32 accumulate).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
constexpr int BM = 64, BN = 64, BK = 32, LDS = 40, LDC = BM + 4;

template <bool MX>
__device__ void gemm_body(const void* A, const uint8_t* S, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    using namespace nvcuda;
    __shared__ __align__(128) __nv_bfloat16 Ws[BM * LDS];
    __shared__ __align__(128) __nv_bfloat16 Xs[BN * LDS];
    __shared__ __align__(128) float Cs[BN * LDC];
    const int tid = threadIdx.x, warp = tid >> 5;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int G = cols / BK;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) wmma::fill_fragment(acc[t], 0.0f);
    for (int g = 0; g < G; ++g) {
        if (tid < BM) {                                        // one weight row's K-block per thread
            const int r = row0 + tid;
            __nv_bfloat16* w = Ws + tid * LDS;
            if (r < rows) {
                if (MX) {
                    const uint8_t* b = (const uint8_t*) A + ((size_t) r * G + g) * 16;
                    const float sc = ldexpf(1.0f, (int) S[(size_t) r * G + g] - 127);
                    const uint4 raw = *reinterpret_cast<const uint4*>(b);
                    const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
                    for (int k = 0; k < 16; ++k) { w[2 * k] = __float2bfloat16(c_fp4[q[k] & 15] * sc); w[2 * k + 1] = __float2bfloat16(c_fp4[q[k] >> 4] * sc); }
                } else {
                    const unsigned short* b = (const unsigned short*) A + (size_t) r * cols + (size_t) g * BK;
#pragma unroll
                    for (int k = 0; k < BK; ++k) w[k] = __ushort_as_bfloat16(b[k]);
                }
            } else {
#pragma unroll
                for (int k = 0; k < BK; ++k) w[k] = __float2bfloat16(0.0f);
            }
        }
        for (int e = tid; e < BN * BK; e += blockDim.x) {      // x tile, fp32 -> bf16
            const int t = e / BK, k = e % BK, tok = tok0 + t;
            Xs[t * LDS + k] = __float2bfloat16(tok < N ? x[(size_t) tok * cols + (size_t) g * BK + k] : 0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, Ws + (warp * 16) * LDS + ks, LDS);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Xs + (t * 16) * LDS + ks, LDS);
                wmma::mma_sync(acc[t], fa, fb, acc[t]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) wmma::store_matrix_sync(Cs + (t * 16) * LDC + warp * 16, acc[t], LDC, wmma::mem_col_major);
    __syncthreads();
    for (int e = tid; e < BM * BN; e += blockDim.x) {
        const int t = e / BM, r = e % BM;
        if (tok0 + t < N && row0 + r < rows) y[(size_t) (tok0 + t) * rows + row0 + r] = Cs[t * LDC + r];
    }
}
#endif  // sm_80+

// ---------------------------------------------------------------- kernels
__global__ void k_bf16(const unsigned short* W, int rows, int cols, const float* x, int N, float* y) { bf16_body(W, rows, cols, x, N, y); }
__global__ void k_mx(const uint8_t* B, const uint8_t* S, int rows, int cols, const float* x, int N, float* y) { mx_body(B, S, rows, cols, x, N, y); }

__global__ void __launch_bounds__(128) k_gemm_bf16(const unsigned short* W, int rows, int cols, const float* x, int N, float* y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    gemm_body<false>(W, nullptr, rows, cols, x, N, y);
#endif
}
__global__ void __launch_bounds__(128) k_gemm_mx(const uint8_t* B, const uint8_t* S, int rows, int cols, const float* x, int N, float* y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    gemm_body<true>(B, S, rows, cols, x, N, y);
#endif
}

// ---- expert pipeline kernels (pointers come from the descriptor in device memory)
__global__ void k_exp_gu(const DDesc* d, const float* x, float* out) { mx_body(d->gu_b, d->gu_s, 2 * d->I, d->H, x, d->N, out); }
__global__ void k_exp_dn(const DDesc* d, const float* act, float* eb) { mx_body(d->dn_b, d->dn_s, d->H, d->I, act, d->N, eb); }
__global__ void __launch_bounds__(128) k_exp_gu_gemm(const DDesc* d, const float* x, float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    gemm_body<true>(d->gu_b, d->gu_s, 2 * d->I, d->H, x, d->N, out);
#endif
}
__global__ void __launch_bounds__(128) k_exp_dn_gemm(const DDesc* d, const float* act, float* eb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    gemm_body<true>(d->dn_b, d->dn_s, d->H, d->I, act, d->N, eb);
#endif
}

__global__ void k_exp_act(const DDesc* d, const float* gu, float* act) {          // + bias, clamp, swiglu
    const int I = d->I, N = d->N;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t) N * I; idx += (size_t) gridDim.x * blockDim.x) {
        const size_t n = idx / I;
        const int j = (int) (idx % I);
        const float* r = gu + n * 2 * (size_t) I;
        float g = (d->halves ? r[j] : r[2 * j]) + bf2f(d->gu_bias[d->halves ? j : 2 * j]);
        float u = (d->halves ? r[I + j] : r[2 * j + 1]) + bf2f(d->gu_bias[d->halves ? I + j : 2 * j + 1]);
        g = fminf(g, d->limit);
        u = fminf(fmaxf(u, -d->limit), d->limit);
        act[idx] = (u + 1.0f) * (g / (1.0f + expf(-d->alpha * g)));
    }
}

__global__ void k_exp_fin(const DDesc* d, const float* eb, float* y) {            // wt[n] * (down + bias)
    const int H = d->H, N = d->N;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t) N * H; idx += (size_t) gridDim.x * blockDim.x) {
        const size_t n = idx / H;
        const int j = (int) (idx % H);
        y[idx] = d->wt[n] * (eb[idx] + bf2f(d->dn_bias[j]));
    }
}

// ---------------------------------------------------------------- dense scratch (sync matmat API)
float *g_dx = nullptr, *g_dy = nullptr, *g_hx = nullptr, *g_hy = nullptr;
size_t g_cx = 0, g_cy = 0;

bool scratch(size_t nx, size_t ny) {
    if (nx > g_cx) {
        cudaFree(g_dx); cudaFreeHost(g_hx); g_dx = nullptr; g_hx = nullptr; g_cx = 0;
        if (!ck(cudaMalloc((void**) &g_dx, nx * sizeof(float)), "cudaMalloc x") || !ck(cudaMallocHost((void**) &g_hx, nx * sizeof(float)), "cudaMallocHost x")) return false;
        g_cx = nx;
    }
    if (ny > g_cy) {
        cudaFree(g_dy); cudaFreeHost(g_hy); g_dy = nullptr; g_hy = nullptr; g_cy = 0;
        if (!ck(cudaMalloc((void**) &g_dy, ny * sizeof(float)), "cudaMalloc y") || !ck(cudaMallocHost((void**) &g_hy, ny * sizeof(float)), "cudaMallocHost y")) return false;
        g_cy = ny;
    }
    return true;
}

bool use_gemm(int N, int cols) { return N >= 32 && g_cc >= 80 && cols % 32 == 0; }

bool run_matmat(bool mx, const void* W, const void* S, int rows, int cols, const float* hx, int N, float* hy) {
    if (N <= 0) return true;
    if (!scratch((size_t) N * cols, (size_t) N * rows)) return false;
    std::memcpy(g_hx, hx, (size_t) N * cols * sizeof(float));
    if (!ck(cudaMemcpyAsync(g_dx, g_hx, (size_t) N * cols * sizeof(float), cudaMemcpyHostToDevice, g_comp), "x h2d")) return false;
    if (use_gemm(N, cols)) {
        const dim3 grid((unsigned) ((rows + 63) / 64), (unsigned) ((N + 63) / 64));
        if (mx) k_gemm_mx<<<grid, 128, 0, g_comp>>>((const uint8_t*) W, (const uint8_t*) S, rows, cols, g_dx, N, g_dy);
        else k_gemm_bf16<<<grid, 128, 0, g_comp>>>((const unsigned short*) W, rows, cols, g_dx, N, g_dy);
    } else {
        const int blocks = (rows + WARPS - 1) / WARPS;
        if (mx) k_mx<<<blocks, WARPS * 32, 0, g_comp>>>((const uint8_t*) W, (const uint8_t*) S, rows, cols, g_dx, N, g_dy);
        else k_bf16<<<blocks, WARPS * 32, 0, g_comp>>>((const unsigned short*) W, rows, cols, g_dx, N, g_dy);
    }
    if (!ck(cudaGetLastError(), "matmat launch")) return false;
    if (!ck(cudaMemcpyAsync(g_hy, g_dy, (size_t) N * rows * sizeof(float), cudaMemcpyDeviceToHost, g_comp), "y d2h")) return false;
    if (!ck(cudaStreamSynchronize(g_comp), "matmat sync")) return false;
    std::memcpy(hy, g_hy, (size_t) N * rows * sizeof(float));
    return true;
}

// ---------------------------------------------------------------- expert contexts
struct ExpCtx {
    cudaStream_t stream = nullptr;
    cudaEvent_t ev = nullptr;
    DDesc *h_desc = nullptr, *d_desc = nullptr;
    float *h_x = nullptr, *h_y = nullptr, *h_wt = nullptr;
    float *d_x = nullptr, *d_wt = nullptr, *d_gu = nullptr, *d_act = nullptr, *d_eb = nullptr, *d_y = nullptr;
    size_t cap = 0;                       // tokens
    int H = 0, I = 0, n = 0;              // n = tokens of the last run
    cudaGraphExec_t graph = nullptr;
    bool graph_ok = false, graph_tried = false;
};
ExpCtx g_ctx[NCTX];

void ctx_free_buffers(ExpCtx& c) {
    if (c.graph) { cudaGraphExecDestroy(c.graph); c.graph = nullptr; }
    c.graph_ok = false; c.graph_tried = false;
    cudaFreeHost(c.h_x); cudaFreeHost(c.h_y); cudaFreeHost(c.h_wt);
    cudaFree(c.d_x); cudaFree(c.d_wt); cudaFree(c.d_gu); cudaFree(c.d_act); cudaFree(c.d_eb); cudaFree(c.d_y);
    c.h_x = c.h_y = c.h_wt = c.d_x = c.d_wt = c.d_gu = c.d_act = c.d_eb = c.d_y = nullptr;
    c.cap = 0;
}

bool ctx_reserve(ExpCtx& c, int N, int H, int I) {
    if (!c.stream) {
        if (!ck(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking), "expert stream") || !ck(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming), "expert event") ||
            !ck(cudaMallocHost((void**) &c.h_desc, sizeof(DDesc)), "desc host") || !ck(cudaMalloc((void**) &c.d_desc, sizeof(DDesc)), "desc dev")) return false;
    }
    if ((size_t) N <= c.cap && c.H == H && c.I == I) return true;
    ctx_free_buffers(c);
    const size_t n = (size_t) N;
    if (!ck(cudaMallocHost((void**) &c.h_x, n * H * 4), "h_x") || !ck(cudaMallocHost((void**) &c.h_y, n * H * 4), "h_y") || !ck(cudaMallocHost((void**) &c.h_wt, n * 4), "h_wt") ||
        !ck(cudaMalloc((void**) &c.d_x, n * H * 4), "d_x") || !ck(cudaMalloc((void**) &c.d_wt, n * 4), "d_wt") || !ck(cudaMalloc((void**) &c.d_gu, n * 2 * I * 4), "d_gu") ||
        !ck(cudaMalloc((void**) &c.d_act, n * I * 4), "d_act") || !ck(cudaMalloc((void**) &c.d_eb, n * H * 4), "d_eb") || !ck(cudaMalloc((void**) &c.d_y, n * H * 4), "d_y")) return false;
    c.cap = n; c.H = H; c.I = I;
    return true;
}

// The whole expert on the context's stream, no host step in between.
bool enqueue_expert(ExpCtx& c, int N) {
    const int H = c.H, I = c.I;
    cudaStream_t st = c.stream;
    if (!ck(cudaMemcpyAsync(c.d_desc, c.h_desc, sizeof(DDesc), cudaMemcpyHostToDevice, st), "desc h2d") ||
        !ck(cudaMemcpyAsync(c.d_x, c.h_x, (size_t) N * H * 4, cudaMemcpyHostToDevice, st), "x h2d") ||
        !ck(cudaMemcpyAsync(c.d_wt, c.h_wt, (size_t) N * 4, cudaMemcpyHostToDevice, st), "wt h2d")) return false;
    const bool gemm = use_gemm(N, H) && use_gemm(N, I);
    if (gemm) k_exp_gu_gemm<<<dim3((unsigned) ((2 * I + 63) / 64), (unsigned) ((N + 63) / 64)), 128, 0, st>>>(c.d_desc, c.d_x, c.d_gu);
    else k_exp_gu<<<(2 * I + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(c.d_desc, c.d_x, c.d_gu);
    k_exp_act<<<(unsigned) std::min<size_t>(((size_t) N * I + 255) / 256, 65535), 256, 0, st>>>(c.d_desc, c.d_gu, c.d_act);
    if (gemm) k_exp_dn_gemm<<<dim3((unsigned) ((H + 63) / 64), (unsigned) ((N + 63) / 64)), 128, 0, st>>>(c.d_desc, c.d_act, c.d_eb);
    else k_exp_dn<<<(H + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(c.d_desc, c.d_act, c.d_eb);
    k_exp_fin<<<(unsigned) std::min<size_t>(((size_t) N * H + 255) / 256, 65535), 256, 0, st>>>(c.d_desc, c.d_eb, c.d_y);
    if (!ck(cudaGetLastError(), "expert kernels")) return false;
    return ck(cudaMemcpyAsync(c.h_y, c.d_y, (size_t) N * H * 4, cudaMemcpyDeviceToHost, st), "y d2h");
}

// CUDA graph of the N == 1 pipeline, captured once per context (thread-local capture: other threads' CUDA calls are not recorded).
bool build_graph(ExpCtx& c) {
    cudaGraph_t g = nullptr;
    bool ok = ck(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (ok) {
        const bool e = enqueue_expert(c, 1);
        const cudaError_t ce = cudaStreamEndCapture(c.stream, &g);
        ok = e && ck(ce, "end capture");
    }
    if (ok) ok = ck(cudaGraphInstantiateWithFlags(&c.graph, g, 0), "graph instantiate");
    if (g) cudaGraphDestroy(g);
    if (!ok) { c.graph = nullptr; cudaGetLastError(); }
    return ok;
}

// ---------------------------------------------------------------- copy side
constexpr int NSTAGE = 4;
constexpr size_t STAGE_BYTES = (size_t) 8 << 20;
void* g_stage[NSTAGE] = {nullptr, nullptr, nullptr, nullptr};
cudaEvent_t g_stage_ev[NSTAGE];
bool g_stage_used[NSTAGE] = {false, false, false, false};
int g_stage_i = 0;
std::vector<std::pair<uintptr_t, size_t>> g_reg;
std::vector<cudaEvent_t> g_events;

bool is_registered(const void* p, size_t n) {
    const uintptr_t a = (uintptr_t) p, b = a + n;
    for (const auto& r : g_reg) if (a >= r.first && b <= r.first + r.second) return true;
    return false;
}

}  // namespace

bool is_emulated() { return false; }
std::string last_error() { std::lock_guard<std::mutex> lk(g_err_mu); return g_err; }
void set_options(bool use_graphs) { g_graphs = use_graphs; }

bool init(int device, std::string& err) {
    int n = 0;
    if (!ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount") || n == 0) { err = last_error().empty() ? "no CUDA device" : last_error(); return false; }
    if (!ck(cudaSetDevice(device), "cudaSetDevice")) { err = last_error(); return false; }
    int maj = 0, min = 0;
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, device);
    g_cc = maj * 10 + min;
    const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    if (!ck(cudaMemcpyToSymbol(c_fp4, lut, sizeof lut), "fp4 table") || !ck(cudaStreamCreateWithFlags(&g_comp, cudaStreamNonBlocking), "compute stream") ||
        !ck(cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking), "copy stream")) { err = last_error(); return false; }
    return true;
}

const char* device_name() {
    static cudaDeviceProp p;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&p, d) != cudaSuccess) return "CUDA device";
    return p.name;
}

void mem_info(size_t* f, size_t* t) {
    *f = 0; *t = 0;
    if (!ck(cudaMemGetInfo(f, t), "cudaMemGetInfo")) { *f = 0; *t = 0; }
}

void* alloc(size_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) { cudaGetLastError(); return nullptr; }   // OOM is not sticky: clear it
    return p;
}

void release(void* p) { if (p) cudaFree(p); }

bool h2d(void* d, const void* h, size_t n) {            // complete when it returns (ordered with every later use, whatever the stream)
    return ck(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, g_comp), "h2d") && ck(cudaStreamSynchronize(g_comp), "h2d sync");
}

bool reserve_scratch(size_t dense_x_floats, size_t dense_y_floats, int expert_batch_n, int H, int I) {
    if (!scratch(dense_x_floats, dense_y_floats)) return false;
    for (int i = 0; i < kExpertCtxBatch; ++i) if (!ctx_reserve(g_ctx[i], 1, H, I)) return false;
    return ctx_reserve(g_ctx[kExpertCtxBatch], std::max(1, expert_batch_n), H, I);
}

bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy) { return run_matmat(false, dW, nullptr, rows, cols, hx, N, hy); }
bool matmat_mx(const void* dB, const void* dS, int rows, int cols, const float* hx, int N, float* hy) {
    if (cols % 32) { std::lock_guard<std::mutex> lk(g_err_mu); g_err = "matmat_mx: cols % 32 != 0"; return false; }
    return run_matmat(true, dB, dS, rows, cols, hx, N, hy);
}

bool expert_run_async(int ci, const ExpertDesc& d, const float* hx, const float* wt, int N) {
    if (ci < 0 || ci > kExpertCtxBatch || N <= 0) return false;
    ExpCtx& c = g_ctx[ci];
    if (!ctx_reserve(c, N, d.H, d.I)) return false;
    std::memcpy(c.h_x, hx, (size_t) N * d.H * sizeof(float));
    std::memcpy(c.h_wt, wt, (size_t) N * sizeof(float));
    DDesc& hd = *c.h_desc;
    hd.gu_b = (const uint8_t*) d.gu_b; hd.gu_s = (const uint8_t*) d.gu_s; hd.gu_bias = (const unsigned short*) d.gu_bias;
    hd.dn_b = (const uint8_t*) d.dn_b; hd.dn_s = (const uint8_t*) d.dn_s; hd.dn_bias = (const unsigned short*) d.dn_bias;
    hd.H = d.H; hd.I = d.I; hd.halves = d.halves; hd.N = N; hd.limit = d.limit; hd.alpha = d.alpha; hd.wt = c.d_wt;
    c.n = N;
    if (N == 1 && g_graphs && ci < kExpertCtxBatch) {
        if (!c.graph_tried) { c.graph_tried = true; c.graph_ok = build_graph(c); }
        if (c.graph_ok) return ck(cudaGraphLaunch(c.graph, c.stream), "graph launch") && ck(cudaEventRecord(c.ev, c.stream), "expert event");
    }
    return enqueue_expert(c, N) && ck(cudaEventRecord(c.ev, c.stream), "expert event");
}

bool expert_collect(int ci, float* hy) {
    if (ci < 0 || ci > kExpertCtxBatch) return false;
    ExpCtx& c = g_ctx[ci];
    if (!ck(cudaEventSynchronize(c.ev), "expert wait")) return false;
    std::memcpy(hy, c.h_y, (size_t) c.n * c.H * sizeof(float));
    return true;
}

int event_create() {
    cudaEvent_t e = nullptr;
    if (!ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event create")) return -1;
    std::lock_guard<std::mutex> lk(g_copy_mu);
    g_events.push_back(e);
    return (int) g_events.size() - 1;
}

bool event_sync(int ev) {
    cudaEvent_t e;
    { std::lock_guard<std::mutex> lk(g_copy_mu); if (ev < 0 || ev >= (int) g_events.size()) return false; e = g_events[(size_t) ev]; }
    return ck(cudaEventSynchronize(e), "event sync");
}

bool event_record_copy(int ev) {
    std::lock_guard<std::mutex> lk(g_copy_mu);
    if (ev < 0 || ev >= (int) g_events.size()) return false;
    return ck(cudaEventRecord(g_events[(size_t) ev], g_copy), "event record");
}

bool h2d_copy_async(void* dev, const void* host, size_t n) {
    std::lock_guard<std::mutex> lk(g_copy_mu);
    if (is_registered(host, n)) return ck(cudaMemcpyAsync(dev, host, n, cudaMemcpyHostToDevice, g_copy), "prefetch copy");     // direct DMA from page-locked RAM
    if (!g_stage[0]) {
        for (int i = 0; i < NSTAGE; ++i)
            if (!ck(cudaMallocHost(&g_stage[i], STAGE_BYTES), "staging") || !ck(cudaEventCreateWithFlags(&g_stage_ev[i], cudaEventDisableTiming), "staging event")) return false;
    }
    size_t off = 0;
    while (off < n) {                                                  // through pinned staging: reuse a buffer only after its previous copy is done
        const int i = g_stage_i;
        g_stage_i = (g_stage_i + 1) % NSTAGE;
        if (g_stage_used[i] && !ck(cudaEventSynchronize(g_stage_ev[i]), "staging wait")) return false;
        const size_t len = std::min(STAGE_BYTES, n - off);
        std::memcpy(g_stage[i], (const uint8_t*) host + off, len);
        if (!ck(cudaMemcpyAsync((uint8_t*) dev + off, g_stage[i], len, cudaMemcpyHostToDevice, g_copy), "staged copy")) return false;
        if (!ck(cudaEventRecord(g_stage_ev[i], g_copy), "staging record")) return false;
        g_stage_used[i] = true;
        off += len;
    }
    return true;
}

bool host_register(const void* p, size_t n) {
    const uintptr_t ps = 4096;
    const uintptr_t a = ((uintptr_t) p + ps - 1) & ~(ps - 1), b = ((uintptr_t) p + n) & ~(ps - 1);
    if (b <= a) return true;
    unsigned flags = cudaHostRegisterDefault;
#ifdef cudaHostRegisterReadOnly
    flags = cudaHostRegisterReadOnly;                                   // the mapping is PROT_READ
#endif
    if (!ck(cudaHostRegister((void*) a, (size_t) (b - a), flags), "cudaHostRegister")) { cudaGetLastError(); return false; }
    std::lock_guard<std::mutex> lk(g_copy_mu);
    g_reg.push_back({a, (size_t) (b - a)});
    return true;
}

}  // namespace gpu
}  // namespace gptoss

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
#include "gptoss/kv_quant.hpp"   // KV formats + pack/unpack: ONE definition, shared with prefill_dev.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace gptoss
{
    namespace gpu
    {
        namespace
        {

            // The KV formats and the pack/unpack templates live in gptoss/kv_quant.hpp: a local copy here
            // would shadow them and stay unreferenced (nvcc #177-D).

            constexpr int TILE = 8;  // tokens whose accumulators live in registers at once (warp-per-row kernels)
            constexpr int WARPS = 4; // output rows per block (warp-per-row kernels)
            constexpr int NCTX = 5;  // expert contexts: 0..3 decode, 4 batch

            std::string g_err;
            bool g_oom = false; // last ck() failure was cudaErrorMemoryAllocation: not sticky, the caller can shrink and retry
            std::mutex g_err_mu, g_copy_mu;
            cudaStream_t g_comp = nullptr, g_copy = nullptr;
            int g_cc = 0;  // compute capability, major*10 + minor
            int g_dev = 0; // device chosen by init(); bind_thread() re-selects it in other host threads (CUDA binds the device per thread)
            bool g_graphs = true;
            __constant__ float c_fp4[16];

            bool ck(cudaError_t e, const char *what)
            {
                if (e == cudaSuccess)
                    return true;
                std::lock_guard<std::mutex> lk(g_err_mu);
                g_oom = (e == cudaErrorMemoryAllocation);
                g_err = std::string(what) + ": " + cudaGetErrorString(e);
                return false;
            }

            // What the expert kernels read from device memory (filled from ExpertDesc by an async H2D at the start of every run).
            struct DDesc
            {
                const uint8_t *gu_b, *gu_s;
                const unsigned short *gu_bias;
                const uint8_t *dn_b, *dn_s;
                const unsigned short *dn_bias;
                int H, I, halves, N;
                float limit, alpha;
                const float *wt;
            };

            __device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(((unsigned)v) << 16); }

            // ---------------------------------------------------------------- warp-per-row bodies (one warp = one output row, TILE tokens at a time)
            __device__ void bf16_body(const unsigned short *__restrict__ W, int rows, int cols, const float *__restrict__ x, int N, float *__restrict__ y)
            {
                const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
                const int lane = threadIdx.x & 31;
                if (row >= rows)
                    return; // the whole warp leaves together: the shuffles below stay legal
                const unsigned short *w = W + (size_t)row * cols;
                for (int n0 = 0; n0 < N; n0 += TILE)
                {
                    const int nn = min(TILE, N - n0);
                    float acc[TILE];
#pragma unroll
                    for (int t = 0; t < TILE; ++t)
                        acc[t] = 0.0f;
                    for (int c = lane; c < cols; c += 32)
                    {
                        const float wv = bf2f(w[c]);
#pragma unroll
                        for (int t = 0; t < TILE; ++t)
                            if (t < nn)
                                acc[t] += wv * x[(size_t)(n0 + t) * cols + c];
                    }
#pragma unroll
                    for (int t = 0; t < TILE; ++t)
                    {
                        float v = acc[t];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1)
                            v += __shfl_xor_sync(0xffffffffu, v, o);
                        if (lane == 0 && t < nn)
                            y[(size_t)(n0 + t) * rows + row] = v;
                    }
                }
            }

            __device__ void mx_body(const uint8_t *__restrict__ B, const uint8_t *__restrict__ S, int rows, int cols, const float *__restrict__ x, int N, float *__restrict__ y)
            {
                const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
                const int lane = threadIdx.x & 31;
                if (row >= rows)
                    return;
                const int G = cols >> 5;
                const uint8_t *b = B + (size_t)row * G * 16;
                const uint8_t *s = S + (size_t)row * G;
                for (int n0 = 0; n0 < N; n0 += TILE)
                {
                    const int nn = min(TILE, N - n0);
                    float acc[TILE];
#pragma unroll
                    for (int t = 0; t < TILE; ++t)
                        acc[t] = 0.0f;
                    for (int g = lane; g < G; g += 32)
                    { // lanes stride over the row's 32-element blocks
                        const float sc = ldexpf(1.0f, (int)s[g] - 127);
                        const uint4 raw = *reinterpret_cast<const uint4 *>(b + (size_t)g * 16); // 16 bytes; slot/row offsets are 16-aligned
                        const uint8_t *q = reinterpret_cast<const uint8_t *>(&raw);
                        float wv[32];
#pragma unroll
                        for (int k = 0; k < 16; ++k)
                        {
                            wv[2 * k] = c_fp4[q[k] & 15] * sc;
                            wv[2 * k + 1] = c_fp4[q[k] >> 4] * sc;
                        }
#pragma unroll
                        for (int t = 0; t < TILE; ++t)
                        {
                            if (t < nn)
                            {
                                const float *xx = x + (size_t)(n0 + t) * cols + (size_t)g * 32;
                                float p = 0.0f;
#pragma unroll
                                for (int j = 0; j < 32; ++j)
                                    p += wv[j] * xx[j];
                                acc[t] += p;
                            }
                        }
                    }
#pragma unroll
                    for (int t = 0; t < TILE; ++t)
                    {
                        float v = acc[t];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1)
                            v += __shfl_xor_sync(0xffffffffu, v, o);
                        if (lane == 0 && t < nn)
                            y[(size_t)(n0 + t) * rows + row] = v;
                    }
                }
            }

// ---------------------------------------------------------------- tensor-core GEMM (sm_80+): Y[N][rows] = X[N][cols] . W[rows][cols]^T
// Block = 128 threads (4 warps) -> a 64 (weight rows) x 64 (tokens) tile. Per 32-wide K step: 64 threads decode one weight row each into shared bf16, all
// threads convert the x tile to bf16; each warp then multiplies its 16 rows x 4 token tiles with bf16 WMMA 16x16x16 (fp32 accumulate).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            constexpr int BM = 64, BN = 64, BK = 32, LDS = 40, LDC = BM + 4;

            template <bool MX>
            __device__ void gemm_body(const void *A, const uint8_t *S, int rows, int cols, const float *__restrict__ x, int N, float *__restrict__ y)
            {
                using namespace nvcuda;
                __shared__ __align__(128) __nv_bfloat16 Ws[BM * LDS];
                __shared__ __align__(128) __nv_bfloat16 Xs[BN * LDS];
                __shared__ __align__(128) float Cs[BN * LDC];
                const int tid = threadIdx.x, warp = tid >> 5;
                const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
                const int G = cols / BK;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    wmma::fill_fragment(acc[t], 0.0f);
                for (int g = 0; g < G; ++g)
                {
                    if (tid < BM)
                    { // one weight row's K-block per thread
                        const int r = row0 + tid;
                        __nv_bfloat16 *w = Ws + tid * LDS;
                        if (r < rows)
                        {
                            if (MX)
                            {
                                const uint8_t *b = (const uint8_t *)A + ((size_t)r * G + g) * 16;
                                const float sc = ldexpf(1.0f, (int)S[(size_t)r * G + g] - 127);
                                const uint4 raw = *reinterpret_cast<const uint4 *>(b);
                                const uint8_t *q = reinterpret_cast<const uint8_t *>(&raw);
#pragma unroll
                                for (int k = 0; k < 16; ++k)
                                {
                                    w[2 * k] = __float2bfloat16(c_fp4[q[k] & 15] * sc);
                                    w[2 * k + 1] = __float2bfloat16(c_fp4[q[k] >> 4] * sc);
                                }
                            }
                            else
                            {
                                const unsigned short *b = (const unsigned short *)A + (size_t)r * cols + (size_t)g * BK;
#pragma unroll
                                for (int k = 0; k < BK; ++k)
                                    w[k] = __ushort_as_bfloat16(b[k]);
                            }
                        }
                        else
                        {
#pragma unroll
                            for (int k = 0; k < BK; ++k)
                                w[k] = __float2bfloat16(0.0f);
                        }
                    }
                    for (int e = tid; e < BN * BK; e += blockDim.x)
                    { // x tile, fp32 -> bf16
                        const int t = e / BK, k = e % BK, tok = tok0 + t;
                        Xs[t * LDS + k] = __float2bfloat16(tok < N ? x[(size_t)tok * cols + (size_t)g * BK + k] : 0.0f);
                    }
                    __syncthreads();
#pragma unroll
                    for (int ks = 0; ks < BK; ks += 16)
                    {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                        wmma::load_matrix_sync(fa, Ws + (warp * 16) * LDS + ks, LDS);
#pragma unroll
                        for (int t = 0; t < 4; ++t)
                        {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                            wmma::load_matrix_sync(fb, Xs + (t * 16) * LDS + ks, LDS);
                            wmma::mma_sync(acc[t], fa, fb, acc[t]);
                        }
                    }
                    __syncthreads();
                }
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    wmma::store_matrix_sync(Cs + (t * 16) * LDC + warp * 16, acc[t], LDC, wmma::mem_col_major);
                __syncthreads();
                for (int e = tid; e < BM * BN; e += blockDim.x)
                {
                    const int t = e / BM, r = e % BM;
                    if (tok0 + t < N && row0 + r < rows)
                        y[(size_t)(tok0 + t) * rows + row0 + r] = Cs[t * LDC + r];
                }
            }
#endif // sm_80+

            // ---------------------------------------------------------------- kernels
            __global__ void k_bf16(const unsigned short *W, int rows, int cols, const float *x, int N, float *y) { bf16_body(W, rows, cols, x, N, y); }
            __global__ void k_mx(const uint8_t *B, const uint8_t *S, int rows, int cols, const float *x, int N, float *y) { mx_body(B, S, rows, cols, x, N, y); }

            __global__ void __launch_bounds__(128) k_gemm_bf16(const unsigned short *W, int rows, int cols, const float *x, int N, float *y)
            {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                gemm_body<false>(W, nullptr, rows, cols, x, N, y);
#endif
            }
            __global__ void __launch_bounds__(128) k_gemm_mx(const uint8_t *B, const uint8_t *S, int rows, int cols, const float *x, int N, float *y)
            {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                gemm_body<true>(B, S, rows, cols, x, N, y);
#endif
            }

            // ---- expert pipeline kernels (pointers come from the descriptor in device memory)
            __global__ void k_exp_gu(const DDesc *d, const float *x, float *out) { mx_body(d->gu_b, d->gu_s, 2 * d->I, d->H, x, d->N, out); }
            __global__ void k_exp_dn(const DDesc *d, const float *act, float *eb) { mx_body(d->dn_b, d->dn_s, d->H, d->I, act, d->N, eb); }
            __global__ void __launch_bounds__(128) k_exp_gu_gemm(const DDesc *d, const float *x, float *out)
            {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                gemm_body<true>(d->gu_b, d->gu_s, 2 * d->I, d->H, x, d->N, out);
#endif
            }
            __global__ void __launch_bounds__(128) k_exp_dn_gemm(const DDesc *d, const float *act, float *eb)
            {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                gemm_body<true>(d->dn_b, d->dn_s, d->H, d->I, act, d->N, eb);
#endif
            }

            __global__ void k_exp_act(const DDesc *d, const float *gu, float *act)
            { // + bias, clamp, swiglu
                const int I = d->I, N = d->N;
                for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t)N * I; idx += (size_t)gridDim.x * blockDim.x)
                {
                    const size_t n = idx / I;
                    const int j = (int)(idx % I);
                    const float *r = gu + n * 2 * (size_t)I;
                    float g = (d->halves ? r[j] : r[2 * j]) + bf2f(d->gu_bias[d->halves ? j : 2 * j]);
                    float u = (d->halves ? r[I + j] : r[2 * j + 1]) + bf2f(d->gu_bias[d->halves ? I + j : 2 * j + 1]);
                    g = fminf(g, d->limit);
                    u = fminf(fmaxf(u, -d->limit), d->limit);
                    act[idx] = (u + 1.0f) * (g / (1.0f + expf(-d->alpha * g)));
                }
            }

            __global__ void k_exp_fin(const DDesc *d, const float *eb, float *y)
            { // wt[n] * (down + bias)
                const int H = d->H, N = d->N;
                for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t)N * H; idx += (size_t)gridDim.x * blockDim.x)
                {
                    const size_t n = idx / H;
                    const int j = (int)(idx % H);
                    y[idx] = d->wt[n] * (eb[idx] + bf2f(d->dn_bias[j]));
                }
            }

            // ---------------------------------------------------------------- async dense contexts (decode's q/k/v: ONE wait for the whole trio)
            struct DenseCtx
            {
                cudaStream_t stream = nullptr;
                cudaEvent_t ev = nullptr;
                float *h_x = nullptr, *h_y = nullptr;
                float *d_x = nullptr, *d_y = nullptr;
                size_t cap_x = 0, cap_y = 0; // floats
                int n = 0, count = 0;
                size_t off[kDenseMaxOps + 1]; // where each op's [N][rows] block lives inside d_y / h_y
                int rows[kDenseMaxOps];
            };
            DenseCtx g_dc[kDenseCtxBatch];

            bool dense_ctx_reserve(DenseCtx &c, size_t nx, size_t ny)
            {
                if (!c.stream && (!ck(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking), "dense stream") ||
                                  !ck(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming), "dense event")))
                    return false;
                if (nx > c.cap_x)
                {
                    cudaFreeHost(c.h_x);
                    cudaFree(c.d_x);
                    if (!ck(cudaMallocHost((void **)&c.h_x, nx * sizeof(float)), "dense h_x") ||
                        !ck(cudaMalloc((void **)&c.d_x, nx * sizeof(float)), "dense d_x"))
                        return false;
                    c.cap_x = nx;
                }
                if (ny > c.cap_y)
                {
                    cudaFreeHost(c.h_y);
                    cudaFree(c.d_y);
                    if (!ck(cudaMallocHost((void **)&c.h_y, ny * sizeof(float)), "dense h_y") ||
                        !ck(cudaMalloc((void **)&c.d_y, ny * sizeof(float)), "dense d_y"))
                        return false;
                    c.cap_y = ny;
                }
                return true;
            }

            // ---------------------------------------------------------------- dense scratch (sync matmat API)
            float *g_dx = nullptr, *g_dy = nullptr, *g_hx = nullptr, *g_hy = nullptr;
            size_t g_cx = 0, g_cy = 0;

            bool scratch(size_t nx, size_t ny)
            {
                if (nx > g_cx)
                {
                    cudaFree(g_dx);
                    cudaFreeHost(g_hx);
                    g_dx = nullptr;
                    g_hx = nullptr;
                    g_cx = 0;
                    if (!ck(cudaMalloc((void **)&g_dx, nx * sizeof(float)), "cudaMalloc x") || !ck(cudaMallocHost((void **)&g_hx, nx * sizeof(float)), "cudaMallocHost x"))
                        return false;
                    g_cx = nx;
                }
                if (ny > g_cy)
                {
                    cudaFree(g_dy);
                    cudaFreeHost(g_hy);
                    g_dy = nullptr;
                    g_hy = nullptr;
                    g_cy = 0;
                    if (!ck(cudaMalloc((void **)&g_dy, ny * sizeof(float)), "cudaMalloc y") || !ck(cudaMallocHost((void **)&g_hy, ny * sizeof(float)), "cudaMallocHost y"))
                        return false;
                    g_cy = ny;
                }
                return true;
            }

            bool use_gemm(int N, int cols) { return N >= 32 && g_cc >= 80 && cols % 32 == 0; }

            bool run_matmat(bool mx, const void *W, const void *S, int rows, int cols, const float *hx, int N, float *hy)
            {
                if (N <= 0)
                    return true;
                if (!scratch((size_t)N * cols, (size_t)N * rows))
                    return false;
                std::memcpy(g_hx, hx, (size_t)N * cols * sizeof(float));
                if (!ck(cudaMemcpyAsync(g_dx, g_hx, (size_t)N * cols * sizeof(float), cudaMemcpyHostToDevice, g_comp), "x h2d"))
                    return false;
                if (use_gemm(N, cols))
                {
                    const dim3 grid((unsigned)((rows + 63) / 64), (unsigned)((N + 63) / 64));
                    if (mx)
                        k_gemm_mx<<<grid, 128, 0, g_comp>>>((const uint8_t *)W, (const uint8_t *)S, rows, cols, g_dx, N, g_dy);
                    else
                        k_gemm_bf16<<<grid, 128, 0, g_comp>>>((const unsigned short *)W, rows, cols, g_dx, N, g_dy);
                }
                else
                {
                    const int blocks = (rows + WARPS - 1) / WARPS;
                    if (mx)
                        k_mx<<<blocks, WARPS * 32, 0, g_comp>>>((const uint8_t *)W, (const uint8_t *)S, rows, cols, g_dx, N, g_dy);
                    else
                        k_bf16<<<blocks, WARPS * 32, 0, g_comp>>>((const unsigned short *)W, rows, cols, g_dx, N, g_dy);
                }
                if (!ck(cudaGetLastError(), "matmat launch"))
                    return false;
                if (!ck(cudaMemcpyAsync(g_hy, g_dy, (size_t)N * rows * sizeof(float), cudaMemcpyDeviceToHost, g_comp), "y d2h"))
                    return false;
                if (!ck(cudaStreamSynchronize(g_comp), "matmat sync"))
                    return false;
                std::memcpy(hy, g_hy, (size_t)N * rows * sizeof(float));
                return true;
            }

            // ---------------------------------------------------------------- device-resident activations (batched prefill, attention on the device)
            // Plain, correctness-first kernels; the numbers follow src/ops.cpp and Model::attn_heads (the emulated device in gpu.cpp is the executable spec).
            __global__ void k_rms(const float *__restrict__ x, const float *__restrict__ g, float eps, int H, float *__restrict__ y)
            { // one block per row
                const float *xr = x + (size_t)blockIdx.x * H;
                float ss = 0.0f;
                for (int i = threadIdx.x; i < H; i += blockDim.x)
                    ss += xr[i] * xr[i];
                __shared__ float red[32];
                __shared__ float tot;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if ((threadIdx.x & 31) == 0)
                    red[threadIdx.x >> 5] = ss;
                __syncthreads();
                if (threadIdx.x < 32)
                {
                    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (threadIdx.x == 0)
                        tot = v;
                }
                __syncthreads();
                const float r = 1.0f / sqrtf(tot / (float)H + eps);
                float *yr = y + (size_t)blockIdx.x * H;
                for (int i = threadIdx.x; i < H; i += blockDim.x)
                    yr[i] = (g ? g[i] : 1.0f) * (xr[i] * r);
            }

            __global__ void k_add_rows(float *__restrict__ X, const float *__restrict__ A, const float *__restrict__ b, size_t total, int H)
            {
                for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
                    X[i] += (A ? A[i] : 0.0f) + (b ? b[i % (size_t)H] : 0.0f);
            }

            // bias + RoPE, in place, on the heads of q (hh < heads) and k (hh >= heads). The rotated k stays in the
            // fp32 stage: qk_rope_kv packs it into the card's KV format right after, so no fp32 row is ever written
            // to the cache. Kc == nullptr keeps the stage only.
            __global__ void k_qk_rope(float *__restrict__ q, float *__restrict__ k, const float *__restrict__ bq, const float *__restrict__ bk, const float *__restrict__ cs,
                                      const float *__restrict__ sn, int N, int p0, int heads, int kv_heads, int D, int sliding, int cap, int pairs, float *__restrict__ Kc)
            {
                const int half = D / 2, nh = heads + kv_heads;
                const size_t total = (size_t)N * nh * half;
                for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x)
                {
                    const int i = (int)(idx % half);
                    const int hh = (int)((idx / half) % nh);
                    const int t = (int)(idx / ((size_t)half * nh));
                    const int pos = p0 + t;
                    const bool isq = hh < heads;
                    const int kh = hh - heads;
                    float *p = isq ? q + (size_t)t * heads * D + (size_t)hh * D : k + (size_t)t * kv_heads * D + (size_t)kh * D;
                    const float *bs = isq ? bq + (size_t)hh * D : bk + (size_t)kh * D;
                    const int ia = pairs ? 2 * i : i, ib = pairs ? 2 * i + 1 : i + half;
                    const float a = p[ia] + bs[ia], b = p[ib] + bs[ib];
                    const float c = cs[(size_t)pos * half + i], s = sn[(size_t)pos * half + i];
                    const float r0 = a * c - b * s;
                    const float r1 = pairs ? a * s + b * c : b * c + a * s;
                    p[ia] = r0;
                    p[ib] = r1;
                    if (!isq && Kc)
                    {
                        const size_t slot = sliding ? (size_t)(pos % cap) : (size_t)pos;
                        float *kc = Kc + slot * kv_heads * D + (size_t)kh * D;
                        kc[ia] = r0;
                        kc[ib] = r1;
                    }
                }
            }

            // The v bias lands in the fp32 stage (in place); qk_rope_kv then packs that stage into the card's
            // V format. v == dst: each element is touched exactly once, so the in-place add is safe.
            __global__ void k_v_bias(float *__restrict__ v, const float *__restrict__ bv, int N, int KD)
            {
                const size_t total = (size_t)N * KD;
                for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x)
                    v[idx] += bv[idx % KD];
            }

            // ---- KV pack / unpack on the card: rows of an fp32 stage -> the packed cache, one power-of-two
            // scale per row (shared by that row's kv_heads). dst/iscl are addressed by SLOT, so a banded
            // layer's run that wraps the ring still lands in the right slots.
            __device__ __forceinline__ float kv_row_scale(float peak, float limit)
            {
                if (!(peak > 0.0f) || !(limit > 0.0f))
                    return 1.0f;
                const float s = exp2f(floorf(log2f(peak / limit)));
                if (!(s > 1e-18f))
                    return 1.0f;
                return (s > 1.0f) ? 1.0f : s;
            }

            __device__ __forceinline__ float kv_row_scale_inv(float scale)
            {
                return (scale > 0.0f) ? 1.0f / scale : 1.0f;
            }

            __device__ __forceinline__ float block_amax(float v, float *__restrict__ red, int nt)
            {
                for (int o = 16; o > 0; o >>= 1)
                    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                const int w = threadIdx.x >> 5;
                if ((threadIdx.x & 31) == 0)
                    red[w] = v;
                __syncthreads();
                const int nw = nt >> 5;
                float tot = 0.0f;
                for (int i = 0; i < nw; ++i)
                    tot = fmaxf(tot, red[i]);
                __syncthreads();
                return tot;
            }

            template <typename T_OUT>
            __global__ void k_kv_pack_at(const float *__restrict__ stage, T_OUT *__restrict__ dst, float *__restrict__ iscl,
                                         int rows, int p0, int kd, int cap, int sliding, float limit)
            {
                __shared__ float red[8];
                const int r = blockIdx.x;
                if (r >= rows)
                    return;
                const int slot = sliding ? (p0 + r) % cap : p0 + r;
                const float *p = stage + (size_t)r * kd;
                float amax = 0.0f;
                for (int i = threadIdx.x; i < kd; i += blockDim.x)
                    amax = fmaxf(amax, fabsf(p[i]));
                amax = block_amax(amax, red, (int)blockDim.x);
                const float sc = kv_row_scale(amax, limit);
                const float isc = kv_row_scale_inv(sc);
                if (threadIdx.x == 0 && iscl)
                    iscl[slot] = isc;
                T_OUT *d = dst + (size_t)slot * kd;
                for (int i = threadIdx.x; i < kd; i += blockDim.x)
                    d[i] = gpu::pack_fp32_to_target<T_OUT>(p[i], isc);
            }

            template <typename T_IN>
            __global__ void k_kv_unpack_at(const T_IN *__restrict__ src, const float *__restrict__ iscl, float *__restrict__ dst,
                                           int rows, int p0, int kd, int cap, int sliding)
            {
                const int r = blockIdx.x;
                if (r >= rows)
                    return;
                const int slot = sliding ? (p0 + r) % cap : p0 + r;
                const float isc = iscl ? iscl[slot] : 1.0f;
                const T_IN *p = src + (size_t)slot * kd;
                float *d = dst + (size_t)r * kd;
                for (int i = threadIdx.x; i < kd; i += blockDim.x)
                    d[i] = gpu::unpack_target_to_fp32<T_IN>(p[i], isc);
            }

            // One warp per (token, head): online softmax over the visible keys (the sink starts the running max /
            // denominator, it has no value), so no score buffer is needed and any context length works. Lane l owns
            // dimensions l, l+32, ... (D % 32 == 0, D <= 128).
            //
            // ON-THE-FLY DEQUANTIZATION: T_K / T_V are the formats chosen INDEPENDENTLY by -k / -v. The cache rows
            // are PACKED (e4m3 = 1B, half = 2B, float = 4B); the kernel reads those bytes and converts them into
            // registers, so the fp32 rows never exist in VRAM and the PCIe bus carries 1/4 or 1/2 the bytes.
            // kinv/vinv hold one power-of-two inverse scale per row (null = 1.0, the F32 case).
            template <typename T_K, typename T_V>
            __global__ void k_attn(const float *__restrict__ q, const T_K *__restrict__ Kc, const T_V *__restrict__ Vc,
                                   const float *__restrict__ kinv, const float *__restrict__ vinv,
                                   const float *__restrict__ sinks, int p0, int heads,
                                   int kv_heads, int D, int window, int cap, int sliding, float scale, float *__restrict__ out)
            {
                const int t = blockIdx.x, hd = blockIdx.y, lane = threadIdx.x;
                const int pos = p0 + t;
                const int QD = heads * D, KD = kv_heads * D, g = hd / (heads / kv_heads), nv = D >> 5;
                const int lo = sliding ? max(0, pos - window + 1) : 0;
                float qv[4], acc[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    qv[i] = i < nv ? q[(size_t)t * QD + (size_t)hd * D + lane + 32 * i] : 0.0f;
                    acc[i] = 0.0f;
                }
                float m = sinks[hd], l = 1.0f;
                for (int u = lo; u <= pos; ++u)
                {
                    const size_t slot = sliding ? (size_t)(u % cap) : (size_t)u;
                    const T_K *kr = Kc + slot * KD + (size_t)g * D;
                    const float kisc = kinv ? kinv[slot] : 1.0f;
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (i < nv)
                            s += qv[i] * gpu::unpack_target_to_fp32<T_K>(kr[lane + 32 * i], kisc);
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        s += __shfl_xor_sync(0xffffffffu, s, o);
                    s *= scale;
                    const float mn = fmaxf(m, s), a = expf(m - mn), p = expf(s - mn);
                    l = l * a + p;
                    const T_V *vr = Vc + slot * KD + (size_t)g * D;
                    const float visc = vinv ? vinv[slot] : 1.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (i < nv)
                            acc[i] = acc[i] * a + p * gpu::unpack_target_to_fp32<T_V>(vr[lane + 32 * i], visc);
                    m = mn;
                }
                const float inv = 1.0f / l;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (i < nv)
                        out[(size_t)t * QD + (size_t)hd * D + lane + 32 * i] = acc[i] * inv;
            }

        void *g_kv_scratch = nullptr;   // device staging for the host<->card KV rows (grown on demand)
        size_t g_kv_scratch_n = 0;      // bytes held

        static bool kv_ensure_scratch(size_t bytes)
        {
            if (bytes <= g_kv_scratch_n)
                return true;
            if (g_kv_scratch)
                cudaFree(g_kv_scratch);
            g_kv_scratch = nullptr;
            if (!ck(cudaMalloc(&g_kv_scratch, bytes), "kv scratch"))
                return false;
            g_kv_scratch_n = bytes;
            return true;
        }


            unsigned grid1d(size_t total, int block) { return (unsigned)std::min<size_t>((total + block - 1) / block, 65535); }

            // ---------------------------------------------------------------- expert contexts
            struct ExpCtx
            {
                cudaStream_t stream = nullptr;
                cudaEvent_t ev = nullptr;
                DDesc *h_desc = nullptr, *d_desc = nullptr;
                float *h_x = nullptr, *h_y = nullptr, *h_wt = nullptr;
                float *d_x = nullptr, *d_wt = nullptr, *d_gu = nullptr, *d_act = nullptr, *d_eb = nullptr, *d_y = nullptr;
                size_t cap = 0;          // tokens
                int H = 0, I = 0, n = 0; // n = tokens of the last run
                cudaGraphExec_t graph = nullptr;
                bool graph_ok = false, graph_tried = false;
            };
            ExpCtx g_ctx[NCTX];

            void ctx_free_buffers(ExpCtx &c)
            {
                if (c.graph)
                {
                    cudaGraphExecDestroy(c.graph);
                    c.graph = nullptr;
                }
                c.graph_ok = false;
                c.graph_tried = false;
                cudaFreeHost(c.h_x);
                cudaFreeHost(c.h_y);
                cudaFreeHost(c.h_wt);
                cudaFree(c.d_x);
                cudaFree(c.d_wt);
                cudaFree(c.d_gu);
                cudaFree(c.d_act);
                cudaFree(c.d_eb);
                cudaFree(c.d_y);
                c.h_x = c.h_y = c.h_wt = c.d_x = c.d_wt = c.d_gu = c.d_act = c.d_eb = c.d_y = nullptr;
                c.cap = 0;
            }

            bool ctx_reserve(ExpCtx &c, int N, int H, int I)
            {
                if (!c.stream)
                {
                    if (!ck(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking), "expert stream") || !ck(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming), "expert event") ||
                        !ck(cudaMallocHost((void **)&c.h_desc, sizeof(DDesc)), "desc host") || !ck(cudaMalloc((void **)&c.d_desc, sizeof(DDesc)), "desc dev"))
                        return false;
                }
                if ((size_t)N <= c.cap && c.H == H && c.I == I)
                    return true;
                ctx_free_buffers(c);
                const size_t n = (size_t)N;
                if (!ck(cudaMallocHost((void **)&c.h_x, n * H * 4), "h_x") || !ck(cudaMallocHost((void **)&c.h_y, n * H * 4), "h_y") || !ck(cudaMallocHost((void **)&c.h_wt, n * 4), "h_wt") ||
                    !ck(cudaMalloc((void **)&c.d_x, n * H * 4), "d_x") || !ck(cudaMalloc((void **)&c.d_wt, n * 4), "d_wt") || !ck(cudaMalloc((void **)&c.d_gu, n * 2 * I * 4), "d_gu") ||
                    !ck(cudaMalloc((void **)&c.d_act, n * I * 4), "d_act") || !ck(cudaMalloc((void **)&c.d_eb, n * H * 4), "d_eb") || !ck(cudaMalloc((void **)&c.d_y, n * H * 4), "d_y"))
                    return false;
                c.cap = n;
                c.H = H;
                c.I = I;
                return true;
            }

            // The expert's KERNELS on the context's stream. The host<->device copies live OUTSIDE this function: a captured
            // graph replays its nodes at fixed addresses, so a copy inside the graph would read the host staging buffer at
            // REPLAY time, not at enqueue time - and experts_decode overwrites c.h_x/c.h_wt/c.h_desc for the next token
            // while this token's graph is still pending, so the expert computes with another token's activations (garbage).
            static bool expert_kernels(ExpCtx &c, int N)
            {
                const int H = c.H, I = c.I;
                cudaStream_t st = c.stream;
                const bool gemm = use_gemm(N, H) && use_gemm(N, I);
                if (gemm)
                    k_exp_gu_gemm<<<dim3((unsigned)((2 * I + 63) / 64), (unsigned)((N + 63) / 64)), 128, 0, st>>>(c.d_desc, c.d_x, c.d_gu);
                else
                    k_exp_gu<<<(2 * I + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(c.d_desc, c.d_x, c.d_gu);
                k_exp_act<<<(unsigned)std::min<size_t>(((size_t)N * I + 255) / 256, 65535), 256, 0, st>>>(c.d_desc, c.d_gu, c.d_act);
                if (gemm)
                    k_exp_dn_gemm<<<dim3((unsigned)((H + 63) / 64), (unsigned)((N + 63) / 64)), 128, 0, st>>>(c.d_desc, c.d_act, c.d_eb);
                else
                    k_exp_dn<<<(H + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(c.d_desc, c.d_act, c.d_eb);
                k_exp_fin<<<(unsigned)std::min<size_t>(((size_t)N * H + 255) / 256, 65535), 256, 0, st>>>(c.d_desc, c.d_eb, c.d_y);
                return !ck(cudaGetLastError(), "expert kernels");
            }

            bool enqueue_expert(ExpCtx &c, int N)
            {
                const int H = c.H;
                cudaStream_t st = c.stream;
                if (!ck(cudaMemcpyAsync(c.d_desc, c.h_desc, sizeof(DDesc), cudaMemcpyHostToDevice, st), "desc h2d") ||
                    !ck(cudaMemcpyAsync(c.d_x, c.h_x, (size_t)N * H * 4, cudaMemcpyHostToDevice, st), "x h2d") ||
                    !ck(cudaMemcpyAsync(c.d_wt, c.h_wt, (size_t)N * 4, cudaMemcpyHostToDevice, st), "wt h2d"))
                    return false;
                if (!expert_kernels(c, N))
                    return false;
                return ck(cudaMemcpyAsync(c.h_y, c.d_y, (size_t)N * H * 4, cudaMemcpyDeviceToHost, st), "y d2h");
            }

            // CUDA graph of the N == 1 KERNELS, captured once per context (thread-local capture: other threads' CUDA calls
            // are not recorded). Only the kernels are captured: the graph's nodes hold fixed device addresses, and the
            // staging buffers are re-uploaded before every launch by expert_run_async.
            bool build_graph(ExpCtx &c)
            {
                cudaGraph_t g = nullptr;
                bool ok = ck(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                if (ok)
                {
                    const bool e = expert_kernels(c, 1);
                    const cudaError_t ce = cudaStreamEndCapture(c.stream, &g);
                    ok = e && ck(ce, "end capture");
                }
                if (ok)
                    ok = ck(cudaGraphInstantiateWithFlags(&c.graph, g, 0), "graph instantiate");
                if (g)
                    cudaGraphDestroy(g);
                if (!ok)
                {
                    c.graph = nullptr;
                    cudaGetLastError();
                }
                return ok;
            }

            // ---------------------------------------------------------------- copy side
            constexpr int NSTAGE = 4;
            constexpr size_t STAGE_BYTES = (size_t)8 << 20;
            void *g_stage[NSTAGE] = {nullptr, nullptr, nullptr, nullptr};
            cudaEvent_t g_stage_ev[NSTAGE];
            bool g_stage_used[NSTAGE] = {false, false, false, false};
            int g_stage_i = 0;
            std::vector<std::pair<uintptr_t, size_t>> g_reg;
            std::vector<cudaEvent_t> g_events;

            bool is_registered(const void *p, size_t n)
            {
                const uintptr_t a = (uintptr_t)p, b = a + n;
                for (const auto &r : g_reg)
                    if (a >= r.first && b <= r.first + r.second)
                        return true;
                return false;
            }

        } // namespace

        bool is_emulated() { return false; }
        std::string last_error()
        {
            std::lock_guard<std::mutex> lk(g_err_mu);
            return g_err;
        }
        bool last_oom()
        {
            std::lock_guard<std::mutex> lk(g_err_mu);
            return g_oom;
        }
        void set_options(bool use_graphs) { g_graphs = use_graphs; }

        bool init(int device, std::string &err)
        {
            int n = 0;
            if (!ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount") || n == 0)
            {
                err = last_error().empty() ? "no CUDA device" : last_error();
                return false;
            }
            if (!ck(cudaSetDevice(device), "cudaSetDevice"))
            {
                err = last_error();
                return false;
            }
            g_dev = device;
            int maj = 0, min = 0;
            cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, device);
            g_cc = maj * 10 + min;
            const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
            if (!ck(cudaMemcpyToSymbol(c_fp4, lut, sizeof lut), "fp4 table") || !ck(cudaStreamCreateWithFlags(&g_comp, cudaStreamNonBlocking), "compute stream") ||
                !ck(cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking), "copy stream"))
            {
                err = last_error();
                return false;
            }
            return true;
        }

        void bind_thread() { cudaSetDevice(g_dev); }

        const char *device_name()
        {
            static cudaDeviceProp p;
            int d = 0;
            if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&p, d) != cudaSuccess)
                return "CUDA device";
            return p.name;
        }

        void mem_info(size_t *f, size_t *t)
        {
            *f = 0;
            *t = 0;
            if (!ck(cudaMemGetInfo(f, t), "cudaMemGetInfo"))
            {
                *f = 0;
                *t = 0;
            }
        }

        void *alloc(size_t n)
        {
            void *p = nullptr;
            if (cudaMalloc(&p, n ? n : 1) != cudaSuccess)
            {
                cudaGetLastError();
                return nullptr;
            } // OOM is not sticky: clear it
            return p;
        }

        void release(void *p)
        {
            if (p)
                cudaFree(p);
        }

        bool h2d(void *d, const void *h, size_t n)
        { // complete when it returns (ordered with every later use, whatever the stream)
            return ck(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, g_comp), "h2d") && ck(cudaStreamSynchronize(g_comp), "h2d sync");
        }

        // The decode's q/k/v of ONE layer: same x, so ONE upload, `count` kernels, ONE copy back and ONE wait.
        // The kernels write into one contiguous output buffer split by rows, so the host gets the blocks back by offset.
        bool dense_run_async(int ci, const float *hx, int N, const DenseOp *ops, int count)
        {
            if (ci < 0 || ci >= kDenseCtxBatch || count <= 0 || count > kDenseMaxOps || N <= 0)
                return false;
            DenseCtx &c = g_dc[ci];
            size_t max_cols = 0, total = 0;
            for (int j = 0; j < count; ++j)
            {
                if (!ops[j].W || ops[j].rows <= 0 || ops[j].cols <= 0)
                    return false;
                if (j > 0 && ops[j].cols != ops[0].cols)
                    return false; // the ops read the SAME x: one row width
                max_cols = std::max(max_cols, (size_t)ops[j].cols);
                c.rows[j] = ops[j].rows;
                c.off[j] = total;
                total += (size_t)N * (size_t)ops[j].rows;
            }
            c.off[count] = total;
            if (!dense_ctx_reserve(c, (size_t)N * max_cols, total))
                return false;
            std::memcpy(c.h_x, hx, (size_t)N * ops[0].cols * sizeof(float));
            cudaStream_t st = c.stream;
            if (!ck(cudaMemcpyAsync(c.d_x, c.h_x, (size_t)N * ops[0].cols * sizeof(float), cudaMemcpyHostToDevice, st), "dense x h2d"))
                return false;
            for (int j = 0; j < count; ++j)
            {
                float *dy = c.d_y + c.off[j];
                if (use_gemm(N, ops[j].cols))
                {
                    const dim3 grid((unsigned)((ops[j].rows + 63) / 64), (unsigned)((N + 63) / 64));
                    k_gemm_bf16<<<grid, 128, 0, st>>>((const unsigned short *)ops[j].W, ops[j].rows, ops[j].cols, c.d_x, N, dy);
                }
                else
                {
                    const int blocks = (ops[j].rows + WARPS - 1) / WARPS;
                    k_bf16<<<blocks, WARPS * 32, 0, st>>>((const unsigned short *)ops[j].W, ops[j].rows, ops[j].cols, c.d_x, N, dy);
                }
                if (!ck(cudaGetLastError(), "dense launch"))
                    return false;
            }
            if (!ck(cudaMemcpyAsync(c.h_y, c.d_y, total * sizeof(float), cudaMemcpyDeviceToHost, st), "dense y d2h"))
                return false;
            c.n = N;
            c.count = count;
            return ck(cudaEventRecord(c.ev, st), "dense event");
        }

        bool dense_collect(int ci, float *const *out)
        {
            if (ci < 0 || ci >= kDenseCtxBatch)
                return false;
            DenseCtx &c = g_dc[ci];
            if (!ck(cudaEventSynchronize(c.ev), "dense wait"))
                return false;
            for (int j = 0; j < c.count; ++j)
                std::memcpy(out[j], c.h_y + c.off[j], (size_t)c.n * (size_t)c.rows[j] * sizeof(float));
            return true;
        }

        bool reserve_scratch(size_t dense_x_floats, size_t dense_y_floats, int expert_batch_n, int H, int I, size_t dense_async_floats)
        {
            if (!scratch(dense_x_floats, dense_y_floats))
                return false;
            for (int i = 0; i < kExpertCtxBatch; ++i)
                if (!ctx_reserve(g_ctx[i], 1, H, I))
                    return false;
            if (!ctx_reserve(g_ctx[kExpertCtxBatch], std::max(1, expert_batch_n), H, I))
                return false;
            if (dense_async_floats > 0)
            {
                // the async dense context: its x holds the widest op's row (H for q/k/v), its y the whole batch of blocks
                if (!dense_ctx_reserve(g_dc[0], (size_t)dense_async_floats, dense_async_floats))
                    return false;
            }
            return true;
        }

        bool matmat_bf16(const void *dW, int rows, int cols, const float *hx, int N, float *hy) { return run_matmat(false, dW, nullptr, rows, cols, hx, N, hy); }
        bool matmat_mx(const void *dB, const void *dS, int rows, int cols, const float *hx, int N, float *hy)
        {
            if (cols % 32)
            {
                std::lock_guard<std::mutex> lk(g_err_mu);
                g_err = "matmat_mx: cols % 32 != 0";
                return false;
            }
            return run_matmat(true, dB, dS, rows, cols, hx, N, hy);
        }

        bool expert_run_async(int ci, const ExpertDesc &d, const float *hx, const float *wt, int N)
        {
            if (ci < 0 || ci > kExpertCtxBatch || N <= 0)
                return false;
            ExpCtx &c = g_ctx[ci];
            if (!ctx_reserve(c, N, d.H, d.I))
                return false;
            std::memcpy(c.h_x, hx, (size_t)N * d.H * sizeof(float));
            std::memcpy(c.h_wt, wt, (size_t)N * sizeof(float));
            DDesc &hd = *c.h_desc;
            hd.gu_b = (const uint8_t *)d.gu_b;
            hd.gu_s = (const uint8_t *)d.gu_s;
            hd.gu_bias = (const unsigned short *)d.gu_bias;
            hd.dn_b = (const uint8_t *)d.dn_b;
            hd.dn_s = (const uint8_t *)d.dn_s;
            hd.dn_bias = (const unsigned short *)d.dn_bias;
            hd.H = d.H;
            hd.I = d.I;
            hd.halves = d.halves;
            hd.N = N;
            hd.limit = d.limit;
            hd.alpha = d.alpha;
            hd.wt = c.d_wt;
            c.n = N;
            if (N == 1 && g_graphs && ci < kExpertCtxBatch)
            {
                if (!c.graph_tried)
                {
                    c.graph_tried = true;
                    c.graph_ok = build_graph(c);
                }
                if (c.graph_ok)
                {
                    // stage this expert's x, wt and descriptor with plain copies, then replay the captured kernels
                    if (!ck(cudaMemcpyAsync(c.d_desc, c.h_desc, sizeof(DDesc), cudaMemcpyHostToDevice, c.stream), "desc h2d") ||
                        !ck(cudaMemcpyAsync(c.d_x, c.h_x, (size_t)N * c.H * 4, cudaMemcpyHostToDevice, c.stream), "x h2d") ||
                        !ck(cudaMemcpyAsync(c.d_wt, c.h_wt, (size_t)N * 4, cudaMemcpyHostToDevice, c.stream), "wt h2d") ||
                        !ck(cudaGraphLaunch(c.graph, c.stream), "graph launch") ||
                        !ck(cudaMemcpyAsync(c.h_y, c.d_y, (size_t)N * c.H * 4, cudaMemcpyDeviceToHost, c.stream), "y d2h"))
                        return false;
                    return ck(cudaEventRecord(c.ev, c.stream), "expert event");
                }
            }
            return enqueue_expert(c, N) && ck(cudaEventRecord(c.ev, c.stream), "expert event");
        }

        bool expert_collect(int ci, float *hy)
        {
            if (ci < 0 || ci > kExpertCtxBatch)
                return false;
            ExpCtx &c = g_ctx[ci];
            if (!ck(cudaEventSynchronize(c.ev), "expert wait"))
                return false;
            std::memcpy(hy, c.h_y, (size_t)c.n * c.H * sizeof(float));
            return true;
        }

        bool d2h(void *h, const void *d, size_t n)
        {
            return ck(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, g_comp), "d2h") && ck(cudaStreamSynchronize(g_comp), "d2h sync");
        }

        bool dmm_bf16(const void *dW, int rows, int cols, const float *dx, int N, float *dy)
        {
            if (N <= 0)
                return true;
            if (use_gemm(N, cols))
                k_gemm_bf16<<<dim3((unsigned)((rows + 63) / 64), (unsigned)((N + 63) / 64)), 128, 0, g_comp>>>((const unsigned short *)dW, rows, cols, dx, N, dy);
            else
                k_bf16<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, g_comp>>>((const unsigned short *)dW, rows, cols, dx, N, dy);
            return ck(cudaGetLastError(), "dmm_bf16 launch");
        }

        bool rms_rows(const float *dx, const float *dg, float eps, int N, int H, float *dy)
        {
            if (N <= 0)
                return true;
            k_rms<<<N, 256, 0, g_comp>>>(dx, dg, eps, H, dy);
            return ck(cudaGetLastError(), "rms_rows launch");
        }

        bool add_rows(float *dX, const float *dA, const float *db, int N, int H)
        {
            if (N <= 0)
                return true;
            k_add_rows<<<grid1d((size_t)N * H, 256), 256, 0, g_comp>>>(dX, dA, db, (size_t)N * H, H);
            return ck(cudaGetLastError(), "add_rows launch");
        }

        bool attn_supported(const AttnShape &a) { return a.D % 32 == 0 && a.D <= 128 && a.heads % a.kv_heads == 0 && a.cap > 0; }

        // The rotated k and the biased v stay in the fp32 stage (dk / dv, in place); this call then packs those
        // rows into the card's formats (a.k_fmt / a.v_fmt, chosen INDEPENDENTLY by -k / -v) and writes the row's
        // inverse scale. With F32 the pack is skipped and the stage IS the cache, exactly as before.
        bool qk_rope_kv(float *dq, float *dk, const float *dv, const float *dbq, const float *dbk, const float *dbv, const float *dcos, const float *dsin,
                        int N, int p0, const AttnShape &a, float *dKc, float *dVc)
        {
            if (N <= 0)
                return true;
            if (a.D % 32 != 0 || a.D > 128 || a.heads % a.kv_heads != 0 || a.cap <= 0)
            {
                std::lock_guard<std::mutex> lk(g_err_mu);
                g_err = "qk_rope_kv: unsupported shape";
                return false;
            }
            const int KD = a.kv_heads * a.D;
            const size_t items = (size_t)N * (a.heads + a.kv_heads) * (a.D / 2);
            k_qk_rope<<<grid1d(items, 256), 256, 0, g_comp>>>((float *)dq, (float *)dk, dbq, dbk, dcos, dsin, N, p0, a.heads, a.kv_heads, a.D, a.sliding ? 1 : 0, a.cap, a.rope_pairs ? 1 : 0, nullptr);
            if (!ck(cudaGetLastError(), "k_qk_rope launch"))
                return false;
            k_v_bias<<<grid1d((size_t)N * KD, 256), 256, 0, g_comp>>>((float *)dv, dbv, N, KD);
            if (!ck(cudaGetLastError(), "k_v_bias launch"))
                return false;
            // F32 on one side still needs the rows copied into that side's cache: with the stage being a_K / a_V
            // and the cache a separate buffer, the copy is the pack with fmt = F32 (a plain row copy).
            auto pack = [&](int fmt, const float *stage, void *dst, float *iscl, float limit) {
                if (fmt == 0) k_kv_pack_at<__nv_fp8_e4m3><<<(unsigned)std::min(N, 65535), 256, 0, g_comp>>>(stage, (__nv_fp8_e4m3 *)dst, iscl, N, p0, KD, a.cap, a.sliding ? 1 : 0, limit);
                else if (fmt == 1) k_kv_pack_at<half><<<(unsigned)std::min(N, 65535), 256, 0, g_comp>>>(stage, (half *)dst, iscl, N, p0, KD, a.cap, a.sliding ? 1 : 0, limit);
                else k_kv_pack_at<float><<<(unsigned)std::min(N, 65535), 256, 0, g_comp>>>(stage, (float *)dst, nullptr, N, p0, KD, a.cap, a.sliding ? 1 : 0, 0.0f);
                return ck(cudaGetLastError(), "kv pack");
            };
            if (!pack(a.k_fmt, dk, dKc, a.dKinv, a.kv_limit))
                return false;
            return pack(a.v_fmt, dv, dVc, a.dVinv, a.kv_limit);
        }

        // Host fp32 rows -> the card's packed cache (what decode wrote since the last device prefill).
        bool kv_upload_rows(int fmt, const void *host, void *dst, float *iscl, int rows, int p0, int kd, int cap, bool sliding, float limit)
        {
            if (!kv_ensure_scratch((size_t)rows * kd * 4)) return false;
            if (rows <= 0)
                return true;
            if (!ck(cudaMemcpyAsync(g_kv_scratch, host, (size_t)rows * kd * 4, cudaMemcpyHostToDevice, g_comp), "kv stage h2d"))
                return false;
            if (fmt == 0) k_kv_pack_at<__nv_fp8_e4m3><<<(unsigned)std::min(rows, 65535), 256, 0, g_comp>>>((const float *)g_kv_scratch, (__nv_fp8_e4m3 *)dst, iscl, rows, p0, kd, cap, sliding ? 1 : 0, limit);
            else if (fmt == 1) k_kv_pack_at<half><<<(unsigned)std::min(rows, 65535), 256, 0, g_comp>>>((const float *)g_kv_scratch, (half *)dst, iscl, rows, p0, kd, cap, sliding ? 1 : 0, limit);
            else if (!ck(cudaMemcpy2DAsync(dst, (size_t)kd * 4, host, (size_t)kd * 4, (size_t)kd * 4, rows, cudaMemcpyHostToDevice, g_comp), "kv f32 h2d"))
                return false;
            return ck(cudaGetLastError(), "kv upload");
        }

        // The card's packed rows -> the host fp32 cache (the host copy is what decode / checkpoints / reuse read).
        bool kv_download_rows(int fmt, void *host, const void *src, const float *iscl, int rows, int p0, int kd, int cap, bool sliding)
        {
            if (!kv_ensure_scratch((size_t)rows * kd * 4)) return false;
            if (rows <= 0)
                return true;
            if (fmt == 0) k_kv_unpack_at<__nv_fp8_e4m3><<<(unsigned)std::min(rows, 65535), 256, 0, g_comp>>>((__nv_fp8_e4m3 *)src, iscl, (float *)g_kv_scratch, rows, p0, kd, cap, sliding ? 1 : 0);
            else if (fmt == 1) k_kv_unpack_at<half><<<(unsigned)std::min(rows, 65535), 256, 0, g_comp>>>((half *)src, iscl, (float *)g_kv_scratch, rows, p0, kd, cap, sliding ? 1 : 0);
            else if (!ck(cudaMemcpy2DAsync(host, (size_t)kd * 4, src, (size_t)kd * 4, (size_t)kd * 4, rows, cudaMemcpyDeviceToHost, g_comp), "kv f32 d2h"))
                return true;
            if (!ck(cudaGetLastError(), "kv unpack"))
                return false;
            if (fmt == 2)
                return true;
            return ck(cudaMemcpyAsync(host, g_kv_scratch, (size_t)rows * kd * 4, cudaMemcpyDeviceToHost, g_comp), "kv d2h");
        }

        bool attn_rows(const float *dq, const float *dKc, const float *dVc, const float *dsinks, int N, int p0, const AttnShape &a, float *dout)
        {
            if (N <= 0)
                return true;
            if (a.D % 32 != 0 || a.D > 128 || a.heads % a.kv_heads != 0 || a.cap <= 0)
            {
                std::lock_guard<std::mutex> lk(g_err_mu);
                g_err = "attn_rows: unsupported shape";
                return false;
            }
            const float scale = 1.0f / sqrtf((float)a.D);
#define ATTN(KT, VT) k_attn<KT, VT><<<dim3((unsigned)N, (unsigned)a.heads), 32, 0, g_comp>>>( \
    dq, (const KT *)dKc, (const VT *)dVc, a.dKinv, a.dVinv, dsinks, p0, a.heads, a.kv_heads, a.D, a.window, a.cap, a.sliding ? 1 : 0, scale, dout)
            if (a.k_fmt == 0 && a.v_fmt == 0) ATTN(__nv_fp8_e4m3, __nv_fp8_e4m3);
            else if (a.k_fmt == 0 && a.v_fmt == 1) ATTN(__nv_fp8_e4m3, half);
            else if (a.k_fmt == 1 && a.v_fmt == 0) ATTN(half, __nv_fp8_e4m3);
            else if (a.k_fmt == 1 && a.v_fmt == 1) ATTN(half, half);
            else if (a.k_fmt == 0) ATTN(__nv_fp8_e4m3, float);
            else if (a.k_fmt == 1) ATTN(half, float);
            else if (a.v_fmt == 0) ATTN(float, __nv_fp8_e4m3);
            else if (a.v_fmt == 1) ATTN(float, half);
            else ATTN(float, float);
#undef ATTN
            return ck(cudaGetLastError(), "k_attn launch");
        }

        int event_create()
        {
            cudaEvent_t e = nullptr;
            if (!ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event create"))
                return -1;
            std::lock_guard<std::mutex> lk(g_copy_mu);
            g_events.push_back(e);
            return (int)g_events.size() - 1;
        }

        bool event_sync(int ev)
        {
            cudaEvent_t e;
            {
                std::lock_guard<std::mutex> lk(g_copy_mu);
                if (ev < 0 || ev >= (int)g_events.size())
                    return false;
                e = g_events[(size_t)ev];
            }
            return ck(cudaEventSynchronize(e), "event sync");
        }

        bool event_record_copy(int ev)
        {
            std::lock_guard<std::mutex> lk(g_copy_mu);
            if (ev < 0 || ev >= (int)g_events.size())
                return false;
            return ck(cudaEventRecord(g_events[(size_t)ev], g_copy), "event record");
        }

        bool h2d_copy_async(void *dev, const void *host, size_t n)
        {
            std::lock_guard<std::mutex> lk(g_copy_mu);
            if (is_registered(host, n))
                return ck(cudaMemcpyAsync(dev, host, n, cudaMemcpyHostToDevice, g_copy), "prefetch copy"); // direct DMA from page-locked RAM
            if (!g_stage[0])
            {
                for (int i = 0; i < NSTAGE; ++i)
                    if (!ck(cudaMallocHost(&g_stage[i], STAGE_BYTES), "staging") || !ck(cudaEventCreateWithFlags(&g_stage_ev[i], cudaEventDisableTiming), "staging event"))
                        return false;
            }
            size_t off = 0;
            while (off < n)
            { // through pinned staging: reuse a buffer only after its previous copy is done
                const int i = g_stage_i;
                g_stage_i = (g_stage_i + 1) % NSTAGE;
                if (g_stage_used[i] && !ck(cudaEventSynchronize(g_stage_ev[i]), "staging wait"))
                    return false;
                const size_t len = std::min(STAGE_BYTES, n - off);
                std::memcpy(g_stage[i], (const uint8_t *)host + off, len);
                if (!ck(cudaMemcpyAsync((uint8_t *)dev + off, g_stage[i], len, cudaMemcpyHostToDevice, g_copy), "staged copy"))
                    return false;
                if (!ck(cudaEventRecord(g_stage_ev[i], g_copy), "staging record"))
                    return false;
                g_stage_used[i] = true;
                off += len;
            }
            return true;
        }

        bool host_register(const void *p, size_t n)
        {
            const uintptr_t ps = 4096;
            const uintptr_t a = ((uintptr_t)p + ps - 1) & ~(ps - 1), b = ((uintptr_t)p + n) & ~(ps - 1);
            if (b <= a)
                return true;
            unsigned flags = cudaHostRegisterDefault;
#ifdef cudaHostRegisterReadOnly
            flags = cudaHostRegisterReadOnly; // the mapping is PROT_READ
#endif
            if (!ck(cudaHostRegister((void *)a, (size_t)(b - a), flags), "cudaHostRegister"))
            {
                cudaGetLastError();
                return false;
            }
            std::lock_guard<std::mutex> lk(g_copy_mu);
            g_reg.push_back({a, (size_t)(b - a)});
            return true;
        }
    } // namespace gpu
} // namespace gptoss

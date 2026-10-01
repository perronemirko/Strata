// src/kernels/cuda/dense_kernels.cu - see include/strata/kernels/dense_kernels.hpp.
#include "strata/kernels/dense_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kWarps = 8;            // warps per attention block: 8 cells in flight
constexpr int kChunk = 256;          // cells per attention block (one split)
constexpr int kMaxHeadDim = 256;     // the merge maps one thread to one output dim

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------------ RMSNorm
// One block per row; the row is read twice (sum of squares, then scale), which for 5120 floats stays in L1.
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                int cols, float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* orow = out + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += xr[i] * xr[i];
    acc = warp_sum(acc);
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
        scale = rsqrtf(s / (float) cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * scale * w[i];
}

// One block per value head: rms over `cols`, then * w[i] * silu(z[i]).
__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ w, float* __restrict__ y, int cols, float eps) {
    const float* orow = o + (size_t) blockIdx.x * cols;
    const float* zrow = z + (size_t) blockIdx.x * cols;
    float* yrow = y + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += orow[i] * orow[i];
    acc = warp_sum(acc);
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
        scale = rsqrtf(s / (float) cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float g = zrow[i];
        yrow[i] = orow[i] * scale * w[i] * (g / (1.0f + __expf(-g)));
    }
}

// One block of 1024 threads: pass 1 finds the largest logit (lowest index on a tie), pass 2 sums exp(l - max).
__global__ void argmax_prob_kernel(const float* __restrict__ logits, int n, int* __restrict__ out_id,
                                   float* __restrict__ out_p) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float best = -FLT_MAX;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }    // NaN compares false: never picked
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    const float gmax = sv[0];
    const int gid = si[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (!isnan(v)) sum += __expf(v - gmax);
    }
    sv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const bool ok = gid != INT_MAX && sv[0] > 0.0f;
        *out_id = ok ? gid : 0;
        *out_p = ok ? 1.0f / sv[0] : 0.0f;
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o,
                              int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    o[i] = (a / (1.0f + __expf(-a))) * u[i];
}

// One warp per output row.
__global__ void gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32) acc += wr[i] * x[i];
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void fill_i32_kernel(int32_t* dst, int n, int32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

// ------------------------------------------------------------------------------------------------ KV cache
__global__ void kv_append_kernel(__half* __restrict__ kc, __half* __restrict__ vc, const float* __restrict__ k,
                                 const float* __restrict__ v, int pos, int head_dim, int max_ctx) {
    const int h = blockIdx.x;
    const size_t cell = ((size_t) h * max_ctx + pos) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        kc[cell + i] = __float2half(k[(size_t) h * head_dim + i]);
        vc[cell + i] = __float2half(v[(size_t) h * head_dim + i]);
    }
}

// ------------------------------------------------------------------------------------------------ attention
// Grid (n_head, n_splits), 256 threads = 8 warps.  A warp owns a cell at a time: its 32 lanes hold head_dim/32
// dims each (strided by 32 so the half loads coalesce), the dot product is a warp reduction, and the warp's
// running (m, l, acc) is the online softmax over the cells it visited (cells t = begin + warp, +8, ...).
__global__ void attn_partial_kernel(const float* __restrict__ q, const __half* __restrict__ K,
                                    const __half* __restrict__ V, float* __restrict__ part_acc,
                                    float* __restrict__ part_ml, int n_head, int n_kv, int head_dim, int n_ctx,
                                    int max_ctx, float scale, int n_splits) {
    const int h = blockIdx.x, split = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;                       // dims per lane, <= 8

    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = i < per ? q[(size_t) h * head_dim + lane + 32 * i] * scale : 0.0f;

    const __half* Kb = K + (size_t) kvh * max_ctx * head_dim;
    const __half* Vb = V + (size_t) kvh * max_ctx * head_dim;
    const int begin = split * kChunk;
    const int end = min(n_ctx, begin + kChunk);

    float m = -FLT_MAX, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int t = begin + warp; t < end; t += kWarps) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) d += qv[i] * __half2float(Kb[(size_t) t * head_dim + lane + 32 * i]);
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn);               // m == -FLT_MAX on the first cell: exp(-huge) = 0
        const float p = __expf(d - mn);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * __half2float(Vb[(size_t) t * head_dim + lane + 32 * i]);
        m = mn;
    }

    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < per) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float gm = -FLT_MAX;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[w]);
        float a = 0.0f, L = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            if (sl[w] == 0.0f) continue;                 // this warp saw no cell of the chunk
            const float f = __expf(sm[w] - gm);
            a += sacc[w][threadIdx.x] * f;
            L += sl[w] * f;
        }
        part_acc[((size_t) h * n_splits + split) * head_dim + threadIdx.x] = a;
        if (threadIdx.x == 0) {
            part_ml[((size_t) h * n_splits + split) * 2] = L > 0.0f ? gm : -FLT_MAX;
            part_ml[((size_t) h * n_splits + split) * 2 + 1] = L;
        }
    }
}

// Grid n_head, head_dim threads: merges the chunks of one head.
__global__ void attn_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                  float* __restrict__ out, int head_dim, int n_splits) {
    const int h = blockIdx.x;
    float gm = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, part_ml[((size_t) h * n_splits + s) * 2]);
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float l = part_ml[((size_t) h * n_splits + s) * 2 + 1];
        if (l == 0.0f) continue;
        const float f = __expf(part_ml[((size_t) h * n_splits + s) * 2] - gm);
        a += part_acc[((size_t) h * n_splits + s) * head_dim + threadIdx.x] * f;
        L += l * f;
    }
    out[(size_t) h * head_dim + threadIdx.x] = L > 0.0f ? a / L : 0.0f;
}

int splits_for(int cells) { return (cells + kChunk - 1) / kChunk; }

// ================================================================================== the batched prompt kernels
// The same arithmetic as the per-token path above, with the token as the slow index, so a chunk of T tokens lands on
// the same logits as feeding those tokens through step() one at a time.

__global__ void split_q_kernel(const float* __restrict__ q_full, float* __restrict__ q, int head_dim) {
    const int row = blockIdx.x;                       // one row per (token, query head)
    const float* src = q_full + (size_t) row * 2 * head_dim;
    float* dst = q + (size_t) row * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) dst[i] = src[i];
}

// out[i] = attn[i] * sigmoid(gate) with the gate the second half of the head's [q | gate] block.
__global__ void gate_apply_kernel(const float* __restrict__ attn, const float* __restrict__ q_full,
                                  float* __restrict__ out, int head_dim, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / head_dim, d = i - row * head_dim;
    const float g = q_full[row * 2 * head_dim + head_dim + d];
    out[i] = attn[i] * (1.0f / (1.0f + __expf(-g)));
}

__global__ void positions_kernel(int32_t* dst, int heads, int32_t pos0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = pos0 + (int32_t) (i / heads);
}

// grid (n_kv, T): one block per (KV head, token) writes that token's cell of that head's cache.
__global__ void kv_append_rows_kernel(__half* __restrict__ kc, __half* __restrict__ vc, const float* __restrict__ k,
                                      const float* __restrict__ v, int pos0, int n_kv, int head_dim, int max_ctx) {
    const int h = blockIdx.x, t = blockIdx.y;
    const size_t cell = ((size_t) h * max_ctx + pos0 + t) * head_dim;
    const float* kh = k + ((size_t) t * n_kv + h) * head_dim;
    const float* vh = v + ((size_t) t * n_kv + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        kc[cell + i] = __float2half(kh[i]);
        vc[cell + i] = __float2half(vh[i]);
    }
}

// One block per (query head, token): the decode kernel's warp-per-cell online softmax, restricted to the causal
// range [0, pos0 + t].  No split-K: the (head, token) grid already gives n_head * T blocks.
__global__ void attn_chunk_kernel(const float* __restrict__ q, const __half* __restrict__ K,
                                  const __half* __restrict__ V, float* __restrict__ out, int n_head, int n_kv,
                                  int head_dim, int pos0, int max_ctx, float scale) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;

    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        qv[i] = i < per ? q[((size_t) t * n_head + h) * head_dim + lane + 32 * i] * scale : 0.0f;

    const __half* Kb = K + (size_t) kvh * max_ctx * head_dim;
    const __half* Vb = V + (size_t) kvh * max_ctx * head_dim;
    const int end = pos0 + t + 1;

    float m = -FLT_MAX, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int c = warp; c < end; c += kWarps) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) d += qv[i] * __half2float(Kb[(size_t) c * head_dim + lane + 32 * i]);
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn);
        const float p = __expf(d - mn);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * __half2float(Vb[(size_t) c * head_dim + lane + 32 * i]);
        m = mn;
    }

    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < per) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float gm = -FLT_MAX;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[w]);
        float a = 0.0f, L = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            if (sl[w] == 0.0f) continue;
            const float f = __expf(sm[w] - gm);
            a += sacc[w][threadIdx.x] * f;
            L += sl[w] * f;
        }
        out[((size_t) t * n_head + h) * head_dim + threadIdx.x] = L > 0.0f ? a / L : 0.0f;
    }
}

// grid (ceil(n_out / 4), rows): one warp per output row of W for one input row.
__global__ void gemv_f32_rows_kernel(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ y,
                                     int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int col = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    const float* xr = X + (size_t) col * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32) acc += wr[i] * xr[i];
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) col * n_out + row] = acc;
}

// One thread per conv channel, walking the chunk inside the launch (the decode path's conv_silu with a token loop).
__global__ void gdn_conv_chunk_kernel(float* __restrict__ history, const float* __restrict__ qkv,
                                      const float* __restrict__ weights, float* __restrict__ h, int channels, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float v[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], 0.0f};
    const float* w = weights + (size_t) c * 4;
    for (int t = 0; t < T; ++t) {
        v[3] = qkv[(size_t) t * channels + c];
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += v[tap] * w[tap];
        // __expf, not expf: the decode path's conv_silu is compiled with --use_fast_math, which turns its expf
        // into this intrinsic.  Matching it keeps a chunk's conv output as close as possible to the token loop's.
        h[(size_t) t * channels + c] = sum / (1.0f + __expf(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) v[tap] = v[tap + 1];
    }
    history[c * 3] = v[0];
    history[c * 3 + 1] = v[1];
    history[c * 3 + 2] = v[2];
}

// grid(rows_per_token, T): one block per q or k head of one token, the row starting at `base` inside that token's
// `channels` values.  The same scaling as native_gdn_l2_norm (which is handed epsilon / 128 and multiplies by
// 1 / sqrt(128) afterwards).
__global__ void gdn_l2_norm_kernel(float* x, float eps, int channels, int base) {
    constexpr int W = 128;
    const int col = threadIdx.x;
    x += (size_t) blockIdx.y * channels + base + (size_t) blockIdx.x * W;
    const float value = col < W ? x[col] : 0.0f;
    float partial = value * value;
    __shared__ float sums[4];
    partial = warp_sum(partial);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = partial;
    __syncthreads();
    const float total = sums[0] + sums[1] + sums[2] + sums[3];
    const float scale = rsqrtf(total / (float) W + eps / (float) W);
    if (col < W) x[col] = scale * value * (1.0f / sqrtf((float) W));
}

__global__ void gdn_gates_kernel(const float* __restrict__ alpha, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int heads, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int hh = (int) (i % heads);
    const float value = alpha[i] + dt[hh];
    const float softplus = value > 20.0f ? value : __logf(1.0f + __expf(value));
    gate[i] = softplus * ssm_a[hh];
    beta[i] = 1.0f / (1.0f + __expf(-beta[i]));
}

// The delta-rule recurrence over a chunk: one block per value head, its 128 state columns split over 4 row groups of
// 32 threads (the rows a warp owns stay in registers), the T tokens walked in order, and the output norm
// rms_norm(o) * gamma * SiLU(z) folded in - SiLU, which is what qwen35 does and the MoE path's sigmoid does not.
// The readout is scaled by 1/sqrt(128) exactly as native_gdn_step's `scale`, so the norm sees the same values.
__global__ void __launch_bounds__(128 * 4) gdn_rec_chunk_kernel(
        float* __restrict__ state, const float* __restrict__ h, const float* __restrict__ gate,
        const float* __restrict__ beta, const float* __restrict__ z, const float* __restrict__ gamma, float eps,
        float* __restrict__ y, int k_heads, int v_heads, int T) {
    constexpr int W = 128, RG = 4, RPG = W / RG;
    __shared__ float sq[W], sk[W], red[RG][W], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * W + col;
    const int qh = head % k_heads;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * v_heads + head) * W + col;
    const size_t rs = (size_t) v_heads * W;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    const float rts = 1.0f / sqrtf((float) W);      // native_gdn_step's readout scale, the same expression
    const size_t hstride = (size_t) (2 * k_heads + v_heads) * W;
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * hstride;
        __syncthreads();
        if (tid < W) {
            sq[tid] = ht[qh * W + tid];
            sk[tid] = ht[k_heads * W + qh * W + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * v_heads + head]);   // the decode step's fast-math expf
        float kv = 0.0f;
        // The expressions are native_gdn_step's, written the same way, so the compiler contracts them the same way
        // (native_gdn.cu is compiled with --use_fast_math like this file): a chunk lands on the same bits as the
        // token loop wherever the reduction order allows.
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv += s[r] * sk[rg * RPG + r];
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * k_heads * W + head * W + col] - g * kv_col) * beta[(size_t) t * v_heads + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = g * s[r] + sk[rg * RPG + r] * delta;
            o += s[r] * sq[rg * RPG + r];
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rts;
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float gz = z[(size_t) t * v_heads * W + head * W + col];
            y[(size_t) t * v_heads * W + head * W + col] =
                oc * rsqrtf(ss / (float) W + eps) * g_col * (gz / (1.0f + __expf(-gz)));
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

}  // namespace

void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream) {
    rms_norm_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, out, cols, eps);
    check("dense_rms_norm");
}

void dense_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int heads, int cols, float eps,
                        void* stream) {
    gdn_out_norm_kernel<<<heads, 128, 0, (cudaStream_t) stream>>>(o, z, w, y, cols, eps);
    check("dense_gdn_out_norm");
}

void dense_argmax_prob(const float* logits, int n, int* out_id, float* out_p, void* stream) {
    argmax_prob_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(logits, n, out_id, out_p);
    check("dense_argmax_prob");
}

void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream) {
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, n);
    check("dense_swiglu");
}

void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream) {
    const int warps = 4;
    gemv_f32_kernel<<<(n_out + warps - 1) / warps, warps * 32, 0, (cudaStream_t) stream>>>(W, x, y, n_in, n_out);
    check("dense_gemv_f32");
}

void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream) {
    fill_i32_kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t) stream>>>(dst, n, value);
    check("dense_fill_i32");
}

void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream) {
    if (pos < 0 || pos >= max_ctx) throw std::runtime_error("dense_kv_append: position outside the cache");
    kv_append_kernel<<<n_kv, 128, 0, (cudaStream_t) stream>>>((__half*) k_cache, (__half*) v_cache, k, v, pos,
                                                              head_dim, max_ctx);
    check("dense_kv_append");
}

uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx) {
    const uint64_t splits = (uint64_t) splits_for(max_ctx);
    return (uint64_t) n_head * splits * ((uint64_t) head_dim + 2) * sizeof(float);
}

void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream) {
    if (n_ctx < 1 || n_ctx > max_ctx) throw std::runtime_error("dense_attn_decode: n_ctx outside the cache");
    if (head_dim % 32 != 0 || head_dim > kMaxHeadDim || n_head % n_kv != 0)
        throw std::runtime_error("dense_attn_decode: head_dim must be a multiple of 32 up to 256 and n_head a multiple of n_kv");
    const int n_splits = splits_for(n_ctx);
    const int max_splits = splits_for(max_ctx);
    // the partials sit at a stride of n_splits of THIS call; the scratch was sized for max_splits
    float* part_acc = scratch;
    float* part_ml = scratch + (size_t) n_head * max_splits * head_dim;
    dim3 grid(n_head, n_splits);
    attn_partial_kernel<<<grid, kWarps * 32, 0, (cudaStream_t) stream>>>(
        q, (const __half*) k_cache, (const __half*) v_cache, part_acc, part_ml, n_head, n_kv, head_dim, n_ctx,
        max_ctx, scale, n_splits);
    check("dense_attn_partial");
    attn_merge_kernel<<<n_head, head_dim, 0, (cudaStream_t) stream>>>(part_acc, part_ml, out, head_dim, n_splits);
    check("dense_attn_merge");
}

// ============================================================================ the batched prompt path
void dense_split_q(const float* q_full, float* q, int rows, int head_dim, void* stream) {
    if (rows < 1) return;
    split_q_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(q_full, q, head_dim);
    check("dense_split_q");
}

void dense_gate_apply(const float* attn, const float* q_full, float* out, int rows, int n_head, int head_dim,
                      void* stream) {
    const int64_t n = (int64_t) rows * head_dim;
    if (n <= 0) return;
    (void) n_head;
    gate_apply_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out, head_dim, n);
    check("dense_gate_apply");
}

void dense_positions_i32(int32_t* dst, int rows, int heads, int32_t pos0, void* stream) {
    if (rows < 1) return;
    positions_kernel<<<(rows + 127) / 128, 128, 0, (cudaStream_t) stream>>>(dst, heads, pos0);
    check("dense_positions_i32");
}

void dense_kv_append_rows(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int T, int pos0,
                          int n_kv, int head_dim, int max_ctx, void* stream) {
    if (T < 1 || pos0 < 0 || pos0 + T > max_ctx)
        throw std::runtime_error("dense_kv_append_rows: positions outside the cache");
    const dim3 grid(n_kv, T);
    kv_append_rows_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>((__half*) k_cache, (__half*) v_cache, k, v, pos0,
                                                                   n_kv, head_dim, max_ctx);
    check("dense_kv_append_rows");
}

void dense_attn_chunk(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int T, int pos0,
                      int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream) {
    if (T < 1 || pos0 < 0 || pos0 + T > max_ctx)
        throw std::runtime_error("dense_attn_chunk: positions outside the cache");
    if (head_dim % 32 != 0 || head_dim > kMaxHeadDim || n_head % n_kv != 0)
        throw std::runtime_error("dense_attn_chunk: head_dim must be a multiple of 32 up to 256 and n_head a multiple of n_kv");
    const dim3 grid(n_head, T);
    attn_chunk_kernel<<<grid, kWarps * 32, 0, (cudaStream_t) stream>>>(
        q, (const __half*) k_cache, (const __half*) v_cache, out, n_head, n_kv, head_dim, pos0, max_ctx, scale);
    check("dense_attn_chunk");
}

void dense_gemv_f32_rows(const float* W, const float* X, float* y, int n_in, int n_out, int rows, void* stream) {
    if (rows < 1) return;
    const int warps = 4;
    const dim3 grid((n_out + warps - 1) / warps, rows);
    gemv_f32_rows_kernel<<<grid, warps * 32, 0, (cudaStream_t) stream>>>(W, X, y, n_in, n_out);
    check("dense_gemv_f32_rows");
}

void dense_gdn_conv_chunk(float* history, const float* qkv, const float* conv_w, float* h, int channels, int T,
                          void* stream) {
    if (T < 1) return;
    gdn_conv_chunk_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, T);
    check("dense_gdn_conv_chunk");
}

void dense_gdn_l2_norm(float* h, int T, int channels, int base, int rows_per_token, int cols, float eps,
                       void* stream) {
    if (T < 1 || rows_per_token < 1) return;
    if (cols != 128) throw std::runtime_error("dense_gdn_l2_norm: only the 128-wide GDN state rows are supported");
    if (base < 0 || base + (int64_t) rows_per_token * cols > channels)
        throw std::runtime_error("dense_gdn_l2_norm: the head rows are outside the token's channels");
    const dim3 grid(rows_per_token, T);
    gdn_l2_norm_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(h, eps, channels, base);
    check("dense_gdn_l2_norm");
}

void dense_gdn_gates(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int heads,
                     int rows, void* stream) {
    const int64_t n = (int64_t) rows * heads;
    if (n <= 0) return;
    gdn_gates_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(alpha, dt, ssm_a, gate, beta,
                                                                                      heads, n);
    check("dense_gdn_gates");
}

void dense_gdn_rec_chunk(float* state, const float* h, const float* gate, const float* beta, const float* z,
                         const float* gamma, float eps, float* y, int k_heads, int v_heads, int T, void* stream) {
    if (T < 1 || k_heads < 1 || v_heads < 1 || v_heads % k_heads != 0)
        throw std::runtime_error("dense_gdn_rec_chunk: T >= 1 and v_heads a positive multiple of k_heads");
    const dim3 block(128, 4);
    gdn_rec_chunk_kernel<<<v_heads, block, 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y,
                                                                       k_heads, v_heads, T);
    check("dense_gdn_rec_chunk");
}

}  // namespace strata::kernels

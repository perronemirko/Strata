// gptoss prefill_dev - DEVICE-RESIDENT PREFILL (real CUDA). Build: sh build.sh --cuda.
//
// Why this exists: the batched prefill of src/model.cpp asks the device for one matmat per projection and per
// touched expert. A 512-token chunk touches 36 layers x (5 projections + up to 128 experts) ~ 9000 device calls,
// each an H2D + stream sync + D2H + two host memcpys: the round-trip latency, not the math, is what caps prefill
// at 8-16 tok/s. Here the chunk's residual stream stays ON THE CARD for the whole layer loop. Per chunk the host
// touches the device only for X in/out, the router logits down and the chunk's new K/V rows back (the decode and
// the KV checkpoint keep living in the host cache, so nothing else has to move). Per layer of MoE: ONE grouped
// launch for all 128 experts instead of 128 calls.
//
// Scope: a dense layer's attention reads the whole history, so its K/V must be on the card; init() sizes that to
// what actually fits and run_layer refuses a chunk that would read past it - the Model then finishes the prompt
// with its old path from the same host cache (prefill_pdev returns the leading count it completed). Banded
// layers only need window+chunk rows, so they always fit.
//
// The arithmetic mirrors src/model.cpp operation for operation: rmsnorm -> q/k/v -> bias+rope -> attention
// (sink in the softmax denominator only, window or full) -> o -> residual -> post-LN -> router -> host routing
// -> grouped experts -> scatter. Same fp32 order, double sums where the host has them.
#include "gptoss/prefill_dev.hpp"
#include "gptoss/kv_quant.hpp"   // KVFormat + pack/unpack: ONE definition, shared with src/gpu_cuda.cu

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace gptoss {
namespace pdev {

namespace {

constexpr int WARPS = 4;              // output rows per block for the warp-per-row kernels
constexpr int TILE = 8;             // token rows whose accumulators live in registers at once
constexpr int ATTN_THREADS = 32;    // one block per (token, head): each thread owns one key of a 32-key tile
constexpr int MAXD = 64;            // head_dim: gpt-oss is always 64; init() refuses anything else
constexpr int BM = 64, BN = 64, BK = 32, LDS = 40, LDC = BM + 4;

std::string g_err;
bool g_bad = false;                 // an allocation or launch failed: the Model turns the whole path off
int g_cc = 0;
cudaStream_t g_st = nullptr;
__constant__ float c_fp4[16];

Cfg g_cfg;
int g_chunk = 0, g_rmax = 0, g_qd = 0, g_kd = 0, g_half = 0, g_layers = 0;
int g_cur_pos0 = 0, g_cur_T = 0;

// Asymmetric KV formats (-k / -v): the card cache is packed per side, independently.
int g_kfmt = 2, g_vfmt = 2;         // gptoss::gpu::KVFormat
int g_kb = 4, g_vb = 4;             // bytes per element of the packed rows
float g_klim = 0.0f, g_vlim = 0.0f; // e4m3 saturation (0 = no rescaling: f16/f32)
float *g_KQ = nullptr, *g_VQ = nullptr;   // fp32 staging for the H2D upload (host bytes cannot be read by a kernel)
int g_upmax = 0;                    // rows the staging can hold: max(chunk, window)

float *g_X = nullptr, *g_HN = nullptr, *g_Q = nullptr, *g_KS = nullptr, *g_VS = nullptr;
float *g_ATT = nullptr, *g_OB = nullptr, *g_GL = nullptr;
float *g_GU = nullptr, *g_ACT = nullptr, *g_EB = nullptr;
int *g_SRC = nullptr, *g_CNT = nullptr, *g_RO = nullptr;
float *g_RW = nullptr;
void *g_PE = nullptr;               // the E experts' weight pointers, as one device array of ExpertPtrs
std::vector<int> g_hro;             // host staging for the row offsets (uploaded with one copy)

struct LDev {
    float *in_ln = nullptr, *post_ln = nullptr, *sinks = nullptr;
    float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr, *rbias = nullptr;
    const void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dobj = nullptr, *drt = nullptr;
};
std::vector<LDev> g_L;

// The card's K/V rows are PACKED (fp8 e4m3 = 1B, half = 2B, float = 4B), K and V independently: the
// host cache and the KV checkpoint stay fp32, only the card's copy is compressed. The per-row scale
// (one power of two per position, shared by that position's kv_heads) lives in kinv/vinv; the
// attention kernels multiply the raw packed value by that inverse scale, in registers.
struct KVDev {
    void *K = nullptr, *V = nullptr;
    float *kinv = nullptr, *vinv = nullptr;   // [cap] inverse scales, one per row
    int cap = 0;                    // rows held on the card
    bool ring = false;              // the HOST cache is a ring (banded layer)
    long long valid = 0;            // dense: leading rows already mirrored on the card
    int base = 0;                   // banded: the card holds the linear window [base, base+cap)
    int cov_lo = 0, cov_hi = 0;     // positions already mirrored (banded), to skip needless uploads
    int cbase = 0;                  // banded: the `base` the rows now on the card were written with (a row sits at p - cbase)
};
std::vector<KVDev> g_kv;

float *g_rope_c = nullptr, *g_rope_s = nullptr;
int g_rope_len = 0;

inline bool bad(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return false;
    g_bad = true;
    g_err = std::string(what) + ": " + cudaGetErrorString(e);
    return true;
}

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(((unsigned) v) << 16); }

// ---------------------------------------------------------------- math kernels
__device__ void bf16_body(const unsigned short* __restrict__ W, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
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

// ============================================================================
// ON-THE-FLY DEQUANTIZATION: the card's K/V rows are PACKED (T_K for K, T_V for V,
// chosen INDEPENDENTLY by -k / -v). The attention kernel reads the packed bytes and
// dequantizes them into registers (unpack_target_to_fp32 from gptoss/kv_quant.hpp),
// so the fp32 rows never exist in VRAM: the PCIe bus carries 1/4 (e4m3) or 1/2 (f16)
// the bytes. The nine combinations are instantiated by the dispatch in run_layer.
// ============================================================================

// One row of the card cache, packed: row = position, KD elements per row.
// The per-row inverse scale (one power of two per position, shared by its kv_heads)
// lives in kinv/vinv; the host cache and the KV checkpoint stay fp32.
__device__ __forceinline__ float kv_row_scale(float peak, float limit) {
    if (!(peak > 0.0f) || !(limit > 0.0f)) return 1.0f;
    const float s = exp2f(floorf(log2f(peak / limit)));
    if (!(s > 1e-18f)) return 1.0f;
    return (s > 1.0f) ? 1.0f : s;
}

__device__ __forceinline__ float kv_row_scale_inv(float scale) {
    return (scale > 0.0f) ? 1.0f / scale : 1.0f;
}

template<typename T_K, typename T_V>
__global__ void k_attn(
    const float* __restrict__ q, const float* __restrict__ stage_k, const float* __restrict__ stage_v,
    const T_K* __restrict__ cache_k, const T_V* __restrict__ cache_v,
    const float* __restrict__ kinv, const float* __restrict__ vinv,
    const float* __restrict__ sinks, float* __restrict__ att, int T, int pos0, int window, bool banded,
    int QD, int KD, int D, int heads, int rep, int base) {
    const int wid = blockIdx.x;
    const int t = wid / heads, h = wid % heads;
    if (t >= T) return;
    const int pos = pos0 + t;
    const int lo = banded ? max(0, pos - window + 1) : 0;
    const int g = h / rep;
    const float scale = 1.0f / sqrtf((float) D);
    const float* qq = q + (size_t) t * QD + (size_t) h * D;
    __shared__ float qv[MAXD];
    __shared__ float vb[ATTN_THREADS * MAXD];
    __shared__ float pw[ATTN_THREADS];
    for (int j = threadIdx.x; j < D; j += ATTN_THREADS) qv[j] = qq[j];
    __syncthreads();
    const int d0 = threadIdx.x * 2, d1 = d0 + 1;
    float o0 = 0.0f, o1 = 0.0f;
    float m = sinks[h];
    double z = 1.0;                       // the sink's own term, exp(sink - m) with m == sink at the start
    for (int u0 = lo; u0 <= pos; u0 += ATTN_THREADS) {
        const int u = u0 + threadIdx.x;
        const bool live = (u <= pos);
        float s = -1e30f;
        if (live) {
            s = 0.0f;
            if (u >= pos0) {   // this chunk's keys: still fp32 in the stage, never packed
                const float* kr = stage_k + (size_t) (u - pos0) * KD + (size_t) g * D;
                for (int j = 0; j < D; ++j) s += qv[j] * kr[j];
            } else {           // history: packed on the card, dequantized into registers
                const int crow = banded ? u - base : u;
                const T_K* kr = cache_k + (size_t) crow * KD + (size_t) g * D;
                const float isc = kinv[crow];
                for (int j = 0; j < D; ++j) s += qv[j] * gpu::unpack_target_to_fp32<T_K>(kr[j], isc);
            }
            s *= scale;
        }
        float mt = s;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
        const float mn = fmaxf(m, mt);
        const float resc = expf(m - mn);
        const float p = live ? expf(s - mn) : 0.0f;
        float tz = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tz += __shfl_xor_sync(0xffffffffu, tz, o);
        if (live) {
            if (u >= pos0) {
                const float* vr = stage_v + (size_t) (u - pos0) * KD + (size_t) g * D;
                for (int j = 0; j < D; ++j) vb[threadIdx.x * MAXD + j] = vr[j];
            } else {
                const int crow = banded ? u - base : u;
                const T_V* vr = cache_v + (size_t) crow * KD + (size_t) g * D;
                const float isc = vinv[crow];
                for (int j = 0; j < D; ++j) vb[threadIdx.x * MAXD + j] = gpu::unpack_target_to_fp32<T_V>(vr[j], isc);
            }
        }
        pw[threadIdx.x] = p;
        z = z * (double) resc + (double) tz;
        o0 *= resc;
        o1 *= resc;
        m = mn;
        __syncthreads();
        for (int i = 0; i < ATTN_THREADS; ++i) {
            const float pi = pw[i];
            if (pi > 0.0f) { o0 += pi * vb[i * MAXD + d0]; o1 += pi * vb[i * MAXD + d1]; }
        }
        __syncthreads();
    }
    const float zi = (float) (1.0 / z);
    float* out = att + (size_t) t * QD + (size_t) h * D;
    out[d0] = o0 * zi;
    out[d1] = o1 * zi;
}

// amax of a row across the block, then the power-of-two scale that keeps it inside the format.
__device__ __forceinline__ float block_amax(float v, float* __restrict__ red, int nt) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[w] = v;
    __syncthreads();
    const int nw = nt >> 5;
    float tot = 0.0f;
    for (int i = 0; i < nw; ++i) tot = fmaxf(tot, red[i]);
    __syncthreads();
    return tot;
}

// Pack a run of fp32 rows into the card's format, writing the row's inverse scale too.
template<typename T_OUT>
__global__ void k_kv_pack(const float* __restrict__ src, T_OUT* __restrict__ dst, float* __restrict__ iscl,
                          int rows, int kd, float limit) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* p = src + (size_t) r * kd;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) amax = fmaxf(amax, fabsf(p[i]));
    amax = block_amax(amax, red, (int) blockDim.x);
    const float sc = kv_row_scale(amax, limit);
    const float isc = kv_row_scale_inv(sc);
    if (threadIdx.x == 0) iscl[r] = isc;
    T_OUT* d = dst + (size_t) r * kd;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) d[i] = gpu::pack_fp32_to_target<T_OUT>(p[i], isc);
}

// Back to fp32: only the host cache / the checkpoint need it, the card never stores fp32 KV.
template<typename T_IN>
__global__ void k_kv_unpack(const T_IN* __restrict__ src, const float* __restrict__ iscl, float* __restrict__ dst,
                            int rows, int kd) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float isc = iscl[r];
    const T_IN* p = src + (size_t) r * kd;
    float* d = dst + (size_t) r * kd;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) d[i] = gpu::unpack_target_to_fp32<T_IN>(p[i], isc);
}

__global__ void k_fill_one(float* d, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = 1.0f;
}


// ---------------------------------------------------------------- tensor-core GEMM (sm_80+): Y[N][rows] = X[N][cols] . W[rows][cols]^T
// Grouped MXFP4 matmul over the row space: one block-column per expert, its rows reached through src[] when the
// input is in token space (gate_up reads the post-LN activations), addressed directly when it is not (down reads
// activations already produced in the row space). The expert's weights are streamed ONCE whatever the number of
// rows it got, which is the whole point of batching a chunk.
template <bool GU>
__global__ void k_moe_mx(const ExpertPtrs* pe, const int* rowoff, const int* cnt, const int* src,
                         const float* x, float* y, int rows, int cols, int E) {
    const int e = blockIdx.y;
    const int n_e = cnt[e];
    if (n_e <= 0) return;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int G = cols >> 5;
    const uint8_t* b = (GU ? (const uint8_t*) pe[e].gu_b : (const uint8_t*) pe[e].dn_b) + (size_t) row * G * 16;
    const uint8_t* s = (GU ? (const uint8_t*) pe[e].gu_s : (const uint8_t*) pe[e].dn_s) + (size_t) row * G;
    const int base = rowoff[e];
    for (int n0 = 0; n0 < n_e; n0 += TILE) {
        const int nn = min(TILE, n_e - n0);
        float acc[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;
        for (int g = lane; g < G; g += 32) {
            const float sc = ldexpf(1.0f, (int) s[g] - 127);
            const uint4 raw = *reinterpret_cast<const uint4*>(b + (size_t) g * 16);
            const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
            float wv[32];
#pragma unroll
            for (int k = 0; k < 16; ++k) { wv[2 * k] = c_fp4[q[k] & 15] * sc; wv[2 * k + 1] = c_fp4[q[k] >> 4] * sc; }
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                if (t < nn) {
                    const int r = base + n0 + t;
                    const float* xx = x + (size_t) (GU ? src[r] : r) * cols + (size_t) g * 32;
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
            if (lane == 0 && t < nn) y[(size_t) (base + n0 + t) * rows + row] = v;
        }
    }
}

__device__ __forceinline__ int expert_of_row(const int* rowoff, int E, int r) {
    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int m = (lo + hi + 1) >> 1;
        if (rowoff[m] <= r) lo = m; else hi = m - 1;
    }
    return lo;
}

__global__ void k_moe_act(const ExpertPtrs* pe, const int* rowoff, int E, int R, int I, int halves,
                          float limit, float alpha, const float* gu, float* act) {
    const size_t n = (size_t) R * I;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x) {
        const size_t r = idx / I;
        const int j = (int) (idx % I);
        const int e = expert_of_row(rowoff, E, (int) r);
        const unsigned short* gb = (const unsigned short*) pe[e].gu_bias;
        const float* g = gu + r * 2 * (size_t) I;
        const int gi = halves ? j : 2 * j, ui = halves ? I + j : 2 * j + 1;
        float gt = g[gi] + bf2f(gb[gi]);
        float up = g[ui] + bf2f(gb[ui]);
        gt = fminf(gt, limit);
        up = fminf(fmaxf(up, -limit), limit);
        act[idx] = (up + 1.0f) * (gt / (1.0f + expf(-alpha * gt)));
    }
}

__global__ void k_moe_fin(const ExpertPtrs* pe, const int* rowoff, int E, const int* src, const float* rw,
                          int H, int R, float* x, const float* eb) {
    const size_t n = (size_t) R * H;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x) {
        const size_t r = idx / H;
        const int j = (int) (idx % H);
        const int e = expert_of_row(rowoff, E, (int) r);
        const unsigned short* db = (const unsigned short*) pe[e].dn_bias;
        atomicAdd(&x[(size_t) src[r] * H + j], rw[r] * (eb[idx] + bf2f(db[j])));
    }
}

__global__ void k_rmsnorm(const float* x, const float* ln, float eps, int H, int T, float* y) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* r = x + (size_t) t * H;
    __shared__ double part[ATTN_THREADS];
    double ss = 0.0;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += (double) r[i] * r[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    const int nw = (int) (blockDim.x >> 5);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    double tot = 0.0;
    for (int w = 0; w < nw; ++w) tot += part[w];
    const float rr = 1.0f / sqrtf((float) (tot / H) + eps);
    float* o = y + (size_t) t * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) o[i] = (ln ? ln[i] : 1.0f) * (r[i] * rr);
}

__global__ void k_bias(float* q, float* k, float* v, const float* bq, const float* bk, const float* bv,
                       int T, int QD, int KD) {
    const size_t nq = (size_t) T * QD, nk = (size_t) T * KD;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < nq + 2 * nk; idx += (size_t) gridDim.x * blockDim.x) {
        if (idx < nq) q[idx] += bq[idx % QD];
        else if (idx < nq + nk) k[idx - nq] += bk[(idx - nq) % KD];
        else v[idx - nq - nk] += bv[(idx - nq - nk) % KD];
    }
}

__global__ void k_rope(float* q, float* k, const float* rc, const float* rs, int T, int pos0, int QD, int KD,
                       int D, int half, int heads, int kv_heads, bool pairs) {
    const size_t rq = (size_t) T * heads * half, rk = (size_t) T * kv_heads * half;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < rq + rk; idx += (size_t) gridDim.x * blockDim.x) {
        int t, hd, i;
        float* base;
        if (idx < rq) {
            const size_t m = idx % ((size_t) heads * half);
            t = (int) (idx / ((size_t) heads * half)); hd = (int) (m / half); i = (int) (m % half);
            base = q + (size_t) t * QD + (size_t) hd * D;
        } else {
            const size_t m0 = idx - rq;
            const size_t m = m0 % ((size_t) kv_heads * half);
            t = (int) (m0 / ((size_t) kv_heads * half)); hd = (int) (m / half); i = (int) (m % half);
            base = k + (size_t) t * KD + (size_t) hd * D;
        }
        const int off = (pos0 + t) * half;
        const float c = rc[off + i], s = rs[off + i];
        float* a = base + (pairs ? 2 * i : i), *b = base + (pairs ? 2 * i + 1 : i + half);
        const float xa = *a, xb = *b;
        *a = xa * c - xb * s;
        *b = pairs ? (xa * s + xb * c) : (xb * c + xa * s);
    }
}

__global__ void k_resid(float* x, const float* ob, const float* bo, int H, int T) {
    const size_t n = (size_t) T * H;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x)
        x[idx] += ob[idx] + bo[idx % H];
}

__global__ void k_router_bias(float* gl, const float* rbias, int E, int T) {
    const size_t n = (size_t) T * E;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x)
        gl[idx] += rbias[idx % E];
}


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
        if (tid < BM) {
            const int r = row0 + tid;
            __nv_bfloat16* w = Ws + tid * LDS;
            if (r < rows) {
                const unsigned short* b = (const unsigned short*) A + (size_t) r * cols + (size_t) g * BK;
#pragma unroll
                for (int k = 0; k < BK; ++k) w[k] = __ushort_as_bfloat16(b[k]);
            } else {
#pragma unroll
                for (int k = 0; k < BK; ++k) w[k] = __float2bfloat16(0.0f);
            }
        }
        for (int e = tid; e < BN * BK; e += blockDim.x) {
            const int tt = e / BK, k = e % BK, tok = tok0 + tt;
            Xs[tt * LDS + k] = __float2bfloat16(tok < N ? x[(size_t) tok * cols + (size_t) g * BK + k] : 0.0f);
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
        const int tt = e / BM, r = e % BM;
        if (tok0 + tt < N && row0 + r < rows) y[(size_t) (tok0 + tt) * rows + row0 + r] = Cs[tt * LDC + r];
    }
}

__global__ void k_gemm_bf16(const unsigned short* W, int rows, int cols, const float* x, int N, float* y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    gemm_body<false>(W, nullptr, rows, cols, x, N, y);
#endif
}
__global__ void k_bf16(const unsigned short* W, int rows, int cols, const float* x, int N, float* y) { bf16_body(W, rows, cols, x, N, y); }

// ---------------------------------------------------------------- host helpers
bool use_gemm(int N, int cols) { return N >= 32 && g_cc >= 80 && cols % 32 == 0; }

bool dense_mm(const void* dW, int rows, int cols, const float* dx, int N, float* dy) {
    if (N <= 0) return true;
    if (use_gemm(N, cols)) k_gemm_bf16<<<dim3((unsigned) ((rows + BM - 1) / BM), (unsigned) ((N + BN - 1) / BN)), 128, 0, g_st>>>((const unsigned short*) dW, rows, cols, dx, N, dy);
    else k_bf16<<<(unsigned) ((rows + WARPS - 1) / WARPS), WARPS * 32, 0, g_st>>>((const unsigned short*) dW, rows, cols, dx, N, dy);
    return !bad(cudaGetLastError(), "dense mm");
}

bool copy_async(void* d, const void* h, size_t n, cudaMemcpyKind kind) {
    if (n == 0) return true;
    if (bad(cudaMemcpyAsync(d, h, n, kind, g_st), "pdev copy")) return false;
    return !bad(cudaStreamSynchronize(g_st), "pdev copy sync");
}

bool malloc_(void** p, size_t n, const char* what) {
    *p = nullptr;
    if (n == 0) n = 1;
    if (bad(cudaMalloc(p, n), what)) return false;
    return true;
}

// Move the KV rows of positions [lo, lo+n) between the host cache and the card cache. The host cache of a banded
// layer is a ring (rows wrap at cap_host); the card cache is linear, so the host side is split into runs.
// Move the KV rows of positions [lo, lo+n) between the host cache (fp32) and the card cache (packed).
// The host side cannot be read by a kernel, so an upload goes through the fp32 staging: H2D of the raw
// floats, then k_kv_pack writes the packed bytes + the row's inverse scale on the card. A download
// (the emulated path only) runs k_kv_unpack back into the host cache.
static bool kv_pack_run(int layer, const float* kq, const float* vq, int dst_row, int len, int kd) {
    KVDev& kv = g_kv[(size_t) layer];
    const int blk = std::min(len, 65535);
    bool ok = true;
    switch (g_kfmt) {
        case 0: k_kv_pack<__nv_fp8_e4m3><<<blk, 256, 0, g_st>>>(kq, (__nv_fp8_e4m3*) kv.K + (size_t) dst_row * kd, kv.kinv + dst_row, len, kd, g_klim); break;
        case 1: k_kv_pack<half><<<blk, 256, 0, g_st>>>(kq, (half*) kv.K + (size_t) dst_row * kd, kv.kinv + dst_row, len, kd, g_klim); break;
        default: k_kv_pack<float><<<blk, 256, 0, g_st>>>(kq, (float*) kv.K + (size_t) dst_row * kd, kv.kinv + dst_row, len, kd, 0.0f); break;
    }
    ok = ok && !bad(cudaGetLastError(), "kv pack k");
    switch (g_vfmt) {
        case 0: k_kv_pack<__nv_fp8_e4m3><<<blk, 256, 0, g_st>>>(vq, (__nv_fp8_e4m3*) kv.V + (size_t) dst_row * kd, kv.vinv + dst_row, len, kd, g_vlim); break;
        case 1: k_kv_pack<half><<<blk, 256, 0, g_st>>>(vq, (half*) kv.V + (size_t) dst_row * kd, kv.vinv + dst_row, len, kd, g_vlim); break;
        default: k_kv_pack<float><<<blk, 256, 0, g_st>>>(vq, (float*) kv.V + (size_t) dst_row * kd, kv.vinv + dst_row, len, kd, 0.0f); break;
    }
    return ok && !bad(cudaGetLastError(), "kv pack v");
}

bool move_kv(int layer, const float* host_K, const float* host_V, int cap_host, bool ring, int lo, int n, cudaMemcpyKind kind) {
    if (g_bad || n <= 0) return !g_bad;
    KVDev& kv = g_kv[(size_t) layer];
    const int kd = g_kd;
    if (kind == cudaMemcpyHostToDevice) {
        for (int p = lo; p < lo + n;) {
            int a = ring ? p % cap_host : p;
            int len = lo + n - p;
            if (ring) len = std::min(len, cap_host - a);
            // the staging holds at most g_upmax rows: split longer runs
            while (len > 0) {
                const int take = std::min(len, g_upmax);
                if (!copy_async(g_KQ, host_K + (size_t) a * kd * 4, (size_t) take * kd * 4, cudaMemcpyHostToDevice)) return false;
                if (!copy_async(g_VQ, host_V + (size_t) a * kd * 4, (size_t) take * kd * 4, cudaMemcpyHostToDevice)) return false;
                if (!kv_pack_run(layer, g_KQ, g_VQ, p - kv.base, take, kd)) return false;
                a += take; len -= take; p += take;
            }
        }
        return !bad(cudaStreamSynchronize(g_st), "kv sync");
    }
    // download: unpack on the card into the staging, then a plain D2H into the host cache
    for (int p = lo; p < lo + n;) {
        int a = ring ? p % cap_host : p;
        int len = lo + n - p;
        if (ring) len = std::min(len, cap_host - a);
        while (len > 0) {
            const int take = std::min(len, g_upmax);
            switch (g_kfmt) {
                case 0: k_kv_unpack<__nv_fp8_e4m3><<<std::min(take, 65535), 256, 0, g_st>>>((__nv_fp8_e4m3*) kv.K + (size_t) (p - kv.base) * kd, kv.kinv + (p - kv.base), g_KQ, take, kd); break;
                case 1: k_kv_unpack<half><<<std::min(take, 65535), 256, 0, g_st>>>((half*) kv.K + (size_t) (p - kv.base) * kd, kv.kinv + (p - kv.base), g_KQ, take, kd); break;
                default: break;
            }
            if (g_kfmt != 2 && bad(cudaGetLastError(), "kv unpack k")) return false;
            switch (g_vfmt) {
                case 0: k_kv_unpack<__nv_fp8_e4m3><<<std::min(take, 65535), 256, 0, g_st>>>((__nv_fp8_e4m3*) kv.V + (size_t) (p - kv.base) * kd, kv.vinv + (p - kv.base), g_VQ, take, kd); break;
                case 1: k_kv_unpack<half><<<std::min(take, 65535), 256, 0, g_st>>>((half*) kv.V + (size_t) (p - kv.base) * kd, kv.vinv + (p - kv.base), g_VQ, take, kd); break;
                default: break;
            }
            if (g_vfmt != 2 && bad(cudaGetLastError(), "kv unpack v")) return false;
            const size_t bytes = (size_t) take * kd * 4;
            float* hk = (float*) (void*) host_K, *hv = (float*) (void*) host_V;
            if (!bad(cudaMemcpyAsync(hk + (size_t) a * kd * 4, g_KQ, bytes, kind, g_st), "kv d2h")) return false;
            if (!bad(cudaMemcpyAsync(hv + (size_t) a * kd * 4, g_VQ, bytes, kind, g_st), "kv d2h v")) return false;
            a += take; len -= take; p += take;
        }
    }
    return !bad(cudaStreamSynchronize(g_st), "kv sync");
}
void free_all() {
#define FR(p) if ((p)) { cudaFree((void*) (p)); (p) = nullptr; }
    FR(g_X)
    FR(g_HN)
    FR(g_Q)
    FR(g_KS)
    FR(g_VS)
    FR(g_ATT)
    FR(g_OB)
    FR(g_GL)
    FR(g_GU)
    FR(g_ACT)
    FR(g_EB)
    FR(g_SRC)
    FR(g_CNT)
    FR(g_RO)
    FR(g_RW)
    FR(g_PE)
    FR(g_rope_c)
    FR(g_rope_s)
    FR(g_KQ)
    FR(g_VQ)
#undef FR
    for (size_t i = 0; i < g_L.size(); ++i) {
        LDev& d = g_L[i];
#define FR2(p) if ((p)) { cudaFree((void*) (p)); (p) = nullptr; }
        FR2(d.in_ln)
        FR2(d.post_ln)
        FR2(d.sinks)
        FR2(d.bq)
        FR2(d.bk)
        FR2(d.bv)
        FR2(d.bo)
        FR2(d.rbias)
#undef FR2
    }
    for (size_t i = 0; i < g_kv.size(); ++i) {
#define FR3(p) if ((p)) { cudaFree((void*) (p)); (p) = nullptr; }
        FR3(g_kv[i].K)
        FR3(g_kv[i].V)
        FR3(g_kv[i].kinv)
        FR3(g_kv[i].vinv)
#undef FR3
    }
    g_L.clear();
    g_kv.clear();
    g_hro.clear();
}

// dense_rows = how many positions of a dense layer's KV the card mirrors: the whole ctx when it fits, otherwise
// the longest prefix that fits, and run_layer refuses a chunk that would read past it (the Model then finishes
// the prompt with its old path from the same host cache).
size_t plan_bytes(const Cfg& c, const char* layer_sliding, int chunk, int dense_rows) {
    const int H = c.H, I = c.I, E = c.E, topk = c.topk, hd = c.head_dim;
    const int qd = c.heads * hd, kd = c.kv_heads * hd;
    const int T = std::max(1, chunk), R = T * topk;
    const int kb = (c.k_fmt == 0) ? 1 : (c.k_fmt == 1) ? 2 : 4;
    const int vb = (c.v_fmt == 0) ? 1 : (c.v_fmt == 1) ? 2 : 4;
    const int up = std::max(T, std::max(1, c.window));   // rows the H2D staging must hold
    size_t s = 0;
    s += (size_t) T * (3 * H + 2 * qd + 2 * kd + E) * 4;
    s += (size_t) R * (2 * I + I + H) * 4;
    s += (size_t) R * 8 + (size_t) (E + 1) * 4 + (size_t) R * 4 + (size_t) R * 4 + (size_t) E * sizeof(ExpertPtrs);
    s += (size_t) c.layers * (2 * H + qd + 2 * kd + H + E) * 4;
    for (int l = 0; l < c.layers; ++l) {
        const bool sliding = layer_sliding[l] != 0;
        const size_t rows = sliding ? (size_t) (c.window + T) : (size_t) dense_rows;
        // the whole point of the task: the card holds PACKED rows, never fp32
        s += (size_t) rows * (kd * (kb + vb) + 2 * 4);
    }
    s += (size_t) 2 * up * kd * 4;   // the fp32 staging for the uploads
    s += (size_t) 2 * (size_t) c.ctx * (hd / 2) * 4;
    return s + ((size_t) 16 << 20);
}

size_t g_reserved = 0;

}  // namespace

// ---------------------------------------------------------------- the API the Model calls
bool available() { return true; }

int chosen_chunk() { return g_chunk; }

size_t reserved_bytes() { return g_reserved; }

bool init(const Cfg& cfg, const char* layer_sliding, size_t budget_bytes) {
    if (g_bad) return false;
    if (cfg.head_dim != MAXD || cfg.layers < 1 || cfg.chunk_max < 1) { g_err = "prefill_dev: unsupported geometry"; g_bad = true; return false; }
    const int H = cfg.H, I = cfg.I, E = cfg.E, topk = cfg.topk, layers = cfg.layers, ctx = cfg.ctx;
    const int qd = cfg.heads * cfg.head_dim, kd = cfg.kv_heads * cfg.head_dim;
    int T = std::max(1, cfg.chunk_max);
    // A dense layer's attention reads the WHOLE history, so its card cache must hold ctx rows: dense_rows is never
    // shrunk below cfg.ctx. run_layer() refuses a chunk whose history is past the card cache, and that refusal is
    // NOT an allocation that did not fit - if init accepted a small card cache the path would fail mid-prefill and
    // the session would lose the device. Better to refuse the whole path here, where the caller falls back to the
    // old prefill and the device stays on.
    int dense_rows = cfg.ctx;
    for (;;) {
        if (plan_bytes(cfg, layer_sliding, T, dense_rows) <= budget_bytes) break;
        if (T == 1) { g_err = "prefill_dev: the reservation does not fit the VRAM left"; g_bad = true; return false; }
        T /= 2;
    }
    g_reserved = plan_bytes(cfg, layer_sliding, T, dense_rows);
    g_cfg = cfg;
    g_cfg.chunk_max = T;
    g_chunk = T;
    g_rmax = T * topk;
    g_qd = qd; g_kd = kd; g_half = cfg.head_dim / 2; g_layers = layers;
    // Asymmetric KV formats: K and V are packed INDEPENDENTLY (-k / -v).
    g_kfmt = (cfg.k_fmt >= 0 && cfg.k_fmt <= 2) ? cfg.k_fmt : 2;
    g_vfmt = (cfg.v_fmt >= 0 && cfg.v_fmt <= 2) ? cfg.v_fmt : 2;
    g_kb = (g_kfmt == 0) ? 1 : (g_kfmt == 1) ? 2 : 4;
    g_vb = (g_vfmt == 0) ? 1 : (g_vfmt == 1) ? 2 : 4;
    g_klim = (g_kfmt == 0) ? cfg.kv_limit : 0.0f;
    g_vlim = (g_vfmt == 0) ? cfg.kv_limit : 0.0f;
    g_upmax = std::max(T, std::max(1, cfg.window));   // rows the H2D staging holds
    const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    if (bad(cudaMemcpyToSymbol(c_fp4, lut, sizeof lut), "fp4 table")) return false;
    if (bad(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking), "pdev stream")) return false;
    int maj = 0, min_ = 0;
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&min_, cudaDevAttrComputeCapabilityMinor, 0);
    g_cc = maj * 10 + min_;
#define A(p, n) if (!malloc_((void**) & (p), (size_t) (n), "pdev alloc")) { (p) = nullptr; return false; }
    A(g_X, (size_t) T * H * 4)
    A(g_HN, (size_t) T * H * 4)
    A(g_Q, (size_t) T * qd * 4)
    A(g_KS, (size_t) T * kd * 4)
    A(g_VS, (size_t) T * kd * 4)
    A(g_ATT, (size_t) T * qd * 4)
    A(g_OB, (size_t) T * H * 4)
    A(g_GL, (size_t) T * E * 4)
    A(g_GU, (size_t) g_rmax * 2 * I * 4)
    A(g_ACT, (size_t) g_rmax * I * 4)
    A(g_EB, (size_t) g_rmax * H * 4)
    A(g_SRC, (size_t) g_rmax * 4)
    A(g_RW, (size_t) g_rmax * 4)
    A(g_CNT, (size_t) E * 4)
    A(g_RO, (size_t) (E + 1) * 4)
    A(g_PE, (size_t) E * sizeof(ExpertPtrs))
    A(g_rope_c, (size_t) ctx * g_half * 4)
    A(g_rope_s, (size_t) ctx * g_half * 4)
    A(g_KQ, (size_t) g_upmax * kd * 4)
    A(g_VQ, (size_t) g_upmax * kd * 4)
#undef A
    g_hro.assign((size_t) E + 1, 0);
    g_L.assign((size_t) layers, LDev());
    g_kv.assign((size_t) layers, KVDev());
    for (int l = 0; l < layers; ++l) {
        KVDev& kv = g_kv[(size_t) l];
        const bool sliding = layer_sliding[l] != 0;
        kv.ring = sliding;
        kv.cap = sliding ? std::max(cfg.window + T, 1) : std::max(dense_rows, 1);
        kv.valid = 0;
        kv.base = 0;
        kv.cov_lo = 0;
        kv.cov_hi = 0;
        if (!malloc_((void**) & kv.K, (size_t) kv.cap * kd * g_kb, "pdev kv alloc")) return false;
        if (!malloc_((void**) & kv.V, (size_t) kv.cap * kd * g_vb, "pdev kv alloc")) return false;
        if (!malloc_((void**) & kv.kinv, (size_t) kv.cap * 4, "pdev kv scale")) return false;
        if (!malloc_((void**) & kv.vinv, (size_t) kv.cap * 4, "pdev kv scale")) return false;
        // rows never uploaded must still read a defined scale: 1.0 = no rescaling
        k_fill_one<<<(unsigned) std::min<size_t>(kv.cap, 65535), 256, 0, g_st>>>(kv.kinv, kv.cap);
        k_fill_one<<<(unsigned) std::min<size_t>(kv.cap, 65535), 256, 0, g_st>>>(kv.vinv, kv.cap);
        if (bad(cudaGetLastError(), "kv scale init")) return false;
    }
    return true;
}

void shutdown() {
    free_all();
    if (g_st) { cudaStreamDestroy(g_st); g_st = nullptr; }
    g_bad = false;
    g_err.clear();
}

bool upload_layer(int layer, bool sliding, const float* in_ln, const float* post_ln, const float* sinks,
                  const float* bq, const float* bk, const float* bv, const float* bo, const float* router_bias,
                  const void* dq, const void* dk, const void* dv, const void* dobj, const void* drt) {
    if (g_bad) return false;
    if (!dq || !dk || !dv || !dobj || !drt) return false;      // the layer's projections must already be on the card
    const int H = g_cfg.H, E = g_cfg.E, hd = g_cfg.head_dim;
    const int qd = g_cfg.heads * hd, kd = g_cfg.kv_heads * hd;
    LDev& d = g_L[(size_t) layer];
    d.dq = dq; d.dk = dk; d.dv = dv; d.dobj = dobj; d.drt = drt;
#define U(p, host, n) if (!malloc_((void**) & (p), (size_t) (n), "pdev alloc")) { (p) = nullptr; return false; } if (!copy_async((p), (host), (size_t) (n), cudaMemcpyHostToDevice)) return false;
    U(d.in_ln, in_ln, (size_t) H * 4)
    U(d.post_ln, post_ln, (size_t) H * 4)
    U(d.sinks, sinks, (size_t) qd * 4)
    U(d.bq, bq, (size_t) qd * 4)
    U(d.bk, bk, (size_t) kd * 4)
    U(d.bv, bv, (size_t) kd * 4)
    U(d.bo, bo, (size_t) H * 4)
    U(d.rbias, router_bias, (size_t) E * 4)
#undef U
    (void) sliding;
    return true;
}

void set_rope(const float* cos_, const float* sin_, int seqlen) {
    if (g_bad) return;
    const size_t n = (size_t) seqlen * g_half * 4;
    if (!copy_async(g_rope_c, cos_, n, cudaMemcpyHostToDevice) || !copy_async(g_rope_s, sin_, n, cudaMemcpyHostToDevice)) return;
    g_rope_len = seqlen;
}

bool x_upload(const float* hx, int T) { return !g_bad && copy_async(g_X, hx, (size_t) T * g_cfg.H * 4, cudaMemcpyHostToDevice); }
bool x_download(float* hx, int T) { return !g_bad && copy_async(hx, g_X, (size_t) T * g_cfg.H * 4, cudaMemcpyDeviceToHost); }

bool kv_prefetch(int layer, const float* host_K, const float* host_V, int cap, bool sliding, int pos0) {
    if (g_bad) return false;
    KVDev& kv = g_kv[(size_t) layer];
    if (sliding) {
        const int lo = std::max(0, pos0 - g_cfg.window + 1);
        int base = std::max(0, pos0 - g_cfg.window);
        if (base + kv.cap > pos0 + g_chunk) base = pos0 + g_chunk - kv.cap;
        if (base > lo) base = lo;
        if (base < 0) base = 0;
        kv.base = base;
        // The card rows of a banded layer sit at (p - base). When the window moved (base changed since the rows were written)
        // every mirrored row is at the wrong index: forget the coverage so the window is rebuilt from the host cache below.
        if (base != kv.cbase) { kv.cbase = base; kv.cov_lo = kv.cov_hi = base; }
        const int n = pos0 - base;
        if (n <= 0) return true;
        if (base >= kv.cov_lo && pos0 <= kv.cov_hi) return true;   // already mirrored
        if (!move_kv(layer, host_K, host_V, cap, true, base, n, cudaMemcpyHostToDevice)) return false;
        kv.cov_lo = std::min(kv.cov_lo, base);
        kv.cov_hi = std::max(kv.cov_hi, pos0);
        return true;
    }
    if (kv.valid > pos0) kv.valid = pos0;   // a rewind: rows from pos0 on are about to be rewritten, none of them is trusted any more
    if ((long long) pos0 > kv.valid) {
        if (!move_kv(layer, host_K, host_V, cap, false, (int) kv.valid, pos0 - (int) kv.valid, cudaMemcpyHostToDevice)) return false;
        kv.valid = pos0;
    }
    return true;
}

// The host cache was written by the decode (or replaced by a checkpoint) from position `pos` on: whatever the card mirrors
// at or past it is stale. Dense layers lower their watermark; banded layers shorten the covered range, so the next
// prefetch rebuilds the window from the host cache.
void kv_trim(int pos) {
    if (pos < 0) pos = 0;
    for (KVDev& kv : g_kv) {
        if (kv.valid > pos) kv.valid = pos;
        if (kv.cov_hi > pos) kv.cov_hi = pos;
        if (kv.cov_lo > kv.cov_hi) kv.cov_lo = kv.cov_hi;
    }
}

bool run_layer(const Cfg& cfg, int layer, const LayerIO& w, int T, int pos0, bool sliding, float* router_logits) {
    (void) w;
    if (g_bad) return false;
    const int H = cfg.H, E = cfg.E, hd = cfg.head_dim;
    const int qd = g_qd, kd = g_kd, hdim2 = g_half, D = hd;
    const int heads = cfg.heads, kvh = cfg.kv_heads, rep = heads / kvh;
    const LDev& d = g_L[(size_t) layer];
    if (!d.in_ln) { g_err = "layer " + std::to_string(layer) + " was never uploaded (its dense weights are not on the card)"; return false; }
    g_cur_pos0 = pos0;
    g_cur_T = T;
    k_rmsnorm<<<(unsigned) T, 256, 0, g_st>>>(g_X, d.in_ln, cfg.eps, H, T, g_HN);
    if (bad(cudaGetLastError(), "pre_ln")) return false;
    if (!dense_mm(d.dq, qd, H, g_HN, T, g_Q)) return false;
    if (!dense_mm(d.dk, kd, H, g_HN, T, g_KS)) return false;
    if (!dense_mm(d.dv, kd, H, g_HN, T, g_VS)) return false;
    k_bias<<<(unsigned) std::min<size_t>(((size_t) T * (qd + 2 * (size_t) kd) + 255) / 256, 65535), 256, 0, g_st>>>(
        g_Q, g_KS, g_VS, d.bq, d.bk, d.bv, T, qd, kd);
    if (bad(cudaGetLastError(), "bias")) return false;
    k_rope<<<(unsigned) std::min<size_t>(((size_t) T * (qd + kd) / 2 + 255) / 256, 65535), 256, 0, g_st>>>(
        g_Q, g_KS, g_rope_c, g_rope_s, T, pos0, qd, kd, D, hdim2, heads, kvh, cfg.rope_pairs);
    if (bad(cudaGetLastError(), "rope")) return false;
    KVDev& kv = g_kv[(size_t) layer];
    // a dense layer reads the whole history: past what the card holds, this chunk is not device-resident
    if (!sliding && pos0 + T > kv.cap) { g_err = "prefill_dev: dense history past the card cache"; return false; }   // cannot happen: init keeps dense_rows == ctx
    // The nine (K,V) format combinations are instantiated by this dispatch: each kernel
    // body exists exactly once, and the history rows are read PACKED and dequantized in registers.
#define ATTN(KT, VT) k_attn<KT, VT><<<(unsigned) (size_t) T * heads, ATTN_THREADS, 0, g_st>>>( \
    g_Q, g_KS, g_VS, (const KT*) kv.K, (const VT*) kv.V, kv.kinv, kv.vinv, d.sinks, g_ATT, \
    T, pos0, cfg.window, sliding, qd, kd, D, heads, rep, kv.base)
    if (g_kfmt == 0 && g_vfmt == 0) ATTN(__nv_fp8_e4m3, __nv_fp8_e4m3);
    else if (g_kfmt == 0 && g_vfmt == 1) ATTN(__nv_fp8_e4m3, half);
    else if (g_kfmt == 1 && g_vfmt == 0) ATTN(half, __nv_fp8_e4m3);
    else if (g_kfmt == 1 && g_vfmt == 1) ATTN(half, half);
    else if (g_kfmt == 0) ATTN(__nv_fp8_e4m3, float);
    else if (g_kfmt == 1) ATTN(half, float);
    else if (g_vfmt == 0) ATTN(float, __nv_fp8_e4m3);
    else if (g_vfmt == 1) ATTN(float, half);
    else ATTN(float, float);
#undef ATTN
    if (bad(cudaGetLastError(), "attn")) return false;
    if (!dense_mm(d.dobj, H, qd, g_ATT, T, g_OB)) return false;
    k_resid<<<(unsigned) std::min<size_t>(((size_t) T * H + 255) / 256, 65535), 256, 0, g_st>>>(g_X, g_OB, d.bo, H, T);
    if (bad(cudaGetLastError(), "resid")) return false;
    k_rmsnorm<<<(unsigned) T, 256, 0, g_st>>>(g_X, d.post_ln, cfg.eps, H, T, g_HN);
    if (bad(cudaGetLastError(), "post_ln")) return false;
    if (!dense_mm(d.drt, E, H, g_HN, T, g_GL)) return false;
    k_router_bias<<<(unsigned) std::min<size_t>(((size_t) T * E + 255) / 256, 65535), 256, 0, g_st>>>(g_GL, d.rbias, E, T);
    if (bad(cudaGetLastError(), "router_bias")) return false;
    return copy_async(router_logits, g_GL, (size_t) T * E * 4, cudaMemcpyDeviceToHost);
}

bool run_experts(const Cfg& cfg, int T, const ExpertPtrs* per_expert, const int* cnt, const int* src_row,
                 const float* row_weight, const int32_t* idx_tok) {
    (void) T;
    (void) idx_tok;
    if (g_bad) return false;
    const int E = cfg.E, I = cfg.I, H = cfg.H;
    int R = 0;
    for (int e = 0; e < E; ++e) R += cnt[e];
    if (R <= 0) return true;
    if (R > g_rmax) { g_err = "prefill_dev: more rows than the reservation"; g_bad = true; return false; }
    g_hro[0] = 0;
    for (int e = 0; e < E; ++e) g_hro[e + 1] = g_hro[e] + cnt[e];
    if (!copy_async(g_PE, per_expert, (size_t) E * sizeof(ExpertPtrs), cudaMemcpyHostToDevice)) return false;
    if (!copy_async(g_CNT, cnt, (size_t) E * sizeof(int), cudaMemcpyHostToDevice)) return false;
    if (!copy_async(g_RO, g_hro.data(), (size_t) (E + 1) * sizeof(int), cudaMemcpyHostToDevice)) return false;
    if (!copy_async(g_SRC, src_row, (size_t) R * sizeof(int), cudaMemcpyHostToDevice)) return false;
    if (!copy_async(g_RW, row_weight, (size_t) R * sizeof(float), cudaMemcpyHostToDevice)) return false;
    const dim3 gu((unsigned) ((2 * I + WARPS - 1) / WARPS), (unsigned) E);
    k_moe_mx<true><<<gu, WARPS * 32, 0, g_st>>>((const ExpertPtrs*) g_PE, g_RO, g_CNT, g_SRC, g_HN, g_GU, 2 * I, H, E);
    if (bad(cudaGetLastError(), "moe gate_up")) return false;
    k_moe_act<<<(unsigned) std::min<size_t>(((size_t) R * I + 255) / 256, 65535), 256, 0, g_st>>>(
        (const ExpertPtrs*) g_PE, g_RO, E, R, I, cfg.glu_halves ? 1 : 0, cfg.limit, cfg.alpha, g_GU, g_ACT);
    if (bad(cudaGetLastError(), "moe act")) return false;
    const dim3 dn((unsigned) ((H + WARPS - 1) / WARPS), (unsigned) E);
    k_moe_mx<false><<<dn, WARPS * 32, 0, g_st>>>((const ExpertPtrs*) g_PE, g_RO, g_CNT, g_SRC, g_ACT, g_EB, H, I, E);
    if (bad(cudaGetLastError(), "moe down")) return false;
    k_moe_fin<<<(unsigned) std::min<size_t>(((size_t) R * H + 255) / 256, 65535), 256, 0, g_st>>>(
        (const ExpertPtrs*) g_PE, g_RO, E, g_SRC, g_RW, H, R, g_X, g_EB);
    return !bad(cudaGetLastError(), "moe fin");
}

void kv_store_back(int layer, float* host_K, float* host_V, int cap, bool sliding, int T, int pos0) {
    if (g_bad) return;
    // This layer's attention has read the stage, so the chunk's rows can be committed now: to the host cache
    // (the decode and the KV checkpoint read it) and to the card cache (the next chunk reads it from there).
    // The commit is per layer because the stage is reused by the next layer.
    const int kd = g_kd;
    for (int p = pos0; p < pos0 + T;) {
        int a = sliding ? p % cap : p;
        int len = pos0 + T - p;
        if (sliding) len = std::min(len, cap - a);
        const size_t bytes = (size_t) len * kd * 4;
        const bool ok = !bad(cudaMemcpyAsync(host_K + (size_t) a * kd * 4, g_KS + (size_t) (p - pos0) * kd * 4, bytes, cudaMemcpyDeviceToHost, g_st), "kv stage d2h") &&
                        !bad(cudaMemcpyAsync(host_V + (size_t) a * kd * 4, g_VS + (size_t) (p - pos0) * kd * 4, bytes, cudaMemcpyDeviceToHost, g_st), "kv stage d2h v");
        if (!ok) { g_bad = true; return; }
        p += len;
    }
    KVDev& kv = g_kv[(size_t) layer];
    // the stage is already on the card: pack it straight into the card's format (no fp32 rows in VRAM)
    if (!kv_pack_run(layer, g_KS, g_VS, pos0 - kv.base, T, kd)) { g_bad = true; return; }
    if (bad(cudaStreamSynchronize(g_st), "kv commit sync")) { g_bad = true; return; }
    if (!kv.ring) kv.valid = std::max<long long>(kv.valid, (long long) pos0 + T);
    else { kv.cov_lo = std::min(kv.cov_lo, kv.base); kv.cov_hi = std::max(kv.cov_hi, pos0 + T); }
}

void advance(int T) {
    // nothing to move: each layer's rows were committed by kv_store_back right after that layer read them
    (void) T;
}

bool sync_all() {
    if (g_bad) return false;
    return !bad(cudaStreamSynchronize(g_st), "pdev sync");
}

std::string last_error() { return g_err; }

}  // namespace pdev
}  // namespace gptoss

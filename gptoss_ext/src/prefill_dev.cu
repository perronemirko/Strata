// gptoss/prefill_dev.cu - device-resident prefill (the contract is include/gptoss/prefill_dev.hpp).
//
// One chunk of T tokens stays on the card for the whole layer loop: X is uploaded once at the chunk start and read
// back once at the end. Per layer, in order: rmsnorm -> q/k/v GEMM (bf16 tensor cores) -> bias + RoPE + K/V ring write
// -> attention against the device KV window (fp32 rings, sinks, sliding window; bf16 WMMA tiles with an online softmax)
// -> o GEMM -> residual add + post-norm -> router GEMM -> router logits down to the host (the layer's only sync). The
// routed experts then run in grouped passes over the touched experts: one launch per projection reads each expert's
// MXFP4 blob where it already is, decodes 32-value groups to shared bf16 and multiplies with bf16 WMMA; swiglu (bias
// included) and the down projection follow, and a combine kernel adds the weighted rows into X in k order, exactly as
// the host path does. The new K/V rows go back to the host cache on the copy stream at the chunk end.
//
// C++14. sm_80+ (the build targets sm_89). Every failure returns false and leaves the device usable: Model falls back
// to the old prefill for the session.

#include "gptoss/prefill_dev.hpp"

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
namespace pdev {

// (no anonymous namespace: the public API below shares this TU's state; the stub links into a different binary, so no ODR clash)

std::string g_err;
bool g_oom = false;
std::mutex g_err_mu;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_oom = (e == cudaErrorMemoryAllocation);
    g_err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(((unsigned) v) << 16); }
// MXFP4 value table (src/weights.cpp contract): the same values gpu_cuda.cu loads into its own file-local copy.
__constant__ float c_fp4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// ---------------------------------------------------------------- state
Cfg g_c;
bool g_on = false;
cudaStream_t g_comp = nullptr, g_copy = nullptr;
cudaEvent_t g_ev_prefetch = nullptr, g_ev_store = nullptr, g_ev_sync = nullptr;

struct LayerDev {
    bool sliding = false;
    int cap_dev = 0;                       // ring positions on the device: ctx (dense) or window + chunk_max (banded: a chunk must not overwrite its own past)
    float *K = nullptr, *V = nullptr;      // [cap_dev][kv_dim] fp32
    float* small = nullptr;                // in_ln[H], post_ln[H], sinks[heads], bq[QD], bk[KD], bv[KD], bo[H], rbias[E]
};
std::vector<LayerDev> g_L;

float *g_dX = nullptr, *g_dHn = nullptr, *g_dQb = nullptr, *g_dKb = nullptr, *g_dVb = nullptr, *g_dAb = nullptr, *g_dOb = nullptr, *g_dGl = nullptr;
float* g_hGl = nullptr;                    // pinned [chunk_max][E]: the router logits come back here (run_layer's sync point)

// expert scratch: TK = chunk_max * topk routed rows at most
float* g_dGU = nullptr;                    // [TK][2I] fp32   gate_up output (bias is added in swiglu, not here)
__nv_bfloat16* g_dACT = nullptr;           // [TK][I] bf16    swiglu output: the down projection's A operand
float* g_dYslot = nullptr;                 // [T*K][H] fp32   one row per (token,k) pair: wt * (down + bias); no atomics

// routing tables, uploaded per layer through ONE pinned staging block
struct TDev {                              // one entry per expert index (untouched experts stay zeroed)
    const uint8_t* gu_b; const uint8_t* dn_b; const uint8_t* gu_s; const uint8_t* dn_s;
    const unsigned short* gu_bias; const unsigned short* dn_bias;
};
TDev* g_dTd = nullptr;                     // [E]
int* g_dRowoff = nullptr;                  // [E+1] grouped-row prefix over the FULL expert index (row r belongs to e with rowoff[e] <= r < rowoff[e+1])
int* g_dGio = nullptr, *g_dDio = nullptr;  // [E+1] work-item prefixes for gate_up / down
int* g_dSrc = nullptr;                     // [R_tot] token of grouped row r
float* g_dWt = nullptr;                    // [R_tot] router weight of the (token,k) pair of row r
int* g_dRowof = nullptr;                   // [T*K] global row of token t's k-th expert (the combine sums in k order)
int* g_dRowexp = nullptr;                  // [R_tot] expert index of grouped row r (swiglu bias lookup)
void* g_hStage = nullptr;                  // pinned staging for all the tables above

float *g_dCos = nullptr, *g_dSin = nullptr;   // rope table: cos [len][half] then sin [len][half], one allocation
int g_rope_len = 0;

int g_fill = 0;                           // device KV positions written so far (all layers share the position space)

// ---------------------------------------------------------------- small kernels

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float eps, int H, int T, float* __restrict__ y) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* xr = x + (size_t) t * H;
    float* yr = y + (size_t) t * H;
    extern __shared__ float red[];
    float p = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { const float v = xr[i]; p += v * v; }
    // block reduce: warp shuffles, one value per warp in red[], then a final warp
    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = p;
    __syncthreads();
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = rsqrtf(v / H + eps);
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = xr[i] * s * w[i];
}

// Y[T][rows] = X[T][cols] . W^T, W BF16 [rows][cols] on the device. 64x64 tile, bf16 WMMA (same body as gpu_cuda.cu).
constexpr int BM = 64, BN = 64, BK = 32, LDS = 40, LDC = BM + 4;

__device__ void gemm_body(const unsigned short* __restrict__ W, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
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
                const unsigned short* b = W + (size_t) r * cols + (size_t) g * BK;
#pragma unroll
                for (int k = 0; k < BK; ++k) w[k] = __ushort_as_bfloat16(b[k]);
            } else {
#pragma unroll
                for (int k = 0; k < BK; ++k) w[k] = __float2bfloat16(0.0f);
            }
        }
        for (int e = tid; e < BN * BK; e += blockDim.x) {
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

__global__ void __launch_bounds__(128) k_gemm(const unsigned short* W, int rows, int cols, const float* x, int N, float* y) { gemm_body(W, rows, cols, x, N, y); }

// bias on q/k/v in place (fp32): one element per thread over the three tensors concatenated.
__global__ void k_bias(float* __restrict__ q, const float* __restrict__ bq, int QD, float* __restrict__ k, const float* __restrict__ bk,
                       int KD, float* __restrict__ v, const float* __restrict__ bv, int T) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nq = (size_t) T * QD, nk = (size_t) T * KD;
    if (idx < nq) q[idx] += bq[idx % QD];
    else if (idx < nq + nk) { const size_t j = idx - nq; k[j] += bk[j % KD]; }
    else if (idx < nq + 2 * nk) { const size_t j = idx - nq - nk; v[j] += bv[j % KD]; }
}

// RoPE per head, in place: one warp per (token, q-head or k-head); lane i < D/2 owns the pair (i, i+D/2) of the
// half-split layout or (2i, 2i+1) of the pairs layout - exactly ops.cpp's rotary(). K heads also write their row to
// the ring slot pos % cap_dev. V rows are copied by k_vwrite. The rope table is [len][half] cos then [len][half] sin.
__global__ void __launch_bounds__(32) k_rope_kv(float* __restrict__ q, float* __restrict__ k,   // in place: RoPE rewrites the rows
                                                const float* __restrict__ rope, int rope_len, bool pairs,
                                                int T, int pos0, int D, int heads, int kv_heads, int KD, int cap_dev,
                                                float* __restrict__ Kring) {
    // the table is [rope_len][half] cos followed by [rope_len][half] sin (one allocation: set_rope)
    const int half = D / 2;
    const int idx = blockIdx.x * 32 + threadIdx.x;
    const int nq = T * heads;
    if (idx >= nq + T * kv_heads) return;
    float* x;
    bool is_k;
    int t, h;
    if (idx < nq) { t = idx / heads; h = idx % heads; x = q + (size_t) t * (heads * D) + (size_t) h * D; is_k = false; }
    else { const int j = idx - nq; t = j / kv_heads; h = j % kv_heads; x = k + (size_t) t * KD + (size_t) h * D; is_k = true; }
    const int lane = threadIdx.x;
    if (lane >= half) return;
    const int pos = pos0 + t;
    const float cs = rope[(size_t) pos * half + lane];
    const float sn = rope[((size_t) rope_len + (size_t) pos) * half + lane];
    const int i1 = pairs ? 2 * lane : lane, i2 = pairs ? 2 * lane + 1 : lane + half;
    const float a = x[i1], b = x[i2];
    if (pairs) { x[i1] = a * cs - b * sn; x[i2] = a * sn + b * cs; }
    else { x[i1] = a * cs - b * sn; x[i2] = b * cs + a * sn; }
    if (is_k) {
        const size_t slot = (size_t) pos % (size_t) cap_dev;
        Kring[slot * KD + (size_t) h * D + i1] = x[i1];
        Kring[slot * KD + (size_t) h * D + i2] = x[i2];
    }
}

__global__ void k_vwrite(const float* __restrict__ v, int T, int pos0, int KD, int cap_dev, float* __restrict__ Vring) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) T * KD) return;
    const int t = (int) (idx / KD), j = (int) (idx % KD);
    Vring[(((size_t) pos0 + t) % (size_t) cap_dev) * KD + j] = v[idx];
}

// ---------------------------------------------------------------- attention: one block per (token, kv group). bf16 WMMA tiles of KT keys with an online
// softmax; the sink joins the denominator only (m starts at -inf, its exp(sink - m_final) is added to l after the last
// tile), exactly as attn_heads() does on the host. D = head_dim must be a multiple of 16 and <= KT.
constexpr int KT = 64;                     // keys per tile (>= every gpt-oss head_dim: 64 real, 16 tiny)

__global__ void __launch_bounds__(128) k_attn(const float* __restrict__ Qb, const float* __restrict__ Kring,
                                              const float* __restrict__ Vring, const float* __restrict__ sinks,
                                              int T, int pos0, int D, int heads, int kv_heads, int rep, int KD,
                                              int cap_dev, int window, bool sliding, float* __restrict__ Ab) {
    using namespace nvcuda;
    extern __shared__ char smem_raw[];
    // flat layouts (D is a runtime value: no array-of-runtime-dim in CUDA C). Offsets in elements.
    const int oK = 16 * D, oV = (16 + KT) * D, oS = (16 + 2 * KT) * D;           // bf16 region: Qs[16*D] Ks[KT*D] Vs[KT*D]
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);              // [16][D], rows >= rep zero
    __nv_bfloat16* Ks = Qs + oK;                                                 // [KT][D]
    __nv_bfloat16* Vs = Qs + oV;                                                 // [KT][D]
    float* Ssm = reinterpret_cast<float*>(smem_raw + oS * 2);                    // [16][KT] fp32 scores
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(smem_raw + (oS * 2 + 16 * KT * 4));   // [16][KT] bf16
    float* Stg = Ssm;                                                            // [16][D] fp32 staging for the P@V tile (D <= KT)
    float* Osm = reinterpret_cast<float*>(smem_raw + oS * 2 + 16 * KT * 4 + 16 * KT * 2);   // [rep][D] fp32 running output
    float* st = Osm + rep * D;                                                   // m[8], l[8], alpha[8]

    const int t = blockIdx.x / kv_heads, g = blockIdx.x % kv_heads;
    if (t >= T) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pos = pos0 + t;
    const int nkeys = sliding ? ((pos + 1 < window) ? pos + 1 : window) : pos + 1;
    const int lo = pos - nkeys + 1;

    // Q tile: the rep q heads of this kv group -> rows 0..rep-1, pad rows zero (the A fragments read all 16 rows).
    {
        const float* qb = Qb + (size_t) t * (heads * D) + (size_t) g * rep * D;
        for (int e2 = tid; e2 < 16 * D; e2 += blockDim.x) {
            const int h = e2 / D, d = e2 % D;
            Qs[h * D + d] = __float2bfloat16(h < rep ? qb[(size_t) h * D + d] : 0.0f);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { st[i] = -INFINITY; st[8 + i] = 0.0f; st[16 + i] = 1.0f; }

    const float* kbase = Kring + (size_t) g * D;
    const float* vbase = Vring + (size_t) g * D;
    const int slot_lo = sliding ? lo % cap_dev : lo;   // consecutive positions: the ring walk wraps at most once inside a window

    for (int k0 = 0; k0 < nkeys; k0 += KT) {
        const int nt = (KT < nkeys - k0) ? KT : nkeys - k0;
        for (int e2 = tid; e2 < KT * D; e2 += blockDim.x) {          // K/V tile -> smem bf16 (invalid slots zero: masked before the softmax)
            const int i = e2 / D, d = e2 % D;
            if (i < nt) {
                const int slot = sliding ? (slot_lo + i) % cap_dev : slot_lo + i;
                Ks[i * D + d] = __float2bfloat16(kbase[(size_t) slot * KD + d]);
                Vs[i * D + d] = __float2bfloat16(vbase[(size_t) slot * KD + d]);
            } else {
                Ks[i * D + d] = __float2bfloat16(0.0f);
                Vs[i * D + d] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // scores: S[16][KT] = Qs . Ks^T; warp w owns the n-tile of 16 keys at columns [w*16, w*16+16)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int ks = 0; ks < D; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, Qs + ks, D);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Ks + warp * 16 * D + ks, D);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(Ssm + warp * 16 * KT, acc, KT, wmma::mem_row_major);   // element (m,n) at Ssm[m*KT + n]
        }
        __syncthreads();

        // online softmax per row: warp w owns rows w and w+4; lane i covers columns (2i, 2i+1) of the whole tile.
        {
            const float sc = rsqrtf((float) D);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int h = warp + 4 * r;
                if (h >= rep) break;
                const int c0 = 2 * lane, c1 = 2 * lane + 1;
                float a = Ssm[h * KT + c0] * sc, b = Ssm[h * KT + c1] * sc;
                if (c0 >= nt) a = -INFINITY;
                if (c1 >= nt) b = -INFINITY;
                const float m_old = st[h];
                float tm = fmaxf(a, b);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
                const float m_new = fmaxf(m_old, tm);
                const float alpha = expf(m_old - m_new);      // first tile: m_old == -inf -> alpha == 1 (l is 0)
                const float p0 = expf(a - m_new), p1 = expf(b - m_new);
                float rs2 = p0 + p1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rs2 += __shfl_xor_sync(0xffffffffu, rs2, o);
                if (lane == 0) { st[h] = m_new; st[8 + h] = st[8 + h] * alpha + rs2; st[16 + h] = alpha; }
                Ps[h * KT + c0] = __float2bfloat16(p0);
                Ps[h * KT + c1] = __float2bfloat16(p1);
            }
        }
        __syncthreads();

        // output tile: O[16][D] += Ps . Vs  (A row_major ld=KT, B row_major ld=D); warp w owns the n-tile of 16 dims
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> vacc;
            wmma::fill_fragment(vacc, 0.0f);
#pragma unroll
            for (int ks = 0; ks < KT; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, Ps + ks, KT);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, Vs + ks * D, D);
                wmma::mma_sync(vacc, fa, fb, vacc);
            }
            wmma::store_matrix_sync(Stg + warp * 16 * D, vacc, D, wmma::mem_row_major);   // element (m,n) at Stg[m*D + n]
        }
        __syncthreads();
        for (int e2 = tid; e2 < rep * D; e2 += blockDim.x) {          // rescale the running output by alpha[h], then add the tile
            const int h = e2 / D, d = e2 % D;
            Osm[e2] = Osm[e2] * st[16 + h] + Stg[h * D + d];
        }
        __syncthreads();
    }

    // out = O / (l + exp(sink - m)); the sink has no value of its own.
    for (int e2 = tid; e2 < rep * D; e2 += blockDim.x) {
        const int h = e2 / D, d = e2 % D;
        const float z = st[8 + h] + expf(sinks[(size_t) g * rep + h] - st[h]);
        Ab[(size_t) t * (heads * D) + ((size_t) g * rep + h) * D + d] = Osm[e2] / z;
    }
}

// ---------------------------------------------------------------- grouped expert kernels. Work item = (expert e, 16 routed rows, 16 weight rows); the whole block
// decodes each group's MXFP4 bytes and gathers/converts the x chunk into shared memory while warp 0 runs the bf16 WMMA.
// Items are distributed with a device-side prefix (gio/dio), so ONE launch covers every touched expert of the layer.

struct ExpArgs {
    const TDev* td;
    const int* rowoff, *gio, *dio;
    int total_gu, total_dn;
    float* X;             // [T][H] post-norm: the experts' input (read-only here)
    float* Xres;          // [T][H] residual stream: where the combine adds the weighted sum
    const int* src;       // [R_tot] token of grouped row r
    const float* wt;      // [R_tot] router weight of row r
    const int* rowof;     // [T*K] global row of (t,k)
    const int* rowexp;    // [R_tot] expert of row r
    float* GU;            // [R_tot][2I] fp32
    __nv_bfloat16* ACT;   // [R_tot][I] bf16
    float* Yslot;         // [T*K][H] fp32
    int H, I, E, T, K, halves;
    float limit, alpha;
};

__device__ __forceinline__ int item_expert(const int* io, int E, int item) {   // first e with io[e+1] > item (io is a prefix over [0..E])
    int lo = 0, hi = E - 1;
    while (lo < hi) { const int m = (lo + hi) >> 1; if (item < io[m + 1]) hi = m; else lo = m + 1; }
    return lo;
}

__device__ void expert_gu_item(const ExpArgs& a, int e, int rt, int wb, __nv_bfloat16* Ws, __nv_bfloat16* Xs, float* Out) {
    using namespace nvcuda;
    const int H = a.H, I = a.I, G = H / 32;
    const TDev& t = a.td[e];
    const int cnt = a.rowoff[e + 1] - a.rowoff[e];
    const int r0l = rt * 16;
    const int n_rows = (16 < cnt - r0l) ? 16 : cnt - r0l;
    const size_t rg_base = (size_t) a.rowoff[e] + r0l;      // global row of the tile's first row
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int g = 0; g < G; ++g) {
        if (threadIdx.x < 16) {                             // one MXFP4 group per thread: 16 bytes + scale -> 32 bf16 in smem
            const size_t wr = (size_t) wb * 16 + threadIdx.x;
            const uint8_t* b = t.gu_b + ((wr * G + g) * 16);
            const float sc = ldexpf(1.0f, (int) t.gu_s[wr * G + g] - 127);
            const uint4 raw = *reinterpret_cast<const uint4*>(b);
            const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
            __nv_bfloat16* w = Ws + threadIdx.x * LDS;
#pragma unroll
            for (int k = 0; k < 16; ++k) { w[2 * k] = __float2bfloat16(c_fp4[q[k] & 15] * sc); w[2 * k + 1] = __float2bfloat16(c_fp4[q[k] >> 4] * sc); }
        }
        for (int e2 = threadIdx.x; e2 < 16 * BK; e2 += blockDim.x) {   // x chunk: gather the tile's tokens, fp32 -> bf16
            const int rr = e2 / BK, k = e2 % BK;
            const float xv = (rr < n_rows) ? a.X[(size_t) a.src[rg_base + rr] * H + g * BK + k] : 0.0f;
            Xs[rr * LDS + k] = __float2bfloat16(xv);
        }
        __syncthreads();
        if ((threadIdx.x >> 5) == 0) {                      // warp 0 multiplies the whole [16][16] tile (M=routed rows, N=weight rows)
#pragma unroll
            for (int ks = 0; ks < BK; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, Xs + ks, LDS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Ws + ks, LDS);
                wmma::mma_sync(acc, fa, fb, acc);
            }
        }
        __syncthreads();
    }
    if ((threadIdx.x >> 5) == 0) wmma::store_matrix_sync(Out, acc, LDC, wmma::mem_row_major);   // Out[16][LDC] fp32
    __syncthreads();
    for (int e2 = threadIdx.x; e2 < 16 * 16; e2 += blockDim.x) {
        const int rr = e2 / 16, jj = e2 % 16;
        if (rr < n_rows) a.GU[(rg_base + rr) * (size_t) (2 * I) + wb * 16 + jj] = Out[rr * LDC + jj];
    }
    __syncthreads();
}

__device__ void expert_dn_item(const ExpArgs& a, int e, int rt, int wb, __nv_bfloat16* Ws, __nv_bfloat16* Xs, float* Out) {
    using namespace nvcuda;
    const int H = a.H, I = a.I, G = I / 32;
    const TDev& t = a.td[e];
    const int cnt = a.rowoff[e + 1] - a.rowoff[e];
    const int r0l = rt * 16;
    const int n_rows = (16 < cnt - r0l) ? 16 : cnt - r0l;
    const size_t rg_base = (size_t) a.rowoff[e] + r0l;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int g = 0; g < G; ++g) {
        if (threadIdx.x < 16) {                             // weight row = hidden dim h = wb*16 + threadIdx.x
            const size_t wr = (size_t) wb * 16 + threadIdx.x;
            const uint8_t* b = t.dn_b + ((wr * G + g) * 16);
            const float sc = ldexpf(1.0f, (int) t.dn_s[wr * G + g] - 127);
            const uint4 raw = *reinterpret_cast<const uint4*>(b);
            const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
            __nv_bfloat16* w = Ws + threadIdx.x * LDS;
#pragma unroll
            for (int k = 0; k < 16; ++k) { w[2 * k] = __float2bfloat16(c_fp4[q[k] & 15] * sc); w[2 * k + 1] = __float2bfloat16(c_fp4[q[k] >> 4] * sc); }
        }
        for (int e2 = threadIdx.x; e2 < 16 * BK; e2 += blockDim.x) {   // A chunk: the ACT rows are contiguous in the grouped layout
            const int rr = e2 / BK, k = e2 % BK;
            Xs[rr * LDS + k] = (rr < n_rows) ? a.ACT[(rg_base + rr) * I + g * BK + k] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        if ((threadIdx.x >> 5) == 0) {
#pragma unroll
            for (int ks = 0; ks < BK; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, Xs + ks, LDS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Ws + ks, LDS);
                wmma::mma_sync(acc, fa, fb, acc);
            }
        }
        __syncthreads();
    }
    if ((threadIdx.x >> 5) == 0) wmma::store_matrix_sync(Out, acc, LDC, wmma::mem_row_major);
    __syncthreads();
    for (int e2 = threadIdx.x; e2 < 16 * 16; e2 += blockDim.x) {
        const int rr = e2 / 16, jj = e2 % 16;               // Yslot[row][h] = wt[row] * (down + bias[h]): one writer per cell, no atomics
        if (rr < n_rows) {
            const size_t rg = rg_base + rr;
            const int h = wb * 16 + jj;
            a.Yslot[rg * H + h] = a.wt[rg] * (Out[rr * LDC + jj] + bf2f(t.dn_bias[h]));
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(128) k_expert_gu(const ExpArgs a) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [16][LDS]
    __nv_bfloat16* Xs = Ws + 16 * LDS;                                       // [16][LDS]
    float* Out = reinterpret_cast<float*>(Xs + 16 * LDS);                    // [16][LDC] fp32
    const int wtiles = (2 * a.I) / 16;
    for (int item = blockIdx.x; item < a.total_gu; item += gridDim.x) {
        const int e = item_expert(a.gio, a.E, item);
        const int local = item - a.gio[e];
        expert_gu_item(a, e, local / wtiles, local % wtiles, Ws, Xs, Out);
    }
}

__global__ void __launch_bounds__(128) k_expert_dn(const ExpArgs a) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Xs = Ws + 16 * LDS;
    float* Out = reinterpret_cast<float*>(Xs + 16 * LDS);
    const int wtiles = a.H / 16;
    for (int item = blockIdx.x; item < a.total_dn; item += gridDim.x) {
        const int e = item_expert(a.dio, a.E, item);
        const int local = item - a.dio[e];
        expert_dn_item(a, e, local / wtiles, local % wtiles, Ws, Xs, Out);
    }
}

__global__ void k_swiglu(const ExpArgs a) {
    const size_t total = (size_t) a.T * a.K * a.I;      // R_tot == T*K rows: every token has K experts
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const size_t r = idx / a.I;
        const int j = (int) (idx % a.I);
        const TDev& t = a.td[a.rowexp[r]];
        const float* row = a.GU + r * (size_t) (2 * a.I);
        float g = (a.halves ? row[j] : row[2 * j]) + bf2f(t.gu_bias[a.halves ? j : 2 * j]);
        float u = (a.halves ? row[a.I + j] : row[2 * j + 1]) + bf2f(t.gu_bias[a.halves ? a.I + j : 2 * j + 1]);
        g = fminf(g, a.limit);
        u = fminf(fmaxf(u, -a.limit), a.limit);
        a.ACT[idx] = __float2bfloat16((u + 1.0f) * (g / (1.0f + expf(-a.alpha * g))));
    }
}

__global__ void k_combine(const ExpArgs a) {
    const size_t total = (size_t) a.T * a.H;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int t = (int) (idx / a.H), h = (int) (idx % a.H);
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) if (k < a.K) s += a.Yslot[(size_t) a.rowof[t * a.K + k] * a.H + h];   // k order, as the host path sums
        a.Xres[idx] += s;
    }
}

// residual add (+ o bias) and post-norm in one pass: X += Ob + bo; Hn = rmsnorm(X).
__global__ void k_res_post(float* __restrict__ x, const float* __restrict__ ob, const float* __restrict__ bo,   // x is updated in place: X += Ob + bo
                           const float* __restrict__ w, float eps, int H, int T, float* __restrict__ y) {
    const int t = blockIdx.x;
    if (t >= T) return;
    float* xr = x + (size_t) t * H;
    const float* orr = ob + (size_t) t * H;
    extern __shared__ float red[];
    float p = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        xr[i] += orr[i] + bo[i];
        const float v = xr[i];
        p += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = p;
    __syncthreads();
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = rsqrtf(v / H + eps);
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[i] = xr[i] * s * w[i];
}

// ---------------------------------------------------------------- host side

bool alloc_all() {
    const size_t H = (size_t) g_c.H, I = (size_t) g_c.I, E = (size_t) g_c.E, Tm = (size_t) std::max(1, g_c.chunk_max);
    const size_t QD = (size_t) g_c.heads * g_c.head_dim, KD = (size_t) g_c.kv_heads * g_c.head_dim;
    const size_t TK = Tm * (size_t) g_c.topk;
    auto M = [](void** p, size_t n, const char* w) { return ck(cudaMalloc(p, n ? n : 1), w); };
    if (!M((void**) &g_dX, Tm * H * 4, "d_X") || !M((void**) &g_dHn, Tm * H * 4, "d_Hn") ||
        !M((void**) &g_dQb, Tm * QD * 4, "d_Qb") || !M((void**) &g_dKb, Tm * KD * 4, "d_Kb") || !M((void**) &g_dVb, Tm * KD * 4, "d_Vb") ||
        !M((void**) &g_dAb, Tm * QD * 4, "d_Ab") || !M((void**) &g_dOb, Tm * H * 4, "d_Ob") || !M((void**) &g_dGl, Tm * E * 4, "d_Gl")) return false;
    if (!ck(cudaMallocHost((void**) &g_hGl, Tm * E * 4), "h_Gl")) return false;
    if (!M((void**) &g_dGU, TK * 2 * I * 4, "d_GU") || !M((void**) &g_dACT, TK * I * 2, "d_ACT") || !M((void**) &g_dYslot, TK * H * 4, "d_Yslot")) return false;
    if (!M((void**) &g_dTd, E * sizeof(TDev), "d_Td") || !M((void**) &g_dRowoff, (E + 1) * 4, "d_rowoff") ||
        !M((void**) &g_dGio, (E + 1) * 4, "d_gio") || !M((void**) &g_dDio, (E + 1) * 4, "d_dio") ||
        !M((void**) &g_dSrc, TK * 4, "d_src") || !M((void**) &g_dWt, TK * 4, "d_wt") || !M((void**) &g_dRowof, TK * 4, "d_rowof") ||
        !M((void**) &g_dRowexp, TK * 4, "d_rowexp")) return false;
    const size_t stage = E * sizeof(TDev) + (size_t) (E + 1) * 12 + TK * 16 + 256;
    if (!ck(cudaMallocHost(&g_hStage, stage), "h_stage")) return false;
    g_L.resize((size_t) g_c.layers);
    for (auto& y : g_L) {
        y.cap_dev = y.sliding ? std::max(1, g_c.window) + std::max(1, g_c.chunk_max) : g_c.ctx;
        if (!M((void**) &y.K, (size_t) y.cap_dev * KD * 4, "KV K") || !M((void**) &y.V, (size_t) y.cap_dev * KD * 4, "KV V")) return false;
    }
    return true;
}

void free_all() {
    for (auto& y : g_L) { cudaFree(y.K); cudaFree(y.V); cudaFree(y.small); }
    g_L.clear();
    cudaFree(g_dX); cudaFree(g_dHn); cudaFree(g_dQb); cudaFree(g_dKb); cudaFree(g_dVb); cudaFree(g_dAb); cudaFree(g_dOb); cudaFree(g_dGl);
    cudaFreeHost(g_hGl);
    cudaFree(g_dGU); cudaFree(g_dACT); cudaFree(g_dYslot);
    cudaFree(g_dTd); cudaFree(g_dRowoff); cudaFree(g_dGio); cudaFree(g_dDio);
    cudaFree(g_dSrc); cudaFree(g_dWt); cudaFree(g_dRowof); cudaFree(g_dRowexp);
    cudaFreeHost(g_hStage);
    if (g_dCos) { cudaFree(g_dCos); g_dCos = nullptr; g_dSin = nullptr; }
    g_dX = g_dHn = g_dQb = g_dKb = g_dVb = g_dAb = g_dOb = g_dGl = nullptr;
    g_hGl = nullptr;
    g_dGU = nullptr; g_dACT = nullptr; g_dYslot = nullptr;
    g_dTd = nullptr; g_dRowoff = nullptr; g_dGio = nullptr; g_dDio = nullptr;
    g_dSrc = nullptr; g_dWt = nullptr; g_dRowof = nullptr; g_dRowexp = nullptr;
    g_hStage = nullptr;
    g_dCos = nullptr;
}

bool launch_gemm(const void* dW, int rows, int cols, const float* x, int N, float* y) {
    if (N <= 0 || rows <= 0 || cols % BK != 0) return false;
    k_gemm<<<dim3((unsigned) ((rows + BM - 1) / BM), (unsigned) ((N + BN - 1) / BN)), 128, 0, g_comp>>>((const unsigned short*) dW, rows, cols, x, N, y);
    return !ck(cudaGetLastError(), "gemm launch");
}

// Copy n consecutive positions [lo, lo+n) of a host ring (cap_h) into the device ring (cap_d), or the reverse. The span is
// shorter than both caps (the caller guarantees it), so each side wraps at most once: at most two contiguous copies per tensor.
template <bool H2D>
void copy_ring(float* dev, int cap_d, const float* host, int cap_h, int lo, int n) {
    if (n <= 0) return;
    const size_t KD = (size_t) g_c.kv_heads * g_c.head_dim;
    const int a = lo % cap_h, b = lo % cap_d;
    const int seg1 = std::min(n, H2D ? cap_h - a : cap_d - b);
    if (H2D) {
        ck(cudaMemcpyAsync(dev + (size_t) b * KD, host + (size_t) a * KD, (size_t) seg1 * KD * 4, cudaMemcpyHostToDevice, g_copy), "ring copy");
        if (n > seg1) ck(cudaMemcpyAsync(dev + ((b + seg1) % cap_d) * KD, host, (size_t) (n - seg1) * KD * 4, cudaMemcpyHostToDevice, g_copy), "ring copy");
    } else {
        ck(cudaMemcpyAsync((void*) (host + (size_t) a * KD), dev + (size_t) b * KD, (size_t) seg1 * KD * 4, cudaMemcpyDeviceToHost, g_copy), "ring copy");
        if (n > seg1) ck(cudaMemcpyAsync((void*) host, dev + ((b + seg1) % cap_d) * KD, (size_t) (n - seg1) * KD * 4, cudaMemcpyDeviceToHost, g_copy), "ring copy");
    }
}

bool run_layer(const Cfg& c, int l, const LayerIO& w, int T, int pos0, bool sliding, float* gl_host) {
    if (!g_on || l < 0 || l >= (int) g_L.size() || T <= 0 || T > g_c.chunk_max || !w.dq || !w.dk || !w.dv || !w.dobj || !w.drt) return false;
    const LayerDev& y = g_L[(size_t) l];
    if (y.sliding != sliding) return false;
    const int H = c.H, E = c.E, D = c.head_dim, QD = c.q_dim(), KD = c.kv_dim();
    const int rep = c.heads / c.kv_heads;
    const float* small = y.small;
    const size_t o_inln = 0, o_postln = (size_t) H, o_sinks = 2 * (size_t) H, o_bq = 2 * (size_t) H + (size_t) c.heads,
                 o_bk = o_bq + (size_t) QD, o_bv = o_bk + (size_t) KD, o_bo = o_bv + (size_t) KD;

    if (!ck(cudaStreamWaitEvent(g_comp, g_ev_prefetch, 0), "wait prefetch")) return false;   // the KV rows this chunk needs are in
    const int want_w = (std::max(32, (H + 7) / 8) + 31) >> 5;   // whole warps: the reduction buffer is sized by this count
    const int nw = (want_w < 32) ? want_w : 32;
    k_rmsnorm<<<T, nw * 32, nw * 4, g_comp>>>(g_dX, small + o_inln, c.eps, H, T, g_dHn);
    if (!launch_gemm(w.dq, QD, H, g_dHn, T, g_dQb)) return false;
    if (!launch_gemm(w.dk, KD, H, g_dHn, T, g_dKb)) return false;
    if (!launch_gemm(w.dv, KD, H, g_dHn, T, g_dVb)) return false;
    const size_t n_bias = (size_t) T * (QD + 2 * KD);
    k_bias<<<(unsigned) std::min((n_bias + 255) / 256, (size_t) 65535), 256, 0, g_comp>>>(g_dQb, small + o_bq, QD, g_dKb, small + o_bk, KD,
                                                                                          g_dVb, small + o_bv, T);
    const int n_rope = T * (c.heads + c.kv_heads);
    k_rope_kv<<<(unsigned) ((n_rope + 31) / 32), 32, 0, g_comp>>>(g_dQb, g_dKb, g_dCos, g_rope_len, c.rope_pairs, T, pos0, D, c.heads,
                                                                  c.kv_heads, KD, y.cap_dev, y.K);
    k_vwrite<<<(unsigned) ((T * KD + 255) / 256), 256, 0, g_comp>>>(g_dVb, T, pos0, KD, y.cap_dev, y.V);

    const size_t smem = (16 + 2 * KT) * D * 2 + 16 * KT * 4 + 16 * KT * 2 + rep * D * 4 + 8 * 3 * 4;
    k_attn<<<T * c.kv_heads, 128, smem, g_comp>>>(g_dQb, y.K, y.V, small + o_sinks, T, pos0, D, c.heads, c.kv_heads, rep, KD,
                                                  y.cap_dev, std::max(1, c.window), sliding, g_dAb);
    if (!launch_gemm(w.dobj, H, QD, g_dAb, T, g_dOb)) return false;
    k_res_post<<<T, nw * 32, nw * 4, g_comp>>>(g_dX, g_dOb, small + o_bo, small + o_postln, c.eps, H, T, g_dHn);
    if (!launch_gemm(w.drt, E, H, g_dHn, T, g_dGl)) return false;
    if (!ck(cudaMemcpyAsync(g_hGl, g_dGl, (size_t) T * E * 4, cudaMemcpyDeviceToHost, g_comp), "router d2h")) return false;
    if (!ck(cudaStreamSynchronize(g_comp), "layer sync")) return false;      // the layer's only sync: the host routes from gl_host
    std::memcpy(gl_host, g_hGl, (size_t) T * E * 4);
    return true;
}

// ---------------------------------------------------------------- public API

bool available() { return true; }   // this translation unit is linked only into the CUDA build

bool init(const Cfg& c, const char* sliding, size_t budget_bytes) {
    if (g_on) shutdown();
    if (!sliding || c.H <= 0 || c.H % 32 != 0 || c.I % 32 != 0 || c.E <= 0 || c.topk < 1 || c.topk > 8 || c.layers <= 0 || c.ctx <= 0 || c.window <= 0 ||
        c.heads <= 0 || c.kv_heads <= 0 || c.head_dim % 16 != 0 || c.head_dim > KT || c.heads % c.kv_heads != 0 || c.chunk_max < 1) return false;
    int n = 0;
    if (!ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount") || n == 0) return false;
    int maj = 0, mino = 0;
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&mino, cudaDevAttrComputeCapabilityMinor, 0);
    if (maj * 10 + mino < 80) return false;                       // the WMMA path needs sm_80+

    const size_t H = (size_t) c.H, I = (size_t) c.I, E = (size_t) c.E, Tm = (size_t) std::max(1, c.chunk_max);
    const size_t QD = (size_t) c.heads * c.head_dim, KD = (size_t) c.kv_heads * c.head_dim;
    const size_t TK = Tm * (size_t) c.topk;
    size_t need = 0;
    for (int l = 0; l < c.layers; ++l) {                          // per-layer ring: dense = ctx, banded = window + chunk_max
        const int cap = sliding[l] ? std::max(1, c.window) + std::max(1, c.chunk_max) : c.ctx;
        need += 2 * (size_t) cap * KD * 4;
    }
    need += Tm * H * 4 * 2 + Tm * QD * 4 * 2 + Tm * KD * 4 * 2 + Tm * E * 4 + TK * (2 * I * 4 + I * 2 + H * 4);
    need += E * sizeof(TDev) + (size_t) (E + 1) * 12 + TK * 16 + 256;   // tables + pinned staging
    need += (size_t) c.ctx * (c.head_dim / 2) * 4 * 2;            // rope table
    if (need > budget_bytes) return false;

    g_c = c;
    for (int l = 0; l < c.layers; ++l) { LayerDev yd; yd.sliding = sliding[l] != 0; g_L.push_back(yd); }   // sizes are set by alloc_all
    if (!ck(cudaStreamCreateWithFlags(&g_comp, cudaStreamNonBlocking), "pdev compute stream") ||
        !ck(cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking), "pdev copy stream") ||
        !ck(cudaEventCreateWithFlags(&g_ev_prefetch, cudaEventDisableTiming), "prefetch event") ||
        !ck(cudaEventCreateWithFlags(&g_ev_store, cudaEventDisableTiming), "store event") ||
        !ck(cudaEventCreateWithFlags(&g_ev_sync, cudaEventDisableTiming), "sync event")) return false;
    if (!alloc_all()) { free_all(); g_L.clear(); return false; }
    g_on = true;
    g_fill = 0;
    return true;
}

void shutdown() {
    if (g_comp) { cudaStreamSynchronize(g_comp); cudaStreamDestroy(g_comp); }
    if (g_copy) { cudaStreamSynchronize(g_copy); cudaStreamDestroy(g_copy); }
    if (g_ev_prefetch) cudaEventDestroy(g_ev_prefetch);
    if (g_ev_store) cudaEventDestroy(g_ev_store);
    if (g_ev_sync) cudaEventDestroy(g_ev_sync);
    g_comp = g_copy = nullptr;
    g_ev_prefetch = g_ev_store = g_ev_sync = nullptr;
    free_all();
    g_L.clear();
    g_on = false;
}

void set_rope(const float* cos_t, const float* sin_t, int len) {
    if (!g_on || !cos_t || !sin_t || len <= 0) return;
    const size_t n = (size_t) len * (size_t) (g_c.head_dim / 2);
    if (len > g_rope_len) {
        cudaFree(g_dCos);
        if (!ck(cudaMalloc((void**) &g_dCos, n * 4 * 2), "rope alloc")) return;
        g_dSin = (float*) g_dCos + n;
    }
    ck(cudaMemcpyAsync(g_dCos, cos_t, n * 4, cudaMemcpyHostToDevice, g_comp), "rope h2d");
    ck(cudaMemcpyAsync(g_dSin, sin_t, n * 4, cudaMemcpyHostToDevice, g_comp), "rope h2d");
    ck(cudaStreamSynchronize(g_comp), "rope sync");
    g_rope_len = len;
}

bool upload_layer(int l, bool sliding, const float* in_ln, const float* post_ln, const float* sinks,
                  const float* bq, const float* bk, const float* bv, const float* bo, const float* rbias,
                  const void* dq, const void* dk, const void* dv, const void* dobj, const void* drt) {
    if (!g_on || l < 0 || l >= (int) g_L.size() || !in_ln || !post_ln || !sinks || !bq || !bk || !bv || !bo || !rbias ||
        !dq || !dk || !dv || !dobj || !drt) return false;
    LayerDev& y = g_L[(size_t) l];
    if (y.sliding != sliding) { std::lock_guard<std::mutex> lk(g_err_mu); g_err = "upload_layer: sliding flag changed after init"; return false; }
    const size_t H = (size_t) g_c.H, QD = (size_t) g_c.heads * g_c.head_dim, KD = (size_t) g_c.kv_heads * g_c.head_dim, E = (size_t) g_c.E;
    const size_t nsmall = 2 * H + (size_t) g_c.heads + QD + 2 * KD + H + E;
    if (!y.small && !ck(cudaMalloc((void**) &y.small, nsmall * 4), "layer small")) return false;
    float* htmp = (float*) std::malloc(nsmall * 4);
    if (!htmp) { std::lock_guard<std::mutex> lk(g_err_mu); g_err = "upload_layer: malloc"; return false; }
    size_t o = 0;
    std::memcpy(htmp + o, in_ln, H * 4); o += H;
    std::memcpy(htmp + o, post_ln, H * 4); o += H;
    std::memcpy(htmp + o, sinks, (size_t) g_c.heads * 4); o += g_c.heads;
    std::memcpy(htmp + o, bq, QD * 4); o += QD;
    std::memcpy(htmp + o, bk, KD * 4); o += KD;
    std::memcpy(htmp + o, bv, KD * 4); o += KD;
    std::memcpy(htmp + o, bo, H * 4); o += H;
    std::memcpy(htmp + o, rbias, E * 4); o += E;
    const bool ok = ck(cudaMemcpyAsync(y.small, htmp, nsmall * 4, cudaMemcpyHostToDevice, g_comp), "layer small h2d") &&
                    ck(cudaStreamSynchronize(g_comp), "layer small sync");
    std::free(htmp);
    return ok;
}

float* x_dev() { return g_on ? g_dX : nullptr; }

bool x_upload(const float* host_x, int T) {
    if (!g_on || !host_x || T <= 0 || T > g_c.chunk_max) return false;
    return ck(cudaMemcpyAsync(g_dX, host_x, (size_t) T * g_c.H * 4, cudaMemcpyHostToDevice, g_comp), "x h2d");
}

bool x_download(float* host_x, int T) {
    if (!g_on || !host_x || T <= 0 || T > g_c.chunk_max) return false;
    if (!ck(cudaMemcpyAsync(host_x, g_dX, (size_t) T * g_c.H * 4, cudaMemcpyDeviceToHost, g_comp), "x d2h")) return false;
    return ck(cudaStreamSynchronize(g_comp), "x sync");
}

void kv_prefetch(int l, const float* k_host, const float* v_host, int cap_host, bool sliding, int pos0) {
    if (!g_on || l < 0 || l >= (int) g_L.size() || !k_host || !v_host || pos0 < 0) return;
    LayerDev& y = g_L[(size_t) l];
    if (sliding) {                                                // the window-1 past rows: cheap, and it makes rewinds correct
        const int lo = std::max(0, pos0 - std::max(1, g_c.window) + 1);
        copy_ring<true>(y.K, y.cap_dev, k_host, cap_host, lo, pos0 - lo);
        copy_ring<true>(y.V, y.cap_dev, v_host, cap_host, lo, pos0 - lo);
    } else if (pos0 > g_fill) {                                   // dense: only the gap [fill, pos0) is missing on the device
        copy_ring<true>(y.K, y.cap_dev, k_host, cap_host, g_fill, pos0 - g_fill);
        copy_ring<true>(y.V, y.cap_dev, v_host, cap_host, g_fill, pos0 - g_fill);
    }
    ck(cudaEventRecord(g_ev_prefetch, g_copy), "prefetch event");
}

bool run_experts(const Cfg& c, int T, const ExpertPtrs* per_expert, const int* cnt_host,
                 const int* src_host, const float* wt_host, const int* rowof_host) {
    if (!g_on || !per_expert || !cnt_host || !src_host || !wt_host || !rowof_host || T <= 0 || T > g_c.chunk_max) return false;
    const int E = c.E, K = c.topk;
    std::vector<int> rowoff((size_t) E + 1), gio((size_t) E + 1), dio((size_t) E + 1);
    std::vector<TDev> td((size_t) E);
    int R_tot = 0, total_gu = 0, total_dn = 0;
    const int gu_wtiles = (2 * c.I) / 16, dn_wtiles = c.H / 16;
    for (int e = 0; e < E; ++e) {                       // prefixes: rowoff[e+1] includes expert e's rows
        const int cnt = cnt_host[e];
        if (cnt > 0) {
            const ExpertPtrs& p = per_expert[e];
            if (!p.gu_b || !p.dn_b || !p.gu_s || !p.dn_s || !p.gu_bias || !p.dn_bias) return false;   // a routed expert with no device blob: the caller must not have sent it
            td[(size_t) e].gu_b = (const uint8_t*) p.gu_b; td[(size_t) e].dn_b = (const uint8_t*) p.dn_b;
            td[(size_t) e].gu_s = (const uint8_t*) p.gu_s; td[(size_t) e].dn_s = (const uint8_t*) p.dn_s;
            td[(size_t) e].gu_bias = (const unsigned short*) p.gu_bias; td[(size_t) e].dn_bias = (const unsigned short*) p.dn_bias;
            total_gu += ((cnt + 15) / 16) * gu_wtiles;
            total_dn += ((cnt + 15) / 16) * dn_wtiles;
        }
        R_tot += cnt;
        rowoff[(size_t) e + 1] = R_tot;
        gio[(size_t) e + 1] = total_gu;
        dio[(size_t) e + 1] = total_dn;
    }
    if (R_tot > T * K) return false;

    // one pinned staging block: td[E] | rowoff[E+1] | gio[E+1] | dio[E+1] | src[R_tot] | wt[R_tot] | rowof[T*K] | rowexp[R_tot]
    uint8_t* st = (uint8_t*) g_hStage;
    std::memcpy(st, td.data(), (size_t) E * sizeof(TDev)); st += (size_t) E * sizeof(TDev);
    std::memcpy(st, rowoff.data(), ((size_t) E + 1) * 4); st += ((size_t) E + 1) * 4;
    std::memcpy(st, gio.data(), ((size_t) E + 1) * 4); st += ((size_t) E + 1) * 4;
    std::memcpy(st, dio.data(), ((size_t) E + 1) * 4); st += ((size_t) E + 1) * 4;
    std::memcpy(st, src_host, (size_t) R_tot * 4); st += (size_t) R_tot * 4;
    std::memcpy(st, wt_host, (size_t) R_tot * 4); st += (size_t) R_tot * 4;
    std::memcpy(st, rowof_host, (size_t) T * K * 4); st += (size_t) T * K * 4;
    // rowexp: expert of each grouped row (the swiglu bias lookup)
    int r = 0;
    for (int e = 0; e < E; ++e) { const int cnt = cnt_host[e]; for (int i = 0; i < cnt; ++i, ++r) ((int*) st)[(size_t) r] = e; }
    st += (size_t) R_tot * 4;

    if (!ck(cudaMemcpyAsync(g_dTd, g_hStage, (size_t) E * sizeof(TDev), cudaMemcpyHostToDevice, g_comp), "td h2d")) return false;
    const size_t off_rowoff = (size_t) E * sizeof(TDev);
    const size_t off_gio = off_rowoff + ((size_t) E + 1) * 4;
    const size_t off_dio = off_gio + ((size_t) E + 1) * 4;
    if (!ck(cudaMemcpyAsync(g_dRowoff, (uint8_t*) g_hStage + off_rowoff, ((size_t) E + 1) * 4, cudaMemcpyHostToDevice, g_comp), "rowoff h2d")) return false;
    if (!ck(cudaMemcpyAsync(g_dGio, (uint8_t*) g_hStage + off_gio, ((size_t) E + 1) * 4, cudaMemcpyHostToDevice, g_comp), "gio h2d")) return false;
    if (!ck(cudaMemcpyAsync(g_dDio, (uint8_t*) g_hStage + off_dio, ((size_t) E + 1) * 4, cudaMemcpyHostToDevice, g_comp), "dio h2d")) return false;
    const size_t off_src = off_dio + ((size_t) E + 1) * 4;
    if (!ck(cudaMemcpyAsync(g_dSrc, (uint8_t*) g_hStage + off_src, (size_t) R_tot * 4, cudaMemcpyHostToDevice, g_comp), "src h2d")) return false;
    const size_t off_wt = off_src + (size_t) R_tot * 4;
    if (!ck(cudaMemcpyAsync(g_dWt, (uint8_t*) g_hStage + off_wt, (size_t) R_tot * 4, cudaMemcpyHostToDevice, g_comp), "wt h2d")) return false;
    const size_t off_rowof = off_wt + (size_t) R_tot * 4;
    if (!ck(cudaMemcpyAsync(g_dRowof, (uint8_t*) g_hStage + off_rowof, (size_t) T * K * 4, cudaMemcpyHostToDevice, g_comp), "rowof h2d")) return false;
    const size_t off_rowexp = off_rowof + (size_t) T * K * 4;
    if (!ck(cudaMemcpyAsync(g_dRowexp, (uint8_t*) g_hStage + off_rowexp, (size_t) R_tot * 4, cudaMemcpyHostToDevice, g_comp), "rowexp h2d")) return false;

    ExpArgs a;
    a.td = g_dTd; a.rowoff = g_dRowoff; a.gio = g_dGio; a.dio = g_dDio;
    a.total_gu = total_gu; a.total_dn = total_dn;
    a.X = g_dHn; a.Xres = g_dX; a.src = g_dSrc; a.wt = g_dWt; a.rowof = g_dRowof; a.rowexp = g_dRowexp;
    a.GU = g_dGU; a.ACT = g_dACT; a.Yslot = g_dYslot;
    a.H = c.H; a.I = c.I; a.E = E; a.T = T; a.K = K; a.halves = c.glu_halves ? 1 : 0;
    a.limit = c.limit; a.alpha = c.alpha;

    const size_t esmem = (size_t) 16 * LDS * 2 * 2 + 16 * LDC * 4;
    if (total_gu > 0) {
        const int grid = std::min(total_gu, 2048);
        k_expert_gu<<<grid, 128, esmem, g_comp>>>(a);
        if (!ck(cudaGetLastError(), "expert gu launch")) return false;
    }
    {
        const int grid = (int) std::min<size_t>(((size_t) T * K * c.I + 255) / 256, 8192);
        k_swiglu<<<grid, 256, 0, g_comp>>>(a);
        if (!ck(cudaGetLastError(), "swiglu launch")) return false;
    }
    if (total_dn > 0) {
        const int grid = std::min(total_dn, 2048);
        k_expert_dn<<<grid, 128, esmem, g_comp>>>(a);
        if (!ck(cudaGetLastError(), "expert dn launch")) return false;
    }
    {
        const int grid = (int) std::min<size_t>(((size_t) T * c.H + 255) / 256, 8192);
        k_combine<<<grid, 256, 0, g_comp>>>(a);
        if (!ck(cudaGetLastError(), "combine launch")) return false;
    }
    return true;
}

void kv_store_back(int l, float* k_host, float* v_host, int cap_host, bool sliding, int T, int pos0) {
    if (!g_on || l < 0 || l >= (int) g_L.size() || !k_host || !v_host || T <= 0) return;
    LayerDev& y = g_L[(size_t) l];
    // the rows were written on g_comp by run_layer: order the copy stream after them before reading the ring
    ck(cudaEventRecord(g_ev_sync, g_comp), "store sync event");
    ck(cudaStreamWaitEvent(g_copy, g_ev_sync, 0), "store wait");
    copy_ring<false>(k_host, cap_host, y.K, y.cap_dev, pos0, T);
    copy_ring<false>(v_host, cap_host, y.V, y.cap_dev, pos0, T);
    ck(cudaEventRecord(g_ev_store, g_copy), "store event");
}

bool sync_all() {
    if (!g_on) return false;
    bool ok = ck(cudaStreamSynchronize(g_copy), "pdev copy sync") && ck(cudaStreamSynchronize(g_comp), "pdev compute sync");
    cudaGetLastError();   // clear any sticky launch error left by a failed enqueue
    return ok;
}

void advance(int T) { if (g_on && T > 0) g_fill += T; }

}  // namespace pdev
}  // namespace gptoss

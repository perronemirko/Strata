#include "gptoss/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__x86_64__)
#include <immintrin.h>
#define GPT_X86 1
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gptoss {

float bf16_to_f32(uint16_t v) { uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return f; }
float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float r;
    if (e == 0) r = std::ldexp((float) m, -24);
    else if (e == 31) r = m ? std::nanf("") : INFINITY;
    else r = std::ldexp((float) (m | 1024), (int) e - 25);
    return s ? -r : r;
}

const float* fp4_lut() {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t;
}

namespace {

int g_req = 0;                                          // CpuIsa requested

int detect_isa() {
#ifdef GPT_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 3;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return 2;
#endif
    return 1;
}
int best_isa() { static const int b = detect_isa(); return b; }
int isa() { const int b = best_isa(); return g_req == 0 ? b : (g_req <= b ? g_req : b); }

struct E8 { float v[256]; E8() { for (int i = 0; i < 256; ++i) v[i] = std::ldexp(1.0f, i - 127); } };
const float* e8_table() { static const E8 t; return t.v; }

// ---------------------------------------------------------------- scalar reference kernels
float dot_row_scalar(const WView& w, const uint8_t* r, const float* x) {
    const int n = w.cols;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (w.dt == DType::BF16) {
        const uint16_t* q = (const uint16_t*) r;
        int j = 0;
        for (; j + 4 <= n; j += 4) { a0 += bf16_to_f32(q[j]) * x[j]; a1 += bf16_to_f32(q[j + 1]) * x[j + 1]; a2 += bf16_to_f32(q[j + 2]) * x[j + 2]; a3 += bf16_to_f32(q[j + 3]) * x[j + 3]; }
        for (; j < n; ++j) a0 += bf16_to_f32(q[j]) * x[j];
    } else if (w.dt == DType::F32) {
        const float* q = (const float*) r;
        for (int j = 0; j < n; ++j) a0 += q[j] * x[j];
    } else if (w.dt == DType::F16) {
        const uint16_t* q = (const uint16_t*) r;
        for (int j = 0; j < n; ++j) a0 += f16_to_f32(q[j]) * x[j];
    }
    return (a0 + a1) + (a2 + a3);
}

void mx_rows_scalar(const MxView& w, const float* x, int r0, int r1, float* y) {
    const float* L = fp4_lut();
    const float* e8 = e8_table();
    const int G = w.cols / 32;
    for (int i = r0; i < r1; ++i) {
        const uint8_t* b = w.blocks + (size_t) i * G * 16;
        const uint8_t* s = w.scales + (size_t) i * G;
        float acc = 0;
        for (int g = 0; g < G; ++g) {
            const float* xx = x + g * 32;
            const uint8_t* bb = b + g * 16;
            float p = 0;
            for (int k = 0; k < 16; ++k) p += L[bb[k] & 15] * xx[2 * k] + L[bb[k] >> 4] * xx[2 * k + 1];
            acc += p * e8[s[g]];
        }
        y[i] = acc;
    }
}

// ---------------------------------------------------------------- SIMD kernels (no OpenMP inside: the callers split the rows)
#ifdef GPT_X86
__attribute__((target("avx2,fma")))
float dot_bf16_avx2(const uint16_t* w, const float* x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (w + j))), 16));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (w + j + 8))), 16));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + j), a0);
        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + j + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    float r = _mm_cvtss_f32(q);
    for (; j < n; ++j) r += bf16_to_f32(w[j]) * x[j];
    return r;
}

__attribute__((target("avx512f,avx2,fma")))
float dot_bf16_avx512(const uint16_t* w, const float* x, int n) {
    __m512 a = _mm512_setzero_ps();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) (w + j))), 16));
        a = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + j), a);
    }
    float r = _mm512_reduce_add_ps(a);
    for (; j < n; ++j) r += bf16_to_f32(w[j]) * x[j];
    return r;
}

// MXFP4 rows [r0, r1). xe/xo: x split per 32-block into its even / odd elements (16 + 16): the low nibbles of a block's 16 bytes are the even
// elements, the high nibbles the odd ones. The int8 table holds 2x the fp4 value, hence the 0.5 in the scale.
__attribute__((target("avx2,fma,ssse3")))
void mx_rows_avx2(const MxView& w, const float* xe, const float* xo, int r0, int r1, float* y) {
    const float* e8 = e8_table();
    const int G = w.cols / 32;
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m0f = _mm_set1_epi8(0x0F);
    for (int i = r0; i < r1; ++i) {
        const uint8_t* b = w.blocks + (size_t) i * G * 16;
        const uint8_t* s = w.scales + (size_t) i * G;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < G; ++g) {
            const __m128i raw = _mm_loadu_si128((const __m128i*) (b + (size_t) g * 16));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m0f));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m0f));
            const float* xE = xe + (size_t) g * 16;
            const float* xO = xo + (size_t) g * 16;
            __m256 blk = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xE));
            blk = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(xE + 8), blk);
            blk = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_loadu_ps(xO), blk);
            blk = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(xO + 8), blk);
            acc = _mm256_fmadd_ps(blk, _mm256_set1_ps(e8[s[g]] * 0.5f), acc);
        }
        __m128 q = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        q = _mm_hadd_ps(q, q);
        q = _mm_hadd_ps(q, q);
        y[i] = _mm_cvtss_f32(q);
    }
}

__attribute__((target("avx512f,avx2,fma,ssse3")))
void mx_rows_avx512(const MxView& w, const float* xe, const float* xo, int r0, int r1, float* y) {
    const float* e8 = e8_table();
    const int G = w.cols / 32;
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m0f = _mm_set1_epi8(0x0F);
    for (int i = r0; i < r1; ++i) {
        const uint8_t* b = w.blocks + (size_t) i * G * 16;
        const uint8_t* s = w.scales + (size_t) i * G;
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < G; ++g) {
            const __m128i raw = _mm_loadu_si128((const __m128i*) (b + (size_t) g * 16));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m0f));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m0f));
            __m512 blk = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_loadu_ps(xe + (size_t) g * 16));
            blk = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_loadu_ps(xo + (size_t) g * 16), blk);
            acc = _mm512_fmadd_ps(blk, _mm512_set1_ps(e8[s[g]] * 0.5f), acc);
        }
        y[i] = _mm512_reduce_add_ps(acc);
    }
}
#endif  // GPT_X86

float dot_row(const WView& w, const uint8_t* r, const float* x, int lv) {
#ifdef GPT_X86
    if (w.dt == DType::BF16) {
        if (lv >= 3) return dot_bf16_avx512((const uint16_t*) r, x, w.cols);
        if (lv == 2) return dot_bf16_avx2((const uint16_t*) r, x, w.cols);
    }
#endif
    (void) lv;
    return dot_row_scalar(w, r, x);
}

}  // namespace

void set_cpu_isa(CpuIsa v) { g_req = (int) v; }
const char* cpu_isa_name() { switch (isa()) { case 3: return "AVX-512"; case 2: return "AVX2+FMA"; default: return "scalar"; } }

void matvec(const WView& w, const float* x, float* y) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
    const int lv = isa();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < w.rows; ++i) y[i] = dot_row(w, w.p + (size_t) i * rb, x, lv);
}

void matvec_scalar(const WView& w, const float* x, float* y) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
    for (int i = 0; i < w.rows; ++i) y[i] = dot_row_scalar(w, w.p + (size_t) i * rb, x);
}

void matmat(const WView& w, const float* x, int N, float* y) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
    const int lv = isa();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < w.rows; ++i) {
        const uint8_t* r = w.p + (size_t) i * rb;                         // the row stays in cache for all N tokens
        for (int t = 0; t < N; ++t) y[(size_t) t * w.rows + i] = dot_row(w, r, x + (size_t) t * w.cols, lv);
    }
}

void matvec_mx_scalar(const MxView& w, const float* x, float* y) { mx_rows_scalar(w, x, 0, w.rows, y); }

void matvec_mx(const MxView& w, const float* x, float* y) {
    const int lv = isa();
#ifdef GPT_X86
    if (lv >= 2) {
        const int G = w.cols / 32;
        thread_local std::vector<float> xe, xo;                           // read by the worker threads below through the pointers
        xe.resize((size_t) G * 16); xo.resize((size_t) G * 16);
        for (int g = 0; g < G; ++g)
            for (int k = 0; k < 16; ++k) { xe[(size_t) g * 16 + k] = x[g * 32 + 2 * k]; xo[(size_t) g * 16 + k] = x[g * 32 + 2 * k + 1]; }
        const float* pe = xe.data();
        const float* po = xo.data();
#pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < w.rows; i0 += 16) {
            const int i1 = std::min(i0 + 16, w.rows);
            if (lv >= 3) mx_rows_avx512(w, pe, po, i0, i1, y);
            else mx_rows_avx2(w, pe, po, i0, i1, y);
        }
        return;
    }
#endif
    (void) lv;
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < w.rows; i0 += 16) mx_rows_scalar(w, x, i0, std::min(i0 + 16, w.rows), y);
}

std::vector<float> decode_all(const StTensor& t) {
    std::vector<float> o((size_t) t.numel());
    for (size_t i = 0; i < o.size(); ++i) {
        switch (t.dtype) {
            case DType::BF16: o[i] = bf16_to_f32(((const uint16_t*) t.data)[i]); break;
            case DType::F16: o[i] = f16_to_f32(((const uint16_t*) t.data)[i]); break;
            case DType::F32: o[i] = ((const float*) t.data)[i]; break;
            default: o[i] = 0;
        }
    }
    return o;
}

}  // namespace gptoss

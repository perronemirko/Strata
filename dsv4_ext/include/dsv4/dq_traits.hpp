/**
 * @file dq_traits.hpp
 * @brief Per-ggml-type dequantization traits for the CUDA matvec kernel.
 *
 * This header is intentionally designed to compile on BOTH host and device:
 *   - Under nvcc (`__CUDACC__`), each trait method gets `__device__ __host__ __forceinline__`.
 *   - On the host, it compiles as plain `inline` so multiple TUs don't cause ODR violations.
 *
 * @par Device constant memory mapping
 *   The IQ* traits read six codebook arrays through abstract `G_*` macro names. By default these
 *   resolve to the host tables in `dsv4::iq::*`. `src/gpu_cuda.cu` redefines the macros to point
 *   at its `__constant__` memory copies BEFORE including this header, so the exact same source
 *   lines read device constant memory on GPU and host RAM in tests.
 *
 * @par Generic kernel design
 *   Each trait exposes `BE` (block elements), `BB` (block bytes), and `at(w, e)` (element e of a row).
 *   The `at()` method derives its own block offset from `e`, enabling one generic kernel to serve
 *   every type. A thread may read any element independently without knowing the block structure.
 *
 * @par Verification
 *   [`tests/test_dq_traits.cpp`](../../tests/test_dq_traits.cpp) compares every trait, element by
 *   element, against `dequant_row()`/`row_dot()` from `src/dequant.cpp` — on CPU, no GPU or nvcc
 *   needed. This catches wrong nibble shifts before they become silent garbage in the logits.
 *
 * @par Precision agreement
 *   Host reference accumulates in `double` left-to-right; kernel uses `float` tree reduction.
 *   Device and host agree to float rounding, not bit-for-bit.
 */
#pragma once

#include "dsv4/iq_tables.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

// Host macro: supplies `inline`. Under nvcc, the #else branch provides device+host qualifiers.
// Never write `inline` next to DSV4_HOST_DEVICE — it becomes a duplicate specifier.
#ifndef DSV4_HOST_DEVICE
#ifdef __CUDACC__
#define DSV4_HOST_DEVICE __device__ __host__ __forceinline__
#else
#define DSV4_HOST_DEVICE inline
#endif
#endif

// Abstract codebook names: default to host tables; gpu_cuda.cu redefines these for __constant__ memory.
#ifndef G_iq1s_grid
#define G_iq1s_grid dsv4::iq::iq1s_grid
#endif
#ifndef G_iq2xxs_grid
#define G_iq2xxs_grid dsv4::iq::iq2xxs_grid
#endif
#ifndef G_iq3xxs_grid
#define G_iq3xxs_grid dsv4::iq::iq3xxs_grid
#endif
#ifndef G_ksigns_iq2xs
#define G_ksigns_iq2xs dsv4::iq::ksigns_iq2xs
#endif
#ifndef G_kmask_iq2xs
#define G_kmask_iq2xs dsv4::iq::kmask_iq2xs
#endif

/**
 * @def G_kvalues_mxfp4
 * @brief MXFP4 mantissa lookup function (not an array).
 *
 * ggml has no `kvalues_mxfp4` in its `ggml-common.h` IQ table set, so it is defined here once
 * and shared by both backends. Implemented as an inline function because a namespace-scope array
 * has internal linkage and nvcc refuses to read it from device code.
 *
 * Values: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}, where bit 3 = sign.
 * The e8m0 scale already carries the factor of 2 (`e8m0_to_fp32_half`).
 */
#ifndef G_kvalues_mxfp4
#define G_kvalues_mxfp4 dqt::mxfp4_value
#endif

namespace dqt {

/**
 * @brief Convert MXFP4 quantized value (4-bit mantissa) to float.
 *
 * Handles the special case m==0 returning +0.0f (not -0.0f), since ggml stores a literal 0
 * and the trait test compares float BITS where -0.0f != 0.0f.
 * @param q 4-bit quantized value (0-15).
 * @return Float mantissa value.
 */
DSV4_HOST_DEVICE float mxfp4_value(int q) {
    static const int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return (float)values[q & 0xF];
}

// ================================================================= scalar helpers (mirror dequant.cpp)
// DSV4_HOST_DEVICE on every one: nvcc rejects a __host__-only helper called from a kernel, and
// these are the leaf calls of every at() below. The macro already carries `inline` on host.

/**
 * @brief Load 16-bit little-endian unsigned integer via memcpy (no alignment requirement).
 */
DSV4_HOST_DEVICE uint16_t ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Load 32-bit little-endian unsigned integer via memcpy.
 */
DSV4_HOST_DEVICE uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Load 32-bit float via memcpy (reinterprets bytes).
 */
DSV4_HOST_DEVICE float ldff(const uint8_t* p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Convert half-precision (F16) bits to single precision (F32).
 *
 * Handles subnormal (exp==0), Inf/NaN (exp==31), and normal cases.
 */
DSV4_HOST_DEVICE float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x3ffu;
    
    float out;
    if (exp == 0) {
        // Subnormal: value = (-1)^sign * 2^-14 * (frac / 1024)
        out = (float)frac * (1.0f / 16777216.0f);  // frac * 2^-24
    } else if (exp == 31) {
        // Inf or NaN
        out = frac ? NAN : INFINITY;
    } else {
        // Normal: value = (-1)^sign * 2^(exp-15) * (1 + frac/1024)
        const uint32_t bits = (exp << 23) + (frac << 13) + ((127u - 15u) << 23);
        std::memcpy(&out, &bits, 4);
    }
    return sign ? -out : out;
}

/**
 * @brief Convert BFloat16 bits to single precision (F32).
 *
 * Simply sign-extends the 16-bit pattern to 32 bits.
 */
DSV4_HOST_DEVICE float bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/**
 * @brief Convert E8M0 exponent-only float half to F32.
 *
 * Value = 0.5 * 2^(x-127). Used by MXFP4 dequantization where the scale is already in e8m0 form.
 */
DSV4_HOST_DEVICE float e8m0_to_fp32_half(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/**
 * @brief Extract scale and min for K-quants (Q4_K, Q5_K, Q6_K, Q8_K).
 *
 * Decodes the per-sub-block scale (`*d`) and minimum (`*m`) from the packed byte layout.
 * For j < 4: direct extraction; for j >= 4: combines bits from adjacent bytes.
 */
DSV4_HOST_DEVICE void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { 
        *d = q[j] & 63; 
        *m = q[j + 4] & 63; 
    } else { 
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); 
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); 
    }
}

// ================================================================= dequantization traits
/**
 * @struct F32T
 * @brief Full float32 trait: 1 element per block, 4 bytes per block.
 */
struct F32T  { static constexpr int BE = 1,  BB = 4;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) { return ldff(w + 4 * (int64_t) e); } };

/**
 * @struct BF16T
 * @brief BFloat16 trait: 1 element per block, 2 bytes per block.
 */
struct BF16T { static constexpr int BE = 1,  BB = 2;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) { return bf16_to_f32(ld16(w + 2 * (int64_t) e)); } };

/**
 * @struct Q8_0T
 * @brief Q8_0 trait: 32 elements per block, 34 bytes per block (2-byte scale + 32 int8 values).
 */
struct Q8_0T { static constexpr int BE = 32, BB = 34;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 5) * BB;
        return f16_to_f32(ld16(b)) * (float)((const int8_t*)(b + 2))[e & 31];
    } };

/**
 * @struct Q4_KT
 * @brief Q4_K trait: 256 elements per block, 144 bytes.
 *
 * Layout: [d f16 | min f16 | scales[3*8] | qs[128]] — four sub-blocks of 64 per 256-element block.
 */
struct Q4_KT { static constexpr int BE = 256, BB = 144;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; 
        const uint8_t* q = b + 16;
        
        // Determine which sub-block and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int chunk = j >> 6;        // Which 64-element chunk (0-3)
        const int l = j & 63;            // Position within chunk (0-63)
        
        // Get scale and min for this sub-block
        uint8_t s, mm;
        get_scale_min_k4(2 * chunk + (l >= 32 ? 1 : 0), sc, &s, &mm);
        const float d1 = d * (float)s;
        const float m1 = min * (float)mm;
        
        // Get the quantized value
        const uint8_t* q_ptr = q + chunk * 32;
        const int idx = l & 31;          // Index within 32-byte half
        const bool upper_half = (l >= 32);
        
        if (!upper_half) {
            return d1 * (float)(q_ptr[idx] & 0xF) - m1;
        } else {
            return d1 * (float)(q_ptr[idx] >> 4) - m1;
        }
    } };

/**
 * @struct Q5_KT
 * @brief Q5_K trait: 256 elements per block, 176 bytes.
 *
 * Layout: [d f16 | min f16 | scales[3*8] | qh[32] | ql[64]] — five bits split into 4-bit low + 1-bit high.
 *
 * The 32 qh bytes carry one high bit per element of their 64-element chunk: the lower half reads
 * bit 2c, the upper half bit 2c+1 (chunk c = 0..3), exactly as ggml's u1/u2 shift by two each round.
 */
struct Q5_KT { static constexpr int BE = 256, BB = 176;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; 
        const uint8_t* qh = b + 16; 
        const uint8_t* ql = b + 48;
        
        // Determine which sub-block and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int chunk = j >> 6;        // Which 64-element chunk (0-3)
        const int l = j & 63;            // Position within chunk (0-63)
        
        // Get scale and min for this sub-block
        uint8_t s, mm;
        get_scale_min_k4(2 * chunk + (l >= 32 ? 1 : 0), sc, &s, &mm);
        const float d1 = d * (float)s;
        const float m1 = min * (float)mm;
        
        // Get the quantized value (5 bits: 4 low + 1 high). The high bit is NOT fixed per half:
        // ggml's u1/u2 advance two positions in qh for every 64-element chunk, so chunk c reads
        // bit 2c of the lower half and bit 2c+1 of the upper half (u1 = 1 << 2c, u2 = 2 << 2c).
        const uint8_t* ql_ptr = ql + chunk * 32;
        const int idx = l & 31;          // Index within 32-byte half
        const bool upper_half = (l >= 32);
        const uint8_t u1 = (uint8_t) (1u << (2 * chunk)), u2 = (uint8_t) (2u << (2 * chunk));

        if (!upper_half) {
            const uint8_t low_nib = ql_ptr[idx] & 0xF;
            const uint8_t high_bit = qh[idx] & u1 ? 16 : 0;
            return d1 * (float)(low_nib + high_bit) - m1;
        } else {
            const uint8_t low_nib = ql_ptr[idx] >> 4;
            const uint8_t high_bit = qh[idx] & u2 ? 16 : 0;
            return d1 * (float)(low_nib + high_bit) - m1;
        }
    } };

/**
 * @struct Q6_KT
 * @brief Q6_K trait: 256 elements per block, 210 bytes.
 *
 * Two 128-element halves per block; each half: 64 ql bytes, 32 qh bytes, 4 super-block scales.
 * Six bits split into 4-bit low nibble + 2-bit high nibble.
 */
struct Q6_KT { static constexpr int BE = 256, BB = 210;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b + 208));
        
        // Determine which half and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int half = j >> 7;         // Which 128-element half (0 or 1)
        const int rem = j & 127;         // Position within half (0-127)
        
        // Get the quantized value (6 bits: 4 low + 2 high)
        const uint8_t* ql = b + half * 64;
        const uint8_t* qh = b + 128 + half * 32;
        const int8_t* sc = (const int8_t*)(b + 192) + half * 8;
        
        const int grp = rem >> 5;        // Which group of 32 (0-3)
        const int l = rem & 31;          // Position within group
        
        const int qidx = (grp & 1) ? (l + 32) : l;
        const int nsh = (grp >> 1) * 4;
        const int hsh = grp * 2;
        
        const int8_t q = (int8_t)(((ql[qidx] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32;
        return d * (float)sc[(l >> 4) + 2 * grp] * (float)q;
    } };

/**
 * @struct IQ3_XXST
 * @brief IQ3_XXS trait: 256 elements per block, 98 bytes.
 *
 * Uses ksigns_iq2xs and iq3xxs_grid codebooks from the generated tables.
 */
struct IQ3_XXST { static constexpr int BE = 256, BB = 98;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b));
        const uint8_t* qs = b + 2; 
        const uint8_t* sss = qs + 64;
        
        // Determine which sub-block and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int ib32 = j >> 5;         // Which 32-element group (0-7)
        const int l = (j & 31) >> 3;     // Which 8-element sub-group (0-3)
        const int jj = j & 7;            // Position within 8 elements
        
        uint32_t aux32; 
        std::memcpy(&aux32, sss + 4 * ib32, sizeof(aux32));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        
        const uint8_t signs = G_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t* g = (const uint8_t*)(G_iq3xxs_grid + qs[ib32 * 8 + 2 * l + ((jj >= 4) ? 1 : 0)]);
        
        return db * (float)g[jj & 3] * ((signs & G_kmask_iq2xs[jj]) ? -1.f : 1.f);
    } };

/**
 * @struct IQ2_XXST
 * @brief IQ2_XXS trait: 256 elements per block, 66 bytes.
 */
struct IQ2_XXST { static constexpr int BE = 256, BB = 66;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)); 
        const uint8_t* qs = b + 2;
        
        // Determine which sub-block and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int ib32 = j >> 5;         // Which 32-element group (0-7)
        const int l = (j & 31) >> 3;     // Which 8-element sub-group (0-3)
        const int jj = j & 7;            // Position within 8 elements
        
        uint32_t aux32[2]; 
        std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));
        const uint8_t* aux8 = (const uint8_t*)aux32;
        
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        const uint8_t* g = (const uint8_t*)(G_iq2xxs_grid + aux8[l]);
        const uint8_t signs = G_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
        
        return db * (float)g[jj] * ((signs & G_kmask_iq2xs[jj]) ? -1.f : 1.f);
    } };

/**
 * @struct IQ1_MT
 * @brief IQ1_M trait: 256 elements per block, 56 bytes. DeepSeek-V4-Flash primary quantization.
 *
 * Four 16-bit scales packed into a single uint16_t array, with sub-block delta multipliers
 * and a ±0.125 offset (kIq1sDelta) applied per sub-block.
 */
struct IQ1_MT { static constexpr int BE = 256, BB = 56;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const uint8_t* qs = b; 
        const uint8_t* qhb = b + 32; 
        const uint8_t* scb = b + 48;
        
        // Determine which sub-block and position within it
        const int j = e & 255;           // Position in block (0-255)
        const int ib = j >> 5;           // Which 32-element group (0-7)
        const int l = (j & 31) >> 3;     // Which 8-element sub-group (0-3)
        const int jj = j & 7;            // Position within 8 elements
        
        const uint16_t sc[4] = {ld16(scb), ld16(scb + 2), ld16(scb + 4), ld16(scb + 6)};
        const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = f16_to_f32(su);
        
        const uint8_t qh0 = qhb[2 * ib], qh1 = qhb[2 * ib + 1];
        const int sel = (l < 2) ? 0 : 3;
        const float dl = d * (2.f * (float)((sc[ib / 2] >> (6 * (ib % 2) + sel)) & 0x7) + 1.f);
        
        const uint16_t idx = (uint16_t)(qs[ib * 4 + l] |
                                        ((l == 0) ? ((qh0 << 8) & 0x700) : (l == 1) ? ((qh0 << 4) & 0x700) :
                                         (l == 2) ? ((qh1 << 8) & 0x700) : ((qh1 << 4) & 0x700)));
        
        const bool neg = (l == 0) ? (qh0 & 0x08) : (l == 1) ? (qh0 & 0x80) :
                         (l == 2) ? (qh1 & 0x08) : (qh1 & 0x80);
        const float delta = neg ? -0.125f : 0.125f;
        
        const int8_t* g = (const int8_t*)(G_iq1s_grid + idx);
        return dl * ((float)g[jj] + delta);
    } };

/**
 * @struct MXFP4T
 * @brief MXFP4 trait: 32 elements per block, 17 bytes.
 *
 * E8M0 scale byte + 16 quantized mantissa values (4-bit each, low nibble then high nibble).
 */
struct MXFP4T { static constexpr int BE = 32, BB = 17;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 5) * BB;
        const float d = e8m0_to_fp32_half(b[0]); 
        const uint8_t* qs = b + 1;
        
        // Determine which nibble within the byte
        const int q = e & 15;            // Which quantized value (0-15)
        const int sh = (e & 16) ? 4 : 0; // Shift for high/low nibble
        
        return d * G_kvalues_mxfp4((qs[q] >> sh) & 0x0F);
    } };

}  // namespace dqt

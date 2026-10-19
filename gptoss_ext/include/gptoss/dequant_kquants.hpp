// dequant_kquants.hpp - dequantizzazione di Q8_0, Q4_K, Q6_K verso float.
// Port 1:1 di dequantize_row_q8_0 / q4_K / q6_K di ggml-quants.c (stesse espressioni, stesso ordine).
// Header-only. Le strutture sono identiche a ggml-common.h.
#pragma once
#include <cstdint>
#include <stdexcept>

#include "dequant_iq.hpp" // per iq_fp16_to_fp32

namespace gptoss {

constexpr int64_t kKQ_QK_K = 256;
constexpr int64_t kKQ_QK8_0 = 32;
constexpr int64_t kKQ_K_SCALE_SIZE = 12;

struct block_q8_0 { uint16_t d; int8_t qs[kKQ_QK8_0]; };
struct block_q4_K { uint16_t d; uint16_t dmin; uint8_t scales[kKQ_K_SCALE_SIZE]; uint8_t qs[kKQ_QK_K / 2]; };
struct block_q6_K { uint8_t ql[kKQ_QK_K / 2]; uint8_t qh[kKQ_QK_K / 4]; int8_t scales[kKQ_QK_K / 16]; uint16_t d; };

static_assert(sizeof(block_q8_0) == 2 + 32, "q8_0 layout");
static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "q4_K layout");
static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 2, "q6_K layout");

inline void kq_get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

inline void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
    if (k % kKQ_QK8_0) throw std::invalid_argument("q8_0: k non multiplo di 32");
    const int64_t nb = k / kKQ_QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int j = 0; j < kKQ_QK8_0; ++j) {
            y[i * kKQ_QK8_0 + j] = x[i].qs[j] * d;
        }
    }
}

inline void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k) {
    if (k % kKQ_QK_K) throw std::invalid_argument("q4_K: k non multiplo di QK_K");
    const int64_t nb = k / kKQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d = iq_fp16_to_fp32(x[i].d);
        const float min = iq_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < kKQ_QK_K; j += 64) {
            kq_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            kq_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

inline void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k) {
    if (k % kKQ_QK_K) throw std::invalid_argument("q6_K: k non multiplo di QK_K");
    const int64_t nb = k / kKQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int n = 0; n < kKQ_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

}  // namespace gptoss

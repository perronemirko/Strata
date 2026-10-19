// dequant_q5k.hpp - dequantizzazione di Q5_K verso float. Port 1:1 di dequantize_row_q5_K di ggml-quants.c.
#pragma once
#include <cstdint>
#include <stdexcept>

#include "dequant_iq.hpp"
#include "dequant_kquants.hpp"

namespace gptoss {

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[kKQ_K_SCALE_SIZE];
    uint8_t qh[kKQ_QK_K / 8];
    uint8_t qs[kKQ_QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 2 + 2 + 12 + 32 + 128, "q5_K layout");

inline void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k) {
    if (k % kKQ_QK_K) throw std::invalid_argument("q5_K: k non multiplo di QK_K");
    const int64_t nb = k / kKQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = iq_fp16_to_fp32(x[i].d);
        const float min = iq_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < kKQ_QK_K; j += 64) {
            kq_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            kq_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

}  // namespace gptoss

// dequant_fp4.hpp - dequantizzazione di Q2_0 e MXFP4 verso float.
// Port 1:1 di dequantize_row_q2_0 / dequantize_row_mxfp4 di ggml-quants.c e di ggml_e8m0_to_fp32_half di ggml-impl.h.
// Header-only. Le strutture sono identiche a ggml-common.h.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "dequant_iq.hpp"  // per iq_fp16_to_fp32 e iq::kvalues_fp4 (via iq_tables.hpp)

namespace gptoss {

constexpr int64_t kFP4_QK2_0 = 64;
constexpr int64_t kFP4_QK_MXFP4 = 32;

struct block_q2_0 { uint16_t d; uint8_t qs[kFP4_QK2_0 / 4]; };
struct block_mxfp4 { uint8_t e; uint8_t qs[kFP4_QK_MXFP4 / 2]; };

static_assert(sizeof(block_q2_0) == 2 + 16, "q2_0 layout");
static_assert(sizeof(block_mxfp4) == 1 + 16, "mxfp4 layout");

// E8M0 a float, dimezzato (come ggml_e8m0_to_fp32_half). I valori MXFP4 sono raddoppiati.
inline float ggml_e8m0_to_fp32_half_port(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline void dequantize_row_q2_0(const block_q2_0* x, float* y, int64_t k) {
    if (k % kFP4_QK2_0) throw std::invalid_argument("q2_0: k non multiplo di 64");
    const int64_t nb = k / kFP4_QK2_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int j = 0; j < kFP4_QK2_0; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i * kFP4_QK2_0 + j] = ((int) q - 1) * d;
        }
    }
}

inline void dequantize_row_mxfp4(const block_mxfp4* x, float* y, int64_t k) {
    if (k % kFP4_QK_MXFP4) throw std::invalid_argument("mxfp4: k non multiplo di 32");
    const int64_t nb = k / kFP4_QK_MXFP4;
    for (int64_t i = 0; i < nb; i++) {
        const float d = ggml_e8m0_to_fp32_half_port(x[i].e);
        for (int j = 0; j < kFP4_QK_MXFP4 / 2; ++j) {
            const int8_t x0 = iq::kvalues_fp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = iq::kvalues_fp4[x[i].qs[j] >> 4];
            y[i * kFP4_QK_MXFP4 + j + 0] = x0 * d;
            y[i * kFP4_QK_MXFP4 + j + kFP4_QK_MXFP4 / 2] = x1 * d;
        }
    }
}

}  // namespace gptoss

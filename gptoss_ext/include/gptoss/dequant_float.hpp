// dequant_float.hpp - conversione di F32, F16, BF16 verso float (elementi singoli, nessun blocco).
#pragma once
#include <cstdint>
#include <cstring>

#include "dequant_iq.hpp"

namespace gptoss {

inline void dequantize_row_f32(const float* x, float* y, int64_t k) {
    std::memcpy(y, x, sizeof(float) * static_cast<size_t>(k));
}

// fp16 IEEE -> float: esatto, come iq_fp16_to_fp32.
inline void dequantize_row_f16(const uint16_t* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k; ++i) y[i] = iq_fp16_to_fp32(x[i]);
}

// bfloat16 -> float: i 16 bit alti del float, azzerando i bit bassi.
inline void dequantize_row_bf16(const uint16_t* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        const uint32_t bits = static_cast<uint32_t>(x[i]) << 16;
        std::memcpy(&y[i], &bits, sizeof(float));
    }
}

}  // namespace gptoss

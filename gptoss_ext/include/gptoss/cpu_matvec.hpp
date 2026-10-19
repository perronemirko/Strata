// cpu_matvec.hpp - prodotto matrice-vettore su pesi GGUF quantizzati, in CPU, riga per riga.
// Convenzione GGUF: un tensore con dims [in, out] ha la riga j (lunga `in`, contigua) al byte j*row_bytes.
// Dequantizza una riga alla volta: la matrice intera non viene mai espansa in float.
#pragma once
#include <cstdint>
#include <vector>

#include "dequant_registry.hpp"

namespace gptoss {

// y[j] = sum_i W[j][i] * x[i], per j in [0, out). W è il puntatore ai byte del tensore (o di un esperto).
inline void matvec_quant(int ggml_type, const uint8_t* W, int64_t in, int64_t out, const float* x, float* y,
                         std::vector<float>& row_buf) {
    const size_t rb = dequant_row_bytes(ggml_type, in);
    row_buf.resize(static_cast<size_t>(in));
    for (int64_t j = 0; j < out; ++j) {
        dequantize(ggml_type, W + static_cast<size_t>(j) * rb, row_buf.data(), in);
        float s = 0.0f;
        for (int64_t i = 0; i < in; ++i) s += row_buf[static_cast<size_t>(i)] * x[i];
        y[j] = s;
    }
}

}  // namespace gptoss

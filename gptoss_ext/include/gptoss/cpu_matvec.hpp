// cpu_matvec.hpp - prodotto matrice-vettore su pesi GGUF quantizzati, in CPU, riga per riga.
// Convenzione GGUF: un tensore con dims [in, out] ha la riga j (lunga `in`, contigua) al byte j*row_bytes.
// Dequantizza una riga alla volta: la matrice intera non viene mai espansa in float.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "dequant_registry.hpp"

namespace gptoss {

// y[j] = sum_i W[j][i] * x[i], per j in [0, out). W è il puntatore ai byte del tensore (o di un esperto).
// A/B (DSV4_Q8ACT=1): arrotonda l'attivazione a Q8_K come fa ggml nelle moltiplicazioni con pesi quantizzati.
// Blocchi da 256, scala = -128/max con segno (come ggml_quantize_row_q8_K_ref). Default: spento.
inline bool q8act_enabled() {
    // Default ON: ggml quantizza sempre l'attivazione. DSV4_Q8ACT=0 la disattiva (solo per A/B).
    static const bool on = [] {
        const char* e = std::getenv("DSV4_Q8ACT");
        return !e || std::atoi(e) != 0;
    }();
    return on;
}

inline void q8k_roundtrip(const float* x, int64_t n, float* out) {
    for (int64_t b = 0; b < n; b += 256) {
        float amax = 0.0f, maxv = 0.0f;
        for (int64_t j = 0; j < 256; ++j) {
            const float a = std::fabs(x[b + j]);
            if (a > amax) { amax = a; maxv = x[b + j]; }
        }
        if (amax == 0.0f) {
            for (int64_t j = 0; j < 256; ++j) out[b + j] = 0.0f;
            continue;
        }
        const float iscale = -127.0f / maxv;  // come quantize_row_q8_K_ref in ggml-quants.c
        const float d = 1.0f / iscale;
        for (int64_t j = 0; j < 256; ++j) {
            int v = (int) std::nearbyint(iscale * x[b + j]);
            v = std::min(127, std::max(-128, v));
            out[b + j] = (float) v * d;
        }
    }
}

// Arrotonda a precisione fp16 (11 bit di mantissa, subnormali con passo 2^-24), round-half-even.
inline float round_fp16(float f) {
    if (f == 0.0f) return 0.0f;
    int e = 0;
    std::frexp(f, &e);  // f = m * 2^e, 0.5 <= |m| < 1
    const int qe = std::max(e - 11, -24);
    const float q = std::ldexp(1.0f, qe);
    return std::nearbyint(f / q) * q;
}

// Q8_0 come quantize_row_q8_0_ref: blocchi da 32, d = amax/127 (in fp16), qs = round-half-away(x / d).
inline void q8_0_roundtrip(const float* x, int64_t n, float* out) {
    for (int64_t b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int64_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const float dh = round_fp16(d);
        for (int64_t j = 0; j < 32; ++j) out[b + j] = std::round(x[b + j] * id) * dh;
    }
}

inline void matvec_quant(int ggml_type, const uint8_t* W, int64_t in, int64_t out, const float* x, float* y,
                         std::vector<float>& row_buf) {
    const size_t rb = dequant_row_bytes(ggml_type, in);
    row_buf.resize(static_cast<size_t>(in));
    const float* xp = x;
    static thread_local std::vector<float> xq;
    if (q8act_enabled() && ggml_type != 0 && ggml_type != 1 && ggml_type != 30) {
        if (ggml_type == 8 && in % 32 == 0) {  // Q8_0: attivazione a Q8_0
            xq.resize(static_cast<size_t>(in));
            q8_0_roundtrip(x, in, xq.data());
            xp = xq.data();
        } else if (in % 256 == 0) {  // k-quant e IQ: attivazione a Q8_K
            xq.resize(static_cast<size_t>(in));
            q8k_roundtrip(x, in, xq.data());
            xp = xq.data();
        }
    }
    for (int64_t j = 0; j < out; ++j) {
        dequantize(ggml_type, W + static_cast<size_t>(j) * rb, row_buf.data(), in);
        float s = 0.0f;
        for (int64_t i = 0; i < in; ++i) s += row_buf[static_cast<size_t>(i)] * xp[i];
        y[j] = s;
    }
}

}  // namespace gptoss

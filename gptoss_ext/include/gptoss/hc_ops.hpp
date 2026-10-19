// hc_ops.hpp - primitive delle hyper-connections di DeepSeek-V4 in CPU (port di build_hc_* di deepseek4.cpp).
// Layout: il residuo x ha hc slice da n_embd: x[h*n_embd + e]. La matrice comb ha C[src*hc + dst]
// (dst varia più in fretta, come nel grafo: offset dst + src*hc).
// hc_fn ha GGUF dims [hc*n_embd, mix] (riga m lunga hc*n_embd, contigua), quindi fn[m*hc_dim + k].
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace gptoss {

// RMS norm senza pesi; y = x / sqrt(mean(x^2) + eps).
inline void rms_norm_plain(const float* x, int64_t n, float eps, float* y) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) x[i] * x[i];
    const float inv = 1.0f / std::sqrt((float) (s / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv;
}

inline float sigmoid_f(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// Sinkhorn come nel grafo: softmax sui dst per ogni src, +eps, poi colonne (sui src), poi righe/colonne ripetute.
inline void hc_sinkhorn(float* C, int64_t hc, int iters, float eps) {
    for (int64_t src = 0; src < hc; ++src) {  // softmax lungo dst
        float* row = C + src * hc;
        float mx = row[0];
        for (int64_t d = 1; d < hc; ++d) mx = row[d] > mx ? row[d] : mx;
        float s = 0.0f;
        for (int64_t d = 0; d < hc; ++d) { row[d] = std::exp(row[d] - mx); s += row[d]; }
        for (int64_t d = 0; d < hc; ++d) row[d] /= s;
    }
    for (int64_t k = 0; k < hc * hc; ++k) C[k] += eps;

    auto norm_cols = [&]() {  // per ogni dst, divide per la somma sui src (+eps)
        for (int64_t dst = 0; dst < hc; ++dst) {
            float s = 0.0f;
            for (int64_t src = 0; src < hc; ++src) s += C[src * hc + dst];
            s += eps;
            for (int64_t src = 0; src < hc; ++src) C[src * hc + dst] /= s;
        }
    };
    auto norm_rows = [&]() {  // per ogni src, divide per la somma sui dst (+eps)
        for (int64_t src = 0; src < hc; ++src) {
            float s = 0.0f;
            for (int64_t dst = 0; dst < hc; ++dst) s += C[src * hc + dst];
            s += eps;
            for (int64_t dst = 0; dst < hc; ++dst) C[src * hc + dst] /= s;
        }
    };
    norm_cols();
    for (int i = 1; i < iters; ++i) { norm_rows(); norm_cols(); }
}

// Pre-mix: calcola il residuo ridotto, i pesi di post e la matrice comb.
// fn: [mix][hc*n_embd]; scale: 3 valori; base: mix valori; mix = (2 + hc) * hc.
inline void hc_pre(const float* x, int64_t hc, int64_t n_embd, const float* fn, const float* scale, const float* base,
                   float eps, int sinkhorn_iters, float* out_x, float* out_post, float* out_comb) {
    const int64_t hc_dim = hc * n_embd;
    const int64_t mix = (2 + hc) * hc;
    std::vector<float> flat_norm(static_cast<size_t>(hc_dim));
    rms_norm_plain(x, hc_dim, eps, flat_norm.data());
    std::vector<float> mixes(static_cast<size_t>(mix), 0.0f);
    for (int64_t m = 0; m < mix; ++m) {
        float s = 0.0f;
        for (int64_t k = 0; k < hc_dim; ++k) s += fn[m * hc_dim + k] * flat_norm[k];
        mixes[m] = s;
    }
    // pre: sigmoid(mix*scale0 + base) + eps
    std::vector<float> pre(static_cast<size_t>(hc));
    for (int64_t h = 0; h < hc; ++h) pre[h] = sigmoid_f(mixes[h] * scale[0] + base[h]) + eps;
    // post: 2 * sigmoid(mix*scale1 + base)
    for (int64_t h = 0; h < hc; ++h) out_post[h] = 2.0f * sigmoid_f(mixes[hc + h] * scale[1] + base[hc + h]);
    // comb: mix*scale2 + base, poi Sinkhorn
    for (int64_t q = 0; q < hc * hc; ++q) out_comb[q] = mixes[2 * hc + q] * scale[2] + base[2 * hc + q];
    hc_sinkhorn(out_comb, hc, sinkhorn_iters, eps);
    // residuo ridotto: sum_h pre[h] * x_h
    for (int64_t e = 0; e < n_embd; ++e) {
        float s = 0.0f;
        for (int64_t h = 0; h < hc; ++h) s += pre[h] * x[h * n_embd + e];
        out_x[e] = s;
    }
}

// Post-mix: out[dst] = x_new * post[dst] + sum_src residual[src] * comb[src][dst].
inline void hc_post(const float* x_new, const float* residual, const float* post, const float* comb, int64_t hc,
                    int64_t n_embd, float* out) {
    for (int64_t dst = 0; dst < hc; ++dst) {
        for (int64_t e = 0; e < n_embd; ++e) {
            float s = x_new[e] * post[dst];
            for (int64_t src = 0; src < hc; ++src) s += residual[src * n_embd + e] * comb[src * hc + dst];
            out[dst * n_embd + e] = s;
        }
    }
}

// Head di uscita: fn [hc][hc*n_embd], scale 1 valore, base hc valori. Restituisce il residuo ridotto a n_embd.
inline void hc_head(const float* x, int64_t hc, int64_t n_embd, const float* fn, const float* scale, const float* base,
                    float eps, float* out_x) {
    const int64_t hc_dim = hc * n_embd;
    std::vector<float> flat_norm(static_cast<size_t>(hc_dim));
    rms_norm_plain(x, hc_dim, eps, flat_norm.data());
    std::vector<float> pre(static_cast<size_t>(hc));
    for (int64_t m = 0; m < hc; ++m) {
        float s = 0.0f;
        for (int64_t k = 0; k < hc_dim; ++k) s += fn[m * hc_dim + k] * flat_norm[k];
        pre[m] = sigmoid_f(s * scale[0] + base[m]) + eps;
    }
    for (int64_t e = 0; e < n_embd; ++e) {
        float s = 0.0f;
        for (int64_t h = 0; h < hc; ++h) s += pre[h] * x[h * n_embd + e];
        out_x[e] = s;
    }
}

}  // namespace gptoss

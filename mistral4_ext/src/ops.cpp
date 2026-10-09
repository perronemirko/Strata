#include "m4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace m4 {

void rmsnorm(const float* x, const float* w, float eps, int n, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = (w ? w[i] : 1.0f) * (x[i] * r);
}

void yarn_freqs(int dim, double base, double factor, int64_t orig, double bf, double bs, std::vector<double>& f) {
    const int h = dim / 2;
    f.assign((size_t) h, 0);
    auto cdim = [&](double rot) { return dim * std::log((double) orig / (rot * 2 * M_PI)) / (2 * std::log(base)); };
    double low = std::max(std::floor(cdim(bf)), 0.0), high = std::min(std::ceil(cdim(bs)), (double) (dim - 1));
    if (low == high) high += 0.001;
    for (int i = 0; i < h; ++i) {
        const double pos = std::pow(base, (double) (2 * i) / dim);
        const double extra = 1.0 / pos, inter = 1.0 / (factor * pos);
        const double ramp = std::min(std::max((i - low) / (high - low), 0.0), 1.0);
        f[(size_t) i] = inter * ramp + extra * (1.0 - ramp);
    }
}

void rope_table(const std::vector<double>& f, int seqlen, std::vector<float>& c, std::vector<float>& s) {
    const size_t h = f.size();
    c.assign((size_t) seqlen * h, 0); s.assign((size_t) seqlen * h, 0);
    for (int t = 0; t < seqlen; ++t)
        for (size_t i = 0; i < h; ++i) { const double a = (double) t * f[i]; c[(size_t) t * h + i] = (float) std::cos(a); s[(size_t) t * h + i] = (float) std::sin(a); }
}

void rotary_pairs(float* x, int dim, const float* c, const float* s) {
    for (int i = 0; i < dim / 2; ++i) {
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c[i] - b * s[i];
        x[2 * i + 1] = a * s[i] + b * c[i];
    }
}

void route(const float* logits, int n, int topk, Router fn, bool norm, float scale, int32_t* idx, float* w) {
    std::vector<double> sc((size_t) n);
    if (fn == Router::Softmax) {
        const double m = *std::max_element(logits, logits + n);
        double z = 0;
        for (int i = 0; i < n; ++i) { sc[(size_t) i] = std::exp(logits[i] - m); z += sc[(size_t) i]; }
        for (auto& v : sc) v /= z;
    } else {
        for (int i = 0; i < n; ++i) sc[(size_t) i] = 1.0 / (1.0 + std::exp(-(double) logits[i]));
    }
    std::vector<int> o((size_t) n);
    std::iota(o.begin(), o.end(), 0);
    std::partial_sort(o.begin(), o.begin() + topk, o.end(), [&](int a, int b) {
        return sc[(size_t) a] != sc[(size_t) b] ? sc[(size_t) a] > sc[(size_t) b] : a < b;
    });
    double sum = 0;
    for (int k = 0; k < topk; ++k) { idx[k] = o[(size_t) k]; sum += sc[(size_t) idx[k]]; }
    for (int k = 0; k < topk; ++k) {
        double v = sc[(size_t) idx[k]];
        if (norm) v /= sum;
        w[k] = (float) (v * scale);
    }
}

double llama4_scale(int64_t pos, double beta, int64_t orig) {
    return 1.0 + beta * std::log(1.0 + std::floor((double) pos / (double) orig));
}

double mla_softmax_scale(int qk_head, double factor, double mscale_all_dim, bool with_mscale) {
    double s = 1.0 / std::sqrt((double) qk_head);
    if (with_mscale) { const double m = 0.1 * mscale_all_dim * std::log(factor) + 1.0; s *= m * m; }
    return s;
}

}  // namespace m4

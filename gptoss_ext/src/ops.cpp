#include "gptoss/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gptoss {

void rmsnorm(const float* x, const float* w, float eps, int n, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = (w ? w[i] : 1.0f) * (x[i] * r);
}

double yarn_concentration(double factor) { return 0.1 * std::log(factor) + 1.0; }

void yarn_freqs(int dim, double base, double factor, int64_t orig, double bf, double bs, bool truncate, std::vector<double>& f) {
    const int h = dim / 2;
    f.assign((size_t) h, 0);
    double low = h * std::log((double) orig / (bf * 2 * M_PI)) / std::log(base);
    double high = h * std::log((double) orig / (bs * 2 * M_PI)) / std::log(base);
    if (truncate) { low = std::floor(low); high = std::ceil(high); }
    if (low == high) high += 0.001;
    for (int i = 0; i < h; ++i) {
        const double fr = std::pow(base, (double) i / h);
        const double ramp = std::min(std::max((i - low) / (high - low), 0.0), 1.0);
        f[(size_t) i] = (1.0 / (factor * fr)) * ramp + (1.0 / fr) * (1.0 - ramp);
    }
}

void rope_table(const std::vector<double>& f, int seqlen, double conc, std::vector<float>& c, std::vector<float>& s) {
    const size_t h = f.size();
    c.assign((size_t) seqlen * h, 0); s.assign((size_t) seqlen * h, 0);
    for (int t = 0; t < seqlen; ++t)
        for (size_t i = 0; i < h; ++i) { const double a = (double) t * f[i]; c[(size_t) t * h + i] = (float) (std::cos(a) * conc); s[(size_t) t * h + i] = (float) (std::sin(a) * conc); }
}

void rotary(float* x, int dim, const float* c, const float* s, bool pairs) {
    const int h = dim / 2;
    if (!pairs) {
        for (int i = 0; i < h; ++i) { const float a = x[i], b = x[i + h]; x[i] = a * c[i] - b * s[i]; x[i + h] = b * c[i] + a * s[i]; }
    } else {
        for (int i = 0; i < h; ++i) { const float a = x[2 * i], b = x[2 * i + 1]; x[2 * i] = a * c[i] - b * s[i]; x[2 * i + 1] = a * s[i] + b * c[i]; }
    }
}

void route(const float* logits, int n, int topk, int32_t* idx, float* w) {
    std::vector<int> o((size_t) n);
    std::iota(o.begin(), o.end(), 0);
    std::partial_sort(o.begin(), o.begin() + topk, o.end(), [&](int a, int b) { return logits[a] != logits[b] ? logits[a] > logits[b] : a < b; });
    double mx = logits[o[0]], z = 0;
    for (int k = 0; k < topk; ++k) { idx[k] = o[(size_t) k]; z += std::exp((double) logits[idx[k]] - mx); }
    for (int k = 0; k < topk; ++k) w[k] = (float) (std::exp((double) logits[idx[k]] - mx) / z);
}

float swiglu(float gate, float up, float limit, float alpha) {
    const float g = std::min(gate, limit);
    const float u = std::min(std::max(up, -limit), limit);
    return (u + 1.0f) * (g / (1.0f + std::exp(-alpha * g)));
}

}  // namespace gptoss

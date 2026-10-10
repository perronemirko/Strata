#include "gptoss/weights.hpp"

#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gptoss {

float bf16_to_f32(uint16_t v) { uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return f; }
float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float r;
    if (e == 0) r = std::ldexp((float) m, -24);
    else if (e == 31) r = m ? std::nanf("") : INFINITY;
    else r = std::ldexp((float) (m | 1024), (int) e - 25);
    return s ? -r : r;
}

const float* fp4_lut() {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t;
}

void matvec(const WView& w, const float* x, float* y) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < w.rows; ++i) {
        const uint8_t* r = w.p + (size_t) i * rb;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const int n = w.cols;
        if (w.dt == DType::BF16) {
            const uint16_t* q = (const uint16_t*) r;
            int j = 0;
            for (; j + 4 <= n; j += 4) { a0 += bf16_to_f32(q[j]) * x[j]; a1 += bf16_to_f32(q[j + 1]) * x[j + 1]; a2 += bf16_to_f32(q[j + 2]) * x[j + 2]; a3 += bf16_to_f32(q[j + 3]) * x[j + 3]; }
            for (; j < n; ++j) a0 += bf16_to_f32(q[j]) * x[j];
        } else if (w.dt == DType::F32) {
            const float* q = (const float*) r; for (int j = 0; j < n; ++j) a0 += q[j] * x[j];
        } else if (w.dt == DType::F16) {
            const uint16_t* q = (const uint16_t*) r; for (int j = 0; j < n; ++j) a0 += f16_to_f32(q[j]) * x[j];
        }
        y[i] = (a0 + a1) + (a2 + a3);
    }
}

void matvec_mx(const MxView& w, const float* x, float* y) {
    const float* L = fp4_lut();
    static float e8[256]; static bool init = false;
    if (!init) { for (int i = 0; i < 256; ++i) e8[i] = std::ldexp(1.0f, i - 127); init = true; }
    const int G = w.cols / 32;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < w.rows; ++i) {
        const uint8_t* b = w.blocks + (size_t) i * G * 16;
        const uint8_t* s = w.scales + (size_t) i * G;
        float acc = 0;
        for (int g = 0; g < G; ++g) {
            const float* xx = x + g * 32;
            const uint8_t* bb = b + g * 16;
            float p = 0;
            for (int k = 0; k < 16; ++k) p += L[bb[k] & 15] * xx[2 * k] + L[bb[k] >> 4] * xx[2 * k + 1];
            acc += p * e8[s[g]];
        }
        y[i] = acc;
    }
}

std::vector<float> decode_all(const StTensor& t) {
    std::vector<float> o((size_t) t.numel());
    for (size_t i = 0; i < o.size(); ++i) {
        switch (t.dtype) {
            case DType::BF16: o[i] = bf16_to_f32(((const uint16_t*) t.data)[i]); break;
            case DType::F16: o[i] = f16_to_f32(((const uint16_t*) t.data)[i]); break;
            case DType::F32: o[i] = ((const float*) t.data)[i]; break;
            default: o[i] = 0;
        }
    }
    return o;
}

}  // namespace gptoss

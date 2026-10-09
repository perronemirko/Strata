#include "m4/weights.hpp"

#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace m4 {

const float* fp8_e4m3_lut() {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int c = 0; c < 256; ++c) {
            const float s = (c & 0x80) ? -1.0f : 1.0f;
            const int e = (c >> 3) & 15, m = c & 7;
            if (e == 15 && m == 7) t[c] = std::nanf("");
            else if (e == 0) t[c] = s * (float) m / 8.0f * std::ldexp(1.0f, -6);
            else t[c] = s * (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, e - 7);
        }
        init = true;
    }
    return t;
}

float bf16_to_f32(uint16_t v) { uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return f; }
float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float r;
    if (e == 0) r = std::ldexp((float) m, -24);
    else if (e == 31) r = m ? std::nanf("") : INFINITY;
    else r = std::ldexp((float) (m | 1024), (int) e - 25);
    return s ? -r : r;
}
float fp8_scale_apply(float s, bool div) { return div ? 1.0f / s : s; }

static inline float dot_row(const WView& w, const uint8_t* r, const float* x) {
    const int n = w.cols;
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    switch (w.dt) {
        case DType::F8E4M3: {
            const float* L = fp8_e4m3_lut();
            int i = 0;
            for (; i + 4 <= n; i += 4) { acc0 += L[r[i]] * x[i]; acc1 += L[r[i + 1]] * x[i + 1]; acc2 += L[r[i + 2]] * x[i + 2]; acc3 += L[r[i + 3]] * x[i + 3]; }
            for (; i < n; ++i) acc0 += L[r[i]] * x[i];
            break;
        }
        case DType::BF16: {
            const uint16_t* q = (const uint16_t*) r;
            int i = 0;
            for (; i + 4 <= n; i += 4) { acc0 += bf16_to_f32(q[i]) * x[i]; acc1 += bf16_to_f32(q[i + 1]) * x[i + 1]; acc2 += bf16_to_f32(q[i + 2]) * x[i + 2]; acc3 += bf16_to_f32(q[i + 3]) * x[i + 3]; }
            for (; i < n; ++i) acc0 += bf16_to_f32(q[i]) * x[i];
            break;
        }
        case DType::F16: { const uint16_t* q = (const uint16_t*) r; for (int i = 0; i < n; ++i) acc0 += f16_to_f32(q[i]) * x[i]; break; }
        case DType::F32: { const float* q = (const float*) r; for (int i = 0; i < n; ++i) acc0 += q[i] * x[i]; break; }
        default: break;
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

void matvec(const WView& w, const float* x, float* y) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < w.rows; ++i) y[i] = w.scale * dot_row(w, w.p + (size_t) i * rb, x);
}

void decode_row(const WView& w, int row, float* out) {
    const size_t rb = (size_t) w.cols * (size_t) dtype_bytes(w.dt);
    const uint8_t* r = w.p + (size_t) row * rb;
    for (int i = 0; i < w.cols; ++i) {
        float v = 0;
        switch (w.dt) {
            case DType::F8E4M3: v = fp8_e4m3_lut()[r[i]]; break;
            case DType::BF16: v = bf16_to_f32(((const uint16_t*) r)[i]); break;
            case DType::F16: v = f16_to_f32(((const uint16_t*) r)[i]); break;
            case DType::F32: v = ((const float*) r)[i]; break;
            default: break;
        }
        out[i] = v * w.scale;
    }
}

std::vector<float> decode_all(const StTensor& t) {
    WView v; v.p = t.data; v.dt = t.dtype; v.rows = 1; v.cols = (int) t.numel();
    std::vector<float> o((size_t) v.cols);
    decode_row(v, 0, o.data());
    return o;
}

}  // namespace m4

/**
 * @file kv_quant.hpp
 * @brief KV cache quantization primitives for the dsv4 engine (host side).
 *
 * Ported from gptoss_ext's `kv_quant` feature set (`-k` / `-v` / `--key-format` /
 * `--value-format`, `--kv-unified`): the KV cache rows can be stored packed in three
 * formats, each with a per-row scale that attention dequantizes on read:
 *
 *   e4m3 = 1 byte (FP8 E4M3, saturating at +-448)   -> cuts the cache 4x vs f32
 *   f16  = 2 bytes (IEEE half)                       -> cuts it 2x
 *   f32  = 4 bytes (the default; byte-identical to the old engine)
 *
 * DSv4-Flash caches ONE fused vector per token that serves as both key and value, so a row is
 * stored in a single format: `-k` names it (the key role), `-v` must agree with it or the load
 * refuses. `--kv-unified` is accepted for llama.cpp parity (one sequence, one shared cache).
 * The checkpoint always stores f32 rows, so a quantized run restores to full precision.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cctype>
#include <string>

namespace dsv4 {

enum class KVFormat : int {
    E4M3 = 0,   // FP8, 1 byte per element
    F16  = 1,   // IEEE half, 2 bytes per element
    F32  = 2    // float, 4 bytes per element (default)
};

inline size_t kv_format_bytes(KVFormat f) {
    switch (f) {
        case KVFormat::E4M3: return 1;
        case KVFormat::F16:  return 2;
        default:             return 4;
    }
}

/// "e4m3" / "f16" / "f32" (case-insensitive) -> format. Anything else returns false.
inline const char* kv_format_name(KVFormat f) {
    switch (f) { case KVFormat::E4M3: return "e4m3"; case KVFormat::F16: return "f16"; default: return "f32"; }
}
inline bool parse_kv_format(const std::string& s, KVFormat& out) {
    std::string t;
    for (unsigned char ch : s) t.push_back((char) std::tolower(ch));
    if (t == "e4m3" || t == "fp8")  out = KVFormat::E4M3;
    else if (t == "f16" || t == "half") out = KVFormat::F16;
    else if (t == "f32" || t == "float") out = KVFormat::F32;
    else return false;
    return true;
}

// ---------------------------------------------------------------------------
// Software FP8 E4M3 / half conversions (no CUDA dependency: the host cache is packed here,
// and the same byte layout is what a device kernel would dequantize in registers).
// ---------------------------------------------------------------------------

/// float -> e4m3 byte. Round-to-nearest-even on the quantized value; saturates to +-448.
inline uint8_t f32_to_e4m3(float v) {
    if (std::isnan(v)) return 0x7E;                       // NaN payload
    const float a = (float) std::fabs((double) v);
    if (a > 448.f) return (uint8_t)((v < 0.f ? 0x80 : 0x00) | 0x7E);   // saturate to +-448
    uint32_t b = 0; std::memcpy(&b, &v, 4);
    const bool neg = (b >> 31) != 0;
    int e = (int)((b >> 23) & 0xff) - 127;
    uint32_t m = b & 0x7fffffu;
    if (e < -6) {                                          // subnormal: value = m' * 2^-9, m' in [0,8)
        if (e < -10) return (uint8_t)(neg ? 0x80 : 0x00); // rounds to zero
        uint32_t sh = (uint32_t) (-6 - e + 1);             // shift so bit0 of m lands on the 2^-9 place
        uint32_t mm = (m | 0x800000u) >> sh;               // round-to-nearest-even via the guard bit
        return (uint8_t)((neg ? 0x80 : 0x00) | (mm & 0x7));
    }
    int E = e + 7;                                   // bias 7: exponent field = e+7 in [-6..8]
    uint32_t qm = m >> 4;                                  // keep the top 3 mantissa bits
    const uint32_t rem = m & 0xfu, half_bit = (qm & 1u);
    if (rem > 8 || (rem == 8 && half_bit)) ++qm;           // round-to-nearest-even
    if (qm >= 8) { qm -= 8; ++E; }                         // mantissa overflow -> exponent up
    uint32_t out = ((uint32_t)(E & 0xf) << 3) | (qm & 7u);
    return (uint8_t)((neg ? 0x80 : 0x00) | out);
}

/// e4m3 byte -> float.
inline float e4m3_to_f32(uint8_t b) {
    const bool neg = (b & 0x80) != 0;
    const int E = (int)((b >> 3) & 0xf), M = b & 7u;
    float v;
    if (E == 0) v = (float) M * std::ldexp(1.f, -9);       // subnormal: M * 2^-9
    else        v = (1.f + (float) M / 8.f) * std::ldexp(1.f, E - 7);
    return neg ? -v : v;
}

/// float -> IEEE half bits (round-to-nearest-even, saturates to +-inf on overflow).
inline uint16_t f32_to_f16(float v) {
    uint32_t b = 0; std::memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int e = (int)((b >> 23) & 0xff) - 127 + 15;
    uint32_t m = b & 0x7fffffu;
    if (((b >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0)); // inf/nan
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);     // overflow -> +-inf
    if (e <= 0) {                                          // subnormal / zero in half
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t sh = (uint32_t)(1 - e);
        uint32_t qm = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half_bit = (qm & 1u);
        if (rem > ((1u << (sh - 1))) || (rem == (1u << (sh - 1)) && half_bit)) ++qm;
        return (uint16_t)(sign | (qm & 0x3ffu));
    }
    uint32_t qm = m >> 13;
    const uint32_t rem = m & 0x1fffu, half_bit = (qm & 1u);
    if (rem > 0x1000 || (rem == 0x1000 && half_bit)) ++qm;
    if (qm >= 0x400) { qm -= 0x400; ++e; }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (qm & 0x3ffu));
}

/// IEEE half bits -> float.
inline float f16_to_f32(uint16_t h) {
    const bool neg = (h & 0x8000) != 0;
    int e = (int)((h >> 10) & 0x1f);
    uint32_t m = h & 0x3ffu;
    float v;
    if (e == 0)      v = (float) m * std::ldexp(1.f, -24);
    else if (e < 0x1f) v = (1.f + (float) m / 1024.f) * std::ldexp(1.f, e - 15);
    else             v = m ? NAN : INFINITY;
    return neg ? -v : v;
}

// ---------------------------------------------------------------------------
// Row packing: one scale per row (per token), shared by all its elements.
// ---------------------------------------------------------------------------

/// Pack `n` floats into packed bytes of `fmt`. Returns the row scale s such that
/// unpack(packed, 1/s) ~= src. An all-zero row gets scale 0 and an all-zero output.
inline float kv_pack_row(const float* src, int n, KVFormat fmt, uint8_t* dst) {
    if (fmt == KVFormat::F32) { std::memcpy(dst, src, (size_t) n * 4); return 1.f; }
    double amax = 0.0;
    for (int i = 0; i < n; ++i) amax = std::max(amax, (double) std::fabs(src[i]));
    if (amax == 0.0) { std::memset(dst, 0, kv_format_bytes(fmt) * (size_t) n); return 0.f; }
    const float target = (fmt == KVFormat::E4M3) ? 448.f : 65504.f;   // e4m3 max / f16 max
    const float s = (float) (target / amax);                          // quantize v * s
    if (fmt == KVFormat::E4M3) {
        for (int i = 0; i < n; ++i) dst[i] = f32_to_e4m3(src[i] * s);
    } else {
        uint16_t* h = (uint16_t*) dst;
        for (int i = 0; i < n; ++i) h[i] = f32_to_f16(src[i] * s);
    }
    return s;
}

/// Unpack packed bytes of `fmt` with row scale `s`: out[i] = unpacked(i) / s.
inline void kv_unpack_row(const uint8_t* src, int n, KVFormat fmt, float s, float* dst) {
    if (fmt == KVFormat::F32) { std::memcpy(dst, src, (size_t) n * 4); return; }
    const float inv = (s != 0.f) ? 1.f / s : 0.f;
    if (fmt == KVFormat::E4M3) {
        for (int i = 0; i < n; ++i) dst[i] = e4m3_to_f32(src[i]) * inv;
    } else {
        const uint16_t* h = (const uint16_t*) src;
        for (int i = 0; i < n; ++i) dst[i] = f16_to_f32(h[i]) * inv;
    }
}

/// Worst-case relative error budget of a format, used by tests: e4m3 has 3 mantissa bits.
inline double kv_format_rel_err(KVFormat fmt) {
    switch (fmt) {
        case KVFormat::E4M3: return 1.0 / 8.0;   // half an ulp at the top of a binade
        case KVFormat::F16:  return 1.0 / 2048.0;
        default:             return 0.0;
    }
}

}  // namespace dsv4

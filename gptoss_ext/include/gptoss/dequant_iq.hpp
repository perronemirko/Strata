// dequant_iq.hpp - dequantizzazione dei formati IQ (2-3 bit non lineari) verso float.
// Port 1:1 delle funzioni dequantize_row_* di ggml-quants.c: stesse espressioni nello stesso ordine,
// quindi risultati identici bit per bit. Le tabelle vengono da iq_tables.hpp (generato da ggml-common.h).
// Header-only, nessuna dipendenza da llama.cpp a runtime.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "iq_tables.hpp"

namespace gptoss {

constexpr int64_t kIQ_QK_K = 256;
constexpr int64_t kIQ_QK4_NL = 32;
constexpr float kIQ1S_DELTA = 0.125f;

// Layout dei blocchi, identico a ggml-common.h (ggml_half = uint16_t, little-endian).
struct block_iq2_xxs { uint16_t d; uint16_t qs[kIQ_QK_K / 8]; };
struct block_iq2_xs  { uint16_t d; uint16_t qs[kIQ_QK_K / 8]; uint8_t scales[kIQ_QK_K / 32]; };
struct block_iq2_s   { uint16_t d; uint8_t qs[kIQ_QK_K / 4]; uint8_t qh[kIQ_QK_K / 32]; uint8_t scales[kIQ_QK_K / 32]; };
struct block_iq3_xxs { uint16_t d; uint8_t qs[3 * kIQ_QK_K / 8]; };
struct block_iq3_s   { uint16_t d; uint8_t qs[kIQ_QK_K / 4]; uint8_t qh[kIQ_QK_K / 32]; uint8_t signs[kIQ_QK_K / 8]; uint8_t scales[kIQ_QK_K / 64]; };
struct block_iq1_s   { uint16_t d; uint8_t qs[kIQ_QK_K / 8]; uint16_t qh[kIQ_QK_K / 32]; };
struct block_iq1_m   { uint8_t qs[kIQ_QK_K / 8]; uint8_t qh[kIQ_QK_K / 16]; uint8_t scales[kIQ_QK_K / 32]; };
struct block_iq4_nl  { uint16_t d; uint8_t qs[kIQ_QK4_NL / 2]; };
struct block_iq4_xs  { uint16_t d; uint16_t scales_h; uint8_t scales_l[kIQ_QK_K / 64]; uint8_t qs[kIQ_QK_K / 2]; };

static_assert(sizeof(block_iq2_xxs) == 2 + 64, "iq2_xxs layout");
static_assert(sizeof(block_iq2_xs) == 2 + 64 + 8, "iq2_xs layout");
static_assert(sizeof(block_iq2_s) == 2 + 64 + 8 + 8, "iq2_s layout");
static_assert(sizeof(block_iq3_xxs) == 2 + 96, "iq3_xxs layout");
static_assert(sizeof(block_iq3_s) == 2 + 64 + 8 + 32 + 4, "iq3_s layout");
static_assert(sizeof(block_iq1_s) == 2 + 32 + 16, "iq1_s layout");
static_assert(sizeof(block_iq1_m) == 32 + 16 + 8, "iq1_m layout");
static_assert(sizeof(block_iq4_nl) == 2 + 16, "iq4_nl layout");
static_assert(sizeof(block_iq4_xs) == 2 + 2 + 4 + 128, "iq4_xs layout");

// fp16 IEEE -> float, esatto (gestisce subnormali, infiniti, NaN).
inline float iq_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        const float v = (float) mant * (1.0f / 16777216.0f);  // mant * 2^-24, esatto
        uint32_t bits; std::memcpy(&bits, &v, 4);
        bits |= sign;
        float r; std::memcpy(&r, &bits, 4);
        return r;
    }
    uint32_t out;
    if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else           out = sign | ((exp + 112) << 23) | (mant << 13);
    float r; std::memcpy(&r, &out, 4);
    return r;
}

// Semplice helper per leggere i byte di una voce di tabella come nel sorgente (little-endian).
template <typename T>
inline const uint8_t* iq_bytes(const T* table, uint32_t idx) {
    return reinterpret_cast<const uint8_t*>(table + idx);
}

inline void dequantize_row_iq2_xxs(const block_iq2_xxs* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq2_xxs: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    uint32_t aux32[2];
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < kIQ_QK_K / 32; ++ib32) {
            std::memcpy(aux32, reinterpret_cast<const uint8_t*>(x[i].qs) + 8 * ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = iq_bytes(iq::iq2xxs_grid, aux8[l]);
                const uint8_t signs = iq::ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

inline void dequantize_row_iq2_xs(const block_iq2_xs* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq2_xs: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    float db[2];
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < kIQ_QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = x[i].qs[4 * ib32 + l];
                const uint8_t* grid = iq_bytes(iq::iq2xs_grid, q & 511);
                const uint8_t signs = iq::ksigns_iq2xs[q >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * (signs & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

inline void dequantize_row_iq2_s(const block_iq2_s* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq2_s: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    float db[2];
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + kIQ_QK_K / 8;
        for (int ib32 = 0; ib32 < kIQ_QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = iq_bytes(iq::iq2s_grid, qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

inline void dequantize_row_iq3_xxs(const block_iq3_xxs* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq3_xxs: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + kIQ_QK_K / 4;
        for (int ib32 = 0; ib32 < kIQ_QK_K / 32; ++ib32) {
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = iq::ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = iq_bytes(iq::iq3xxs_grid, qs[2 * l + 0]);
                const uint8_t* grid2 = iq_bytes(iq::iq3xxs_grid, qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

inline void dequantize_row_iq3_s(const block_iq3_s* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq3_s: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib32 = 0; ib32 < kIQ_QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = iq_bytes(iq::iq3s_grid, qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256));
                const uint8_t* grid2 = iq_bytes(iq::iq3s_grid, qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = iq_bytes(iq::iq3s_grid, qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256));
                const uint8_t* grid2 = iq_bytes(iq::iq3s_grid, qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

inline void dequantize_row_iq1_s(const block_iq1_s* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq1_s: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = iq_fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint16_t* qh = x[i].qh;
        for (int ib = 0; ib < kIQ_QK_K / 32; ++ib) {
            const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -kIQ1S_DELTA : kIQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(
                        iq_bytes(iq::iq1s_grid, qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

inline void dequantize_row_iq1_m(const block_iq1_m* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq1_m: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    float delta[4];
    uint16_t idx[4];
    for (int64_t i = 0; i < nb; i++) {
        uint16_t sc[4];
        std::memcpy(sc, x[i].scales, sizeof(sc));
        // scala super-blocco: nibble sparsi dei quattro uint16 (come nel sorgente, f16 via u16)
        const uint16_t scale_u16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                               ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = iq_fp16_to_fp32(scale_u16);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        for (int ib = 0; ib < kIQ_QK_K / 32; ++ib) {
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -kIQ1S_DELTA : kIQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -kIQ1S_DELTA : kIQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -kIQ1S_DELTA : kIQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -kIQ1S_DELTA : kIQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(iq_bytes(iq::iq1s_grid, idx[l]));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(iq_bytes(iq::iq1s_grid, idx[l]));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

inline void dequantize_row_iq4_nl(const block_iq4_nl* x, float* y, int64_t k) {
    if (k % kIQ_QK4_NL) throw std::invalid_argument("iq4_nl: k non multiplo di 32");
    const int64_t nb = k / kIQ_QK4_NL;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int j = 0; j < kIQ_QK4_NL / 2; ++j) {
            y[j + 0] = d * iq::kvalues_iq4nl[qs[j] & 0xf];
            y[j + kIQ_QK4_NL / 2] = d * iq::kvalues_iq4nl[qs[j] >> 4];
        }
        y += kIQ_QK4_NL;
    }
}

inline void dequantize_row_iq4_xs(const block_iq4_xs* x, float* y, int64_t k) {
    if (k % kIQ_QK_K) throw std::invalid_argument("iq4_xs: k non multiplo di QK_K");
    const int64_t nb = k / kIQ_QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = iq_fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < kIQ_QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * iq::kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * iq::kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

}  // namespace gptoss

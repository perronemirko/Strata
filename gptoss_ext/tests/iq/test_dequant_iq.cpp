// Test: port (gptoss::dequantize_row_*) vs riferimento (ggml-quants.c, copiato in ref_iq.inc). Confronto bit per bit.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "gptoss/dequant_iq.hpp"

#define GGML_RESTRICT __restrict__
#define GGML_FP16_TO_FP32(x) gptoss::iq_fp16_to_fp32(x)
#define IQ1S_DELTA 0.125f
#define QK_K 256
#define QK4_NL 32
typedef uint16_t ggml_half;
typedef union { ggml_half f16; uint16_t u16; } iq1m_scale_t;

namespace ref {
using namespace gptoss;
using namespace gptoss::iq;
#include "ref_iq.inc"
}

static std::mt19937 rng(12345);

template <typename B>
static void fill_blocks(std::vector<B>& v) {
    for (auto& b : v) {
        uint8_t* p = reinterpret_cast<uint8_t*>(&b);
        for (size_t i = 0; i < sizeof(B); ++i) p[i] = (uint8_t) rng();
    }
}

// Forza il campo d (primi 2 byte) a un fp16 finito, per evitare NaN nel confronto.
template <typename B>
static void fix_d(std::vector<B>& v) {
    for (auto& b : v) {
        uint16_t d = (uint16_t) rng();
        d &= 0x7bff;
        std::memcpy(&b, &d, 2);
    }
}

template <typename B, typename RefFn, typename MyFn>
static bool check(const char* name, int64_t k, RefFn ref_fn, MyFn my_fn, int iters) {
    for (int it = 0; it < iters; ++it) {
        const int64_t per = (name[2] == '4' && name[4] == 'n') ? 32 : 256;
        const int64_t nblk = k / per;
        std::vector<B> blk(nblk);
        fill_blocks(blk);
        fix_d(blk);
        std::vector<float> a(k), b(k);
        ref_fn(blk.data(), a.data(), k);
        my_fn(blk.data(), b.data(), k);
        if (std::memcmp(a.data(), b.data(), sizeof(float) * k) != 0) {
            for (int64_t i = 0; i < k; ++i) {
                if (std::memcmp(&a[i], &b[i], 4) != 0) {
                    std::printf("FAIL %s iter=%d idx=%lld ref=%.9g mio=%.9g\n", name, it, (long long) i, a[i], b[i]);
                    break;
                }
            }
            return false;
        }
    }
    std::printf("OK   %s (%d iterazioni, k=%lld)\n", name, iters, (long long) k);
    return true;
}

int main() {
    const int N = 300;
    const int64_t K = 256 * 16;  // 16 super-blocchi per iterazione
    bool ok = true;
    ok &= check<gptoss::block_iq2_xxs>("iq2_xxs", K, ref::dequantize_row_iq2_xxs, gptoss::dequantize_row_iq2_xxs, N);
    ok &= check<gptoss::block_iq2_xs>("iq2_xs", K, ref::dequantize_row_iq2_xs, gptoss::dequantize_row_iq2_xs, N);
    ok &= check<gptoss::block_iq2_s>("iq2_s", K, ref::dequantize_row_iq2_s, gptoss::dequantize_row_iq2_s, N);
    ok &= check<gptoss::block_iq3_xxs>("iq3_xxs", K, ref::dequantize_row_iq3_xxs, gptoss::dequantize_row_iq3_xxs, N);
    ok &= check<gptoss::block_iq3_s>("iq3_s", K, ref::dequantize_row_iq3_s, gptoss::dequantize_row_iq3_s, N);
    ok &= check<gptoss::block_iq1_s>("iq1_s", K, ref::dequantize_row_iq1_s, gptoss::dequantize_row_iq1_s, N);
    ok &= check<gptoss::block_iq1_m>("iq1_m", K, ref::dequantize_row_iq1_m, gptoss::dequantize_row_iq1_m, N);
    ok &= check<gptoss::block_iq4_nl>("iq4_nl", 32 * 64, ref::dequantize_row_iq4_nl, gptoss::dequantize_row_iq4_nl, N);
    ok &= check<gptoss::block_iq4_xs>("iq4_xs", K, ref::dequantize_row_iq4_xs, gptoss::dequantize_row_iq4_xs, N);
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

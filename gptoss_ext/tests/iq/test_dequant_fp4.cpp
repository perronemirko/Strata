// Test bit-exact: gptoss::dequantize_row_{q2_0,mxfp4} vs riferimento ggml-quants.c / ggml-impl.h (ref_fp4.inc).
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gptoss/dequant_fp4.hpp"

#define GGML_FP16_TO_FP32(x) gptoss::iq_fp16_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)
#define QK2_0 64
#define QK_MXFP4 32
#define kvalues_mxfp4 kvalues_fp4
#define GGML_RESTRICT __restrict__
#include <cstdint>

namespace ref {
using namespace gptoss;
using namespace gptoss::iq;
#include "ref_fp4.inc"
}

static std::mt19937 rng(4242);

template <typename B, typename RefFn, typename MyFn>
static bool check(const char* name, int64_t per_block, RefFn ref_fn, MyFn my_fn, int64_t blocks, int iters) {
    for (int it = 0; it < iters; ++it) {
        std::vector<B> blk(blocks);
        uint8_t* p = reinterpret_cast<uint8_t*>(blk.data());
        for (size_t i = 0; i < sizeof(B) * blocks; ++i) p[i] = (uint8_t) rng();
        for (auto& b : blk) {  // d fp16 finito (solo per q2_0; per mxfp4 il primo byte è l'esponente E8M0)
            uint16_t d = (uint16_t) rng() & 0x7bff;
            if (sizeof(B) == sizeof(gptoss::block_q2_0)) std::memcpy(&b, &d, 2);
        }
        const int64_t k = blocks * per_block;
        std::vector<float> a(k), b(k);
        ref_fn(blk.data(), a.data(), k);
        my_fn(blk.data(), b.data(), k);
        if (std::memcmp(a.data(), b.data(), sizeof(float) * k) != 0) {
            std::printf("FAIL %s iter=%d\n", name, it);
            return false;
        }
    }
    std::printf("OK   %s (%d iterazioni)\n", name, iters);
    return true;
}

int main() {
    bool ok = true;
    ok &= check<gptoss::block_q2_0>("q2_0", 64, ref::dequantize_row_q2_0, gptoss::dequantize_row_q2_0, 32, 300);
    ok &= check<gptoss::block_mxfp4>("mxfp4", 32, ref::dequantize_row_mxfp4, gptoss::dequantize_row_mxfp4, 64, 300);
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

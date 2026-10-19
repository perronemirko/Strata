// Test bit-exact: gptoss::dequantize_row_{q8_0,q4_K,q6_K} vs riferimento ggml-quants.c (ref_kq.inc).
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gptoss/dequant_kquants.hpp"

#define GGML_RESTRICT __restrict__
#define GGML_FP16_TO_FP32(x) gptoss::iq_fp16_to_fp32(x)
#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
#define GGML_COMMON_AGGR_S
#define GGML_COMMON_AGGR_U
#define GGML_EXTENSION

namespace ref {
using namespace gptoss;
#include "ref_kq.inc"
}

static std::mt19937 rng(777);

template <typename B, typename RefFn, typename MyFn>
static bool check(const char* name, int64_t per_block, RefFn ref_fn, MyFn my_fn, int64_t blocks, int iters) {
    for (int it = 0; it < iters; ++it) {
        std::vector<B> blk(blocks);
        uint8_t* p = reinterpret_cast<uint8_t*>(blk.data());
        for (size_t i = 0; i < sizeof(B) * blocks; ++i) p[i] = (uint8_t) rng();
        // d (e dmin per q4_K) finiti: maschera del campo fp16 a inizio blocco
        for (auto& b : blk) {
            uint16_t d = (uint16_t) rng() & 0x7bff;
            std::memcpy(&b, &d, 2);
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
    ok &= check<gptoss::block_q8_0>("q8_0", 32, ref::dequantize_row_q8_0, gptoss::dequantize_row_q8_0, 64, 300);
    ok &= check<gptoss::block_q4_K>("q4_K", 256, ref::dequantize_row_q4_K, gptoss::dequantize_row_q4_K, 16, 300);
    ok &= check<gptoss::block_q6_K>("q6_K", 256, ref::dequantize_row_q6_K, gptoss::dequantize_row_q6_K, 16, 300);
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

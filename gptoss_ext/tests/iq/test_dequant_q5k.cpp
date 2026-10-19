// Test bit-exact: gptoss::dequantize_row_q5_K vs riferimento ggml-quants.c (ref_q5k.inc).
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gptoss/dequant_q5k.hpp"

#define GGML_RESTRICT __restrict__
#define GGML_FP16_TO_FP32(x) gptoss::iq_fp16_to_fp32(x)
#define QK_K 256
#define K_SCALE_SIZE 12

namespace ref {
using namespace gptoss;
#include "ref_q5k.inc"
}

int main() {
    std::mt19937 rng(2024);
    bool ok = true;
    const int iters = 300;
    const int64_t blocks = 16;
    for (int it = 0; it < iters; ++it) {
        std::vector<gptoss::block_q5_K> blk(blocks);
        uint8_t* p = reinterpret_cast<uint8_t*>(blk.data());
        for (size_t i = 0; i < sizeof(gptoss::block_q5_K) * blocks; ++i) p[i] = (uint8_t) rng();
        for (auto& b : blk) {  // d e dmin finiti
            uint16_t d = (uint16_t) rng() & 0x7bff, dm = (uint16_t) rng() & 0x7bff;
            std::memcpy(&b.d, &d, 2);
            std::memcpy(&b.dmin, &dm, 2);
        }
        const int64_t k = blocks * 256;
        std::vector<float> a(k), b(k);
        ref::dequantize_row_q5_K(blk.data(), a.data(), k);
        gptoss::dequantize_row_q5_K(blk.data(), b.data(), k);
        if (std::memcmp(a.data(), b.data(), sizeof(float) * k) != 0) { std::printf("FAIL q5_K iter=%d\n", it); ok = false; break; }
    }
    if (ok) std::printf("OK   q5_K (%d iterazioni)\n", iters);
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

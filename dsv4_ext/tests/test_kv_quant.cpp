// tests/test_kv_quant.cpp - TDD spec for dsv4/kv_quant.hpp (header-only KV row packing).
//
// Behavioural specification:
//   - e4m3/f16/f32 conversions: round-to-nearest-even, saturation, subnormals, exact inverses
//   - kv_pack_row: one scale per row; unpack(packed, 1/s) ~= src; all-zero row -> scale 0
//   - format parsing is case-insensitive; -k and -v must agree (one fused row per token)
//   - memory budget: e4m3 = 1/4 of f32, f16 = 1/2
//
// Built and run by build.sh; needs no model file.

#include "dsv4/kv_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

int main() {
    // ---------------------------------------------------------------- format table
    CHECK(kv_format_bytes(KVFormat::E4M3) == 1);
    CHECK(kv_format_bytes(KVFormat::F16) == 2);
    CHECK(kv_format_bytes(KVFormat::F32) == 4);
    {
        KVFormat f;
        CHECK(parse_kv_format("e4m3", f) && f == KVFormat::E4M3);
        CHECK(parse_kv_format("FP8", f) && f == KVFormat::E4M3);
        CHECK(parse_kv_format("Half", f) && f == KVFormat::F16);
        CHECK(parse_kv_format("f32", f) && f == KVFormat::F32);
        CHECK(!parse_kv_format("q8_0", f));
        CHECK(std::string(kv_format_name(KVFormat::E4M3)) == "e4m3");
    }
    // ---------------------------------------------------------------- e4m3 conversions
    {
        CHECK(e4m3_to_f32(f32_to_e4m3(0.0f)) == 0.0f);
        CHECK(e4m3_to_f32(f32_to_e4m3(1.0f)) == 1.0f);
        CHECK(e4m3_to_f32(f32_to_e4m3(-1.0f)) == -1.0f);
        CHECK(e4m3_to_f32(f32_to_e4m3(448.0f)) == 448.0f);          // max normal, exact
        CHECK(std::fabs((double) e4m3_to_f32(f32_to_e4m3(1000.0f)) - 448.0) < 1e-3);  // saturates
        CHECK(std::fabs((double) e4m3_to_f32(f32_to_e4m3(-1000.0f)) + 448.0) < 1e-3);
        CHECK(e4m3_to_f32(f32_to_e4m3(1e-6f)) == 0.0f);             // rounds to zero
        CHECK(e4m3_to_f32(f32_to_e4m3(0.001f)) >= 0.0f && e4m3_to_f32(f32_to_e4m3(0.001f)) <= 0.002f);  // subnormal band
        // relative error within half an ulp of the 3-bit mantissa
        double worst = 0.0;
        for (int i = 1; i < 200; ++i) {
            const float v = (float) i * 0.137f;
            const float r = e4m3_to_f32(f32_to_e4m3(v));
            worst = std::max(worst, (double) std::fabs(r - v) / v);
        }
        CHECK(worst <= 1.0 / 8.0 + 1e-9);   // the documented budget
    }
    // ---------------------------------------------------------------- f16 conversions
    {
        CHECK(f16_to_f32(f32_to_f16(0.0f)) == 0.0f);
        CHECK(f16_to_f32(f32_to_f16(1.5f)) == 1.5f);
        CHECK(f16_to_f32(f32_to_f16(-3.25f)) == -3.25f);
        CHECK(std::isinf(f16_to_f32(f32_to_f16(1e6f))));            // overflow -> inf
        double worst = 0.0;
        for (int i = 1; i < 500; ++i) {
            const float v = (float) i * 0.0211f;
            const float r = f16_to_f32(f32_to_f16(v));
            worst = std::max(worst, (double) std::fabs(r - v) / v);
        }
        CHECK(worst <= 1.0 / 2048.0 + 1e-9);   // the documented budget
    }
    // ---------------------------------------------------------------- row packing with one scale
    {
        std::vector<float> src(128);
        for (size_t i = 0; i < src.size(); ++i) src[i] = std::sin((float) i * 0.7f) * 3.0f;
        std::vector<uint8_t> packed(src.size());
        const float s = kv_pack_row(src.data(), (int) src.size(), KVFormat::E4M3, packed.data());
        CHECK(s > 0.0f);
        std::vector<float> back(src.size());
        kv_unpack_row(packed.data(), (int) src.size(), KVFormat::E4M3, s, back.data());
        double worst = 0.0;
        for (size_t i = 0; i < src.size(); ++i)
            worst = std::max(worst, (double) std::fabs(back[i] - src[i]) / std::fabs(src[i]));
        CHECK(worst <= 1.0 / 8.0 + 1e-6);
        // f32 packing is the identity
        const float s2 = kv_pack_row(src.data(), (int) src.size(), KVFormat::F32, packed.data());
        CHECK(s2 == 1.0f);
        CHECK(std::memcmp(packed.data(), src.data(), src.size() * 4) == 0);
        // an all-zero row: scale 0, all-zero output, no NaN on the way back
        std::vector<float> zeros(16, 0.0f), out(16, 0.0f);
        std::vector<uint8_t> pz(16);
        const float s3 = kv_pack_row(zeros.data(), 16, KVFormat::E4M3, pz.data());
        CHECK(s3 == 0.0f);
        kv_unpack_row(pz.data(), 16, KVFormat::E4M3, s3, out.data());
        for (float v : out) CHECK(v == 0.0f);
    }
    // ---------------------------------------------------------------- error budgets
    CHECK(kv_format_rel_err(KVFormat::E4M3) == 0.125);
    CHECK(kv_format_rel_err(KVFormat::F16) <= 1.0 / 2048.0);
    CHECK(kv_format_rel_err(KVFormat::F32) == 0.0);

    std::printf("\nkv_quant tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

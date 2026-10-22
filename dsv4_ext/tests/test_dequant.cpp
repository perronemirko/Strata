// tests/test_dequant.cpp - TDD spec for dsv4/dequant.cpp primitives (row_bytes, dequant_row,
// row_dot, row_dots, dequant_supported, dequant_block_elems).
//
// The golden tests (test_ops, test_dq_traits) check the decode math against llama.cpp/numpy.
// This file specifies the CONTRACT of the primitives themselves:
//   - row_bytes: exact block arithmetic, 0 when the element count is not a whole number of blocks
//   - dequant_row == the values row_dot must see (self-consistency per type)
//   - row_dots with T==1 and rows==null is BIT-IDENTICAL to row_dot() (the prefill contract)
//   - unsupported types: row_bytes 0, row_dot 0, dequant_row leaves the buffer untouched
//
// Built and run by build.sh; needs no model file.

#include "dsv4/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

// Build one Q8_0 block (32 int8 + f16 scale) and one Q4_0 block (16 nibble bytes + f16 scale).
static void put_f16(std::vector<uint8_t>& v, float f) {
    uint32_t b = 0; std::memcpy(&b, &f, 4);
    int e = (int)((b >> 23) & 0xff) - 127 + 15;
    uint32_t m = b & 0x7fffffu;
    uint16_t h = (uint16_t)((((b >> 31) & 1u) << 15) | ((uint32_t)(e & 0x1f) << 10) | ((m >> 13) & 0x3ffu));
    v.push_back((uint8_t) h); v.push_back((uint8_t)(h >> 8));
}

int main() {
    // ---------------------------------------------------------------- row_bytes arithmetic
    CHECK(row_bytes(T_Q8_0, 32) == 34);
    CHECK(row_bytes(T_Q8_0, 96) == 3 * 34);
    CHECK(row_bytes(T_Q8_0, 33) == 0);                 // not a whole number of blocks
    CHECK(row_bytes(T_Q4_0, 32) == 18);
    CHECK(row_bytes(T_F32, 7) == 28);
    CHECK(row_bytes(T_IQ1_M, 256) == 56);               // ggml's IQ1_M block: 256 elements, 56 bytes
    CHECK(row_bytes(T_IQ1_M, 512) == 112);
    CHECK(row_bytes(T_IQ1_M, 257) == 0);
    CHECK(dequant_block_elems(T_Q8_0) == 32);
    CHECK(dequant_block_elems(T_F32) == 1);
    CHECK(!dequant_supported(999));

    // ---------------------------------------------------------------- Q8_0 self-consistency
    {
        std::vector<uint8_t> blk;
        put_f16(blk, 0.5f);
        for (int i = 0; i < 32; ++i) blk.push_back((uint8_t)(int8_t)(i - 16));
        std::vector<float> y(32);
        dequant_row(T_Q8_0, blk.data(), 32, y.data());
        CHECK(y[0] == -8.0f);   // (-16) * 0.5
        CHECK(y[31] == 7.5f);   // 15 * 0.5
        // row_dot must see exactly these values
        std::vector<float> x(32, 0.0f);
        for (int i = 0; i < 32; ++i) x[i] = (float) i;
        const float got = row_dot(T_Q8_0, blk.data(), x.data(), 32);
        double ref = 0.0;
        for (int i = 0; i < 32; ++i) ref += (double) y[i] * x[i];
        CHECK(std::fabs((double) got - ref) < 1e-3);
        // row_dots with T==1 is bit-identical to row_dot (the prefill contract)
        float out1 = 0.f;
        row_dots(T_Q8_0, blk.data(), x.data(), 32, nullptr, 1, 32, &out1, 1);
        CHECK(out1 == got);
        // and against gathered rows: row k of x equals row_dot for that row
        std::vector<float> x2(64);
        for (int i = 0; i < 32; ++i) { x2[i] = x[i]; x2[32 + i] = (float)(31 - i); }
        float outs[2] = {};
        row_dots(T_Q8_0, blk.data(), x2.data(), 32, nullptr, 2, 32, outs, 1);
        std::vector<float> xb(32);
        for (int i = 0; i < 32; ++i) xb[i] = x2[32 + i];
        CHECK(outs[1] == row_dot(T_Q8_0, blk.data(), xb.data(), 32));
    }
    // ---------------------------------------------------------------- Q4_0 self-consistency
    {
        std::vector<uint8_t> blk;
        put_f16(blk, 1.0f);
        for (int i = 0; i < 16; ++i) blk.push_back((uint8_t)(((15 - i) << 4) | i));   // lo=i, hi=15-i
        std::vector<float> y(32);
        dequant_row(T_Q4_0, blk.data(), 32, y.data());
        CHECK(y[0] == -8.0f);    // nibble 0 - 8
        CHECK(y[1] == -7.0f);
        CHECK(y[31] == 7.0f);    // nibble 15 - 8
        std::vector<float> x(32, 1.0f);
        double ref = 0.0; for (float v : y) ref += v;
        CHECK(std::fabs((double) row_dot(T_Q4_0, blk.data(), x.data(), 32) - ref) < 1e-3);
    }
    // ---------------------------------------------------------------- unsupported types are inert
    {
        CHECK(row_dot(999, nullptr, nullptr, 32) == 0.0f);
        std::vector<float> y(4, -7.0f);
        dequant_row(999, nullptr, 4, y.data());
        CHECK(y[0] == -7.0f);   // untouched
    }

    std::printf("\ndequant primitive tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

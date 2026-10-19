// Test: F16 contro il tipo _Float16 del compilatore (su tutti i 65536 pattern, NaN esclusi); BF16 e F32 su tutti i pattern.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "gptoss/dequant_float.hpp"

int main() {
    bool ok = true;
    int bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint16_t bits = (uint16_t) h;
        const bool nan = ((bits >> 10) & 0x1f) == 0x1f && (bits & 0x3ff) != 0;
        if (nan) continue;
        _Float16 ref16;
        std::memcpy(&ref16, &bits, 2);
        const float ref = (float) ref16;
        float mine;
        gptoss::dequantize_row_f16(&bits, &mine, 1);
        if (std::memcmp(&ref, &mine, 4) != 0) bad++;
        const uint32_t bf = (uint32_t) bits << 16;
        float ref_bf; std::memcpy(&ref_bf, &bf, 4);
        float mine_bf;
        gptoss::dequantize_row_bf16(&bits, &mine_bf, 1);
        if (std::memcmp(&ref_bf, &mine_bf, 4) != 0) bad++;
    }
    std::printf("%s F16/BF16 su 65536 pattern: %d differenze\n", bad ? "FAIL" : "OK  ", bad);
    ok = bad == 0;
    float f[4] = {1.5f, -2.25f, 0.0f, 3e38f}, g[4];
    gptoss::dequantize_row_f32(f, g, 4);
    ok = ok && std::memcmp(f, g, sizeof f) == 0;
    std::printf("%s F32 copia\n", ok ? "OK  " : "FAIL");
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

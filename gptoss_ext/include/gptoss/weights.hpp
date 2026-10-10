// gptoss/weights.hpp - BF16/F16/F32 matvec and the MXFP4 matvec used by every expert.
#pragma once
#include "gptoss/safetensors.hpp"

#include <cstdint>
#include <vector>

namespace gptoss {

float bf16_to_f32(uint16_t v);
float f16_to_f32(uint16_t v);

/// Row-major [rows x cols] dense matrix over mapped bytes.  y = W x
struct WView { const uint8_t* p = nullptr; int rows = 0, cols = 0; DType dt = DType::BF16; };
void matvec(const WView& w, const float* x, float* y);              // OpenMP over rows (serial when already inside a parallel region)
std::vector<float> decode_all(const StTensor& t);

/// FP4 E2M1 values: {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, ..., -6}
const float* fp4_lut();

/// MXFP4: rows x cols (cols % 32 == 0). blocks: [rows][cols/32][16] bytes (element 2i = low nibble, 2i+1 = high nibble);
/// scales: [rows][cols/32] uint8, value = fp4 * 2^(scale - 127).
struct MxView { const uint8_t* blocks = nullptr; const uint8_t* scales = nullptr; int rows = 0, cols = 0; };
void matvec_mx(const MxView& w, const float* x, float* y);

}  // namespace gptoss

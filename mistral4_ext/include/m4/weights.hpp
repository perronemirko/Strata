// m4/weights.hpp - dtype decoding (F32/F16/BF16/FP8-E4M3) and the matvec every projection goes through.
#pragma once
#include "m4/safetensors.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace m4 {

/// 256-entry E4M3FN decode table (bias 7, 0x7f/0xff = NaN, max 448), built from the format definition.
const float* fp8_e4m3_lut();
float bf16_to_f32(uint16_t v);
float f16_to_f32(uint16_t v);

/// A row-major [rows x cols] matrix view over mapped bytes plus ONE scale: y = scale * (W x).
/// Expert slices and the gate/up halves of a fused tensor are just views with another pointer and scale.
struct WView {
    const uint8_t* p = nullptr;
    int rows = 0, cols = 0;
    DType dt = DType::F32;
    float scale = 1.0f;
};

void matvec(const WView& w, const float* x, float* y);      // OpenMP over rows
void decode_row(const WView& w, int row, float* out);        // one row as float (embeddings, norms)
std::vector<float> decode_all(const StTensor& t);            // a whole (small) tensor as float

/// How the *_scale_inv tensors act on the weights. "mul" (w = fp8 * s) is the DeepSeek/HF-FP8 convention; "div" is its inverse.
/// The sources read for this port do not say which one Mistral's checkpoint uses -> a switch, settled by compare_logits.
float fp8_scale_apply(float s, bool div);

}  // namespace m4

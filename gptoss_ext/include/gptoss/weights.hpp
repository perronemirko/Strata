// gptoss/weights.hpp - BF16/F16/F32 matvec and the MXFP4 matvec used by every expert, with AVX2 / AVX-512 kernels selected at run time.
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
/// y[n][rows] = W x[n][cols] for N inputs: every weight row is streamed ONCE for all N tokens (the CPU analogue of the device matmat).
void matmat(const WView& w, const float* x, int N, float* y);
std::vector<float> decode_all(const StTensor& t);

/// FP4 E2M1 values: {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, ..., -6}
const float* fp4_lut();

/// MXFP4: rows x cols (cols % 32 == 0). blocks: [rows][cols/32][16] bytes (element 2i = low nibble, 2i+1 = high nibble);
/// scales: [rows][cols/32] uint8, value = fp4 * 2^(scale - 127).
struct MxView { const uint8_t* blocks = nullptr; const uint8_t* scales = nullptr; int rows = 0, cols = 0; };
void matvec_mx(const MxView& w, const float* x, float* y);

/// CPU kernels. Auto = the best the host supports (AVX-512F > AVX2+FMA > scalar); the others cap it (never above what the CPU has).
enum class CpuIsa { Auto = 0, Scalar = 1, Avx2 = 2, Avx512 = 3 };
void set_cpu_isa(CpuIsa v);
const char* cpu_isa_name();
/// Reference (always scalar) versions: what the SIMD kernels are tested against.
void matvec_scalar(const WView& w, const float* x, float* y);
void matvec_mx_scalar(const MxView& w, const float* x, float* y);

}  // namespace gptoss

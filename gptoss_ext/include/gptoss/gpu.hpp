// gptoss/gpu.hpp - device layer, TWO interchangeable backends behind one API (same design as dsv4_ext):
//   src/gpu.cpp       CPU-emulated device (malloc "VRAM", memcpy "H2D"). Always buildable; what the tests run against.
//   src/gpu_cuda.cu   real CUDA device (sh build.sh --cuda).
// Both define the same gptoss::gpu symbols; a binary links ONE of them, never both.
// Host-pointer interface on purpose: x/y are host arrays, the backend stages them. Weights live on the device (dalloc + h2d).
// NOT thread-safe: the caller (Model) serialises every call.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace gptoss {
namespace gpu {

bool is_emulated();
bool init(int device, std::string& err);
const char* device_name();
void mem_info(size_t* free_bytes, size_t* total_bytes);
void* alloc(size_t n);                       // nullptr on failure (also clears a non-sticky OOM)
void release(void* p);                       // release(nullptr) is a no-op
bool h2d(void* dev, const void* host, size_t n);

/// y[n][rows] = W x[n][cols], n < N.  W: BF16 row-major [rows][cols] on the device.  One pass over W for all N tokens.
bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy);
/// Same for an MXFP4 matrix: blocks [rows][cols/32][16] bytes, scales [rows][cols/32] bytes (both on the device); cols % 32 == 0.
bool matmat_mx(const void* dBlocks, const void* dScales, int rows, int cols, const float* hx, int N, float* hy);

std::string last_error();

}  // namespace gpu
}  // namespace gptoss

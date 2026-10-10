// CPU-emulated device: the "VRAM" is malloc, the matmats call the host kernels. Same API as gpu_cuda.cu.
#include "gptoss/gpu.hpp"
#include "gptoss/weights.hpp"

#include <cstdlib>
#include <cstring>

namespace gptoss {
namespace gpu {

bool is_emulated() { return true; }
bool init(int, std::string&) { return true; }
const char* device_name() { return "CPU-emulated device"; }
void mem_info(size_t* f, size_t* t) { *t = (size_t) 16 << 30; *f = *t; }
void* alloc(size_t n) { return std::malloc(n ? n : 1); }
void release(void* p) { std::free(p); }
bool h2d(void* d, const void* h, size_t n) { std::memcpy(d, h, n); return true; }
std::string last_error() { return ""; }

bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy) {
    WView w; w.p = (const uint8_t*) dW; w.rows = rows; w.cols = cols; w.dt = DType::BF16;
    for (int n = 0; n < N; ++n) gptoss::matvec(w, hx + (size_t) n * cols, hy + (size_t) n * rows);
    return true;
}

bool matmat_mx(const void* dB, const void* dS, int rows, int cols, const float* hx, int N, float* hy) {
    MxView w; w.blocks = (const uint8_t*) dB; w.scales = (const uint8_t*) dS; w.rows = rows; w.cols = cols;
    for (int n = 0; n < N; ++n) gptoss::matvec_mx(w, hx + (size_t) n * cols, hy + (size_t) n * rows);
    return true;
}

}  // namespace gpu
}  // namespace gptoss

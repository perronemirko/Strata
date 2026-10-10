// Real CUDA device. Build: sh build.sh --cuda   (CUDA_ARCH=sm_89 etc., default "native").
// Layout/decoding contract = src/weights.cpp: BF16 = high half of a float; MXFP4 = 32-element blocks, 16 bytes, element 2k = low nibble,
// 2k+1 = high nibble, value = fp4[nibble] * 2^(scale-127). One warp per output row; the weights are walked ONCE for up to TILE tokens.
#include "gptoss/gpu.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace gptoss {
namespace gpu {
namespace {

constexpr int TILE = 8;                 // tokens whose accumulators live in registers at once
constexpr int WARPS = 4;                // output rows per block

std::string g_err;
float* g_dx = nullptr;
float* g_dy = nullptr;
size_t g_cx = 0, g_cy = 0;              // capacities in floats
__constant__ float c_fp4[16];

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    g_err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

__global__ void k_bf16(const unsigned short* __restrict__ W, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;                                   // whole warp leaves together: the shuffles below stay legal
    const unsigned short* w = W + (size_t) row * cols;
    for (int n0 = 0; n0 < N; n0 += TILE) {
        const int nn = min(TILE, N - n0);
        float acc[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;
        for (int c = lane; c < cols; c += 32) {
            const float wv = __uint_as_float(((unsigned) w[c]) << 16);
#pragma unroll
            for (int t = 0; t < TILE; ++t)
                if (t < nn) acc[t] += wv * x[(size_t) (n0 + t) * cols + c];
        }
#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t < nn) y[(size_t) (n0 + t) * rows + row] = v;
        }
    }
}

__global__ void k_mx(const uint8_t* __restrict__ B, const uint8_t* __restrict__ S, int rows, int cols, const float* __restrict__ x, int N, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int G = cols >> 5;
    const uint8_t* b = B + (size_t) row * G * 16;
    const uint8_t* s = S + (size_t) row * G;
    for (int n0 = 0; n0 < N; n0 += TILE) {
        const int nn = min(TILE, N - n0);
        float acc[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;
        for (int g = lane; g < G; g += 32) {                   // lanes stride over the row's 32-element blocks
            const float sc = ldexpf(1.0f, (int) s[g] - 127);
            const uint4 raw = *reinterpret_cast<const uint4*>(b + (size_t) g * 16);   // 16 bytes; slot/row offsets are 16-aligned
            const uint8_t* q = reinterpret_cast<const uint8_t*>(&raw);
            float wv[32];
#pragma unroll
            for (int k = 0; k < 16; ++k) { wv[2 * k] = c_fp4[q[k] & 15] * sc; wv[2 * k + 1] = c_fp4[q[k] >> 4] * sc; }
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                if (t < nn) {
                    const float* xx = x + (size_t) (n0 + t) * cols + (size_t) g * 32;
                    float p = 0.0f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) p += wv[j] * xx[j];
                    acc[t] += p;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t < nn) y[(size_t) (n0 + t) * rows + row] = v;
        }
    }
}

bool scratch(size_t nx, size_t ny) {
    if (nx > g_cx) { cudaFree(g_dx); g_dx = nullptr; g_cx = 0; if (!ck(cudaMalloc((void**) &g_dx, nx * sizeof(float)), "cudaMalloc x")) return false; g_cx = nx; }
    if (ny > g_cy) { cudaFree(g_dy); g_dy = nullptr; g_cy = 0; if (!ck(cudaMalloc((void**) &g_dy, ny * sizeof(float)), "cudaMalloc y")) return false; g_cy = ny; }
    return true;
}

}  // namespace

bool is_emulated() { return false; }
std::string last_error() { return g_err; }

bool init(int device, std::string& err) {
    int n = 0;
    if (!ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount") || n == 0) { err = g_err.empty() ? "no CUDA device" : g_err; return false; }
    if (!ck(cudaSetDevice(device), "cudaSetDevice")) { err = g_err; return false; }
    const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    if (!ck(cudaMemcpyToSymbol(c_fp4, lut, sizeof lut), "fp4 table")) { err = g_err; return false; }
    return true;
}

const char* device_name() {
    static cudaDeviceProp p;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&p, d) != cudaSuccess) return "CUDA device";
    return p.name;
}

void mem_info(size_t* f, size_t* t) {
    *f = 0; *t = 0;
    if (!ck(cudaMemGetInfo(f, t), "cudaMemGetInfo")) { *f = 0; *t = 0; }
}

void* alloc(size_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) { cudaGetLastError(); return nullptr; }   // OOM is not sticky: clear it
    return p;
}

void release(void* p) { if (p) cudaFree(p); }

bool h2d(void* d, const void* h, size_t n) { return ck(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice), "h2d"); }

bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy) {
    if (N <= 0) return true;
    if (!scratch((size_t) N * cols, (size_t) N * rows)) return false;
    if (!ck(cudaMemcpy(g_dx, hx, (size_t) N * cols * sizeof(float), cudaMemcpyHostToDevice), "x h2d")) return false;
    const int blocks = (rows + WARPS - 1) / WARPS;
    k_bf16<<<blocks, WARPS * 32>>>((const unsigned short*) dW, rows, cols, g_dx, N, g_dy);
    if (!ck(cudaGetLastError(), "k_bf16 launch")) return false;
    return ck(cudaMemcpy(hy, g_dy, (size_t) N * rows * sizeof(float), cudaMemcpyDeviceToHost), "y d2h");   // synchronises
}

bool matmat_mx(const void* dB, const void* dS, int rows, int cols, const float* hx, int N, float* hy) {
    if (N <= 0) return true;
    if (cols % 32) { g_err = "matmat_mx: cols % 32 != 0"; return false; }
    if (!scratch((size_t) N * cols, (size_t) N * rows)) return false;
    if (!ck(cudaMemcpy(g_dx, hx, (size_t) N * cols * sizeof(float), cudaMemcpyHostToDevice), "x h2d")) return false;
    const int blocks = (rows + WARPS - 1) / WARPS;
    k_mx<<<blocks, WARPS * 32>>>((const uint8_t*) dB, (const uint8_t*) dS, rows, cols, g_dx, N, g_dy);
    if (!ck(cudaGetLastError(), "k_mx launch")) return false;
    return ck(cudaMemcpy(hy, g_dy, (size_t) N * rows * sizeof(float), cudaMemcpyDeviceToHost), "y d2h");
}

}  // namespace gpu
}  // namespace gptoss

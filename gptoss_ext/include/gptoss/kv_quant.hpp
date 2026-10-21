#ifndef GPTOSS_KV_QUANT_HPP
#define GPTOSS_KV_QUANT_HPP
// gptoss/kv_quant.hpp - KV quantization primitives, shared by src/gpu_cuda.cu and src/prefill_dev.cu.
// ONE definition per template here: the two TUs include this header, so a duplicate in a .cu would
// shadow it and produce the "declared but never referenced" #177-D warnings.
//
// The card's KV cache holds packed rows (fp8 e4m3 = 1B, half = 2B, float = 4B); the attention
// kernels read those bytes and dequantize into registers, so the fp32 rows never exist in VRAM.
// The scale is per row (per token, shared by its kv_heads): K and V carry independent scales,
// set by -k / -v on the command line.
#pragma once

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace gptoss {
namespace gpu {

enum class KVFormat : int {
    E4M3 = 0,
    F16  = 1,
    F32  = 2
};

inline size_t kv_format_bytes(KVFormat f) {
    switch (f) {
        case KVFormat::E4M3: return 1;
        case KVFormat::F16:  return 2;
        default:             return 4;
    }
}

// ============================================================================
// LOGICA DI PACKING / UNPACKING (ONE definition, shared)
// ============================================================================
template<typename T_OUT>
__device__ __forceinline__ T_OUT pack_fp32_to_target(float val, float scale) {
    return static_cast<T_OUT>(val * scale);
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 pack_fp32_to_target<__nv_fp8_e4m3>(float val, float scale) {
    return __nv_fp8_e4m3(val * scale);
}

template<>
__device__ __forceinline__ half pack_fp32_to_target<half>(float val, float scale) {
    return __float2half(val * scale);
}

template<typename T_IN>
__device__ __forceinline__ float unpack_target_to_fp32(T_IN val, float inv_scale) {
    return static_cast<float>(val) * inv_scale;
}

template<>
__device__ __forceinline__ float unpack_target_to_fp32<__nv_fp8_e4m3>(__nv_fp8_e4m3 val, float inv_scale) {
    return static_cast<float>(val) * inv_scale;
}

template<>
__device__ __forceinline__ float unpack_target_to_fp32<half>(half val, float inv_scale) {
    return __half2float(val) * inv_scale;
}

}  // namespace gpu
}  // namespace gptoss

#endif  // GPTOSS_KV_QUANT_HPP

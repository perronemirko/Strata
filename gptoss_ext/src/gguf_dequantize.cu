#include "gptoss/gguf_loader.hpp"
#include <cuda_fp16.h>
#include <cstdio>

namespace gptoss {

// Kernel CUDA: Dequantizza i blocchi Q4_K_L
__global__ void dequantize_q4_k_l_kernel(const block_q4_k_l* __restrict__ src, __half* __restrict__ dst, int64_t num_blocks) {
    int64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const block_q4_k_l& blk = src[block_idx];
    __half* dst_block = dst + (block_idx * 32);

    float scale_f = __half2float(blk.scale);
    float min_f   = __half2float(blk.min);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t byte_val = blk.qs[i];
        
        uint8_t low = byte_val & 0x0F;
        float val_low = (static_cast<float>(low) * scale_f) + min_f;
        dst_block[i * 2] = __float2half(val_low);

        uint8_t high = byte_val >> 4;
        float val_high = (static_cast<float>(high) * scale_f) + min_f;
        dst_block[i * 2 + 1] = __float2half(val_high);
    }
}

// Aggiornato il puntatore di destinazione a __half
bool GgufLoader::dequantize_tensor_async(const std::string& tensor_name, __half* dst_gpu_ptr, cudaStream_t stream) {
    if (!tensors_.count(tensor_name)) {
        return false;
    }

    const Gguftensor& t = tensors_[tensor_name];
    
    int64_t num_elements = 1;
    for (int64_t dim : t.shape) num_elements *= dim;
    
    int64_t num_blocks = num_elements / 32;

    const block_q4_k_l* src_gpu_mapped = static_cast<const block_q4_k_l*>(t.data_ptr);

    int threads_per_block = 256;
    int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    dequantize_q4_k_l_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        src_gpu_mapped, 
        dst_gpu_ptr, 
        num_blocks
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess);
}

} // namespace gptoss

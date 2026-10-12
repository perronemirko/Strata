#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // <--- AGGIUNGI QUESTO HEADER

namespace gptoss {

// Struttura di un blocco Q4_K_L (32 elementi quantizzati a 4-bit con scala e minimo a 16-bit)
struct alignas(8) block_q4_k_l {
    __half scale;       // <--- MODIFICA DA half A __half
    __half min;         // <--- MODIFICA DA half A __half
    u_int8_t qs[16];     
};

struct Gguftensor {
    std::string name;
    std::vector<int64_t> shape;
    u_int32_t type;    
    size_t offset;    
    size_t size;      
    void* data_ptr;   
};

class GgufLoader {
public:
    GgufLoader();
    ~GgufLoader();

    bool open_file(const std::string& path, std::string& err);
    void close_file();

    bool get_metadata(const std::string& key, std::string& val_out) const;
    bool get_metadata(const std::string& key, int32_t& val_out) const;

    // Modificato anche qui in __half
    bool dequantize_tensor_async(const std::string& tensor_name, __half* dst_gpu_ptr, cudaStream_t stream);

private:
    int fd_;
    size_t file_size_;
    void* mmap_addr_;
    std::unordered_map<std::string, std::string> metadata_str_;
    std::unordered_map<std::string, int32_t> metadata_int_;
    std::unordered_map<std::string, Gguftensor> tensors_;

    bool parse_header(std::string& err);
};

} // namespace gptoss

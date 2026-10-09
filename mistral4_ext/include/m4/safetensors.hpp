// m4/safetensors.hpp - multi-shard safetensors reader (header only + mmap). The Mistral analogue of gguf_header.hpp.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace m4 {

enum class DType { F32, F16, BF16, F8E4M3, I32, Unknown };
const char* dtype_name(DType d);
inline int dtype_bytes(DType d) { switch (d) { case DType::F32: case DType::I32: return 4; case DType::F16: case DType::BF16: return 2; case DType::F8E4M3: return 1; default: return 0; } }

struct StTensor {
    std::string name, dtype_str;
    DType dtype = DType::Unknown;
    std::vector<int64_t> shape;
    int shard = 0;
    uint64_t begin = 0, end = 0;          // offsets inside the shard's data area
    const uint8_t* data = nullptr;        // set once the shard is mapped
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
    uint64_t nbytes() const { return end - begin; }
};

class StFiles {
public:
    StFiles() = default;
    ~StFiles();
    StFiles(const StFiles&) = delete;
    StFiles& operator=(const StFiles&) = delete;

    /// Reads every shard's header, checks offsets against the file size, merges the tensor tables (a name in two shards is an error)
    /// and mmaps the shards read-only. The weights are NOT read: pages come in on first touch.
    bool open(const std::vector<std::string>& paths, std::string& err);
    const StTensor* find(const std::string& name) const;
    const std::vector<StTensor>& tensors() const { return tensors_; }
    uint64_t total_bytes() const;
    void advise_random(bool on);

private:
    struct Map { void* base = nullptr; uint64_t size = 0; int fd = -1; };
    std::vector<Map> maps_;
    std::vector<StTensor> tensors_;
    std::map<std::string, size_t> by_name_;
};

/// A directory (model-*.safetensors preferred, then *.safetensors that are not "consolidated-*") or a single file.
bool discover_shards(const std::string& path, std::vector<std::string>& out, std::string& err);

}  // namespace m4

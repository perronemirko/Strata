#include "gptoss/gguf_loader.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cstdint>

namespace gptoss {

// Magic number standard dei file GGUF ("GGUF" in ASCII)
const u_int32_t GGUF_MAGIC = 0x46554747; 

GgufLoader::GgufLoader() : fd_(-1), file_size_(0), mmap_addr_(nullptr) {}

GgufLoader::~GgufLoader() { close_file(); }

void GgufLoader::close_file() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
    mmap_addr_ = nullptr;
    fd_ = -1;
    tensors_.clear();
}

bool GgufLoader::open_file(const std::string& path, std::string& err) {
    close_file();
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        err = "Impossibile aprire il file GGUF: " + path;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        err = "Impossibile leggere la dimensione del file.";
        close(fd_);
        return false;
    }
    file_size_ = sb.st_size;

    mmap_addr_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        err = "Errore durante l'esecuzione di mmap sul file GGUF.";
        close(fd_);
        return false;
    }

    return parse_header(err);
}

bool GgufLoader::parse_header(std::string& err) {
    u_int8_t* ptr = static_cast<u_int8_t*>(mmap_addr_);
    u_int8_t* end = ptr + file_size_;

    // 1. Validazione Magic Number e Versione
    u_int32_t magic;
    std::memcpy(&magic, ptr, 4); ptr += 4;
    if (magic != GGUF_MAGIC) {
        err = "File non valido: Magic number GGUF non trovato.";
        return false;
    }

    u_int32_t version;
    std::memcpy(&version, ptr, 4); ptr += 4;
    if (version < 2) {
        err = "Versione GGUF non supportata (richiesta >= 2).";
        return false;
    }

    u_int64_t tensor_count, kv_count;
    std::memcpy(&tensor_count, ptr, 8); ptr += 8;
    std::memcpy(&kv_count, ptr, 8); ptr += 8;

    // Helper per leggere stringhe GGUF (lunghezza u_int64 + caratteri)
    auto read_gguf_string = [&](u_int8_t*& p) -> std::string {
        u_int64_t len;
        std::memcpy(&len, p, 8); p += 8;
        std::string s(reinterpret_cast<char*>(p), len);
        p += len;
        return s;
    };

    // 2. Parsing dei Key-Value Metadata
    for (u_int64_t i = 0; i < kv_count; ++i) {
        if (ptr >= end) { err = "Header corrotto durante il parsing dei KV."; return false; }
        std::string key = read_gguf_string(ptr);
        u_int32_t val_type;
        std::memcpy(&val_type, ptr, 4); ptr += 4;

        if (val_type == 4) { // UINT32 / INT32
            int32_t val;
            std::memcpy(&val, ptr, 4); ptr += 4;
            metadata_int_[key] = val;
        } else if (val_type == 8) { // STRING
            std::string val = read_gguf_string(ptr);
            metadata_str_[key] = val;
        } else {
            // Saltiamo i tipi non essenziali (array, bool, float) per snellezza backend
            u_int64_t skip_len = (val_type == 9 || val_type == 5) ? 8 : 4;
            ptr += skip_len; 
        }
    }

    // 3. Parsing delle informazioni sui Tensori
    for (u_int64_t i = 0; i < tensor_count; ++i) {
        if (ptr >= end) { err = "Header corrotto durante il parsing dei tensori."; return false; }
        Gguftensor t;
        t.name = read_gguf_string(ptr);
        
        u_int32_t n_dimensions;
        std::memcpy(&n_dimensions, ptr, 4); ptr += 4;
        
        t.shape.resize(n_dimensions);
        for (u_int32_t d = 0; d < n_dimensions; ++d) {
            std::memcpy(&t.shape[d], ptr, 8); ptr += 8;
        }
        
        std::memcpy(&t.type, ptr, 4); ptr += 4;
        std::memcpy(&t.offset, ptr, 8); ptr += 8;
        
        tensors_[t.name] = t;
    }

    // Calcolo dell'allineamento dei dati binari dei tensori (default GGUF = 32 byte)
    size_t alignment = 32;
    if (metadata_int_.count("general.alignment")) {
        alignment = metadata_int_["general.alignment"];
    }
    
    size_t body_offset = (reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(mmap_addr_));
    body_offset = ((body_offset + alignment - 1) / alignment) * alignment;
    u_int8_t* body_start = static_cast<u_int8_t*>(mmap_addr_) + body_offset;

    // Assegnazione dei puntatori ai dati reali dei tensori
    for (auto& pair : tensors_) {
        pair.second.data_ptr = body_start + pair.second.offset;
    }

    return true;
}

bool GgufLoader::get_metadata(const std::string& key, std::string& val_out) const {
    if (!metadata_str_.count(key)) return false;
    val_out = metadata_str_.at(key);
    return true;
}

bool GgufLoader::get_metadata(const std::string& key, int32_t& val_out) const {
    if (!metadata_int_.count(key)) return false;
    val_out = metadata_int_.at(key);
    return true;
}

} // namespace gptoss

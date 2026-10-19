// gguf_reader.hpp - legge un file GGUF (metadati e tensori) via mmap, senza copiare i pesi.
// Per ogni tensore: nome, tipo GGML, forme, numero di elementi, offset assoluto e dimensione in byte.
// La dimensione è nota per i tipi F32, F16, BF16 e per quelli nel registro dei dequantizzatori;
// per gli altri bytes = 0 (il motore lo segnala, non lo indovina).
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dequant_registry.hpp"

namespace gptoss {

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dims;  // ne[0] è il primo (più interno), come in GGUF
    uint64_t n_elems = 0;
    uint64_t offset = 0;         // offset assoluto nel file
    uint64_t bytes = 0;          // 0 se il tipo non è noto al motore
};

class GgufFile {
public:
    explicit GgufFile(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("gguf: impossibile aprire " + path);
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { release(); throw std::runtime_error("gguf: fstat fallito"); }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ < 24) { release(); throw std::runtime_error("gguf: file troppo piccolo"); }
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) { release(); throw std::runtime_error("gguf: mmap fallito"); }
        base_ = static_cast<const uint8_t*>(p);
        try {
            parse();
        } catch (...) {
            release();
            throw;
        }
    }
    ~GgufFile() { release(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    uint32_t version() const { return version_; }
    uint64_t data_start() const { return data_start_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

    const GgufTensor* find(const std::string& name) const {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &tensors_[it->second];
    }

    // Puntatore ai byte grezzi del tensore (dentro il file mappato, nessuna copia).
    const uint8_t* data(const GgufTensor& t) const { return base_ + t.offset; }

    uint64_t meta_int(const std::string& key, uint64_t def) const {
        auto it = ints_.find(key);
        return it == ints_.end() ? def : static_cast<uint64_t>(it->second);
    }
    double meta_float(const std::string& key, double def) const {
        auto it = floats_.find(key);
        return it == floats_.end() ? def : it->second;
    }
    std::string meta_str(const std::string& key, const std::string& def = "") const {
        auto it = strs_.find(key);
        return it == strs_.end() ? def : it->second;
    }

private:
    void release() {
        if (base_) ::munmap(const_cast<uint8_t*>(base_), size_);
        base_ = nullptr;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

    void need(size_t n) const {
        if (n > size_ - pos_) throw std::runtime_error("gguf: file troncato o corrotto");
    }

    // Lettura little-endian di n byte (n <= 8). Il motore gira su host little-endian.
    uint64_t rd(size_t n) {
        need(n);
        uint64_t v = 0;
        std::memcpy(&v, base_ + pos_, n);
        pos_ += n;
        return v;
    }

    std::string rd_str() {
        const uint64_t n = rd(8);
        if (n > size_ - pos_) throw std::runtime_error("gguf: stringa oltre la fine del file");
        std::string s(reinterpret_cast<const char*>(base_ + pos_), static_cast<size_t>(n));
        pos_ += static_cast<size_t>(n);
        return s;
    }

    static size_t scalar_size(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: return 1;
            case 2: case 3: return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: throw std::runtime_error("gguf: tipo scalare sconosciuto " + std::to_string(t));
        }
    }

    void skip_element(uint32_t t) {
        if (t == 8) { rd_str(); return; }
        if (t == 9) throw std::runtime_error("gguf: array annidato non supportato");
        const size_t n = scalar_size(t);
        need(n);
        pos_ += n;
    }

    void read_kv() {
        const std::string key = rd_str();
        const uint32_t t = static_cast<uint32_t>(rd(4));
        switch (t) {
            case 0:  ints_[key] = static_cast<int64_t>(rd(1)); break;
            case 1:  ints_[key] = static_cast<int64_t>(static_cast<int8_t>(rd(1))); break;
            case 2:  ints_[key] = static_cast<int64_t>(rd(2)); break;
            case 3:  ints_[key] = static_cast<int64_t>(static_cast<int16_t>(rd(2))); break;
            case 4:  ints_[key] = static_cast<int64_t>(rd(4)); break;
            case 5:  ints_[key] = static_cast<int64_t>(static_cast<int32_t>(rd(4))); break;
            case 6: { const uint32_t b = static_cast<uint32_t>(rd(4)); float f; std::memcpy(&f, &b, 4); floats_[key] = f; break; }
            case 7:  rd(1); break;   // bool
            case 8:  strs_[key] = rd_str(); break;
            case 9: {                // array: tipo elementi, conteggio, elementi (scartati)
                const uint32_t et = static_cast<uint32_t>(rd(4));
                const uint64_t n = rd(8);
                for (uint64_t i = 0; i < n; ++i) skip_element(et);
                break;
            }
            case 10: ints_[key] = static_cast<int64_t>(rd(8)); break;
            case 11: ints_[key] = static_cast<int64_t>(rd(8)); break;
            case 12: { const uint64_t b = rd(8); double d; std::memcpy(&d, &b, 8); floats_[key] = d; break; }
            default: throw std::runtime_error("gguf: tipo di metadato sconosciuto " + std::to_string(t));
        }
    }

    static uint64_t type_bytes(uint32_t type, uint64_t n) {
        switch (type) {
            case 24: return n;        // I8
            case 25: return n * 2;    // I16
            case 26: return n * 4;    // I32
            case 27: return n * 8;    // I64
            case 28: return n * 8;    // F64
            default: break;
        }
        if (type == 0) return n * 4;                  // F32
        if (type == 1 || type == 30) return n * 2;    // F16, BF16
        const DequantType* dt = dequant_lookup(static_cast<int>(type));
        if (!dt) return 0;
        if (n % static_cast<uint64_t>(dt->block_elems))
            throw std::runtime_error(std::string("gguf: elementi non multipli del blocco per ") + dt->name);
        return (n / static_cast<uint64_t>(dt->block_elems)) * dt->block_bytes;
    }

    void parse() {
        if (std::memcmp(base_, "GGUF", 4) != 0) throw std::runtime_error("gguf: magic non valido");
        pos_ = 4;
        version_ = static_cast<uint32_t>(rd(4));
        if (version_ < 2 || version_ > 3)
            throw std::runtime_error("gguf: versione non supportata " + std::to_string(version_));
        const uint64_t n_tensors = rd(8);
        const uint64_t n_kv = rd(8);
        for (uint64_t i = 0; i < n_kv; ++i) read_kv();

        const uint64_t align = meta_int("general.alignment", 32);
        if (align == 0 || (align & (align - 1)))
            throw std::runtime_error("gguf: general.alignment non valido");

        tensors_.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            GgufTensor t;
            t.name = rd_str();
            const uint32_t nd = static_cast<uint32_t>(rd(4));
            if (nd == 0 || nd > 4) throw std::runtime_error("gguf: numero di dimensioni non valido in " + t.name);
            for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(rd(8));
            t.type = static_cast<uint32_t>(rd(4));
            t.offset = rd(8);
            tensors_.push_back(std::move(t));
        }

        data_start_ = (pos_ + align - 1) / align * align;
        for (size_t idx = 0; idx < tensors_.size(); ++idx) {
            GgufTensor& t = tensors_[idx];
            if (index_.count(t.name)) throw std::runtime_error("gguf: tensore duplicato " + t.name);
            index_[t.name] = idx;
            t.n_elems = 1;
            for (uint64_t d : t.dims) t.n_elems *= d;
            t.offset += data_start_;
            if (t.offset > size_) throw std::runtime_error("gguf: tensore oltre la fine del file " + t.name);
            t.bytes = type_bytes(t.type, t.n_elems);
            if (t.bytes > size_ - t.offset)
                throw std::runtime_error("gguf: dati del tensore oltre la fine del file " + t.name);
        }
    }

    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
    size_t pos_ = 0;
    uint32_t version_ = 0;
    uint64_t data_start_ = 0;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> index_;
    std::map<std::string, int64_t> ints_;
    std::map<std::string, std::string> strs_;
    std::map<std::string, double> floats_;
};

}  // namespace gptoss

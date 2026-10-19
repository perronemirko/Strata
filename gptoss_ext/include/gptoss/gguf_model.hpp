// gguf_model.hpp - un modello GGUF composto da uno o più shard: indice unico nome -> tensore.
// Controlla che non ci siano nomi duplicati tra shard e che il totale coincida con split.tensors.count.
#pragma once
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_reader.hpp"

namespace gptoss {

struct GgufRef {
    const GgufFile* file = nullptr;
    const GgufTensor* t = nullptr;
    const uint8_t* data() const { return file->data(*t); }
};

class GgufModel {
public:
    explicit GgufModel(const std::vector<std::string>& paths) {
        if (paths.empty()) throw std::invalid_argument("gguf: nessun file");
        for (const auto& p : paths) files_.push_back(std::make_unique<GgufFile>(p));
        for (const auto& f : files_) {
            for (const auto& t : f->tensors()) {
                if (index_.count(t.name)) throw std::runtime_error("gguf: tensore duplicato tra shard: " + t.name);
                index_[t.name] = GgufRef{f.get(), &t};
            }
            const uint64_t decl = f->meta_int("split.tensors.count", 0);
            if (decl) declared_ = decl;
        }
        if (declared_ && declared_ != index_.size())
            throw std::runtime_error("gguf: tensori trovati " + std::to_string(index_.size()) +
                                     ", dichiarati " + std::to_string(declared_));
    }

    const GgufRef* find(const std::string& name) const {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &it->second;
    }
    size_t size() const { return index_.size(); }
    uint64_t declared() const { return declared_; }
    const std::vector<std::unique_ptr<GgufFile>>& files() const { return files_; }

private:
    std::vector<std::unique_ptr<GgufFile>> files_;
    std::map<std::string, GgufRef> index_;
    uint64_t declared_ = 0;
};

}  // namespace gptoss

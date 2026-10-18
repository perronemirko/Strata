// Gateway (facade): unico punto d'ingresso. Sceglie l'adapter dall'architettura e lo apre.
// Architettura senza adapter registrato: errore che rimanda a arch.json (cosa manca).
#pragma once
#include <map>
#include <stdexcept>
#include <string>

#include "adapter.hpp"

namespace gptoss {

class Gateway {
public:
    static Gateway& instance() {
        static Gateway g;
        return g;
    }
    void add(const std::string& arch, AdapterFactory f) { reg_[arch] = std::move(f); }
    bool has(const std::string& arch) const { return reg_.count(arch) > 0; }

    std::unique_ptr<ModelAdapter> open(const std::string& arch, const std::string& model_dir) const {
        auto it = reg_.find(arch);
        if (it == reg_.end())
            throw std::runtime_error("nessun adapter per l'architettura '" + arch +
                                     "': componenti da implementare, vedi arch.json (schema arch-spec/1)");
        return it->second(model_dir);
    }

private:
    std::map<std::string, AdapterFactory> reg_;
};

}  // namespace gptoss

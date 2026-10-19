// arch_check.hpp - controlla arch.json (arch-spec/1) contro arch/supported.json: quali componenti mancano al motore. Header-only.
#pragma once
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"

namespace gptoss {

struct SpecCheck {
    bool found = false;                    // false = nessun arch.json: il controllo non si e' potuto fare
    std::string arch;
    std::vector<std::string> missing;      // componenti o caratteristiche che il motore non esegue
    bool ok() const { return missing.empty(); }
};

// Componenti eseguiti per architettura, da arch/supported.json. Senza file (o illeggibile) usa la tabella di riserva.
inline std::set<std::string> supported_components(const std::string& arch, const std::string& path = "arch/supported.json") {
    std::ifstream in(path);
    if (in) {
        std::stringstream ss;
        ss << in.rdbuf();
        Json j;
        std::string err;
        if (parse_json(ss.str(), j, err)) {
            std::set<std::string> out;
            const Json* archs = j.get("architectures");
            const Json* e = archs ? archs->get(arch) : nullptr;
            const Json* c = e ? e->get("components") : nullptr;
            if (c)
                for (const auto& x : c->a)
                    if (x.t == Json::Str) out.insert(x.s);
            return out;
        }
    }
    if (arch == "gpt_oss") return {"attention_sinks", "router_bias", "mxfp4_experts", "attention_bias"};
    return {};
}

inline SpecCheck check_arch_spec(const std::string& model_dir, const std::string& arch_expected) {
    SpecCheck r;
    std::ifstream in(model_dir + "/arch.json");
    if (!in) return r;
    r.found = true;
    std::stringstream ss;
    ss << in.rdbuf();
    Json j;
    std::string err;
    if (!parse_json(ss.str(), j, err)) {
        r.missing.push_back("arch.json non leggibile: " + err);
        return r;
    }
    r.arch = j.str("arch", "unknown");
    if (r.arch != arch_expected)
        r.missing.push_back("architettura '" + r.arch + "' (questo motore esegue '" + arch_expected + "')");
    const std::set<std::string> sup = supported_components(r.arch);
    if (const Json* c = j.get("components")) {
        for (const auto& kv : c->o)
            if (kv.second.t == Json::Bool && kv.second.b && !sup.count(kv.first))
                r.missing.push_back("componente " + kv.first);
    }
    if (const Json* lk = j.get("layer_kinds")) {
        if (lk->a.size() > 1 && !sup.count("multiple_layer_kinds"))
            r.missing.push_back("layer di " + std::to_string(lk->a.size()) + " tipi diversi");
    }
    return r;
}

}  // namespace gptoss

// Gateway (facade): unico punto d'ingresso. Sceglie l'adapter dall'architettura and opens it.
// Architettura without adapter registered: error that points to arch.json (what is missing).
// The architecture comes from arch.json (arch-spec/1), never from the directory or the file name.
// No silent fallback: an unsupported architecture stops before the inference.
// The diagnostics distinguish a component NOT SUPPORTED (the engine does not run it) from a component
// whose SEMANTICS ARE UNKNOWN (the extractor did not recognize it): different causes, different fixes.
#pragma once
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "adapter.hpp"
#include "gate.hpp"
#include "json.hpp"
#include "model.hpp"   // RunOpts: the configuration fingerprint is part of the contract

namespace gptoss {

// What the gateway reads from arch.json, and what it refuses.
struct Spec {
    std::string arch;
    std::string schema;
    std::vector<std::string> missing;          // components the engine does not run (arch/supported.json)
    std::vector<std::string> unknown_sem;      // tensors/params the extractor could not recognize
    std::string cfg;                            // fingerprint of the run configuration (gate.hpp)
    unsigned long long spec_hash = 0;           // hash of arch.json: the certification is tied to it
    bool valid = false;
    std::string why_invalid;
};

// Required fields of arch-spec/1. A missing one makes the spec invalid: the gateway refuses, it does not assume.
static const char* spec_required[] = {"schema", "arch", "layers", "components", "layer_kinds", "tensors",
                                       "params", "unknown_params", "layer_types", "unbound_tensors", "routing", nullptr};

static Spec read_spec(const std::string& model_dir, const RunOpts& o) {
    Spec s;
    std::ifstream in(model_dir + "/arch.json");
    if (!in) { s.why_invalid = "arch.json absent: run tools/inspect_model.py on the model first"; return s; }
    std::stringstream ss;
    ss << in.rdbuf();
    s.spec_hash = fnv1a(ss.str());
    Json j;
    std::string err;
    if (!parse_json(ss.str(), j, err)) { s.why_invalid = "arch.json not readable: " + err; return s; }
    s.schema = j.str("schema");
    if (s.schema != "arch-spec/1") {
        s.why_invalid = "schema '" + (s.schema.empty() ? std::string("(none)") : s.schema) + "' not arch-spec/1";
        return s;
    }
    for (const char** k = spec_required; *k; ++k)
        if (!j.get(*k)) { s.why_invalid = std::string("field required by arch-spec/1 missing: ") + *k; return s; }
    s.arch = j.str("arch");
    if (s.arch.empty() || s.arch == "unknown") { s.why_invalid = "architecture not identified"; return s; }
    // The components the engine does not run: cause = the adapter does not exist yet.
    if (const Json* m = j.get("missing"))
        for (const auto& x : m->a)
            if (x.t == Json::Str) s.missing.push_back(x.s);
    // The parts the extractor could not recognize: cause = the semantics are unknown, not "unsupported".
    if (const Json* u = j.get("unbound_tensors"))
        for (const auto& x : u->a)
            if (x.t == Json::Str) s.unknown_sem.push_back("tensor '" + x.s + "' not bound to a component");
    if (const Json* u = j.get("unknown_params"))
        for (const auto& x : u->a)
            if (x.t == Json::Str) s.unknown_sem.push_back("parameter '" + x.s + "' not declared by the model files");
    if (const Json* lt = j.get("layer_types"))
        if (!lt->get("value")) s.unknown_sem.push_back("type of the layers not declared: cannot know which layer is sliding");
    if (const Json* r = j.get("routing"))
        if (r->str("algorithm") == "unknown") s.unknown_sem.push_back("routing algorithm not declared: read it from the reference code");
    s.cfg = config_fingerprint(o);
    s.valid = true;
    return s;
}

class Gateway {
public:
    static Gateway& instance() {
        static Gateway g;
        return g;
    }
    void add(const std::string& arch, AdapterFactory f) { reg_[arch] = std::move(f); }
    bool has(const std::string& arch) const { return reg_.count(arch) > 0; }

    // Path of the original entry point: the caller already knows the architecture. Kept for the regression test.
    std::unique_ptr<ModelAdapter> open(const std::string& arch, const std::string& model_dir) const {
        auto it = reg_.find(arch);
        if (it == reg_.end())
            throw std::runtime_error("nessun adapter per l'architettura '" + arch +
                                     "': components to implement, see arch.json (schema arch-spec/1)");
        return it->second(model_dir);
    }

    // Path of the gateway: the architecture comes from the spec, and the adapter must be APPROVED (MATCH)
    // for the exact configuration. NO_MATCH and INCONCLUSIVE block the admission; no fallback.
    std::unique_ptr<ModelAdapter> open_from_spec(const std::string& model_dir, const RunOpts& o,
                                                 const std::string& ref_path, std::string& diag) const {
        Spec s = read_spec(model_dir, o);
        if (!s.valid) { diag = "spec not valid: " + s.why_invalid; return nullptr; }
        if (!has(s.arch)) {
            diag = "no adapter for '" + s.arch + "'";
            for (const auto& m : s.missing) diag += "; not supported: " + m;
            for (const auto& u : s.unknown_sem) diag += "; semantics unknown: " + u;
            return nullptr;
        }
        const GateStatus st = Gate::instance().status(s.arch, s.cfg, s.spec_hash, ref_path);
        if (st != GateStatus::Match) {
            diag = std::string("adapter '") + s.arch + "' not approved for this configuration (" +
                   gate_status_name(st) + "): the gate requires MATCH against a reference, for the exact flags";
            return nullptr;
        }
        return reg_.find(s.arch)->second(model_dir);
    }

private:
    std::map<std::string, AdapterFactory> reg_;
};

}  // namespace gptoss

// gate.hpp - Certification of the adapters. An adapter is admitted by the gateway only if MATCH.
// A record is valid for the EXACT configuration verified: the fingerprint covers the flags that change the numeric path,
// plus the hash of arch.json (the spec) and of the reference. Different configuration = different record, never reused.
// Status: MATCH = verified; NO_MATCH = verification failed; INCONCLUSIVE = verification incomplete.
// Only MATCH authorizes the admission. The gate certifies functional correctness within the tests run, not a proof for every input.
#pragma once
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "model.hpp"

namespace gptoss {

enum class GateStatus { Unknown, Match, NoMatch, Inconclusive };
inline const char* gate_status_name(GateStatus s) {
    switch (s) { case GateStatus::Match: return "MATCH"; case GateStatus::NoMatch: return "NO_MATCH";
                 case GateStatus::Inconclusive: return "INCONCLUSIVE"; default: return "UNKNOWN"; }
}

// Version of the code that is being certified. Raise it at EVERY change to an adapter or to a numeric path:
// a record of an old version never authorizes a new binary.
#ifndef GATE_CODE_VERSION
#define GATE_CODE_VERSION "gpt_oss/adapter/phase1"
#endif

// FNV-1a 64: stable fingerprint of bytes (spec, reference, config).
inline unsigned long long fnv1a(const std::string& s) {
    unsigned long long h = 1469598125565353939ULL;
    for (size_t i = 0; i < s.size(); ++i) { h ^= (unsigned char) s[i]; h *= 1099511628211ULL; }
    return h;
}
inline unsigned long long hash_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0;
    std::stringstream ss;
    ss << in.rdbuf();
    return fnv1a(ss.str());
}

// The flags that change the numeric path of the gpt_oss adapter. Anything that changes the result of a forward
// changes the fingerprint: a MATCH of one configuration says nothing about another.
inline std::string config_fingerprint(const RunOpts& o) {
    auto f = [](double x) { std::ostringstream ss; ss.precision(6); ss << x; return ss.str(); };
    auto kv = [](const std::string& k, const std::string& v) { return k + "=" + v + ";"; };
    std::string s;
    s += kv("cuda", o.cuda ? "1" : "0");
    s += kv("ctx", std::to_string(o.ctx));
    s += kv("prefill_chunk", std::to_string(o.prefill_chunk));
    s += kv("gpu_attn", o.gpu_attn ? "1" : "0");
    s += kv("prefill_dev_pct", f(o.prefill_dev_pct));
    s += kv("key_format", o.key_format == KVFormat::E4M3 ? "e4m3" : o.key_format == KVFormat::F16 ? "f16" : "f32");
    s += kv("value_format", o.value_format == KVFormat::E4M3 ? "e4m3" : o.value_format == KVFormat::F16 ? "f16" : "f32");
    s += kv("device_graphs", o.device_graphs ? "1" : "0");
    s += kv("async_dense", o.async_dense ? "1" : "0");
    s += kv("glu_halves", o.glu_halves ? "1" : "0");
    s += kv("rope_pairs", o.rope_pairs ? "1" : "0");
    s += kv("no_concentration", o.no_concentration ? "1" : "0");
    s += kv("vram_pct", f(o.vram_pct));
    s += kv("vram_reserve_mib", std::to_string(o.vram_reserve_mib));
    s += kv("pin_ram_gib", f(o.pin_ram_gib));
    s += kv("expert_promote", std::to_string(o.expert_promote));
    return s;
}

struct CertRecord {
    std::string arch, cfg, code;
    unsigned long long spec = 0, ref = 0;
    GateStatus st = GateStatus::Unknown;
};

class Gate {
public:
    static Gate& instance() { static Gate g; return g; }

    // The records loaded from the file (the last one per arch+cfg wins: the file is append-only).
    static std::vector<CertRecord>& loaded() { static std::vector<CertRecord> v; return v; }

    // Read the file: each line "arch|cfg|code|spec|ref|status". A malformed line is INCONCLUSIVE, not ignored.
    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> p;
            std::stringstream ss(line);
            std::string t;
            size_t pos = 0;
            while ((pos = line.find('|')) != std::string::npos) { p.push_back(line.substr(0, pos)); line.erase(0, pos + 1); }
            p.push_back(line);
            if (p.size() != 6) { loaded().push_back(CertRecord{p.empty() ? "" : p[0], "", "", 0, 0, GateStatus::Inconclusive}); continue; }
            CertRecord r;
            r.arch = p[0]; r.cfg = p[1]; r.code = p[2];
            r.spec = std::stoull(p[3].c_str()); r.ref = std::stoull(p[4].c_str());
            r.st = p[5] == "MATCH" ? GateStatus::Match : p[5] == "NO_MATCH" ? GateStatus::NoMatch : GateStatus::Inconclusive;
            loaded().push_back(r);
        }
    }

    GateStatus status(const std::string& arch, const std::string& cfg, unsigned long long spec,
                      const std::string& ref_path) const {
        const unsigned long long ref = ref_path.empty() ? 0 : hash_file(ref_path);
        GateStatus best = GateStatus::Unknown;
        for (const CertRecord& r : loaded()) {
            if (r.arch != arch || r.cfg != cfg) continue;
            if (r.code != GATE_CODE_VERSION) continue;              // the code changed: the record is stale
            if (r.spec != spec) continue;                          // the spec changed: the record is stale
            if (r.ref != ref) continue;                            // the reference changed or is missing
            if (r.st == GateStatus::Match) return GateStatus::Match;
            if (r.st == GateStatus::NoMatch) best = GateStatus::NoMatch;
            else if (best == GateStatus::Unknown) best = GateStatus::Inconclusive;
        }
        return best;
    }

    bool approved(const std::string& arch, const std::string& cfg, unsigned long long spec,
                  const std::string& ref_path) const {
        return status(arch, cfg, spec, ref_path) == GateStatus::Match;
    }

    // Writes the record of the verification just run. Append-only: the newest wins.
    static void record(const std::string& arch, const std::string& cfg, unsigned long long spec,
                       const std::string& ref_path, GateStatus st) {
        std::string path = std::string("arch/certifications.txt");
        std::ofstream out(path, std::ios::app);
        if (!out) return;
        out << arch << "|" << cfg << "|" << GATE_CODE_VERSION << "|" << spec << "|"
            << (ref_path.empty() ? 0ULL : hash_file(ref_path)) << "|" << gate_status_name(st) << "\n";
        loaded().push_back(CertRecord{arch, cfg, GATE_CODE_VERSION, spec,
                                      ref_path.empty() ? 0ULL : hash_file(ref_path), st});
    }
};

}  // namespace gptoss

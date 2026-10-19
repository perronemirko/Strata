// gguf_inspect: elenca i tensori di uno o più file GGUF (anche shard) e segnala i tipi non registrati.
// Uso: gguf_inspect file1.gguf [file2.gguf ...] [--all]
// Con gli shard (es. ...-00001-of-00003.gguf) passa tutti i file: i conteggi vengono sommati.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "gptoss/gguf_reader.hpp"

static const char* type_name(uint32_t t) {
    const gptoss::DequantType* dt = gptoss::dequant_lookup((int) t);
    if (dt) return dt->name;
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 30: return "BF16";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        default: return "?";
    }
}

static bool is_known(uint32_t t) {
    return gptoss::dequant_lookup((int) t) != nullptr || (t >= 24 && t <= 28);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("uso: gguf_inspect file1.gguf [file2.gguf ...] [--all]\n"); return 2; }
    bool all = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--all") == 0) all = true;
        else files.push_back(argv[i]);
    }

    std::map<uint32_t, int> count;
    std::map<uint32_t, uint64_t> bytes;
    size_t total = 0;
    for (const auto& path : files) {
        gptoss::GgufFile g(path);
        std::printf("%s\n  architettura: %s   tensori in questo file: %zu   inizio dati: %llu\n",
                    path.c_str(), g.meta_str("general.architecture", "?").c_str(), g.tensors().size(),
                    (unsigned long long) g.data_start());
        if (g.meta_int("split.count", 0) > 0)
            std::printf("  split: no %llu di %llu, tensori totali dichiarati %llu\n",
                        (unsigned long long) g.meta_int("split.no", 0),
                        (unsigned long long) g.meta_int("split.count", 0),
                        (unsigned long long) g.meta_int("split.tensors.count", 0));
        total += g.tensors().size();
        for (const auto& t : g.tensors()) {
            count[t.type]++;
            bytes[t.type] += t.bytes;
            if (all) {
                std::string dims;
                for (auto d : t.dims) dims += std::to_string(d) + " ";
                std::printf("    %-40s %-8s dims [%s] bytes %llu\n", t.name.c_str(), type_name(t.type),
                            dims.c_str(), (unsigned long long) t.bytes);
            }
        }
    }

    std::printf("\ntensori totali: %zu\nriepilogo per tipo:\n", total);
    int missing = 0;
    for (const auto& kv : count) {
        const bool known = is_known(kv.first);
        if (!known) missing++;
        std::printf("  %-8s (id %2u) %5d tensori, %12llu byte %s\n", type_name(kv.first), kv.first, kv.second,
                    (unsigned long long) bytes[kv.first], known ? "" : "<-- NON REGISTRATO");
    }
    std::printf(missing ? "\nTIPI MANCANTI: %d\n" : "\nTUTTI I TIPI REGISTRATI\n", missing);
    return missing ? 1 : 0;
}

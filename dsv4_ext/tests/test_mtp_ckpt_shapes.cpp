// dsv4_mtp_ckpt: adapt a real MTP safetensors checkpoint against a real GGUF, and print the
// engine tensors that result. Headers + the adapter only: no device is created and no weight of
// the main model is read, so this is safe to run while the serve engine holds the whole VRAM.
//
//   ./build-out/dsv4_mtp_ckpt --model FIRST_SHARD.gguf --ckpt FILE.safetensors|DIR [--quiet]
//
// Exits 0 when the checkpoint adapts, 1 with the adapter's message when it does not.
// This is the diagnostic for "the MTP module of X cannot be used by this model": it reports
// WHICH tensors the checkpoint ships and which engine tensors the adapter produced.
#include "dsv4/config.hpp"
#include "dsv4/gguf_header.hpp"
#include "dsv4/mtp_adapter.hpp"
#include "dsv4/st_header.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace dsv4;

// Same shard expansion the runners use: X-00001-of-00003.gguf pulls all three shards.
static std::vector<std::string> expand_shards(const std::string& p) {
    const size_t of = p.rfind("-of-");
    if (of == std::string::npos || of < 5 || p.size() < of + 4 + 5 + 5) return {p};
    const std::string a = p.substr(of - 5, 5), b = p.substr(of + 4, 5);
    if (a.find_first_not_of("0123456789") != std::string::npos || b.find_first_not_of("0123456789") != std::string::npos) return {p};
    const int n = std::atoi(b.c_str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(p.substr(0, of - 5) + buf + p.substr(of));
    }
    return out;
}

// A safetensors source is one file or a directory holding them (an HF snapshot).
static bool discover(const std::string& path, std::vector<std::string>& out) {
    struct ::stat info;
    if (::stat(path.c_str(), &info) != 0) return false;
    if (S_ISREG(info.st_mode)) { out.push_back(path); return true; }
    if (!S_ISDIR(info.st_mode)) return false;
    // Directory listing without pulling <filesystem> into the extension's dependency set.
    const std::string cmd = "ls -1 \"" + path + "\" 2>/dev/null";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) return false;
    char line[4096];
    while (std::fgets(line, sizeof line, f)) {
        std::string n(line);
        while (!n.empty() && (n.back() == '\n' || n.back() == '\r')) n.pop_back();
        if (n.size() < 13 || n.compare(n.size() - 12, 12, ".safetensors") != 0) continue;
        out.push_back(path + "/" + n);
    }
    ::pclose(f);
    std::sort(out.begin(), out.end());
    return !out.empty();
}

int main(int argc, char** argv) {
    std::string model, ckpt;
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
        else if (a == "--quiet") quiet = true;
        else { std::printf("usage: dsv4_mtp_ckpt --model FIRST_SHARD.gguf --ckpt FILE.safetensors|DIR [--quiet]\n"); return 2; }
    }
    if (model.empty() || ckpt.empty()) { std::printf("usage: dsv4_mtp_ckpt --model FIRST_SHARD.gguf --ckpt FILE.safetensors|DIR\n"); return 2; }

    std::string err;
    GgufHeader h;
    if (!gguf_read_header(expand_shards(model), h, err)) { std::fprintf(stderr, "cannot read the GGUF: %s\n", err.c_str()); return 1; }
    Dsv4Config c;
    if (!config_from_gguf(h, c, err)) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }

    std::vector<std::string> st_shards;
    if (!discover(ckpt, st_shards)) { std::fprintf(stderr, "cannot list the safetensors of %s\n", ckpt.c_str()); return 1; }
    StHeader sh;
    if (!st_read_header(st_shards, sh, err)) { std::fprintf(stderr, "cannot read the safetensors: %s\n", err.c_str()); return 1; }

    if (!quiet) {
        std::printf("checkpoint files: %zu, tensors: %zu\n", sh.files.size(), sh.tensors.size());
        for (const StTensor& t : sh.tensors) {
            std::string d;
            for (uint64_t s : t.shape) { d += "x"; d += std::to_string(s); }
            std::printf("  %-46s %-8s %s\n", t.name.c_str(), t.dtype.c_str(), d.empty() ? "?" : d.c_str());
        }
    }

    MtpAdapted ad;
    if (!mtp_adapt_safetensors(sh, c, ad, err)) { std::fprintf(stderr, "REFUSED: the MTP module cannot be used by this model: %s\n", err.c_str()); return 1; }

    if (!quiet) {
        std::printf("adapted: %zu engine tensors, %d draft layer(s), fusion_mode %d, %.1f MiB\n",
                    ad.items.size(), ad.n_layers, ad.fusion_mode, (double) ad.bytes / 1048576.0);
        for (const auto& it : ad.items) {
            std::string dims;
            for (uint64_t d : it.t.dims) { if (!dims.empty()) dims += "x"; dims += std::to_string(d); }
            std::printf("  %-46s %-8s %s %10.1f KiB\n", it.t.name.c_str(), ggml_type_str(it.t.type).c_str(), dims.c_str(), (double) it.t.nbytes / 1024.0);
        }
        std::printf("draft compression ratios:");
        for (int r : ad.inferred_ratios) std::printf(" %d", r);
        std::printf("\n");
    }
    std::printf("OK\n");
    return 0;
}

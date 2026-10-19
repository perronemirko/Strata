// run_dsv4.cpp - primo avvio del forward DeepSeek-V4 in CPU da GGUF multi-shard.
// Uso: run_dsv4 <shard1.gguf> [shard2.gguf ...] [--tokens 1,2,3] [--top 10]
// Stampa prima le chiavi di metadati mancanti e gli errori di binding, poi i logit dell'ultimo token.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gptoss/dsv4_bind.hpp"
#include "gptoss/dsv4_forward.hpp"
#include "gptoss/gguf_model.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    std::vector<int32_t> tokens;
    int top = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                if (c == std::string::npos) c = s.size();
                tokens.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
                pos = c + 1;
            }
        } else if (a == "--top" && i + 1 < argc) {
            top = std::atoi(argv[++i]);
        } else {
            paths.push_back(a);
        }
    }
    if (paths.empty()) {
        std::fprintf(stderr, "uso: %s <shard.gguf> [...] [--tokens 1,2,3] [--top 10]\n", argv[0]);
        return 2;
    }
    if (tokens.empty()) tokens = {1};

    gptoss::GgufModel model(paths);
    std::printf("tensori: %zu (dichiarati %llu)\n", model.size(), (unsigned long long) model.declared());

    gptoss::dsv4::BindReport rep;
    gptoss::dsv4::DsModel M = gptoss::dsv4::bind_dsv4(model, rep);

    for (const auto& k : rep.missing_kv) std::printf("METADATO MANCANTE  %s\n", k.c_str());
    for (const auto& e : rep.errors) std::printf("ERRORE BINDING     %s\n", e.c_str());
    std::printf("layer legati: %zu, vocab %lld, hc %lld, n_embd %lld\n", M.layers.size(),
                (long long) M.n_vocab, (long long) M.hc, (long long) M.ap.n_embd);
    if (!rep.ok()) {
        std::printf("BINDING NON COMPLETO: correggi le voci sopra prima di eseguire il forward\n");
        return 1;
    }

    std::vector<float> logits(static_cast<size_t>(M.n_vocab));
    const auto t0 = std::chrono::steady_clock::now();
    gptoss::dsv4::forward_logits_last(M, tokens.data(), (int64_t) tokens.size(), logits.data());
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("forward: %lld token in %.1f s\n", (long long) tokens.size(),
                std::chrono::duration<double>(t1 - t0).count());

    std::vector<int64_t> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int64_t) i;
    const int k = std::min<int>(top, (int) idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    std::printf("top %d logit (id, valore):\n", k);
    for (int i = 0; i < k; ++i) std::printf("  %lld  %.4f\n", (long long) idx[i], logits[idx[i]]);
    return 0;
}

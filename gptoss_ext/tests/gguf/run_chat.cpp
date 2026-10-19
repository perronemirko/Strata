// run_chat.cpp - generazione di testo con DeepSeek-V4 in CPU, decodifica incrementale, greedy.
// Uso: run_chat <shard1.gguf> [shard2 shard3] --prompt "testo" [--n 30] [--bos]
// Prompt grezzo, senza chat template (per ora). Il tempo per token è quello del passo CPU.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gptoss/dsv4_bind.hpp"
#include "gptoss/dsv4_decode.hpp"
#include "gptoss/dsv4_tok.hpp"
#include "gptoss/gguf_model.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    std::string prompt;
    int n_gen = 30;
    bool bos = false;
    bool dbg = false;
    bool have_prompt = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prompt" && i + 1 < argc) { prompt = argv[++i]; have_prompt = true; }
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--bos") bos = true;
        else if (a == "--debug") dbg = true;
        else paths.push_back(a);
    }
    if (paths.empty() || !have_prompt) {
        std::fprintf(stderr, "uso: %s <shard.gguf> [...] --prompt \"testo\" [--n 30] [--bos]\n", argv[0]);
        return 2;
    }

    gptoss::GgufModel model(paths);
    gptoss::dsv4::BindReport rep;
    gptoss::dsv4::DsModel M = gptoss::dsv4::bind_dsv4(model, rep);
    for (const auto& k : rep.missing_kv) std::printf("METADATO MANCANTE  %s\n", k.c_str());
    for (const auto& e : rep.errors) std::printf("ERRORE BINDING     %s\n", e.c_str());
    if (!rep.ok()) {
        std::printf("BINDING NON COMPLETO\n");
        return 1;
    }

    gptoss::tok::Tokenizer tk = gptoss::tok::Tokenizer::from_gguf(*model.files()[0]);
    tk.add_bos = bos;
    std::vector<int32_t> ids = tk.encode(prompt);
    if (ids.empty()) {
        std::printf("prompt vuoto dopo la tokenizzazione\n");
        return 1;
    }
    std::printf("vocab %zu, prompt = %zu token:", tk.vocab_size(), ids.size());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    std::vector<float> logits(static_cast<size_t>(M.n_vocab));
    gptoss::dsv4::DsDecoder dec(M);
    dec.debug = dbg;

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < ids.size(); ++i) {
        const bool last = (i + 1 == ids.size());
        dec.step(ids[i], last, logits.data());
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("prefill: %.1f s\n", std::chrono::duration<double>(t1 - t0).count());

    std::vector<int32_t> gen;
    for (int g = 0; g < n_gen; ++g) {
        int32_t best = 0;
        for (int32_t v = 1; v < (int32_t) logits.size(); ++v)
            if (logits[v] > logits[best]) best = v;
        if (best == tk.eos()) break;
        gen.push_back(best);
        const auto ts = std::chrono::steady_clock::now();
        dec.step(best, true, logits.data());
        const auto te = std::chrono::steady_clock::now();
        std::printf("[tok %d] %.1f s\n", best, std::chrono::duration<double>(te - ts).count());
        std::fflush(stdout);
    }

    std::printf("\n--- testo generato ---\n%s\n", tk.decode(gen).c_str());
    return 0;
}

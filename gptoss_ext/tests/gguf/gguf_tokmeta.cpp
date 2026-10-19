// gguf_tokmeta.cpp - stampa i metadati del tokenizer di un GGUF (valori scalari e stringhe).
// Uso: gguf_tokmeta <shard1.gguf> [altri shard ...]
// Gli array (tokens, merges, token_type) non sono ancora letti dal reader: qui se ne vede solo la presenza.
#include <cstdio>
#include <string>
#include <vector>

#include "gptoss/gguf_model.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "uso: %s <shard.gguf> [...]\n", argv[0]);
        return 2;
    }
    std::vector<std::string> paths(argv + 1, argv + argc);
    gptoss::GgufModel m(paths);
    const gptoss::GgufFile& f = *m.files()[0];

    const char* str_keys[] = {"tokenizer.ggml.model", "tokenizer.ggml.pre", "general.architecture", "general.name"};
    for (const char* k : str_keys) std::printf("%-28s = \"%s\"\n", k, f.meta_str(k, "<assente>").c_str());

    const char* int_keys[] = {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                              "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
                              "tokenizer.ggml.add_eos_token"};
    const uint64_t sentinel = ~0ull;
    for (const char* k : int_keys) {
        const uint64_t v = f.meta_int(k, sentinel);
        if (v == sentinel) std::printf("%-28s = <assente>\n", k);
        else std::printf("%-28s = %llu\n", k, (unsigned long long) v);
    }

    const std::string tpl = f.meta_str("tokenizer.chat_template", "");
    std::printf("%-28s = %s (%zu caratteri)\n", "tokenizer.chat_template", tpl.empty() ? "<assente>" : "presente",
                tpl.size());
    return 0;
}

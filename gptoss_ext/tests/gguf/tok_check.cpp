// tok_check.cpp - controllo rapido del tokenizer: non carica i pesi, solo i metadati.
// Uso: tok_check <shard1.gguf> "testo"
// Stampa gli id, il pezzo di ogni id, e il round trip decode(encode(testo)).
#include <cstdio>
#include <string>
#include <vector>

#include "gptoss/dsv4_tok.hpp"
#include "gptoss/gguf_reader.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s <shard1.gguf> \"testo\"\n", argv[0]);
        return 2;
    }
    // Solo metadati: GgufFile diretto, senza il controllo di GgufModel sui tensori (che richiede tutti gli shard).
    gptoss::GgufFile file(argv[1]);
    gptoss::tok::Tokenizer tk = gptoss::tok::Tokenizer::from_gguf(file);
    const std::string text = argv[2];

    const std::vector<int32_t> ids = tk.encode(text);
    std::printf("token: %zu\n", ids.size());
    for (int32_t id : ids) {
        std::printf("  %7d  |%s|\n", id, tk.decode(std::vector<int32_t>{id}).c_str());
    }
    const std::string back = tk.decode(ids);
    std::printf("round trip: %s\n", back == text ? "OK (identico)" : "DIVERSO");
    std::printf("decode: |%s|\n", back.c_str());
    return 0;
}

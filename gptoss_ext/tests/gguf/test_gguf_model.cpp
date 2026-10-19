// Test GgufModel: indice su un file, tensore trovato con i dati corretti, duplicato tra shard rifiutato.
#include <cstdio>
#include "gptoss/gguf_model.hpp"

int main() {
    bool ok = true;
    auto check = [&](bool c, const char* what) { std::printf("%s %s\n", c ? "OK  " : "FAIL", what); ok = ok && c; };
    gptoss::GgufModel m({"test.gguf"});
    check(m.size() == 3, "indice con tre tensori");
    const gptoss::GgufRef* r = m.find("b.q8");
    check(r && r->t->type == 8 && r->data() == r->file->data(*r->t), "tensore trovato per nome con dati dal file");
    check(m.find("non.esiste") == nullptr, "nome inesistente -> null");
    bool threw = false;
    try { gptoss::GgufModel dup({"test.gguf", "test.gguf"}); } catch (const std::exception&) { threw = true; }
    check(threw, "stesso tensore in due shard rifiutato");
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

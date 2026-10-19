// Test del lettore GGUF: metadati, tensori, byte grezzi e dequantizzazione Q8_0 contro i valori attesi calcolati in Python.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "gptoss/gguf_reader.hpp"

static std::vector<uint8_t> readall(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("uso: test_gguf_reader test.gguf expected_f32.bin expected_q8.bin expected_q4k.bin\n");
        return 2;
    }
    bool ok = true;
    auto check = [&](bool c, const char* what) {
        std::printf("%s %s\n", c ? "OK  " : "FAIL", what);
        ok = ok && c;
    };

    gptoss::GgufFile g(argv[1]);
    check(g.version() == 3, "versione 3");
    check(g.meta_str("general.architecture") == "test", "metadato stringa");
    check(g.meta_int("general.alignment", 0) == 64, "metadato alignment = 64");
    check(g.data_start() % 64 == 0, "inizio dati allineato a 64");
    check(g.tensors().size() == 3, "tre tensori");

    const gptoss::GgufTensor* a = g.find("a.f32");
    auto exp_f32 = readall(argv[2]);
    check(a && a->type == 0 && a->dims.size() == 2 && a->dims[0] == 8 && a->dims[1] == 2 && a->n_elems == 16 && a->bytes == 64,
          "a.f32 forma, tipo e dimensione");
    check(a && exp_f32.size() == 64 && std::memcmp(g.data(*a), exp_f32.data(), 64) == 0, "a.f32 byte grezzi");

    const gptoss::GgufTensor* q = g.find("b.q8");
    auto exp_q8 = readall(argv[3]);
    check(q && q->type == 8 && q->n_elems == 128 && q->bytes == 136, "b.q8 forma e dimensione (4 blocchi x 34)");
    std::vector<float> out(128, 0.0f);
    if (q) gptoss::dequantize(8, g.data(*q), out.data(), 128);
    check(exp_q8.size() == 512 && std::memcmp(out.data(), exp_q8.data(), 512) == 0, "b.q8 dequantizzato bit per bit");

    const gptoss::GgufTensor* k = g.find("c.q4k");
    auto exp_q4 = readall(argv[4]);
    check(k && k->type == 12 && k->n_elems == 512 && k->bytes == 288, "c.q4k forma e dimensione (2 x 144)");
    check(k && exp_q4.size() == 288 && std::memcmp(g.data(*k), exp_q4.data(), 288) == 0, "c.q4k byte grezzi");

    check(g.find("non.esiste") == nullptr, "tensore inesistente restituisce null");

    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

// Test del registro: per ogni tipo registrato, dispatch == chiamata diretta (bit per bit); tipi ignoti danno errore con il nome.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gptoss/dequant_registry.hpp"

int main() {
    std::mt19937 rng(99);
    bool ok = true;
    for (int id : {8, 12, 14, 42, 39, 16, 17, 22, 18, 21, 19, 29, 20, 23, 13, 0, 1, 30}) {
        const gptoss::DequantType* t = gptoss::dequant_lookup(id);
        const int64_t k = t->block_elems * 8;
        std::vector<uint8_t> buf(gptoss::dequant_row_bytes(id, k));
        for (auto& b : buf) b = (uint8_t) rng();
        // d fp16 finito: per i tipi con d all'inizio del blocco (non per mxfp4 né iq1_m)
        if (id != 39 && id != 29) {
            const size_t step = t->block_bytes;
            for (size_t off = 0; off + 2 <= buf.size(); off += step) {
                uint16_t d = (uint16_t) rng() & 0x7bff;
                std::memcpy(&buf[off], &d, 2);
            }
        }
        std::vector<float> a(k), b(k);
        t->fn(buf.data(), a.data(), k);
        gptoss::dequantize(id, buf.data(), b.data(), k);
        if (std::memcmp(a.data(), b.data(), sizeof(float) * k) != 0) { std::printf("FAIL dispatch %s\n", t->name); ok = false; }
        else std::printf("OK   dispatch %s\n", t->name);
    }
    for (int id : {10, 11, 2}) {  // non portati: devono dare errore esplicito
        try { std::vector<float> o(256); std::vector<uint8_t> z(4096); gptoss::dequantize(id, z.data(), o.data(), 256); std::printf("FAIL: tipo %d accettato\n", id); ok = false; }
        catch (const std::exception& e) { std::printf("OK   errore atteso: %s\n", e.what()); }
    }
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

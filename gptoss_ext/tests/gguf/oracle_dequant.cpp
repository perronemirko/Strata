// oracle_dequant.cpp - confronta il nostro dequant (gptoss/dequant_registry.hpp) con to_float di ggml,
// riga per riga, su un tensore GGUF. Nessuna attivazione coinvolta.
// Uso: oracle_dequant <shard.gguf> <nome-tensore>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gptoss/dequant_registry.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s <shard.gguf> <nome-tensore>\n", argv[0]);
        return 2;
    }
    ggml_context* ctx = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = false;
    gp.ctx = &ctx;
    gguf_context* g = gguf_init_from_file(argv[1], gp);
    if (!g || !ctx) {
        std::fprintf(stderr, "impossibile aprire %s\n", argv[1]);
        return 1;
    }
    ggml_tensor* W = ggml_get_tensor(ctx, argv[2]);
    if (!W) {
        std::fprintf(stderr, "tensore %s non trovato in questo shard\n", argv[2]);
        return 1;
    }
    const int64_t in = W->ne[0], out = W->ne[1];
    const ggml_type_traits* tr = ggml_get_type_traits(W->type);
    std::printf("tensore %s: tipo=%s ne=[%lld, %lld] row_stride=%zu (nostro=%zu)\n", argv[2], ggml_type_name(W->type),
                (long long) in, (long long) out, (size_t) W->nb[1], gptoss::dequant_row_bytes((int) W->type, in));

    std::vector<float> ours(static_cast<size_t>(in)), ggmlv(static_cast<size_t>(in));
    double worst = 0.0;
    int64_t worst_row = -1;
    for (int64_t j = 0; j < out; ++j) {
        const uint8_t* row = static_cast<const uint8_t*>(W->data) + static_cast<size_t>(j) * W->nb[1];
        gptoss::dequantize((int) W->type, row, ours.data(), in);
        tr->to_float(row, ggmlv.data(), in);
        for (int64_t i = 0; i < in; ++i) {
            const double d = std::fabs((double) ours[i] - (double) ggmlv[i]);
            if (d > worst) { worst = d; worst_row = j; }
        }
        if (j == 0) {
            std::printf("riga 0: nostro = %.9g %.9g %.9g | ggml = %.9g %.9g %.9g\n", ours[0], ours[1], ours[2], ggmlv[0],
                        ggmlv[1], ggmlv[2]);
        }
    }
    std::printf("differenza massima su tutto il tensore = %.3g (riga %lld)\n", worst, (long long) worst_row);
    return 0;
}

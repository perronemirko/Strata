// oracle_kv.cpp - stampa il valore di alcune chiavi metadata di un GGUF, con l'API gguf di ggml.
// Uso: oracle_kv <shard.gguf> <chiave> [chiave ...]
#include <cstdio>

#include "ggml.h"
#include "gguf.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s <shard.gguf> <chiave> [chiave ...]\n", argv[0]);
        return 2;
    }
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(argv[1], gp);
    if (!g) {
        std::fprintf(stderr, "impossibile aprire %s\n", argv[1]);
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        const int64_t id = gguf_find_key(g, argv[i]);
        if (id < 0) {
            std::printf("%-45s = (ASSENTE)\n", argv[i]);
            continue;
        }
        const enum gguf_type t = gguf_get_kv_type(g, id);
        if (t == GGUF_TYPE_FLOAT32) {
            std::printf("%-45s = %.9g (f32)\n", argv[i], (double) gguf_get_val_f32(g, id));
        } else if (t == GGUF_TYPE_UINT32) {
            std::printf("%-45s = %u (u32)\n", argv[i], gguf_get_val_u32(g, id));
        } else if (t == GGUF_TYPE_INT32) {
            std::printf("%-45s = %d (i32)\n", argv[i], gguf_get_val_i32(g, id));
        } else if (t == GGUF_TYPE_STRING) {
            std::printf("%-45s = %s (str)\n", argv[i], gguf_get_val_str(g, id));
        } else {
            std::printf("%-45s = (tipo %d)\n", argv[i], (int) t);
        }
    }
    gguf_free(g);
    return 0;
}

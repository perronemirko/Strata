// oracle_mulmat.cpp - oracolo: calcola W*x con ggml_mul_mat (la stessa moltiplicazione di llama.cpp CPU).
// Uso: oracle_mulmat <shard.gguf> <nome-tensore> <input.bin>
// input.bin: vettore float32 lungo quanto le colonne di W (es. attn_norm_l0.bin scritto da run_chat --debug).
// Stampa i primi 3 valori di y = W x. Nessun test: è solo il valore di riferimento per un tensore.
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#if __has_include("ggml-cpu.h")
#include "ggml-cpu.h"
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "uso: %s <shard.gguf> <nome-tensore> <input.bin>\n", argv[0]);
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
    std::printf("tensore %s: tipo=%s ne=[%lld, %lld]\n", argv[2], ggml_type_name(W->type), (long long) W->ne[0],
                (long long) W->ne[1]);

    const int64_t in = W->ne[0];
    std::vector<float> x(static_cast<size_t>(in));
    FILE* f = std::fopen(argv[3], "rb");
    if (!f || std::fread(x.data(), sizeof(float), static_cast<size_t>(in), f) != static_cast<size_t>(in)) {
        std::fprintf(stderr, "input.bin non letto o di lunghezza sbagliata (attesi %lld float)\n", (long long) in);
        return 1;
    }
    std::fclose(f);

    ggml_init_params ip{};
    ip.mem_size = 256u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context* c2 = ggml_init(ip);
    ggml_tensor* X = ggml_new_tensor_1d(c2, GGML_TYPE_F32, in);
    std::memcpy(X->data, x.data(), x.size() * sizeof(float));
    ggml_tensor* Y = ggml_mul_mat(c2, W, X);
    ggml_cgraph* gf = ggml_new_graph(c2);
    ggml_build_forward_expand(gf, Y);
    ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    const float* y = static_cast<const float*>(Y->data);
    std::printf("ggml mul_mat[0:3] = %.6g %.6g %.6g\n", y[0], y[1], y[2]);
    return 0;
}

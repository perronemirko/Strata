// tests/q3k_mmq_check.cpp - synthetic check that the MMQ path multiplies a Q3_K matrix correctly.
//
// The UD packs mix quantizations layer by layer (a UD-Q4_K_M Qwen3.8 file stores seven ffn matrices as Q3_K), and
// an uncovered type switches the whole dense prompt path off.  This checks one such matrix end to end: the weights
// are quantized with ggml's own quantizer, multiplied through strata's MMQ wrapper, and compared against a CPU
// reference that dequantizes the very same blocks with ggml's own to_float.  No model, no tokenizer: a few MiB.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml.h"

int main() {
    if (!strata::prefill::mmq::built()) { std::printf("this build has no MMQ\n"); return 2; }
    if (!strata::prefill::mmq::supported(GGML_TYPE_Q3_K)) { std::printf("FAIL: Q3_K is not supported\n"); return 1; }

    const int K = 512, N = 256, T = 64;          // [N, K] weights, T activation rows
    std::vector<float> x((size_t) T * K), w((size_t) N * K);
    srand(1234);
    auto frand = [] { return (rand() / (float) RAND_MAX - 0.5f) * 2.0f; };
    for (float& v : x) v = frand();
    for (float& v : w) v = frand() * 0.25f;

    const size_t wbytes = ggml_row_size(GGML_TYPE_Q3_K, K) * (size_t) N;
    std::vector<uint8_t> wq(wbytes);
    ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), wq.data(), 0, (int64_t) N, K, nullptr);

    // CPU reference: dequantize every row with ggml's own to_float and multiply.
    const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_Q3_K);
    if (!tr->to_float) { std::printf("FAIL: ggml has no Q3_K dequantizer\n"); return 2; }
    std::vector<float> ref((size_t) T * N), wrow((size_t) K);
    for (int r = 0; r < N; ++r) {
        tr->to_float(wq.data() + (size_t) r * ggml_row_size(GGML_TYPE_Q3_K, K), wrow.data(), K);
        for (int t = 0; t < T; ++t) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double) x[(size_t) t * K + k] * wrow[k];
            ref[(size_t) t * N + r] = (float) acc;
        }
    }

    // Device.
    float *dx = nullptr, *ddst = nullptr; void* dxq = nullptr; void* dw = nullptr;
    int32_t *dids = nullptr, *dbounds = nullptr;
    cudaMalloc(&dx, sizeof(float) * x.size());
    cudaMalloc(&ddst, sizeof(float) * ref.size());
    cudaMalloc(&dxq, strata::prefill::mmq::q8_bytes(T, K));
    cudaMalloc(&dw, wq.size());
    cudaMalloc(&dids, sizeof(int32_t) * T);
    cudaMalloc(&dbounds, sizeof(int32_t) * 2);
    cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    std::vector<int32_t> ids(T), bounds = {0, T};
    for (int i = 0; i < T; ++i) ids[i] = i;
    cudaMemcpy(dids, ids.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice);
    cudaMemcpy(dbounds, bounds.data(), sizeof(int32_t) * 2, cudaMemcpyHostToDevice);

    strata::prefill::mmq::quantize(dx, nullptr, dxq, GGML_TYPE_Q3_K, K, K, T, nullptr);
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("FAIL quantize: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    strata::prefill::mmq::Context ctx;
    strata::prefill::mmq::Product p;
    p.w = dw;
    p.type = GGML_TYPE_Q3_K;
    p.w_rows = N;
    p.w_cols = K;
    p.expert_bytes = strata::prefill::mmq::matrix_bytes(GGML_TYPE_Q3_K, N, K);
    p.n = 1;
    p.xq = dxq;
    p.bounds = dbounds;
    p.ids = dids;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = ddst;
    p.ld_dst = N;
    ctx.run(p, nullptr);
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("FAIL mul_mat_q: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), ddst, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);

    double worst = 0.0, scale = 1.0;
    size_t at = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        scale = std::max(scale, std::fabs((double) ref[i]));
        if (d > worst) { worst = d; at = i; }
    }
    std::printf("Q3_K MMQ vs CPU dequant: %dx%d, T=%d | worst abs %.4f of max |ref| %.2f at %zu (got %.4f ref %.4f)\n",
                N, K, T, worst, scale, at, got[at], ref[at]);
    // q8_1 activations are 8-bit: a couple of percent of the row's magnitude is the expected error.
    const bool pass = worst <= 0.03 * scale;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

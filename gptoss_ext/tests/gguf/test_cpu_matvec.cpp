// Test matvec_quant su tensori del file di prova: confronto con la matrice intera dequantizzata e moltiplicata in double.
#include <cmath>
#include <cstdio>
#include <vector>
#include "gptoss/gguf_model.hpp"
#include "gptoss/cpu_matvec.hpp"

int main() {
    bool ok = true;
    gptoss::GgufModel m({"test.gguf"});
    const gptoss::GgufRef* r = m.find("b.q8");          // Q8_0, dims [64, 2]: in 64, out 2
    if (!r) { std::printf("FAIL tensore b.q8 mancante\n"); return 1; }
    const int64_t in = 64, out = 2;
    std::vector<float> full(in * out);
    gptoss::dequantize(8, r->data(), full.data(), in * out);
    std::vector<float> x(in);
    for (int64_t i = 0; i < in; ++i) x[i] = std::sin(0.37 * (double) i) + 0.5;
    std::vector<float> y(out), buf;
    gptoss::matvec_quant(8, r->data(), in, out, x.data(), y.data(), buf);
    for (int64_t j = 0; j < out; ++j) {
        double ref = 0.0;
        for (int64_t i = 0; i < in; ++i) ref += (double) full[j * in + i] * x[i];
        const double err = std::fabs(ref - y[j]);
        const bool good = err <= 1e-4 * (1.0 + std::fabs(ref));
        std::printf("%s riga %lld: matvec %.6f, riferimento %.6f\n", good ? "OK  " : "FAIL", (long long) j, y[j], ref);
        ok = ok && good;
    }
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}

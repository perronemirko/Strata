// End-to-end check against tests/tiny/golden.txt (numpy, naive MLA, float64) on the tiny FP8/BF16 safetensors checkpoint.
// Usage: test_m4_core tests/tiny
#include "m4/model.hpp"
#include "m4/safetensors.hpp"
#include "m4/weights.hpp"
#include "m4/ops.hpp"
#include "m4/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/tiny";
    std::ifstream f(dir + "/golden.txt");
    if (!f) { std::fprintf(stderr, "cannot open %s/golden.txt (run python3 tools/tiny_model.py %s)\n", dir.c_str(), dir.c_str()); return 2; }
    std::map<std::string, std::vector<double>> G;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string name, tok; size_t n; is >> name >> n;
        std::vector<double> v; v.reserve(n);
        for (size_t i = 0; i < n; ++i) { is >> tok; v.push_back(tok == "nan" ? NAN : std::stod(tok)); }
        G[name] = v;
    }
    using namespace m4;

    // FP8 E4M3 table: every one of the 256 codes against the Python table
    const float* lut = fp8_e4m3_lut();
    int bad = 0;
    for (int i = 0; i < 256; ++i) { const double r = G["fp8_table"][(size_t) i]; if (std::isnan(r) ? !std::isnan(lut[i]) : (double) lut[i] != r) ++bad; }
    CHECK(bad == 0); std::printf("%s fp8 e4m3 table: %d mismatches\n", bad ? "FAIL" : "ok  ", bad);
    CHECK(llama4_scale(100, 0.1, 8192) == 1.0);
    CHECK(std::fabs(llama4_scale(8192, 0.1, 8192) - (1.0 + 0.1 * std::log(2.0))) < 1e-12);

    std::vector<int> ids; for (double d : G["ids"]) ids.push_back((int) d);
    struct V { const char* name; Router r; bool ms, qpe; } vars[] = {
        {"base", Router::Softmax, false, false}, {"sigmoid", Router::Sigmoid, false, false},
        {"mscale", Router::Softmax, true, false}, {"qpe", Router::Softmax, false, true}};
    std::vector<std::vector<float>> got_base;
    for (auto& v : vars) {
        Model m; RunOpts o; o.ctx = 64; o.router = v.r; o.mscale_softmax = v.ms; o.l4_qpe_only = v.qpe; o.verbose = false;
        std::string err;
        if (!m.load(dir, o, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 2; }
        const int V_ = m.config().vocab;
        const auto& ref = G["logits_" + std::string(v.name)];
        double worst = 0, scale = 0;
        int argmax_bad = 0;
        std::vector<float> lg;
        for (size_t t = 0; t < ids.size(); ++t) {
            m.forward(ids[t], (int) t, &lg);
            int am = 0, rm = 0;
            for (int j = 0; j < V_; ++j) {
                const double r = ref[t * (size_t) V_ + (size_t) j];
                worst = std::max(worst, std::fabs(r - lg[(size_t) j])); scale = std::max(scale, std::fabs(r));
                if (lg[(size_t) j] > lg[(size_t) am]) am = j;
                if (r > ref[t * (size_t) V_ + (size_t) rm]) rm = j;
            }
            if (am != rm) ++argmax_bad;
        }
        const bool ok = worst <= 2e-4 * std::max(scale, 1.0) && argmax_bad == 0;
        std::printf("%s logits[%-7s] max|diff| = %.3g (max|logit| %.3g), argmax mismatches %d\n", ok ? "ok  " : "FAIL", v.name, worst, scale, argmax_bad);
        CHECK(ok);
        if (std::string(v.name) == "base") got_base.push_back(lg);
    }
    // batched prefill == token by token == golden; prefix reuse; KV checkpoint round trip
    {
        const int V_ = 96;
        const auto& ref = G["logits_base"];
        auto worst_vs_ref = [&](const std::vector<float>& lg, size_t t) { double w = 0; for (int j = 0; j < V_; ++j) w = std::max(w, std::fabs(ref[t * (size_t) V_ + (size_t) j] - lg[(size_t) j])); return w; };
        for (int chunk : {1, 5, 7, 64}) {
            Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.prefill_chunk = chunk; std::string err;
            if (!m.load(dir, o, err)) return 2;
            std::vector<float> lg;
            m.prefill(ids.data(), (int) ids.size(), 0, &lg);
            const double w = worst_vs_ref(lg, ids.size() - 1);
            std::printf("%s prefill chunk=%-2d last-token logits max|diff| = %.3g\n", w < 5e-4 ? "ok  " : "FAIL", chunk, w);
            CHECK(w < 5e-4); CHECK(m.cached_len() == (int) ids.size());
            m.forward(ids[3], 3, nullptr);                 // cache must also work for decode after a batched prefill
        }
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.prefill_chunk = 6; std::string err;
        if (!m.load(dir, o, err)) return 2;
        std::vector<float> lg;
        m.prefill(ids.data(), 12, 0, nullptr);
        CHECK(m.save_kv("/tmp/m4_test.kv", err));
        // another process-like instance: restore, extend the SAME prompt by 12 more tokens, compare with the golden of the full run
        Model m2; if (!m2.load(dir, o, err)) return 2;
        CHECK(m2.load_kv("/tmp/m4_test.kv", err));
        CHECK(m2.cached_len() == 12);
        std::vector<int> full = ids;
        const int keep = m2.reuse_prefix(full);
        CHECK(keep == 12);
        m2.prefill(full.data() + keep, (int) full.size() - keep, keep, &lg);
        const double w = worst_vs_ref(lg, ids.size() - 1);
        std::printf("%s KV checkpoint restore + reuse (12 cached, %zu prefilled) max|diff| = %.3g\n", w < 5e-4 ? "ok  " : "FAIL", full.size() - 12, w);
        CHECK(w < 5e-4);
        // a different question after the same prefix: only the tail is recomputed
        std::vector<int> other(ids.begin(), ids.begin() + 10); other.push_back(50); other.push_back(51);
        CHECK(m2.reuse_prefix(other) == 10);
        // checkpoint of another configuration must be refused
        RunOpts o2 = o; o2.router = Router::Sigmoid; Model m3; if (!m3.load(dir, o2, err)) return 2;
        CHECK(!m3.load_kv("/tmp/m4_test.kv", err));
        std::printf("ok   checkpoint from another router setting refused: %s\n", err.c_str());
        // the plan must say it is a plan
        RunOpts o3 = o; o3.verbose = true; o3.vram_pct = 90; o3.vram_total_mib = 24576; Model m4_; CHECK(m4_.load(dir, o3, err));
    }
    // the switches must matter: otherwise the variants above would prove nothing
    {
        const auto &a = G["logits_base"], &b = G["logits_sigmoid"], &c2 = G["logits_mscale"], &d = G["logits_qpe"];
        double da = 0, dm = 0, dq = 0;
        for (size_t i = 0; i < a.size(); ++i) { da = std::max(da, std::fabs(a[i] - b[i])); dm = std::max(dm, std::fabs(a[i] - c2[i])); dq = std::max(dq, std::fabs(a[i] - d[i])); }
        std::printf("variant spread: sigmoid %.3g, mscale %.3g, qpe %.3g\n", da, dm, dq);
        CHECK(da > 1e-3 && dm > 1e-3 && dq > 1e-4);
    }
    // sampler is the dsv4 one, reused untouched: greedy must pick the argmax
    { SampleOpts so; Sampler sm(so); std::vector<float> l = {0.1f, 2.0f, -1.0f}; CHECK(sm.sample(l.data(), 3, {}) == 1); }
    std::printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}

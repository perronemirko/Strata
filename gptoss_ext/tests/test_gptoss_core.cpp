// End-to-end check against tests/tiny/golden.txt (numpy, float64) on the tiny MXFP4/BF16 gpt-oss checkpoint.
#include "gptoss/gpu.hpp"
#include "gptoss/model.hpp"
#include "gptoss/weights.hpp"
#include "gptoss/ops.hpp"
#include "gptoss/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
using namespace gptoss;

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/tiny";
    std::ifstream f(dir + "/golden.txt");
    if (!f) { std::fprintf(stderr, "missing %s/golden.txt (python3 tools/tiny_gptoss.py %s)\n", dir.c_str(), dir.c_str()); return 2; }
    std::map<std::string, std::vector<double>> G; std::string line;
    while (std::getline(f, line)) { std::istringstream is(line); std::string name; size_t n; is >> name >> n; std::vector<double> v(n); for (auto& x : v) is >> x; G[name] = v; }
    std::vector<int> ids; for (double d : G["ids"]) ids.push_back((int) d);
    const int V = 96;

    // unit checks
    CHECK(fp4_lut()[7] == 6.0f && fp4_lut()[15] == -6.0f && fp4_lut()[3] == 1.5f);
    { int32_t idx[4]; float w[4]; const float lg[5] = {0.f, 2.f, 1.f, 2.f, -1.f}; route(lg, 5, 2, idx, w);
      CHECK(idx[0] == 1 && idx[1] == 3 && std::fabs(w[0] - 0.5f) < 1e-6 && std::fabs(w[1] - 0.5f) < 1e-6); }                 // ties -> lower index, softmax over selected
    CHECK(std::fabs(swiglu(100.f, 100.f, 7.f, 1.702f) - 8.f * 7.f / (1.f + std::exp(-1.702f * 7.f))) < 1e-4);              // both clamped
    CHECK(std::fabs(swiglu(-1.f, 0.f, 7.f, 1.702f) - (-1.f / (1.f + std::exp(1.702f)))) < 1e-6);

    struct Var { const char* name; bool halves, pairs, noconc; } vars[] = {
        {"base", false, false, false}, {"glu_halves", true, false, false}, {"rope_pairs", false, true, false}, {"no_conc", false, false, true}};
    for (auto& v : vars) {
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.glu_halves = v.halves; o.rope_pairs = v.pairs; o.no_concentration = v.noconc; std::string err;
        if (!m.load(dir, o, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 2; }
        const auto& ref = G[std::string("logits_") + v.name];
        double worst = 0, scale = 0; int am_bad = 0; std::vector<float> lg;
        for (size_t t = 0; t < ids.size(); ++t) {
            m.forward(ids[t], (int) t, &lg);
            int am = 0, rm = 0;
            for (int j = 0; j < V; ++j) {
                const double r = ref[t * V + (size_t) j];
                worst = std::max(worst, std::fabs(r - lg[(size_t) j])); scale = std::max(scale, std::fabs(r));
                if (lg[(size_t) j] > lg[(size_t) am]) am = j;
                if (r > ref[t * V + (size_t) rm]) rm = j;
            }
            am_bad += am != rm;
        }
        const bool ok = worst <= 3e-4 * std::max(scale, 1.0) && am_bad == 0;
        std::printf("%s logits[%-10s] max|diff| = %.3g (max|logit| %.3g), argmax mismatches %d\n", ok ? "ok  " : "FAIL", v.name, worst, scale, am_bad);
        CHECK(ok);
    }
    { const auto &a = G["logits_base"], &b = G["logits_glu_halves"], &c = G["logits_rope_pairs"], &d = G["logits_no_conc"]; double x = 0, y = 0, z = 0;
      for (size_t i = 0; i < a.size(); ++i) { x = std::max(x, std::fabs(a[i] - b[i])); y = std::max(y, std::fabs(a[i] - c[i])); z = std::max(z, std::fabs(a[i] - d[i])); }
      std::printf("variant spread: glu_halves %.3g, rope_pairs %.3g, no_conc %.3g\n", x, y, z); CHECK(x > 1e-3 && y > 1e-3 && z > 1e-3); }

    // batched prefill == golden, with a FULL cache (no ring) and with a WRAPPING ring (window 4 + rollback 6 = 10 slots < 24 tokens)
    const auto& ref = G["logits_base"];
    auto worst_last = [&](const std::vector<float>& lg) { double w = 0; for (int j = 0; j < V; ++j) w = std::max(w, std::fabs(ref[(ids.size() - 1) * V + (size_t) j] - lg[(size_t) j])); return w; };
    for (int rollback : {8192, 6}) for (int chunk : {1, 3, 64}) {
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.prefill_chunk = chunk; o.kv_rollback = rollback; std::string err;
        if (!m.load(dir, o, err)) return 2;
        std::vector<float> lg; m.prefill(ids.data(), (int) ids.size(), 0, &lg);
        const double w = worst_last(lg);
        std::printf("%s prefill chunk=%-2d rollback=%-4d last-token max|diff| = %.3g\n", w < 1e-3 ? "ok  " : "FAIL", chunk, rollback, w);
        CHECK(w < 1e-3); CHECK(m.cached_len() == (int) ids.size());
    }
    // prefix reuse + checkpoint round trip, ring wrapping (rollback 6 -> 10 slots in banded layers)
    {
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.prefill_chunk = 4; o.kv_rollback = 6; std::string err;
        if (!m.load(dir, o, err)) return 2;
        m.prefill(ids.data(), 16, 0, nullptr);
        CHECK(m.save_kv("/tmp/gptoss_test.kv", err));
        Model m2; if (!m2.load(dir, o, err)) return 2;
        CHECK(m2.load_kv("/tmp/gptoss_test.kv", err)); CHECK(m2.cached_len() == 16);
        const int keep = m2.reuse_prefix(ids); CHECK(keep == 16);
        std::vector<float> lg; m2.prefill(ids.data() + keep, (int) ids.size() - keep, keep, &lg);
        const double w = worst_last(lg);
        std::printf("%s KV checkpoint (ring) restore + reuse: 16 cached, %zu prefilled, max|diff| = %.3g\n", w < 1e-3 ? "ok  " : "FAIL", ids.size() - 16, w); CHECK(w < 1e-3);
        // rewinding 6 tokens is inside the ring slack (6): allowed; 7 would need overwritten positions: must fall back to a full recompute
        std::vector<int> a(ids.begin(), ids.begin() + 10); a.push_back(77);
        Model m3; if (!m3.load(dir, o, err)) return 2; m3.prefill(ids.data(), 16, 0, nullptr);
        CHECK(m3.reuse_prefix(a) == 10);
        Model m4_; if (!m4_.load(dir, o, err)) return 2; m4_.prefill(ids.data(), 24, 0, nullptr);
        std::vector<int> b(ids.begin(), ids.begin() + 12); b.push_back(77);          // cached 24, rewind to 12 = 12 back > slack 6 (+window)
        std::printf("ok   rewind beyond the ring -> %d (0 = recompute)\n", m4_.reuse_prefix(b)); 
        Model m5; if (!m5.load(dir, o, err)) return 2; m5.prefill(ids.data(), 24, 0, nullptr);
        CHECK(m5.reuse_prefix(b) == 0);
        // a checkpoint taken with another option set must be refused
        RunOpts o2 = o; o2.glu_halves = true; Model m6; if (!m6.load(dir, o2, err)) return 2;
        CHECK(!m6.load_kv("/tmp/gptoss_test.kv", err));
        std::printf("ok   checkpoint from another option set refused: %s\n", err.c_str());
        RunOpts o3 = o; o3.verbose = true; o3.vram_pct = 90; o3.vram_total_mib = 24576; Model m7; CHECK(m7.load(dir, o3, err));
    }
    // Device path through the CPU-EMULATED device: same Model code as CUDA (dense upload, expert slot pool + LRU, batched matmats), no GPU needed.
    CHECK(gpu::is_emulated());
    for (int slots : {-1, 5, 0}) {                         // all experts resident / 5 slots (constant eviction) / none (CPU fallback for experts)
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.cuda = true; o.vram_total_mib = 1; o.vram_pct = 90; o.vram_reserve_mib = 0;
        o.gpu_slots = slots; o.prefill_chunk = 5; std::string err;
        if (!m.load(dir, o, err)) { std::fprintf(stderr, "device load: %s\n", err.c_str()); return 2; }
        std::vector<float> lg; m.prefill(ids.data(), 16, 0, nullptr);
        double w = 0;
        for (size_t t = 16; t < ids.size(); ++t) {         // decode steps on the device after a batched device prefill
            m.forward(ids[t], (int) t, &lg);
            for (int j = 0; j < V; ++j) w = std::max(w, (double) std::fabs(ref[t * V + (size_t) j] - lg[(size_t) j]));
        }
        Model m2; if (!m2.load(dir, o, err)) return 2;
        std::vector<float> lg2; m2.prefill(ids.data(), (int) ids.size(), 0, &lg2);
        const double w2 = worst_last(lg2);
        std::printf("%s device(emulated) slots=%-2d decode max|diff| = %.3g, full prefill last-token = %.3g\n", (w < 1e-3 && w2 < 1e-3) ? "ok  " : "FAIL", slots, w, w2);
        CHECK(w < 1e-3); CHECK(w2 < 1e-3);
    }
    // Attention + KV resident on the (emulated) device: prefill in chunks -> host KV mirrored -> decode on it -> a prefill that starts after the decoded positions,
    // so the device KV must be re-synced from the host. Full cache and wrapping ring.
    for (int rollback : {8192, 6}) for (int chunk : {3, 5}) {
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.cuda = true; o.vram_total_mib = 512; o.vram_pct = 90; o.vram_reserve_mib = 0;
        o.prefill_chunk = chunk; o.kv_rollback = rollback; std::string err;
        if (!m.load(dir, o, err)) { std::fprintf(stderr, "device load: %s\n", err.c_str()); return 2; }
        CHECK(m.gpu_attention());
        std::vector<float> lg; m.prefill(ids.data(), 16, 0, nullptr);
        double w = 0;
        for (size_t t = 16; t < ids.size(); ++t) {
            m.forward(ids[t], (int) t, &lg);
            for (int j = 0; j < V; ++j) w = std::max(w, (double) std::fabs(ref[t * V + (size_t) j] - lg[(size_t) j]));
        }
        std::vector<float> lg2; m.prefill(ids.data() + ids.size() - 1, 1, (int) ids.size() - 1, &lg2);
        const double w2 = worst_last(lg2);
        Model m2; if (!m2.load(dir, o, err)) return 2;
        std::vector<float> lg3; m2.prefill(ids.data(), (int) ids.size(), 0, &lg3);
        const double w3 = worst_last(lg3);
        const bool ok = w < 1e-3 && w2 < 1e-3 && w3 < 1e-3;
        std::printf("%s device attention+KV chunk=%d rollback=%-4d: decode %.3g, re-prefill after decode %.3g, full prefill %.3g\n", ok ? "ok  " : "FAIL", chunk, rollback, w, w2, w3);
        CHECK(ok);
    }
    // Background preload of the expert slots (emulated device): every slot is filled by the copy thread, published only when complete; afterwards no expert misses.
    {
        Model m; RunOpts o; o.ctx = 64; o.verbose = false; o.cuda = true; o.vram_total_mib = 512; o.vram_pct = 90; o.vram_reserve_mib = 0; o.prefill_chunk = 5; std::string err;
        if (!m.load(dir, o, err)) return 2;
        m.preload_wait();
        std::string st = m.stats();
        const bool done = st.find("(done)") != std::string::npos;
        const bool all = st.find("experts: " + std::to_string(m.config().layers * m.config().n_exp) + " slots") != std::string::npos;   // every expert has a slot
        std::vector<float> lg; m.prefill(ids.data(), 16, 0, nullptr);
        double w = 0;
        for (size_t t = 16; t < ids.size(); ++t) { m.forward(ids[t], (int) t, &lg); for (int j = 0; j < V; ++j) w = std::max(w, (double) std::fabs(ref[t * V + (size_t) j] - lg[(size_t) j])); }
        st = m.stats();
        // This branch's stats line counts "X of the Y routed experts read over PCIe, Z on the host": a completed
        // preload means the decode found a slot for every expert, so both counters must be 0.
        const bool nomiss = !all || (st.find("; 0 of the") != std::string::npos && st.find(", 0 on the host, oom") != std::string::npos);
        std::printf("%s expert preload: %s | decode max|diff| %.3g\n", (done && nomiss && w < 1e-3) ? "ok  " : "FAIL", st.c_str(), w);
        CHECK(done); CHECK(nomiss); CHECK(w < 1e-3);
    }
    { SampleOpts so; Sampler sm(so); std::vector<float> l = {0.1f, 2.0f, -1.0f}; CHECK(sm.sample(l.data(), 3, {}) == 1); }
    std::printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}

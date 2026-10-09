// m4_run - one-shot generation or the resident --serve engine for Mistral Small 4 (CPU).
#include "m4/model.hpp"
#include "m4/sampler.hpp"
#include "m4/serve_loop.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace m4;

static void usage() {
    std::fprintf(stderr,
        "m4_run --model DIR [--ctx N] [--threads N]\n"
        "       [--prompt-ids 1,2,3 --n-predict N [--temperature T --top-k K --top-p P --seed S]] | [--serve [--eos-ids 2,..]]\n"
        "       [--selfcheck] [--prefill-chunk N] [--kv-checkpoint FILE] [--kv-unified]\n"
        "       [--vram-pct P --vram-total-mib N [--vram-reserve-mib N]]   (plan only: no CUDA backend yet)\n"
        "  switches for facts the sources do not settle (see docs/mistral_small4_porting.md):\n"
        "       --router softmax|sigmoid   --mscale-softmax   --l4-qpe   --fp8-scale-div\n");
}

int main(int argc, char** argv) {
    std::string dir, ids_s, eos_s;
    RunOpts o; SampleOpts so;
    int n_predict = 32; bool serve = false, selfcheck = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--model") dir = val();
        else if (a == "--ctx") o.ctx = std::atoi(val());
        else if (a == "--threads") o.threads = std::atoi(val());
        else if (a == "--prompt-ids") ids_s = val();
        else if (a == "--n-predict") n_predict = std::atoi(val());
        else if (a == "--temperature") so.temperature = (float) std::atof(val());
        else if (a == "--top-k") so.top_k = std::atoi(val());
        else if (a == "--top-p") so.top_p = (float) std::atof(val());
        else if (a == "--seed") so.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--eos-ids") eos_s = val();
        else if (a == "--router") { std::string r = val(); o.router = r == "sigmoid" ? Router::Sigmoid : Router::Softmax; }
        else if (a == "--mscale-softmax") o.mscale_softmax = true;
        else if (a == "--l4-qpe") o.l4_qpe_only = true;
        else if (a == "--fp8-scale-div") o.fp8_scale_div = true;
        else if (a == "--prefill-chunk") o.prefill_chunk = std::atoi(val());
        else if (a == "--kv-checkpoint") o.kv_checkpoint = val();
        else if (a == "--kv-unified") {}      // accepted for llama.cpp parity: this engine has ONE sequence and ONE shared cache already
        else if (a == "--vram-pct") o.vram_pct = std::atof(val());
        else if (a == "--vram-total-mib") o.vram_total_mib = std::atoll(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoll(val());
        else if (a == "--serve") serve = true;
        else if (a == "--selfcheck") selfcheck = true;
        else { usage(); return 2; }
    }
    if (dir.empty()) { usage(); return 2; }
    auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str())); return v; };

    Model m; std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!m.load(dir, o, err)) { std::fprintf(stderr, "m4_run: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "m4_run: loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    if (selfcheck) {     // finite logits for one token: catches a wrong dtype/scale convention long before any text is read
        std::vector<float> lg; m.forward(m.config().bos >= 0 ? m.config().bos : 1, 0, &lg);
        double mn = 1e30, mx = -1e30; int bad = 0;
        for (float v : lg) { if (!(v == v) || v > 1e30f || v < -1e30f) ++bad; else { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); } }
        std::printf("selfcheck: %zu logits, non-finite %d, min %.3f max %.3f\n", lg.size(), bad, mn, mx);
        return bad ? 1 : 0;
    }
    if (serve) {
        std::vector<int> eos = split(eos_s);
        return serve_loop(m, o, eos);
    }
    std::vector<int> ids = split(ids_s);
    if (ids.empty()) { usage(); return 2; }
    Sampler sm(so);
    std::vector<int> hist = ids; std::vector<float> lg; int pos = 0;
    m.prefill(ids.data(), (int) ids.size(), 0, &lg); pos = (int) ids.size();
    for (int n = 0; n < n_predict && pos < o.ctx; ++n) {
        const int tok = sm.sample(lg.data(), (int) lg.size(), hist);
        std::printf("%d%s", tok, n + 1 < n_predict ? "," : "\n"); std::fflush(stdout);
        hist.push_back(tok);
        if (tok == m.config().eos) break;
        m.forward(tok, pos++, &lg);
    }
    return 0;
}

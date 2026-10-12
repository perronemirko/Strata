// gptoss_run - one-shot generation or the resident --serve engine for gpt-oss-120b (CPU).
#include "gptoss/gpu.hpp"
#include "gptoss/model.hpp"
#include "gptoss/sampler.hpp"
#include "gptoss/serve_loop.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace gptoss;

static void usage() {
    std::fprintf(stderr,
        "gptoss_run --model DIR [--ctx N] [--threads N]\n"
        "       [--prompt-ids 1,2,3 --n-predict N [--temperature T --top-k K --top-p P --seed S]] | [--serve [--eos-ids 200002,200012]]\n"
        "       [--selfcheck] [--prefill-chunk N] [--kv-rollback N] [--kv-checkpoint FILE] [--kv-unified]\n"
        "       [--cuda [--cuda-device N] [--vram-pct P] [--vram-total-mib N] [--vram-reserve-mib N] [--gpu-slots N] [--expert-profile FILE] [--expert-profile-save FILE] [--expert-profile-save-every SEC] [--pin-pct P] [--expert-promote N] [--expert-lru-layered] [--profile-decay N]]\n"
        "         expert cache, opt-in (0/off = the previous release): --expert-promote N pins an LRU expert after N decode hits and unpins the coldest pinned one;\n"
        "         --expert-lru-layered evicts inside the requesting layer first; --profile-decay N halves the usage profile past N routed activations\n"
        "         --no-device-graphs: the decode experts run as separate kernel enqueues, no CUDA graph captured (the switch that isolates the graph from the stream/event ordering)\n"
        "  facts the model card does not settle (docs/gptoss_porting.md):  --glu-halves  --rope-pairs  --no-concentration\n");
}

int main(int argc, char** argv) {
    std::string dir, ids_s, eos_s;
    RunOpts o; SampleOpts so;
    int n_predict = 32; bool serve = false, selfcheck = false, allow_emu = false;
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
        else if (a == "--prefill-chunk") o.prefill_chunk = std::atoi(val());
        else if (a == "--kv-rollback") o.kv_rollback = std::atoi(val());
        else if (a == "--kv-checkpoint") o.kv_checkpoint = val();
        else if (a == "--kv-unified") {}      // accepted for llama.cpp parity: one sequence, one shared cache already
        else if (a == "--cuda") o.cuda = true;
        else if (a == "--gpu-emu") { o.cuda = true; allow_emu = true; }     // tests only: the "device" is host memory
        else if (a == "--cuda-device") o.cuda_device = std::atoi(val());
        else if (a == "--gpu-slots") o.gpu_slots = std::atoi(val());
        else if (a == "--expert-profile") o.expert_profile = val();
        else if (a == "--expert-profile-save") o.expert_profile_save = val();
        else if (a == "--expert-profile-save-every") o.expert_profile_save_every = std::atof(val());
        else if (a == "--pin-pct") o.pin_pct = std::atof(val());
        else if (a == "--expert-promote") o.expert_promote = std::atoi(val());
        else if (a == "--expert-lru-layered") o.expert_lru_layered = true;
        else if (a == "--profile-decay") o.profile_decay = std::atoll(val());
        else if (a == "--no-device-graphs") o.device_graphs = false;
        else if (a == "--vram-pct") o.vram_pct = std::atof(val());
        else if (a == "--vram-total-mib") o.vram_total_mib = std::atoll(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoll(val());
        else if (a == "--glu-halves") o.glu_halves = true;
        else if (a == "--rope-pairs") o.rope_pairs = true;
        else if (a == "--no-concentration") o.no_concentration = true;
        else if (a == "--serve") serve = true;
        else if (a == "--selfcheck") selfcheck = true;
        else { usage(); return 2; }
    }
    if (dir.empty()) { usage(); return 2; }
    if (o.cuda && gpu::is_emulated() && !allow_emu) { std::fprintf(stderr, "gptoss_run: this binary has the EMULATED device, not CUDA: rebuild with `sh build.sh --cuda`\n"); return 2; }
    auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str())); return v; };
    Model m; std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!m.load(dir, o, err)) { std::fprintf(stderr, "gptoss_run: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "gptoss_run: loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (selfcheck) {
        std::vector<float> lg; m.forward(m.config().bos >= 0 ? m.config().bos : 1, 0, &lg);
        double mn = 1e30, mx = -1e30; int bad = 0;
        for (float v : lg) { if (!(v == v) || v > 1e30f || v < -1e30f) ++bad; else { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); } }
        std::printf("selfcheck: %zu logits, non-finite %d, min %.3f max %.3f\n", lg.size(), bad, mn, mx);
        return bad ? 1 : 0;
    }
    if (serve) { std::vector<int> eos = split(eos_s); return serve_loop(m, o, eos); }
    std::vector<int> ids = split(ids_s);
    if (ids.empty()) { usage(); return 2; }
    Sampler sm(so);
    std::vector<int> hist = ids; std::vector<float> lg;
    m.prefill(ids.data(), (int) ids.size(), 0, &lg);
    int pos = (int) ids.size();
    for (int n = 0; n < n_predict && pos < o.ctx; ++n) {
        const int tok = sm.sample(lg.data(), (int) lg.size(), hist);
        std::printf("%d%s", tok, n + 1 < n_predict ? "," : "\n"); std::fflush(stdout);
        hist.push_back(tok);
        if (tok == m.config().eos) break;
        m.forward(tok, pos++, &lg);
    }
    {   // one-shot mode: write the profile at exit
        const std::string pp = !o.expert_profile_save.empty() ? o.expert_profile_save : o.expert_profile;
        if (!pp.empty()) {
            std::string perr;
            if (m.save_profile(pp, perr)) { std::fprintf(stderr, "gptoss_run: expert profile saved to %s (exit)\n", pp.c_str()); m.decay_profile(); }
            else std::fprintf(stderr, "gptoss_run: profile save FAILED: %s\n", perr.c_str());
        }
    }
    return 0;
}

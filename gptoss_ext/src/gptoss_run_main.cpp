// gptoss_run - one-shot generation or the resident --serve engine for gpt-oss-120b (CPU).
#include "gptoss/gpu.hpp"
#include "gptoss/model.hpp"
#include "gptoss/adapter_gpt_oss.hpp"
#include "gptoss/sampler.hpp"
#include "gptoss/sampler.hpp"
#include "gptoss/serve_loop.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

using namespace gptoss;

static void usage() {
    std::fprintf(stderr,
        "gptoss_run --model DIR [--ctx N] [--threads N]\n"
        "       [--prompt-ids 1,2,3 --n-predict N [--temperature T --top-k K --top-p P --seed S]] | [--serve [--eos-ids 200002,200012]]\n"
        "       [--selfcheck] [--prefill-chunk N] [--kv-rollback N] [--kv-checkpoint FILE] [--kv-unified]\n"
        "       [--cuda [--cuda-device N] [--vram-pct P] [--vram-total-mib N] [--vram-reserve-mib N] [--gpu-slots N] [--expert-profile FILE] [--expert-profile-save FILE] [--expert-profile-save-every SEC] [--pin-pct P] [--expert-promote N] [--expert-lru-layered] [--profile-decay N] [--prefill-dev-pct P]]\n"
        "         -k FMT / --key-format FMT, -v FMT / --value-format FMT: format of the CARD KV cache, K and V INDEPENDENTLY (e4m3 = 1B, f16 = 2B, f32 = 4B; default f32).\n"
        "         e4m3 cuts the KV windows 4x, f16 2x; the host cache and the KV checkpoint stay f32, the attention dequantizes in registers (no fp32 rows in VRAM).\n"
        "         --prefill-dev-pct P: percent of the VRAM left for the device-resident prefill (its KV windows + chunk scratch); 0 = the old prefill (default 50)\n"
        "         expert cache, opt-in (0/off = the previous release): --expert-promote N pins an LRU expert after N decode hits and unpins the coldest pinned one;\n"
        "         --expert-lru-layered evicts inside the requesting layer first; --profile-decay N halves the usage profile past N routed activations\n"
        "         --no-device-graphs: the decode experts run as separate kernel enqueues, no CUDA graph captured (the switch that isolates the graph from the stream/event ordering)\n"
        "         --no-async-dense: the decode's q/k/v run as three separate round-trips (the control arm for the fused one)\n"
        "  facts the model card does not settle (docs/gptoss_porting.md):  --glu-halves  --rope-pairs  --no-concentration\n");
}

extern char** environ;

// Every GPTOSS_* variable the engine reads. An unknown one is almost always a typo, and until now it was silently ignored.
// Keep this list in step with the getenv calls in model.cpp / prefill_dev.cu / gpu_cuda.cu.
static void check_gptoss_env() {
    static const char* known[] = {"GPTOSS_PROFILE", "GPTOSS_ATTN_DEC", "GPTOSS_ATTN_DEC_CTX", "GPTOSS_ATTN_DEC_MIN", "GPTOSS_ATTN_DEC_SHARED",
                                  "GPTOSS_ATTN_PREFILL", "GPTOSS_CHECK_DECODE_ATTN", "GPTOSS_CHECK_PREFILL", "GPTOSS_CHECK_PREFILL_ATTN",
                                  "GPTOSS_CPU_MISS", "GPTOSS_KV_SLOTS", "GPTOSS_KV_SLOTS_GIB", "GPTOSS_PREFETCH", "GPTOSS_PREFETCH_MAX", nullptr};
    std::string set;
    for (char** e = environ; e && *e; ++e) {
        const std::string kv = *e;
        if (kv.compare(0, 7, "GPTOSS_") != 0) continue;
        const std::string k = kv.substr(0, kv.find('='));
        bool ok = false;
        for (const char** p = known; *p && !ok; ++p) ok = (k == *p);
        if (!ok) std::fprintf(stderr, "gptoss_run: WARNING unknown environment variable %s: it is ignored (typo?)\n", k.c_str());
        else set += (set.empty() ? "" : " ") + kv;
    }
    if (!set.empty()) std::fprintf(stderr, "gptoss_run: environment: %s\n", set.c_str());
}

int main(int argc, char** argv) {
    check_gptoss_env();
    std::string dir, ids_s, eos_s;
    RunOpts o; SampleOpts so;
    int n_predict = 32; bool serve = false, selfcheck = false, allow_emu = false;
    bool gateway_info = false;                  // --gateway-info: open gpt-oss only through the gateway, print arch/vocab, exit
    std::string selfcheck_file;                 // --selfcheck-file: reference sequence (written the first time, compared after)
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
        else if ((a == "-k" || a == "--key-format") && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "e4m3")  o.key_format = KVFormat::E4M3;
            else if (val == "f16") o.key_format = KVFormat::F16;
            else if (val == "f32") o.key_format = KVFormat::F32;
            else {
                std::cerr << "Formato Key non valido: " << val << " (usa e4m3, f16, f32)\n";
                exit(1);
            }
        } 
        else if ((a == "-v" || a == "--value-format") && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "e4m3")  o.value_format = KVFormat::E4M3;
            else if (val == "f16") o.value_format = KVFormat::F16;
            else if (val == "f32") o.value_format = KVFormat::F32;
            else {
                std::cerr << "Formato Value non valido: " << val << " (usa e4m3, f16, f32)\n";
                exit(1);
            }
        }
        else if (a == "--cuda") o.cuda = true;
        else if (a == "--gpu-emu") { o.cuda = true; allow_emu = true; }     // tests only: the "device" is host memory
        else if (a == "--cuda-device") o.cuda_device = std::atoi(val());
        else if (a == "--gpu-slots") o.gpu_slots = std::atoi(val());
        else if (a == "--no-gpu-attn") o.gpu_attn = false;
        else if (a == "--no-preload") o.preload = false;
        else if (a == "--preload-wait") o.preload_wait = true;
        else if (a == "--pin-ram-gib") o.pin_ram_gib = std::atof(val());
        else if (a == "--expert-profile") o.expert_profile = val();
        else if (a == "--expert-profile-save") o.expert_profile_save = val();
        else if (a == "--expert-profile-save-every") o.expert_profile_save_every = std::atof(val());
        else if (a == "--pin-pct") o.pin_pct = std::atof(val());
        else if (a == "--expert-promote") o.expert_promote = std::atoi(val());
        else if (a == "--expert-lru-layered") o.expert_lru_layered = true;
        else if (a == "--profile-decay") o.profile_decay = std::atoll(val());
        else if (a == "--no-device-graphs") o.device_graphs = false;
        else if (a == "--no-async-dense") o.async_dense = false;
        else if (a == "--vram-pct") o.vram_pct = std::atof(val());
        else if (a == "--vram-total-mib") o.vram_total_mib = std::atoll(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoll(val());
        else if (a == "--prefill-dev-pct") o.prefill_dev_pct = std::atof(val());
        else if (a == "--glu-halves") o.glu_halves = true;
        else if (a == "--rope-pairs") o.rope_pairs = true;
        else if (a == "--no-concentration") o.no_concentration = true;
        else if (a == "--serve") serve = true;
        else if (a == "--selfcheck") selfcheck = true;
        else if (a == "--gateway-info") gateway_info = true;
        else if (a == "--selfcheck-file") selfcheck_file = val();
        else { usage(); return 2; }
    }
    if (dir.empty()) { usage(); return 2; }
    if (o.cuda && gpu::is_emulated() && !allow_emu) { std::fprintf(stderr, "gptoss_run: this binary has the EMULATED device, not CUDA: rebuild with `sh build.sh --cuda`\n"); return 2; }
    auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str())); return v; };
    if (gateway_info) {
        gptoss::register_gpt_oss(o);
        try {
            auto ad = gptoss::Gateway::instance().open("gpt_oss", dir);
            std::printf("gateway: arch=%s vocab=%d\n", ad->arch(), ad->vocab());
            // Stessa sequenza del selfcheck diretto, ma attraverso l'adapter.
            const int V = ad->vocab() > 0 ? ad->vocab() : 1;
            std::vector<int> pid(64);
            for (int i = 0; i < 64; ++i) pid[(size_t) i] = (i * 997 + 13) % V;
            ad->reset();
            std::vector<float> lg;
            ad->prefill(pid.data(), (int) pid.size(), 0, &lg);
            gptoss::Sampler greedy(gptoss::SampleOpts{});
            std::vector<int> hist = pid, got;
            int pos = (int) pid.size();
            for (int n = 0; n < 16; ++n) {
                const int tok = greedy.sample(lg.data(), (int) lg.size(), hist);
                got.push_back(tok); hist.push_back(tok);
                if (n + 1 < 16) ad->forward(tok, pos++, &lg);
            }
            std::string line;
            for (size_t i = 0; i < got.size(); ++i) line += (i ? "," : "") + std::to_string(got[i]);
            std::printf("GATEWAY_TOKENS %s\n", line.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "gateway: %s\n", e.what());
            return 1;
        }
    }
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    // Il modello si apre attraverso il gateway (facade -> GptOssAdapter -> Model): stesso Model di prima, nessun calcolo nuovo.
    gptoss::register_gpt_oss(o);
    std::unique_ptr<gptoss::ModelAdapter> base;
    try { base = gptoss::Gateway::instance().open("gpt_oss", dir); }
    catch (const std::exception& e) { std::fprintf(stderr, "gptoss_run: %s\n", e.what()); return 1; }
    gptoss::Model& m = dynamic_cast<gptoss::GptOssAdapter*>(base.get())->model();
    std::fprintf(stderr, "gptoss_run: loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (selfcheck) {
        // Deterministic regression test: 64 fixed prompt ids, then 16 greedy tokens. The sequence is printed; with --selfcheck-file
        // it is compared with the reference in that file (written on the first run). Same flags, same reference.
        const int V = m.config().vocab > 0 ? m.config().vocab : 1;
        std::vector<int> pid(64);
        for (int i = 0; i < 64; ++i) pid[(size_t) i] = (i * 997 + 13) % V;
        m.reset();
        std::vector<float> lg;
        m.prefill(pid.data(), (int) pid.size(), 0, &lg);
        int bad = 0;
        for (float v : lg) if (!(v == v) || v > 1e30f || v < -1e30f) ++bad;
        Sampler greedy(SampleOpts{});
        std::vector<int> hist = pid, got;
        int pos = (int) pid.size();
        for (int n = 0; n < 16; ++n) {
            const int tok = greedy.sample(lg.data(), (int) lg.size(), hist);
            got.push_back(tok); hist.push_back(tok);
            if (n + 1 < 16) m.forward(tok, pos++, &lg);
        }
        std::string line;
        for (size_t i = 0; i < got.size(); ++i) line += (i ? "," : "") + std::to_string(got[i]);
        std::printf("selfcheck: %zu logits, non-finite %d, tokens %s\n", lg.size(), bad, line.c_str());
        int rc = bad ? 1 : 0;
        if (!selfcheck_file.empty()) {
            std::ifstream in(selfcheck_file);
            std::string ref;
            if (in && std::getline(in, ref)) {
                if (ref == line) std::printf("selfcheck: MATCH with the reference %s\n", selfcheck_file.c_str());
                else { std::printf("selfcheck: DIFFERENT from the reference %s (reference: %s)\n", selfcheck_file.c_str(), ref.c_str()); rc = 1; }
            } else {
                std::ofstream out(selfcheck_file);
                out << line << "\n";
                std::printf("selfcheck: reference written to %s\n", selfcheck_file.c_str());
            }
        }
        return rc;
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

#include "gptoss/model.hpp"
#include "gptoss/safetensors.hpp"
#include "gptoss/weights.hpp"
#include "gptoss/gpu.hpp"
#include "gptoss/prefill_dev.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gptoss {

namespace {
struct Layer {
    bool sliding = false;
    int cap = 0;                                       // positions held by K/V (ring when sliding && cap < ctx)
    std::vector<float> in_ln, post_ln, sinks, bq, bk, bv, bo, rbias;
    WView q, k, v, o, router;
    MxView gu, dn;
    const uint16_t *gu_bias = nullptr, *dn_bias = nullptr;   // BF16 [E,2I], [E,H] (mmap)
    std::vector<float> K, V;                           // [cap][kv_dim]
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dobj = nullptr, *drt = nullptr;   // device copies of the BF16 projections (null = host)
    float *d_in_ln = nullptr, *d_post_ln = nullptr, *d_bq = nullptr, *d_bk = nullptr, *d_bv = nullptr, *d_bo = nullptr, *d_sinks = nullptr;   // device copies (attention on the device)
    void *dKc = nullptr, *dVc = nullptr;              // device KV cache of the layer, [cap][kv_dim] PACKED by -k / -v (e4m3 = 1B, f16 = 2B, f32 = 4B); same slots as K / V
    float *dKinv = nullptr, *dVinv = nullptr;         // device [cap] inverse scales, one power of two per row (null = f32, no rescaling)
    size_t slot(int pos) const { return sliding ? (size_t) pos % (size_t) cap : (size_t) pos; }
};
struct Work { std::vector<float> h, q, k, v, att, tmp, glog, y, gu, act, eb, sc, part; };   // part: [topk][H] per-expert contributions of one decode layer
struct KvStage {                                       // a prefill chunk's own K/V, kept outside the ring until the chunk's attention has read it
    const float *K = nullptr, *V = nullptr;            // [T][kv_dim] of positions base..base+T-1
    int base = 0;
};
struct Slot {                                          // one expert in VRAM: [gate_up blocks | gate_up scales | down blocks | down scales | gate_up bias | down bias]
    void* base = nullptr;
    uint8_t *gu_b = nullptr, *gu_s = nullptr, *dn_b = nullptr, *dn_s = nullptr;
    uint16_t *gu_bias = nullptr, *dn_bias = nullptr;         // device copies of the BF16 bias rows: the expert kernels add them, so the async pipeline needs no host step
    int layer = -1, exp = -1;
    uint64_t tick = 0;
    bool pinned = false;                               // chosen by the usage profile: never evicted
    bool busy = false;                                 // an async expert kernel is reading this slot: never evicted until it is collected
};
inline bool in_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}
}  // namespace

struct Model::Impl {
    GptConfig c;
    RunOpts o;
    StFiles st;
    std::vector<Layer> L;
    WView embed, lm_head;
    std::vector<float> final_norm, rope_cos, rope_sin, lastn;
    std::vector<Work> works;
    std::vector<int> cached;
    int chunk_max = 1;
    uint64_t oom_fallbacks = 0;                        // device calls that found the scratch/slot allocation did not fit: host fallback, device stays on
    // --- device state (empty/false unless RunOpts::cuda)
    bool gpu_on = false;
    bool pdev_on = false;                            // device-resident prefill (prefill_dev): the chunk never crosses PCIe except X in/out + router logits
    void* d_head = nullptr;                            // lm_head on the device
    std::vector<Slot> pool;                            // expert slots
    std::unordered_map<long long, int> where;          // layer*E+expert -> slot
    uint64_t tick = 0, hits = 0, host_served = 0, pcie_served = 0, pin_hits = 0;
    bool counting = false;                             // true only inside Model::forward (decode): prefill lookups are not counted
    uint64_t base_hits = 0, base_lookups = 0, base_pcie = 0;   // counters at the start of the request, for the per-request line
    std::vector<uint64_t> usage;                       // [layer*E + expert]: times routed to (the profile)
    std::vector<uint64_t> lru_hits;                    // [layer*E + expert]: decode hits on an LRU slot, for --expert-promote
    uint64_t promotions = 0;                           // LRU <-> pinned swaps done by --expert-promote
    std::mutex gmu;                                    // every device call is serialised

    // --- attention on the device (batched prefill): the activations of one chunk stay in VRAM for every layer
    bool attn_dev = false;
    float *d_cos = nullptr, *d_sin = nullptr;          // RoPE tables [ctx][D/2]
    float *a_X = nullptr, *a_Hn = nullptr, *a_Q = nullptr, *a_K = nullptr, *a_V = nullptr, *a_Att = nullptr, *a_Ob = nullptr, *a_Gl = nullptr, *a_Y = nullptr;
    // --- background preload of the expert slots
    struct PreTarget { int slot, layer, exp; };
    std::vector<PreTarget> pre_targets;
    std::thread pre_th;
    std::atomic<bool> pre_stop{false};
    std::atomic<size_t> pre_done{0};
    std::atomic<int> pre_state{0};                     // 0 off, 1 running, 2 done, 3 stopped on an error
    // --- --pin-ram-gib: page-lock the expert tensors of the busiest layers so every copy out of them is a direct DMA
    std::vector<void*> ram_blocks;                     // page-locked copies of whole layers' expert tensors (freed at exit)
    std::atomic<size_t> ram_layers{0};                 // layers registered so far
    std::atomic<size_t> ram_bytes{0};
    std::atomic<int> ram_state{0};                     // 0 off, 1 running, 2 done, 3 stopped on an error
    bool in_check = false;                             // GPTOSS_CHECK_PREFILL: inside the device-vs-CPU comparison
    int pdev_kv_pos = 0;                               // the same watermark for the pdev (prefill_dev.cu) mirror, which is separate from attn_dev's
    int dev_kv_pos = 0;                                // positions [0, dev_kv_pos) of the device KV are known to equal the host KV (decode writes the host only)
    int dev_kfmt = 2, dev_vfmt = 2;                        // formats the DEVICE KV cache is packed in (the emulated device keeps f32)

    void init_work(Work& w) const;
    Work& my_work();
    void attn_in(Layer& y, Work& w, const float* x, int pos, const KvStage& st = KvStage());
    void attn_out(Layer& y, Work& w, float* x, int pos, const KvStage& st = KvStage());
    void route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt);
    void expert_add(Layer& y, Work& w, const float* h, int e, float wt, float* acc);
    void experts_decode(Layer& y, Work& w, const float* h, const int32_t* idx, const float* wt, float* acc);   // the token's top-k experts, in flight together
    std::string fingerprint() const;
    void plan_vram() const;
    bool gpu_setup(std::string& err);
    void gpu_fail(const std::string& what);
    void pdev_fail_(const std::string& why);             // the device-resident prefill turns itself off: the device stays on, the old prefill continues
    bool upload_expert(Slot& sl, int li, int e);
    void count_use(int li, const int32_t* idx);
    Slot* resident(int layer, int e);                  // expert -> slot (upload on miss = PCIe, LRU eviction); nullptr when there is no pool or the expert runs on the host
    int victim_slot(int li);                           // slot to evict for a miss in layer li: this layer first (--expert-lru-layered), then the whole pool
    void promote_locked(int li, int e, int slot_idx);  // an LRU expert that keeps being used takes the place of the least-used pinned one
    void begin_request_stats();
    void request_stats(unsigned long long*, unsigned long long*, unsigned long long*) const;
    void mv(const WView& w, const void* dW, const float* x, float* y);
    void mm(const WView& w, const void* dW, const float* x, int N, float* y);
    void attn_heads(Layer& y, Work& w, int pos, const KvStage& st = KvStage());
    void prefill_gpu(const int* tokens, int n, int start_pos, std::vector<float>* logits_last);
    pdev::Cfg pdev_cfg() const;
    /// Device-resident prefill. Returns how many leading tokens were fully processed (a multiple of chunk_max): n on success,
    /// the failed chunk's start if a device call failed mid-way - the host cache is consistent up to that point and the caller
    /// resumes the rest with the old path.
    int prefill_pdev(const int* tokens, int n, int start_pos, std::vector<float>* logits_last);
    void moe_prefill_layer(int l, int T, const float* Hn, const int32_t* IDX, const float* WT, float* Y);   // routed experts of one layer for a chunk (Y = sum, zeroed here)
    bool chunk_dev(int T, int p0, const float* X0, float* Xlast);   // one chunk, whole stack on the device; false = redo it on the host
    bool kv_sync_to_dev(int upto);
    bool setup_attn_dev(double& budget);
    void release_attn();
    void preload_plan();
    void preload_run();
    void preload_start();
    void preload_join();
    void preload_wait();
    void ram_pin_start();
};

Model::Model() : p_(new Impl) {}
Model::~Model() {
    if (!p_) return;
    Impl& s = *p_;
    s.preload_join();                                  // before any slot is released: a copy may still be in flight
    if (s.hits + s.host_served + s.pcie_served) std::fprintf(stderr, "gptoss: expert slots (%zu): %llu hits, %llu over PCIe, %llu on the host\n", s.pool.size(),
                   (unsigned long long) s.hits, (unsigned long long) s.pcie_served, (unsigned long long) s.host_served);
    pdev::shutdown();                                // its buffers are separate allocations; free them before the process exits
    for (void* b : s.ram_blocks) gpu::host_free(b);
    for (auto& sl : s.pool) gpu::release(sl.base);
    for (auto& y : s.L) { gpu::release(y.dq); gpu::release(y.dk); gpu::release(y.dv); gpu::release(y.dobj); gpu::release(y.drt); }
    gpu::release(s.d_head);
    s.release_attn();
}
const GptConfig& Model::config() const { return p_->c; }
uint64_t Model::weight_bytes() const { return p_->st.total_bytes(); }
void Model::reset() { p_->cached.clear(); p_->dev_kv_pos = 0; p_->pdev_kv_pos = 0; }
void Model::preload_wait() { p_->preload_wait(); }
bool Model::gpu_attention() const { return p_->attn_dev && p_->gpu_on; }
int Model::cached_len() const { return (int) p_->cached.size(); }

bool Model::load(const std::string& dir, const RunOpts& opts, std::string& err) {
    Impl& s = *p_;
    s.o = opts;
#ifdef _OPENMP
    if (opts.threads > 0) omp_set_num_threads(opts.threads);
#endif
    std::string cfgdir = dir;
    struct stat sb;
    if (stat(dir.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) { const size_t k = dir.find_last_of('/'); cfgdir = k == std::string::npos ? "." : dir.substr(0, k); }
    if (!config_from_file(cfgdir + "/config.json", s.c, err)) return false;
    GptConfig& c = s.c;
    std::vector<std::string> shards;
    if (!discover_shards(dir, shards, err) || !s.st.open(shards, err)) return false;
    if (opts.verbose) std::fprintf(stderr, "gptoss: %s\ngptoss: %zu shard(s), %zu tensors, %.2f GiB mapped (not read yet)\n", config_summary(c).c_str(), shards.size(),
                                   s.st.tensors().size(), (double) s.st.total_bytes() / 1073741824.0);

    auto get = [&](const std::string& n, const StTensor*& t) { t = s.st.find(n); if (!t) err = "missing tensor " + n; return t != nullptr; };
    auto shape_is = [&](const StTensor* t, std::vector<int64_t> want) {
        if (t->shape == want) return true;
        err = "tensor " + t->name + " has shape [";
        for (size_t i = 0; i < t->shape.size(); ++i) err += (i ? "," : "") + std::to_string(t->shape[i]);
        err += "], expected [";
        for (size_t i = 0; i < want.size(); ++i) err += (i ? "," : "") + std::to_string(want[i]);
        err += "]";
        return false;
    };
    auto dense = [&](const std::string& n, int rows, int cols, WView& v) {
        const StTensor* t; if (!get(n, t) || !shape_is(t, {rows, cols})) return false;
        if (t->dtype != DType::BF16 && t->dtype != DType::F32 && t->dtype != DType::F16) { err = n + ": dtype " + t->dtype_str + " not implemented"; return false; }
        v = {t->data, rows, cols, t->dtype}; return true;
    };
    auto vec = [&](const std::string& n, int len, std::vector<float>& out) {
        const StTensor* t; if (!get(n, t) || !shape_is(t, {len})) return false; out = decode_all(*t); return true;
    };
    auto u8 = [&](const std::string& n, std::vector<int64_t> shape, const uint8_t*& p) {
        const StTensor* t; if (!get(n, t) || !shape_is(t, shape)) return false;
        if (t->dtype != DType::U8) { err = n + ": expected U8, got " + t->dtype_str; return false; }
        p = t->data; return true;
    };

    const std::string P = c.prefix;
    const int H = c.hidden, E = c.n_exp, I = c.moe_ff, D = c.head_dim, QD = c.q_dim(), KD = c.kv_dim();
    const StTensor* t;
    if (!dense(P + "embed_tokens.weight", c.vocab, H, s.embed) || !dense("lm_head.weight", c.vocab, H, s.lm_head) || !vec(P + "norm.weight", H, s.final_norm)) return false;
    s.L.resize((size_t) c.layers);
    const int ring = std::min(opts.ctx, c.window + std::max(opts.kv_rollback, 0));
    for (int l = 0; l < c.layers; ++l) {
        Layer& y = s.L[(size_t) l];
        const std::string b = P + "layers." + std::to_string(l) + ".";
        y.sliding = c.sliding[(size_t) l] != 0;
        y.cap = y.sliding ? ring : opts.ctx;
        if (!vec(b + "input_layernorm.weight", H, y.in_ln) || !vec(b + "post_attention_layernorm.weight", H, y.post_ln) ||
            !vec(b + "self_attn.sinks", c.heads, y.sinks)) return false;
        if (!dense(b + "self_attn.q_proj.weight", QD, H, y.q) || !dense(b + "self_attn.k_proj.weight", KD, H, y.k) ||
            !dense(b + "self_attn.v_proj.weight", KD, H, y.v) || !dense(b + "self_attn.o_proj.weight", H, QD, y.o) ||
            !dense(b + "mlp.router.weight", E, H, y.router)) return false;
        if (!vec(b + "self_attn.q_proj.bias", QD, y.bq) || !vec(b + "self_attn.k_proj.bias", KD, y.bk) || !vec(b + "self_attn.v_proj.bias", KD, y.bv) ||
            !vec(b + "self_attn.o_proj.bias", H, y.bo) || !vec(b + "mlp.router.bias", E, y.rbias)) return false;
        y.gu.rows = 2 * I; y.gu.cols = H; y.dn.rows = H; y.dn.cols = I;
        if (!u8(b + "mlp.experts.gate_up_proj_blocks", {E, 2 * I, H / 32, 16}, y.gu.blocks) || !u8(b + "mlp.experts.gate_up_proj_scales", {E, 2 * I, H / 32}, y.gu.scales) ||
            !u8(b + "mlp.experts.down_proj_blocks", {E, H, I / 32, 16}, y.dn.blocks) || !u8(b + "mlp.experts.down_proj_scales", {E, H, I / 32}, y.dn.scales)) return false;
        if (!get(b + "mlp.experts.gate_up_proj_bias", t) || !shape_is(t, {E, 2 * I}) || t->dtype != DType::BF16) { if (err.empty()) err = "gate_up_proj_bias must be BF16"; return false; }
        y.gu_bias = (const uint16_t*) t->data;
        if (!get(b + "mlp.experts.down_proj_bias", t) || !shape_is(t, {E, H}) || t->dtype != DType::BF16) { if (err.empty()) err = "down_proj_bias must be BF16"; return false; }
        y.dn_bias = (const uint16_t*) t->data;
        y.K.assign((size_t) y.cap * KD, 0.0f); y.V.assign((size_t) y.cap * KD, 0.0f);
    }
    std::vector<double> f;
    yarn_freqs(D, c.rope_theta, c.yarn_factor, c.yarn_orig, c.yarn_beta_fast, c.yarn_beta_slow, c.yarn_truncate, f);
    rope_table(f, opts.ctx, opts.no_concentration ? 1.0 : yarn_concentration(c.yarn_factor), s.rope_cos, s.rope_sin);
    // The batch is no longer limited by the ring's margin: a prefill chunk stages its own K/V (KvStage) and commits it to the ring only after
    // the chunk's attention has read it, so a 512-token chunk runs on a 132-position ring. With N >= 32 the device uses the tensor-core GEMM.
    // 0 = auto: a device wants the whole prompt in as few chunks as possible (the per-chunk cost is the PCIe round-trip, not the math),
    // the CPU keeps the old 512. gpu_setup may still halve it, so the chunk's scratch fits the VRAM left after the dense weights.
    const int pchunk = opts.prefill_chunk > 0 ? opts.prefill_chunk : (opts.cuda ? 8192 : 512);
    s.chunk_max = std::max(1, pchunk);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    s.works.resize((size_t) nt);
    for (auto& w : s.works) s.init_work(w);
    s.lastn.assign(H, 0);
    if (opts.verbose) {
        double kv = 0; for (auto& y : s.L) kv += 2.0 * (double) y.cap * KD * 4.0;
        std::fprintf(stderr, "gptoss: KV cache %.2f GiB (fp32; banded layers hold %d positions, dense layers %d)\n", kv / 1073741824.0, ring, opts.ctx);
    }
    s.usage.assign((size_t) c.layers * (size_t) c.n_exp, 0);
    s.lru_hits.assign((size_t) c.layers * (size_t) c.n_exp, 0);
    if (!opts.expert_profile.empty()) {
        if (FILE* fp = std::fopen(opts.expert_profile.c_str(), "rb")) {
            std::fclose(fp);
            std::string perr;
            if (load_profile(opts.expert_profile, perr)) std::fprintf(stderr, "gptoss: expert profile loaded from %s\n", opts.expert_profile.c_str());
            else std::fprintf(stderr, "gptoss: expert profile ignored (%s)\n", perr.c_str());
        } else std::fprintf(stderr, "gptoss: no expert profile at %s yet (it is created after the first request)\n", opts.expert_profile.c_str());
    }
    if (opts.cuda) {
        std::string gerr;
        if (!s.gpu_setup(gerr)) { err = "CUDA: " + gerr; return false; }
    } else if (opts.vram_pct >= 0) s.plan_vram();
    if (!opts.kv_checkpoint.empty()) {
        if (FILE* fp = std::fopen(opts.kv_checkpoint.c_str(), "rb")) {
            std::fclose(fp);
            std::string kerr;
            if (load_kv(opts.kv_checkpoint, kerr)) std::fprintf(stderr, "gptoss: KV checkpoint restored: %zu tokens from %s\n", s.cached.size(), opts.kv_checkpoint.c_str());
            else std::fprintf(stderr, "gptoss: KV checkpoint ignored (%s)\n", kerr.c_str());
        }
    }
    return true;
}

void Model::Impl::init_work(Work& w) const {
    w.h.assign(c.hidden, 0); w.q.assign(c.q_dim(), 0); w.k.assign(c.kv_dim(), 0); w.v.assign(c.kv_dim(), 0); w.att.assign(c.q_dim(), 0);
    w.tmp.assign(c.hidden, 0); w.glog.assign(c.n_exp, 0); w.y.assign(c.hidden, 0); w.gu.assign((size_t) 2 * c.moe_ff, 0);
    w.act.assign(c.moe_ff, 0); w.eb.assign(c.hidden, 0); w.sc.assign((size_t) o.ctx + 1, 0); w.part.assign((size_t) c.topk * c.hidden, 0);
}

Work& Model::Impl::my_work() {
#ifdef _OPENMP
    return works[(size_t) omp_get_thread_num() % works.size()];
#else
    return works[0];
#endif
}

void Model::Impl::plan_vram() const {
    uint64_t dense = 0, experts0 = 0;
    for (const auto& t : st.tensors()) {
        if (t.name.find(".experts.") == std::string::npos) dense += t.nbytes();
        else if (t.name.find("layers.0.") != std::string::npos) experts0 += t.nbytes();
    }
    const uint64_t slot = experts0 / (uint64_t) c.n_exp;
    const double total = (double) o.vram_total_mib * 1048576.0, budget = total * o.vram_pct / 100.0 - (double) o.vram_reserve_mib * 1048576.0 - (double) dense;
    std::fprintf(stderr, "gptoss: VRAM plan only (--cuda not given, nothing is uploaded): card %lld MiB x %.0f%% - reserve %lld MiB - dense weights %.2f GiB\n",
                 o.vram_total_mib, o.vram_pct, o.vram_reserve_mib, (double) dense / 1073741824.0);
    if (o.vram_total_mib <= 0) { std::fprintf(stderr, "gptoss:   --vram-total-mib missing: cannot plan\n"); return; }
    if (budget < (double) slot) { std::fprintf(stderr, "gptoss:   the dense weights alone do not fit in that budget: 0 expert slots\n"); return; }
    const long long slots = (long long) (budget / (double) slot), per_layer = std::min<long long>(slots / c.layers, c.n_exp);
    std::fprintf(stderr, "gptoss:   expert = %.2f MiB -> %lld slots, %lld per layer of %d (%.0f%% of the experts)\n", (double) slot / 1048576.0, slots, per_layer, c.n_exp,
                 100.0 * (double) per_layer / c.n_exp);
}

std::string Model::Impl::fingerprint() const {
    char b[640];
    int ring = 0; for (const auto& y : L) if (y.sliding) { ring = y.cap; break; }
    std::snprintf(b, sizeof b, "%s|glu_halves=%d|rope_pairs=%d|no_conc=%d|ring=%d|bytes=%llu", config_summary(c).c_str(), (int) o.glu_halves, (int) o.rope_pairs,
                  (int) o.no_concentration, ring, (unsigned long long) st.total_bytes());
    return b;
}

// ---- building blocks shared by the single-token path and the batched prefill

void Model::Impl::attn_in(Layer& y, Work& w, const float* x, int pos, const KvStage& st) {
    const int H = c.hidden, D = c.head_dim, half = D / 2, QD = c.q_dim(), KD = c.kv_dim();
    rmsnorm(x, y.in_ln.data(), c.eps, H, w.h.data());
    // The three projections read the SAME x. On the device they run as ONE upload, three kernels and ONE wait
    // instead of three full round-trips: the decode's per-token device calls drop from ~180 to ~108.
    bool fused = false;
    if (o.async_dense && gpu_on && y.dq && y.dk && y.dv && !in_parallel_region()) {
        std::lock_guard<std::mutex> g(gmu);
        gpu::DenseOp ops[3];
        ops[0].W = y.dq; ops[0].rows = y.q.rows; ops[0].cols = y.q.cols;
        ops[1].W = y.dk; ops[1].rows = y.k.rows; ops[1].cols = y.k.cols;
        ops[2].W = y.dv; ops[2].rows = y.v.rows; ops[2].cols = y.v.cols;
        if (gpu::dense_run_async(0, w.h.data(), 1, ops, 3)) {
            float* out[3] = { w.q.data(), w.k.data(), w.v.data() };
            fused = gpu::dense_collect(0, out);
            if (!fused) gpu_fail(gpu::last_error());
        } else {
            gpu_fail(gpu::last_error());
        }
    }
    if (!fused) {
        mv(y.q, y.dq, w.h.data(), w.q.data());
        mv(y.k, y.dk, w.h.data(), w.k.data());
        mv(y.v, y.dv, w.h.data(), w.v.data());
    }
    for (int i = 0; i < QD; ++i) w.q[(size_t) i] += y.bq[(size_t) i];
    for (int i = 0; i < KD; ++i) { w.k[(size_t) i] += y.bk[(size_t) i]; w.v[(size_t) i] += y.bv[(size_t) i]; }
    const float* cs = &rope_cos[(size_t) pos * half];
    const float* sn = &rope_sin[(size_t) pos * half];
    for (int h = 0; h < c.heads; ++h) rotary(&w.q[(size_t) h * D], D, cs, sn, o.rope_pairs);
    for (int h = 0; h < c.kv_heads; ++h) rotary(&w.k[(size_t) h * D], D, cs, sn, o.rope_pairs);
    if (st.K) {                                       // batched prefill: the chunk's K/V goes to the stage, not into the ring
        std::memcpy(const_cast<float*>(st.K) + (size_t) (pos - st.base) * KD, w.k.data(), sizeof(float) * (size_t) KD);
        std::memcpy(const_cast<float*>(st.V) + (size_t) (pos - st.base) * KD, w.v.data(), sizeof(float) * (size_t) KD);
    } else {
        std::memcpy(&y.K[y.slot(pos) * KD], w.k.data(), sizeof(float) * (size_t) KD);
        std::memcpy(&y.V[y.slot(pos) * KD], w.v.data(), sizeof(float) * (size_t) KD);
    }
}

void Model::Impl::attn_heads(Layer& y, Work& w, int pos, const KvStage& st) {
    const int D = c.head_dim, KD = c.kv_dim(), rep = c.heads / c.kv_heads;   // H unused: heads read q/K/V, not the residual
    const int lo = y.sliding ? std::max(0, pos - c.window + 1) : 0;
    const float scale = 1.0f / std::sqrt((float) D);
    // A staged position (this chunk's own K/V, not yet in the ring) is read from the stage; everything older is still in the ring.
    auto key_at = [&](int u, int g) {
        return (st.K && u >= st.base) ? st.K + (size_t) (u - st.base) * KD + (size_t) g * D
                                      : &y.K[y.slot(u) * KD + (size_t) g * D];
    };
    auto val_at = [&](int u, int g) {
        return (st.V && u >= st.base) ? st.V + (size_t) (u - st.base) * KD + (size_t) g * D
                                      : &y.V[y.slot(u) * KD + (size_t) g * D];
    };
    auto one_head = [&](Work& ww, int hd) {
        const float* q = &w.q[(size_t) hd * D];
        const int g = hd / rep;
        float* sc = ww.sc.data();
        float mx = y.sinks[(size_t) hd];                       // the sink joins the softmax denominator only (no value)
        for (int u = lo; u <= pos; ++u) {
            const float* k = key_at(u, g);
            float a = 0;
            for (int j = 0; j < D; ++j) a += q[j] * k[j];
            sc[u - lo] = a * scale;
            mx = std::max(mx, sc[u - lo]);
        }
        double z = std::exp((double) y.sinks[(size_t) hd] - mx);
        for (int u = lo; u <= pos; ++u) { sc[u - lo] = std::exp(sc[u - lo] - mx); z += sc[u - lo]; }
        float* out = &w.att[(size_t) hd * D];
        for (int j = 0; j < D; ++j) out[j] = 0;
        for (int u = lo; u <= pos; ++u) {
            const float p = (float) (sc[u - lo] / z);
            const float* v = val_at(u, g);
            for (int j = 0; j < D; ++j) out[j] += p * v[j];
        }
    };
    bool in_par = false;
#ifdef _OPENMP
    in_par = omp_in_parallel();
#endif
    if (in_par) { for (int hd = 0; hd < c.heads; ++hd) one_head(w, hd); }
    else {
#pragma omp parallel for schedule(static)
        for (int hd = 0; hd < c.heads; ++hd) one_head(my_work(), hd);
    }
}

void Model::Impl::attn_out(Layer& y, Work& w, float* x, int pos, const KvStage& st) {
    attn_heads(y, w, pos, st);
    mv(y.o, y.dobj, w.att.data(), w.tmp.data());
    for (int i = 0; i < c.hidden; ++i) x[i] += w.tmp[(size_t) i] + y.bo[(size_t) i];
}

void Model::Impl::route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt) {
    rmsnorm(x, y.post_ln.data(), c.eps, c.hidden, w.h.data());
    mv(y.router, y.drt, w.h.data(), w.glog.data());
    for (int e = 0; e < c.n_exp; ++e) w.glog[(size_t) e] += y.rbias[(size_t) e];
    route(w.glog.data(), c.n_exp, c.topk, idx, wt);
    count_use((int) (&y - L.data()), idx);
}

void Model::Impl::gpu_fail(const std::string& what) {
    if (what.empty()) {                                // not a CUDA call that failed: a path we turned off ourselves
        // (prefill_dev keeps its own error string and never touches gpu::last_error). An empty message is by
        // definition not a poisoned context: the device stays on and the caller already chose another path.
        return;
    }
    if (gpu::last_oom()) {                             // an allocation that did not fit is NOT a poisoned context: run this call on the host,
        ++oom_fallbacks;                               // shrink the chunk so the next one fits, and keep the device for the rest of the session
        if (chunk_max > 1) {
            chunk_max = std::max(1, chunk_max / 2);
            std::fprintf(stderr, "gptoss: CUDA call failed (%s): prefill chunk halved to %d, the device stays on\n", what.c_str(), chunk_max);
        } else
            std::fprintf(stderr, "gptoss: CUDA call failed (%s): this call runs on the host, the device stays on\n", what.c_str());
        return;
    }
    std::fprintf(stderr, "gptoss: CUDA call failed (%s): continuing on the CPU for the rest of the session\n", what.c_str());
    gpu_on = false;                                    // sticky errors poison every later call: stop using the device
}

bool Model::Impl::gpu_setup(std::string& err) {
    if (!gpu::init(o.cuda_device, err)) return false;
    gpu::set_options(o.device_graphs);   // --no-device-graphs: the same kernels enqueued one by one, no graph captured
    size_t freeb = 0, totb = 0;
    gpu::mem_info(&freeb, &totb);
    const double total = o.vram_total_mib > 0 ? (double) o.vram_total_mib * 1048576.0 : (double) totb;
    const double pct = o.vram_pct > 0 ? o.vram_pct : 90.0;
    double budget = std::min(total * pct / 100.0, (double) freeb) - (double) o.vram_reserve_mib * 1048576.0;
    const double budget0 = budget;
    auto up = [&](const WView& w, void*& d) {          // BF16 only; whatever does not fit stays on the host
        if (w.dt != DType::BF16) return;
        const size_t n = (size_t) w.rows * (size_t) w.cols * 2;
        if ((double) n > budget) return;
        d = gpu::alloc(n);
        if (!d) return;
        if (!gpu::h2d(d, w.p, n)) { gpu::release(d); d = nullptr; return; }
        budget -= (double) n;
    };
    up(lm_head, d_head);                               // biggest single read per token: first
    for (auto& y : L) { up(y.q, y.dq); up(y.k, y.dk); up(y.v, y.dv); up(y.o, y.dobj); up(y.router, y.drt); }
    const double dense_used = budget0 - budget;

    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff;
    // The dense scratch must fit the WHOLE prefill chunk, not one token: prefill_gpu calls mm(..., T = chunk_max) on q/k/v/o/router, so run_matmat
    // grows it to T * cols / T * rows floats. Sized for N == 1 it OOMs at the first prefill call and the session falls back to the CPU.
    // Reserve it BEFORE the slots take the VRAM (no cudaMalloc mid-generation), and halve the chunk until the scratch fits what is left.
    size_t max_rows = (size_t) c.n_exp;                // rows of the projections that run batched: q, k, v, o, router
    for (const Layer& y : L) { max_rows = std::max(max_rows, std::max((size_t) y.q.rows, std::max((size_t) y.k.rows, std::max((size_t) y.v.rows, std::max((size_t) y.o.rows, (size_t) y.router.rows))))); }
    const double ctx_floats = 3.0 * (double) H + 3.0 * (double) I + 1.0;   // one expert context at N == 1 (d_x, d_wt, d_gu, d_act, d_eb, d_y)
    // The batch expert context (context kExpertCtxBatch, used by the pipelined prefill) is the SAME buffers at N == chunk:
    // it is the biggest single reservation of the whole plan (at gpt-oss-120b: 2I = 28800 rows of fp32 GU alone), so it
    // is counted in the loop that halves the chunk - a chunk that survives always leaves room for its own expert batch.
    // The pipelined prefill holds TWO of them at once (expert a computes while a+1 is gathered/uploaded), so the budget
    // counts 2 * chunk * ctx_floats, not 1.
    // The async dense context (decode's fused q/k/v) is a FIXED cost, independent of the chunk: one row of x plus the
    // three output blocks. Count it in the loop that halves the chunk, so the chunk that survives always leaves room
    // for it: reserving it only after the loop would OOM at load and turn the whole device off.
    // reserve_scratch sizes BOTH of its buffers at this number (the y blocks are the bigger one), so the budget counts
    // exactly what is allocated: 2 * (QD + 2*KD) floats.
    const double dense_async_bytes = o.async_dense
        ? 8.0 * (double) ((size_t) c.q_dim() + 2 * (size_t) c.kv_dim())
        : 0.0;
    const int want_chunk = std::max(1, chunk_max);
    int chunk = want_chunk;
    size_t nx = 0, ny = 0;
    for (;;) {
        nx = (size_t) chunk * H;
        ny = std::max((size_t) chunk * max_rows, (size_t) c.vocab);        // the lm_head/router mv() calls run at N == 1 over vocab rows
        // the four decode contexts stay at N == 1; the batch context grows with the chunk, so it is counted at N == chunk
        const double need = 4.0 * ((double) (nx + ny) + (double) gpu::kExpertCtxBatch * ctx_floats + (double) chunk * ctx_floats) + dense_async_bytes;
        if (need <= budget) break;
        if (chunk == 1) { std::fprintf(stderr, "gptoss:   the dense scratch does not fit the VRAM left: the device path stays off\n"); err = "dense scratch does not fit"; return false; }
        chunk /= 2;
    }
    // Reserve it with the rest of the scratch, BEFORE the slots take the VRAM: no cudaMalloc mid-generation.
    const size_t dense_async_y = (size_t) c.q_dim() + 2 * (size_t) c.kv_dim();
    if (!gpu::reserve_scratch(nx, ny, chunk, (int) H, (int) I, o.async_dense ? dense_async_y : (size_t) 0)) { err = gpu::last_error(); return false; }
    budget -= 4.0 * ((double) (nx + ny) + (double) gpu::kExpertCtxBatch * ctx_floats + (double) chunk * ctx_floats) + dense_async_bytes;
    if (chunk < want_chunk) {
        chunk_max = chunk;
        std::fprintf(stderr, "gptoss: prefill chunk %d -> %d so the dense scratch fits the VRAM left\n", want_chunk, chunk);
    }
    // Device-resident prefill (prefill_dev) claims its share of the VRAM BEFORE the expert pool: a dense layer's
    // attention reads the whole history, so its K/V must be on the card for the chunk to stay on the card. It gets
    // --prefill-dev-pct % of what the dense weights and the chunk scratch left; the pool takes the rest. Whatever
    // does not fit turns this path off for the session - the old prefill is still correct, just slower.
    if (pdev::available() && o.prefill_dev_pct > 0) {
        std::vector<char> sl((size_t) c.layers);
        for (int l = 0; l < c.layers; ++l) sl[(size_t) l] = L[(size_t) l].sliding ? 1 : 0;
        const size_t pdev_budget = (size_t) (std::max(0.0, budget) * std::min((double) o.prefill_dev_pct, 90.0) / 100.0);
        if (pdev::init(pdev_cfg(), sl.data(), pdev_budget)) {
            bool up_ok = true;
            for (int l = 0; l < c.layers && up_ok; ++l) {
                Layer& y = L[(size_t) l];
                up_ok = pdev::upload_layer(l, y.sliding, y.in_ln.data(), y.post_ln.data(), y.sinks.data(), y.bq.data(), y.bk.data(),
                                           y.bv.data(), y.bo.data(), y.rbias.data(), y.dq, y.dk, y.dv, y.dobj, y.drt);
            }
            if (up_ok) {
                pdev::set_rope(rope_cos.data(), rope_sin.data(), o.ctx);
                pdev_on = true;
                budget -= (double) pdev::reserved_bytes();
                const int got = pdev::chosen_chunk();
                if (got > 0 && got != chunk_max) {
                    chunk_max = got;
                    std::fprintf(stderr, "gptoss: device-resident prefill chunk %d -> %d so the reservation fits\n", want_chunk, got);
                }
                std::fprintf(stderr, "gptoss: device-resident prefill on (%.2f GiB of KV windows + scratch; the expert pool takes the rest)\n",
                             (double) pdev::reserved_bytes() / 1073741824.0);
                {
                    static const char* fn[] = { "e4m3", "f16", "f32" };
                    std::fprintf(stderr, "gptoss: card KV formats K=%s V=%s (%zu/%zu bytes per element; the host cache and the checkpoint stay f32)\n",
                                 fn[(int) o.key_format], fn[(int) o.value_format], get_kv_format_bytes(o.key_format), get_kv_format_bytes(o.value_format));
                }
            } else {
                pdev::shutdown();
                std::fprintf(stderr, "gptoss: device-resident prefill off (a layer's dense weights are not all on the device): old prefill\n");
            }
        } else {
            std::fprintf(stderr, "gptoss: device-resident prefill off (its KV windows + scratch do not fit its share of the VRAM left): old prefill\n");
        }
    }
    // Attention on the device is for the batched prefill only; the device-resident prefill (prefill_dev) keeps the whole
    // chunk on the card already and runs instead of it, so when that one is on there is nothing to reserve.
    if (o.gpu_attn && !pdev_on) setup_attn_dev(budget);
    auto al = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t s_gub = 2 * I * (H / 32) * 16, s_gus = 2 * I * (H / 32), s_dnb = H * (I / 32) * 16, s_dns = H * (I / 32);
    const size_t s_gubias = 2 * I * 2, s_dnbias = H * 2;   // the BF16 bias rows travel with the expert: the async pipeline has no host step to add them
    const size_t o2 = al(s_gub), o3 = o2 + al(s_gus), o4 = o3 + al(s_dnb), o5 = o4 + al(s_dns), o6 = o5 + al(s_gubias), slot_bytes = o6 + al(s_dnbias);
    long long want = std::min<long long>((long long) c.layers * c.n_exp, budget > 0 ? (long long) (budget / (double) slot_bytes) : 0);
    if (o.gpu_slots >= 0) want = std::min<long long>(want, o.gpu_slots);
    for (long long i = 0; i < want; ++i) {
        void* b = gpu::alloc(slot_bytes);
        if (!b) break;
        Slot sl;
        uint8_t* p = (uint8_t*) b;
        sl.base = b; sl.gu_b = p; sl.gu_s = p + o2; sl.dn_b = p + o3; sl.dn_s = p + o4;
        sl.gu_bias = (uint16_t*) (p + o5); sl.dn_bias = (uint16_t*) (p + o6);
        pool.push_back(sl);
    }
    if (!pool.empty() && o.pin_pct > 0 && !o.preload) {   // (--no-preload) pin the hottest experts of the profile, globally ranked, synchronously
        std::vector<std::pair<uint64_t, int>> rank;
        for (size_t i = 0; i < usage.size(); ++i) if (usage[i] > 0) rank.push_back({usage[i], (int) i});
        std::sort(rank.begin(), rank.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        const size_t npin = std::min(rank.size(), (size_t) ((double) pool.size() * std::min(o.pin_pct, 100.0) / 100.0));
        uint64_t total = 0, covered = 0;
        for (auto v : usage) total += v;
        size_t done = 0;
        for (size_t i = 0; i < npin; ++i) {
            const int li = rank[i].second / c.n_exp, e = rank[i].second % c.n_exp;
            Slot& sl = pool[done];
            if (!upload_expert(sl, li, e)) break;
            sl.layer = li; sl.exp = e; sl.pinned = true; sl.tick = 0;
            where[(long long) li * c.n_exp + e] = (int) done;
            covered += rank[i].first; ++done;
        }
        std::fprintf(stderr, "gptoss: pinned %zu hot experts (%.1f%% of the routed activations in the profile); %zu slots stay LRU\n",
                     done, total ? 100.0 * (double) covered / (double) total : 0.0, pool.size() - done);
    }
    gpu_on = true;
    std::fprintf(stderr, "gptoss: CUDA device '%s' (%s): dense weights %.2f GiB on device%s, %zu expert slots of %.2f MiB (%.0f%% of the %d experts)\n",
                 gpu::device_name(), gpu::is_emulated() ? "EMULATED" : "real", dense_used / 1073741824.0, d_head ? "" : " (lm_head stays on the host: no room)",
                 pool.size(), (double) slot_bytes / 1048576.0, 100.0 * (double) pool.size() / ((double) c.layers * c.n_exp), c.layers * c.n_exp);
    ram_pin_start();   // before the preload: it then reads the pinned copies
    preload_start();
    if (o.preload_wait) {
        const auto t0 = std::chrono::steady_clock::now();
        preload_wait();
        std::fprintf(stderr, "gptoss: preload finished: %zu experts in %.1f s\n", pre_done.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return true;
}

bool Model::Impl::upload_expert(Slot& sl, int li, int e) {
    const Layer& y = L[(size_t) li];
    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff, G1 = H / 32, G2 = I / 32;
    return gpu::h2d(sl.gu_b, y.gu.blocks + (size_t) e * 2 * I * G1 * 16, 2 * I * G1 * 16) &&
           gpu::h2d(sl.gu_s, y.gu.scales + (size_t) e * 2 * I * G1, 2 * I * G1) &&
           gpu::h2d(sl.dn_b, y.dn.blocks + (size_t) e * H * G2 * 16, H * G2 * 16) &&
           gpu::h2d(sl.dn_s, y.dn.scales + (size_t) e * H * G2, H * G2) &&
           gpu::h2d(sl.gu_bias, y.gu_bias + (size_t) e * 2 * I, 2 * I * 2) &&
           gpu::h2d(sl.dn_bias, y.dn_bias + (size_t) e * H, H * 2);
}

// The device-resident prefill found a reason not to run this chunk on the card. That is NOT a CUDA call that
// failed: gpu::last_error() is empty and gpu_on must stay true - the old prefill path does the rest of the chunk
// with the device. pdev::last_error() says which call refused and why (a CUDA error, or a policy refusal such as
// a history longer than the mirrored KV window); it is printed here because it is the only copy.
void Model::Impl::pdev_fail_(const std::string& why) {
    pdev_on = false;
    const std::string perr = pdev::last_error();
    std::fprintf(stderr, "gptoss: device-resident prefill off (%s): %s: the old prefill continues on the device\n",
                 why.c_str(), perr.empty() ? "no CUDA call failed" : perr.c_str());
}

// Everything the device needs to run the whole stack of a prefill chunk without going back to the host between the projections: device KV per layer, the
// per-layer norms/biases/sinks, the RoPE tables and the chunk's activation buffers. All or nothing (one layer on the host would break the resident flow);
// at most 45% of what is left of the budget, and the chunk is shortened if its activations do not fit.
bool Model::Impl::setup_attn_dev(double& budget) {
    const size_t H = (size_t) c.hidden, QD = (size_t) c.q_dim(), KD = (size_t) c.kv_dim(), E = (size_t) c.n_exp, D = (size_t) c.head_dim;
    auto no = [&](const char* why) { if (o.verbose) std::fprintf(stderr, "gptoss: attention stays on the host in the prefill (%s)\n", why); return false; };
    // The shape rule belongs to the BACKEND's kernel, not to the path: the emulated device runs the host math and accepts
    // any shape (it is the executable spec the tests run against), the CUDA kernel owns one dimension per lane. Ask it.
    gpu::AttnShape probe;
    probe.heads = c.heads; probe.kv_heads = c.kv_heads; probe.D = c.head_dim; probe.window = c.window; probe.cap = 1;
    if (!gpu::attn_supported(probe)) return no("the attention kernel needs head_dim a multiple of 32, at most 128");
    for (const auto& y : L) if (!y.dq || !y.dk || !y.dv || !y.dobj || !y.drt) return no("a dense matmul did not fit on the device");
    // Dynamic byte math for the card's KV cache: the format comes from -k / -v, K and V INDEPENDENTLY.
    const bool emu = gpu::is_emulated();
    dev_kfmt = emu ? 2 : (int) o.key_format;
    dev_vfmt = emu ? 2 : (int) o.value_format;
    const size_t kb = get_kv_format_bytes((gptoss::KVFormat) dev_kfmt), vb = get_kv_format_bytes((gptoss::KVFormat) dev_vfmt);
    const size_t single_layer_elements = (size_t) o.ctx * KD;   // ctx x kv_heads x head_dim
    const size_t total_k_bytes = single_layer_elements * (size_t) c.layers * kb;
    const size_t total_v_bytes = single_layer_elements * (size_t) c.layers * vb;
    (void) total_k_bytes; (void) total_v_bytes;   // the same math a raw cudaMalloc pair would use
    double kvb = 0;
    for (const auto& y : L) {
        kvb += (double) y.cap * (double) KD * (double) (kb + vb);   // packed rows, not fp32
        kvb += 2.0 * (double) y.cap * 4.0;                          // the two per-row scale arrays
    }
    const double small = (double) c.layers * (double) (2 * H + QD + 2 * KD + H + (size_t) c.heads) * 4.0 + 2.0 * (double) o.ctx * (double) (D / 2) * 4.0;
    const double ppt = (double) (4 * H + 2 * QD + 2 * KD + E) * 4.0;                       // device bytes per token of the chunk
    const double room = budget * 0.45 - kvb - small;
    const long long tmax = room > 0 ? (long long) (room / ppt) : 0;
    if (tmax < std::min<long long>(chunk_max, 64)) return no("the device KV cache does not fit; try a smaller --ctx");
    if (tmax < chunk_max) {
        if (o.verbose) std::fprintf(stderr, "gptoss: prefill chunk %d -> %lld (the activations stay in VRAM)\n", chunk_max, tmax);
        chunk_max = (int) tmax;
    }
    bool ok = true;
    auto upf = [&](const float* h, size_t n, float*& d) {
        if (!ok) return;
        d = (float*) gpu::alloc(n * 4);
        if (!d || !gpu::h2d(d, h, n * 4)) ok = false;
    };
    auto buf = [&](size_t n, float*& d) { if (!ok) return; d = (float*) gpu::alloc(n * 4); if (!d) ok = false; };
    for (auto& y : L) {
        upf(y.in_ln.data(), H, y.d_in_ln); upf(y.post_ln.data(), H, y.d_post_ln); upf(y.bq.data(), QD, y.d_bq); upf(y.bk.data(), KD, y.d_bk);
        upf(y.bv.data(), KD, y.d_bv); upf(y.bo.data(), H, y.d_bo); upf(y.sinks.data(), (size_t) c.heads, y.d_sinks);
        auto pbufV = [&](size_t n, void*& d) { if (!ok) return; d = gpu::alloc(n); if (!d) ok = false; };
        pbufV((size_t) y.cap * KD * kb, y.dKc); pbufV((size_t) y.cap * KD * vb, y.dVc);
        if (kb < 4) { auto f = [&](size_t n, float*& d) { if (!ok) return; d = (float*) gpu::alloc(n); if (!d) ok = false; }; f((size_t) y.cap * 4, y.dKinv); }
        if (vb < 4) { auto f = [&](size_t n, float*& d) { if (!ok) return; d = (float*) gpu::alloc(n); if (!d) ok = false; }; f((size_t) y.cap * 4, y.dVinv); }
        // rows never uploaded must still read a defined scale: 1.0 = no rescaling
        if (kb < 4) { std::vector<float> one((size_t) y.cap, 1.0f); if (!gpu::h2d(y.dKinv, one.data(), (size_t) y.cap * 4)) ok = false; }
        if (vb < 4) { std::vector<float> one((size_t) y.cap, 1.0f); if (!gpu::h2d(y.dVinv, one.data(), (size_t) y.cap * 4)) ok = false; }
    }
    upf(rope_cos.data(), rope_cos.size(), d_cos); upf(rope_sin.data(), rope_sin.size(), d_sin);
    const size_t T = (size_t) chunk_max;
    buf(T * H, a_X); buf(T * H, a_Hn); buf(T * QD, a_Q); buf(T * KD, a_K); buf(T * KD, a_V); buf(T * QD, a_Att); buf(T * H, a_Ob); buf(T * E, a_Gl); buf(T * H, a_Y);
    if (!ok) { release_attn(); return no("device allocation failed"); }
    budget -= kvb + small + ppt * (double) T;
    attn_dev = true;
    dev_kv_pos = 0;
    if (o.verbose) std::fprintf(stderr, "gptoss: attention + KV on the device for the prefill: KV %.2f GiB, activations for %d tokens %.2f GiB\n", kvb / 1073741824.0, chunk_max, ppt * (double) T / 1073741824.0);
    return true;
}

void Model::Impl::release_attn() {
    for (auto& y : L) {
        float** v[] = {&y.d_in_ln, &y.d_post_ln, &y.d_bq, &y.d_bk, &y.d_bv, &y.d_bo, &y.d_sinks, &y.dKinv, &y.dVinv};
        for (float** p : v) { gpu::release(*p); *p = nullptr; }
        void** w[] = {&y.dKc, &y.dVc};
        for (void** p : w) { gpu::release(*p); *p = nullptr; }
    }
    float** g[] = {&d_cos, &d_sin, &a_X, &a_Hn, &a_Q, &a_K, &a_V, &a_Att, &a_Ob, &a_Gl, &a_Y};
    for (float** p : g) { gpu::release(*p); *p = nullptr; }
    attn_dev = false;
}

// Host KV -> device KV for positions [dev_kv_pos, upto) (what decode wrote since the last device prefill); banded layers only need the last `cap` of them.
bool Model::Impl::kv_sync_to_dev(int upto) {
    const size_t KD = (size_t) c.kv_dim();
    for (auto& y : L) {
        const int a = std::max(dev_kv_pos, y.sliding ? std::max(0, upto - y.cap) : 0);
        for (int pos = a; pos < upto;) {
            const size_t slot = y.slot(pos);
            const int run = y.sliding ? std::min(upto - pos, y.cap - (int) slot) : upto - pos;
            if (!gpu::kv_upload_rows(dev_kfmt, &y.K[slot * KD], y.dKc, y.dKinv, run, pos, (int) KD, y.cap, y.sliding, 448.0f) ||
                !gpu::kv_upload_rows(dev_vfmt, &y.V[slot * KD], y.dVc, y.dVinv, run, pos, (int) KD, y.cap, y.sliding, 448.0f)) return false;
            pos += run;
        }
    }
    dev_kv_pos = std::max(dev_kv_pos, upto);
    return true;
}

// ---- background preload of the expert slots (startup). The copy side of the device API is thread-safe and runs on its own stream, so a thread fills the slots
// while the rest of the startup (and the first requests) go on. Order: the pinned experts of the profile, then the rest of the profile hottest first, then
// layer order. A slot under load is `busy` (neither victim nor free for resident()) and unpublished (not in `where`); the thread publishes it under gmu only after
// the copy is COMPLETE (event_sync), so the compute side never sees half a weight. A request that wants an expert not published yet just takes the normal miss path.
void Model::Impl::preload_plan() {
    const size_t NE = (size_t) c.n_exp, total_keys = (size_t) c.layers * NE;
    std::vector<std::pair<uint64_t, int>> rank;
    for (size_t i = 0; i < usage.size(); ++i) if (usage[i] > 0) rank.push_back({usage[i], (int) i});
    std::sort(rank.begin(), rank.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    const size_t npin = o.pin_pct > 0 ? std::min(rank.size(), (size_t) ((double) pool.size() * std::min(o.pin_pct, 100.0) / 100.0)) : 0;
    std::vector<char> taken(total_keys, 0);
    pre_targets.clear();
    auto add = [&](size_t key, bool pin) {
        if (pre_targets.size() >= pool.size() || taken[key]) return;
        Slot& sl = pool[pre_targets.size()];
        sl.busy = true; sl.pinned = pin; sl.layer = -1;
        pre_targets.push_back({(int) pre_targets.size(), (int) (key / NE), (int) (key % NE)});
        taken[key] = 1;
    };
    for (size_t i = 0; i < npin; ++i) add((size_t) rank[i].second, true);
    for (size_t i = npin; i < rank.size(); ++i) add((size_t) rank[i].second, false);
    for (size_t k = 0; k < total_keys; ++k) add(k, false);
    uint64_t tot = 0, cov = 0;
    for (auto v : usage) tot += v;
    for (size_t i = 0; i < npin; ++i) cov += rank[i].first;
    std::fprintf(stderr, "gptoss: preloading %zu experts in the background (%zu pinned = %.1f%% of the routed activations in the profile)\n", pre_targets.size(), npin,
                 tot ? 100.0 * (double) cov / (double) tot : 0.0);
}

void Model::Impl::preload_run() {
    gpu::bind_thread();
    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff, G1 = H / 32, G2 = I / 32;
    const int ev = gpu::event_create();
    size_t next = 0;                                   // targets [0, next) are published or released
    bool failed = ev < 0;
    constexpr size_t B = 4;                            // experts per event wait
    while (!failed && next < pre_targets.size() && !pre_stop.load()) {
        const size_t n = std::min(B, pre_targets.size() - next);
        bool ok = true;
        for (size_t j = 0; j < n && ok; ++j) {
            const PreTarget& t = pre_targets[next + j];
            Slot& sl = pool[(size_t) t.slot];
            const Layer& y = L[(size_t) t.layer];
            const size_t e = (size_t) t.exp;
            // The slot carries the BF16 bias rows too (this branch's layout): a preloaded expert must get them,
            // or the device kernels would add whatever was in the slot before. Same ranges as upload_expert.
            ok = gpu::h2d_copy_async(sl.gu_b, y.gu.blocks + e * 2 * I * G1 * 16, 2 * I * G1 * 16) && gpu::h2d_copy_async(sl.gu_s, y.gu.scales + e * 2 * I * G1, 2 * I * G1) &&
                 gpu::h2d_copy_async(sl.dn_b, y.dn.blocks + e * H * G2 * 16, H * G2 * 16) && gpu::h2d_copy_async(sl.dn_s, y.dn.scales + e * H * G2, H * G2) &&
                 gpu::h2d_copy_async(sl.gu_bias, y.gu_bias + e * 2 * I, (size_t) 2 * I * 2) && gpu::h2d_copy_async(sl.dn_bias, y.dn_bias + e * H, (size_t) H * 2);
        }
        ok = ok && gpu::event_record_copy(ev) && gpu::event_sync(ev);     // always waited for, even on failure paths below: no DMA may outlive its slot
        std::lock_guard<std::mutex> g(gmu);
        for (size_t j = 0; j < n; ++j) {
            const PreTarget& t = pre_targets[next + j];
            Slot& sl = pool[(size_t) t.slot];
            const long long key = (long long) t.layer * c.n_exp + t.exp;
            if (ok && where.find(key) == where.end()) { sl.layer = t.layer; sl.exp = t.exp; sl.tick = 0; where[key] = t.slot; }
            else sl.pinned = false;                    // failed, or a miss loaded it first: the slot is an ordinary free one
            sl.busy = false;
        }
        next += n;
        pre_done.store(next);
        if (!ok) failed = true;
    }
    {
        std::lock_guard<std::mutex> g(gmu);
        for (size_t k = next; k < pre_targets.size(); ++k) { Slot& sl = pool[(size_t) pre_targets[k].slot]; sl.busy = false; sl.pinned = false; }   // stopped / failed: free the rest
    }
    if (failed) std::fprintf(stderr, "gptoss: expert preload stopped after %zu experts (%s): the slots fill on demand\n", next, gpu::last_error().c_str());
    pre_state.store(failed ? 3 : 2);
}

void Model::Impl::preload_start() {
    if (pool.empty() || !o.preload) return;
    preload_plan();
    pre_state.store(1);
    pre_th = std::thread([this] { preload_run(); });
}

void Model::Impl::preload_join() {
    pre_stop.store(true);
    if (pre_th.joinable()) pre_th.join();
}

// --pin-ram-gib: copies the expert tensors of whole layers (the layers the profile uses most first) into page-locked host
// memory and points the layer at the copy, until the budget is spent. This is the way that works on every card: registering the
// weights' file mapping in place (cudaHostRegister) needs the read-only flag that GeForce drivers often refuse. It runs at load,
// BEFORE the preload and before any request, so nothing reads the old pointers while they move. Copies out of the pinned blocks
// are direct DMA, and the CPU kernels read them as before.
void Model::Impl::ram_pin_start() {
    if (!(o.pin_ram_gib > 0) || !gpu_on) return;
    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff, G1 = H / 32, G2 = I / 32, NE = (size_t) c.n_exp;
    const size_t sz[4] = { NE * 2 * I * G1 * 16, NE * 2 * I * G1, NE * H * G2 * 16, NE * H * G2 };
    const size_t per_layer = sz[0] + sz[1] + sz[2] + sz[3];
    const size_t budget = (size_t) (o.pin_ram_gib * 1073741824.0);
    std::vector<std::pair<uint64_t, int>> order;
    for (int l = 0; l < c.layers; ++l) {
        uint64_t u = 0;
        for (size_t e = 0; e < NE; ++e) u += usage[(size_t) l * NE + e];
        order.push_back({u, l});
    }
    std::stable_sort(order.begin(), order.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first > b.first; });
    ram_state.store(1);
    const auto t0 = std::chrono::steady_clock::now();
    size_t used = 0, n = 0;
    bool failed = false;
    for (const auto& pr : order) {
        if (used + per_layer > budget) break;
        uint8_t* buf = (uint8_t*) gpu::host_alloc(per_layer);
        if (!buf) { failed = true; break; }
        Layer& y = L[(size_t) pr.second];
        const uint8_t* src[4] = { y.gu.blocks, y.gu.scales, y.dn.blocks, y.dn.scales };
        size_t off = 0;
        for (int k = 0; k < 4; ++k) { std::memcpy(buf + off, src[k], sz[k]); off += sz[k]; }
        y.gu.blocks = buf; y.gu.scales = buf + sz[0]; y.dn.blocks = buf + sz[0] + sz[1]; y.dn.scales = buf + sz[0] + sz[1] + sz[2];
        ram_blocks.push_back(buf);
        used += per_layer; ++n;
        ram_layers.store(n); ram_bytes.store(used);
    }
    std::fprintf(stderr, "gptoss: %zu layers of experts in page-locked RAM (%.1f GiB, %.0f s)%s: their copies to the card are direct DMA\n", n, (double) used / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), failed ? "; the system would not give more pinned memory" : "");
    ram_state.store(failed && n == 0 ? 3 : 2);
}

void Model::Impl::preload_wait() {
    while (pre_state.load() == 1) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void Model::Impl::count_use(int li, const int32_t* idx) {
    for (int k = 0; k < c.topk; ++k) __atomic_fetch_add(&usage[(size_t) li * (size_t) c.n_exp + (size_t) idx[k]], (uint64_t) 1, __ATOMIC_RELAXED);
}

Slot* Model::Impl::resident(int li, int e) {
    // The counters are the decode's, per request: only Model::forward sets `counting`, so a prefill that walks
    // every expert of a chunk does not dilute the hit rate the DONE line reports.
    const bool count = counting;
    if (pool.empty()) { if (count) ++host_served; return nullptr; }
    const long long key = (long long) li * c.n_exp + e;
    ++tick;
    auto it = where.find(key);
    if (it != where.end()) {
        Slot& hit = pool[(size_t) it->second];
        if (count) {
            ++hits;
            if (hit.pinned) ++pin_hits;
            else ++lru_hits[(size_t) key];
            // --expert-promote: an LRU expert that keeps being used swaps its PIN with the least-used pinned
            // expert. Both experts are already in VRAM, so the swap costs no PCIe and no slot is added: it only
            // moves WHICH experts hold the slots. Without the flag this branch is never taken.
            if (o.expert_promote > 0 && (long long) lru_hits[(size_t) key] >= (long long) o.expert_promote)
                promote_locked(li, e, (int) it->second);
        }
        hit.tick = tick;
        return &pool[(size_t) it->second];
    }
    // A miss: the bytes go to the card over PCIe if a slot can take them; otherwise this expert runs on the host.
    int v = -1;
    for (size_t i = 0; i < pool.size() && v < 0; ++i) if (pool[i].layer < 0 && !pool[i].busy) v = (int) i;   // a slot being filled by the preload thread is not free
    if (v < 0) v = o.expert_lru_layered ? victim_slot(li) : -1;   // --expert-lru-layered: evict inside this layer first
    if (v < 0) {                                       // LRU
        uint64_t best = ~(uint64_t) 0;
        for (size_t i = 0; i < pool.size(); ++i) if (!pool[i].pinned && !pool[i].busy && pool[i].tick < best) { best = pool[i].tick; v = (int) i; }
    }
    if (v < 0) { if (count) ++host_served; return nullptr; }                         // every slot is pinned or in flight: this expert runs from the host
    Slot& sl = pool[(size_t) v];
    if (sl.layer >= 0) where.erase((long long) sl.layer * c.n_exp + sl.exp);
    sl.layer = -1;
    const bool ok = upload_expert(sl, li, e);
    if (!ok) { if (count) ++host_served; gpu_fail(gpu::last_error()); return nullptr; }
    if (count) ++pcie_served;                            // the upload really crossed the link: the device computes this expert
    sl.layer = li; sl.exp = e; sl.tick = tick;
    where[key] = v;
    return &sl;
}

// Which slot a miss in layer `li` should evict. With --expert-lru-layered the cold expert of one layer may not
// push out the hot expert of another: the model has 36 layers x 128 experts and a card takes ~1400 slots, so a
// global LRU lets one chatty layer evict the LRU of every other layer. Prefer this layer's own LRU slot; if this
// layer has none free, fall back to the global scan in resident().
int Model::Impl::victim_slot(int li) {
    int v = -1;
    uint64_t best = ~(uint64_t) 0;
    for (size_t i = 0; i < pool.size(); ++i)
        if (!pool[i].pinned && !pool[i].busy && pool[i].layer == li && pool[i].tick < best) { best = pool[i].tick; v = (int) i; }
    return v;
}

// The swap behind --expert-promote. Both slots are already filled, so this is a two-line change of bookkeeping:
// no upload, no eviction of a resident expert, no extra slot. The demoted expert goes back to the LRU pool and
// is re-uploaded only if it is routed again.
void Model::Impl::promote_locked(int li, int e, int slot_idx) {
    const long long key = (long long) li * c.n_exp + e;
    const uint64_t mine = usage[(size_t) key];
    // The pinned set stays the size --pin-pct asked for: below that cap a promoted expert only pins, at the cap
    // it trades with the coldest pinned one. With --pin-pct 0 the cap is 0 and this flag does nothing, which is
    // what a pure-LRU run wants: there is no pinned set to correct.
    const size_t cap = (size_t) ((double) pool.size() * std::min(std::max(o.pin_pct, 0.0), 100.0) / 100.0);
    size_t pinned_now = 0;
    for (const Slot& sl : pool) pinned_now += sl.pinned ? 1 : 0;
    int w = -1;
    if (pinned_now < cap) {
        // room left in the pinned set: no expert has to be demoted
    } else {
        uint64_t worst = ~(uint64_t) 0;
        for (size_t i = 0; i < pool.size(); ++i)
            if (pool[i].pinned && !pool[i].busy && usage[(size_t) pool[i].layer * (size_t) c.n_exp + (size_t) pool[i].exp] < worst) {
                worst = usage[(size_t) pool[i].layer * (size_t) c.n_exp + (size_t) pool[i].exp];
                w = (int) i;
            }
        // Only trade if the demoted pinned expert really is colder than the one being promoted: this is what
        // stops two experts of the same rank from changing places on every token.
        if (w < 0 || worst >= mine) { lru_hits[(size_t) key] = 0; return; }
        if (pool[(size_t) w].layer < 0) { lru_hits[(size_t) key] = 0; return; }
        pool[(size_t) w].pinned = false;
        lru_hits[(size_t) pool[(size_t) w].layer * (size_t) c.n_exp + (size_t) pool[(size_t) w].exp] = 0;
    }
    // Neither expert changes slot, so `where` stays exactly as it is: only the PIN flag trades. The promoted
    // expert stops being an eviction candidate; a demoted one becomes evictable and its bytes are dropped
    // only later, if the LRU scan picks it.
    pool[(size_t) slot_idx].pinned = true;
    lru_hits[(size_t) key] = 0;
    ++promotions;
}

void Model::Impl::mv(const WView& w, const void* dW, const float* x, float* y) {
    if (gpu_on && dW && !in_parallel_region()) {
        std::lock_guard<std::mutex> g(gmu);
        if (gpu::matmat_bf16(dW, w.rows, w.cols, x, 1, y)) return;
        gpu_fail(gpu::last_error());
    }
    matvec(w, x, y);
}

void Model::Impl::mm(const WView& w, const void* dW, const float* x, int N, float* y) {
    if (gpu_on && dW && !in_parallel_region()) {
        std::lock_guard<std::mutex> g(gmu);
        if (gpu::matmat_bf16(dW, w.rows, w.cols, x, N, y)) return;
        gpu_fail(gpu::last_error());
    }
    for (int n = 0; n < N; ++n) matvec(w, x + (size_t) n * w.cols, y + (size_t) n * w.rows);
}

void Model::Impl::expert_add(Layer& y, Work& w, const float* h, int e, float wt, float* acc) {
    const int H = c.hidden, I = c.moe_ff, G1 = H / 32, G2 = I / 32;
    Slot* sl = nullptr;
    if (gpu_on && !pool.empty() && !in_parallel_region()) { std::lock_guard<std::mutex> g(gmu); sl = resident((int) (&y - L.data()), e); }
    bool ok = false;
    if (sl) {
        std::lock_guard<std::mutex> g(gmu);
        ok = gpu::matmat_mx(sl->gu_b, sl->gu_s, 2 * I, H, h, 1, w.gu.data());
        if (!ok) gpu_fail(gpu::last_error());
    }
    if (!ok) {
        MxView gu = y.gu; gu.blocks += (size_t) e * 2 * I * G1 * 16; gu.scales += (size_t) e * 2 * I * G1;
        matvec_mx(gu, h, w.gu.data());
    }
    const uint16_t* gb = y.gu_bias + (size_t) e * 2 * I;
    for (int i = 0; i < 2 * I; ++i) w.gu[(size_t) i] += bf16_to_f32(gb[i]);
    for (int i = 0; i < I; ++i) {
        const float g = o.glu_halves ? w.gu[(size_t) i] : w.gu[(size_t) 2 * i];
        const float u = o.glu_halves ? w.gu[(size_t) I + i] : w.gu[(size_t) 2 * i + 1];
        w.act[(size_t) i] = swiglu(g, u, c.swiglu_limit, c.swiglu_alpha);
    }
    ok = false;
    if (sl && gpu_on) {
        std::lock_guard<std::mutex> g(gmu);
        ok = gpu::matmat_mx(sl->dn_b, sl->dn_s, H, I, w.act.data(), 1, w.eb.data());
        if (!ok) gpu_fail(gpu::last_error());
    }
    if (!ok) {
        MxView dn = y.dn; dn.blocks += (size_t) e * H * G2 * 16; dn.scales += (size_t) e * H * G2;
        matvec_mx(dn, w.act.data(), w.eb.data());
    }
    const uint16_t* db = y.dn_bias + (size_t) e * H;
    for (int i = 0; i < H; ++i) acc[i] += wt * (w.eb[(size_t) i] + bf16_to_f32(db[i]));
}

// Decode, one token: the top-k experts of the layer are all enqueued on their own context/stream (x and wt up, 4 kernels or one CUDA graph, y down) and
// only then collected, instead of k times [matmat gate_up, host SwiGLU, matmat down] with a full sync each. A miss uploads over PCIe while the experts
// already enqueued are computing. Whatever has no slot (all pinned/in flight, upload or enqueue failed, k > contexts) runs on the CPU
// while the device works. The sum into acc stays in k order.
void Model::Impl::experts_decode(Layer& y, Work& w, const float* h, const int32_t* idx, const float* wt, float* acc) {
    const int H = c.hidden, I = c.moe_ff, K = c.topk, li = (int) (&y - L.data());
    Slot* fly[gpu::kExpertCtxBatch] = {};
    if (gpu_on && !pool.empty() && !in_parallel_region()) {
        std::lock_guard<std::mutex> g(gmu);
        for (int k = 0; k < K && k < gpu::kExpertCtxBatch && gpu_on; ++k) {
            Slot* sl = resident(li, idx[k]);
            if (!sl) continue;
            gpu::ExpertDesc d;
            d.gu_b = sl->gu_b; d.gu_s = sl->gu_s; d.gu_bias = sl->gu_bias;
            d.dn_b = sl->dn_b; d.dn_s = sl->dn_s; d.dn_bias = sl->dn_bias;
            d.H = H; d.I = I; d.halves = o.glu_halves ? 1 : 0; d.limit = c.swiglu_limit; d.alpha = c.swiglu_alpha;
            sl->busy = true;                                   // set before the enqueue: the next resident() of this loop must not pick it as the victim
            if (gpu::expert_run_async(k, d, h, &wt[k], 1)) fly[k] = sl;
            else { sl->busy = false; gpu_fail(gpu::last_error()); }
        }
    }
    float* part = w.part.data();
    for (int k = 0; k < K; ++k) {                              // CPU share: runs while the device computes the experts above
        if (k < gpu::kExpertCtxBatch && fly[k]) continue;
        float* pk = part + (size_t) k * H;
        std::fill(pk, pk + H, 0.0f);
        expert_add(y, w, h, idx[k], wt[k], pk);
    }
    for (int k = 0; k < K && k < gpu::kExpertCtxBatch; ++k) {  // collect: the device already applied wt[k] and the down bias
        if (!fly[k]) continue;
        float* pk = part + (size_t) k * H;
        bool ok;
        { std::lock_guard<std::mutex> g(gmu); ok = gpu::expert_collect(k, pk); fly[k]->busy = false; if (!ok) gpu_fail(gpu::last_error()); }
        if (!ok) { std::fill(pk, pk + H, 0.0f); expert_add(y, w, h, idx[k], wt[k], pk); }
    }
    for (int k = 0; k < K; ++k) { const float* pk = part + (size_t) k * H; for (int i = 0; i < H; ++i) acc[i] += pk[i]; }
}

// The routed experts of one layer for a whole chunk: Y = the sum of the top-k contributions (zeroed here). Shared by the
// host-resident prefill (prefill_gpu) and the device-resident one (chunk_dev): in both, the experts' input rows come from
// the host, so this is the only part that still crosses PCIe per layer.
void Model::Impl::moe_prefill_layer(int l, int T, const float* Hn, const int32_t* IDX, const float* WT, float* Y) {
    const int H = c.hidden, E = c.n_exp, K = c.topk, I = c.moe_ff;
    Layer& y = L[(size_t) l];
    std::vector<float> Xe, GU, ACT, EB;
    std::fill(Y, Y + (size_t) T * H, 0.0f);
    std::vector<std::vector<int>> by_exp((size_t) E);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) by_exp[(size_t) IDX[(size_t) t * K + k]].push_back(t * K + k);
    auto sync_expert = [&](int e) {                                        // synchronous path: [device matmat | host SwiGLU | device matmat], or all on the host
        const auto& lst = by_exp[(size_t) e];
        const int N = (int) lst.size();
        if (N == 0) return;
        Xe.assign((size_t) N * H, 0.0f);
        for (int i = 0; i < N; ++i) std::copy(&Hn[(size_t) (lst[(size_t) i] / K) * H], &Hn[(size_t) (lst[(size_t) i] / K) * H] + H, &Xe[(size_t) i * H]);
        Slot* sl = nullptr;
        if (gpu_on && !pool.empty()) { std::lock_guard<std::mutex> g(gmu); sl = resident(l, e); }
        GU.assign((size_t) N * 2 * I, 0.0f);
        bool ok = false;
        if (sl) { std::lock_guard<std::mutex> g(gmu); ok = gpu::matmat_mx(sl->gu_b, sl->gu_s, 2 * I, H, Xe.data(), N, GU.data()); if (!ok) gpu_fail(gpu::last_error()); }
        if (!ok) {
            MxView gu = y.gu; gu.blocks += (size_t) e * 2 * I * (H / 32) * 16; gu.scales += (size_t) e * 2 * I * (H / 32);
            for (int i = 0; i < N; ++i) matvec_mx(gu, &Xe[(size_t) i * H], &GU[(size_t) i * 2 * I]);
        }
        ACT.assign((size_t) N * I, 0.0f);
        const uint16_t* gb = y.gu_bias + (size_t) e * 2 * I;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const float* r = &GU[(size_t) i * 2 * I];
            for (int j = 0; j < I; ++j) {
                const float gt = (o.glu_halves ? r[j] : r[2 * j]) + bf16_to_f32(gb[o.glu_halves ? j : 2 * j]);
                const float up = (o.glu_halves ? r[I + j] : r[2 * j + 1]) + bf16_to_f32(gb[o.glu_halves ? I + j : 2 * j + 1]);
                ACT[(size_t) i * I + j] = swiglu(gt, up, c.swiglu_limit, c.swiglu_alpha);
            }
        }
        EB.assign((size_t) N * H, 0.0f);
        ok = false;
        if (sl && gpu_on) { std::lock_guard<std::mutex> g(gmu); ok = gpu::matmat_mx(sl->dn_b, sl->dn_s, H, I, ACT.data(), N, EB.data()); if (!ok) gpu_fail(gpu::last_error()); }
        if (!ok) {
            MxView dn = y.dn; dn.blocks += (size_t) e * H * (I / 32) * 16; dn.scales += (size_t) e * H * (I / 32);
            for (int i = 0; i < N; ++i) matvec_mx(dn, &ACT[(size_t) i * I], &EB[(size_t) i * H]);
        }
        const uint16_t* db = y.dn_bias + (size_t) e * H;
        for (int i = 0; i < N; ++i) {                                      // a token has an expert at most once: no overlap on Y[t]
            const int t = lst[(size_t) i] / K;
            const float wt = WT[(size_t) lst[(size_t) i]];
            for (int j = 0; j < H; ++j) Y[(size_t) t * H + j] += wt * (EB[(size_t) i * H + j] + bf16_to_f32(db[j]));
        }
    };
    std::vector<int> act;
    for (int e = 0; e < E; ++e) if (!by_exp[(size_t) e].empty()) act.push_back(e);
    if (!(gpu_on && !pool.empty())) {
        for (int e : act) sync_expert(e);
    } else {
        // Pipelined: while the device computes expert a (whole expert in one launch on the batch context: x/wt up, gate_up GEMM, SwiGLU, down GEMM,
        // y down), the host gathers the rows of expert a+1 and uploads its weights over PCIe; expert a+1 is launched BEFORE the host scatters a's result.
        struct Stage { int e = -1, N = 0; Slot* sl = nullptr; bool fly = false; std::vector<float> xe, wt, eb; };
        Stage stg[2];
        const int Ke = K;
        auto prep = [&](Stage& s, int e) {
            const auto& lst = by_exp[(size_t) e];
            s.e = e; s.N = (int) lst.size(); s.fly = false; s.sl = nullptr;
            s.xe.resize((size_t) s.N * H); s.wt.resize((size_t) s.N);
            for (int i = 0; i < s.N; ++i) {
                const int r = lst[(size_t) i];
                std::copy(&Hn[(size_t) (r / Ke) * H], &Hn[(size_t) (r / Ke) * H] + H, &s.xe[(size_t) i * H]);
                s.wt[(size_t) i] = WT[(size_t) r];
            }
            if (gpu_on) { std::lock_guard<std::mutex> g(gmu); s.sl = resident(l, e); if (s.sl) s.sl->busy = true; }   // busy: not evictable until collected
        };
        auto launch = [&](Stage& s) {
            if (!s.sl || !gpu_on) return;
            gpu::ExpertDesc d;
            d.gu_b = s.sl->gu_b; d.gu_s = s.sl->gu_s; d.gu_bias = s.sl->gu_bias;      // the bias rows live inside the slot: no per-layer copy
            d.dn_b = s.sl->dn_b; d.dn_s = s.sl->dn_s; d.dn_bias = s.sl->dn_bias;
            d.H = H; d.I = I; d.halves = o.glu_halves ? 1 : 0; d.limit = c.swiglu_limit; d.alpha = c.swiglu_alpha;
            std::lock_guard<std::mutex> g(gmu);
            if (gpu::expert_run_async(gpu::kExpertCtxBatch, d, s.xe.data(), s.wt.data(), s.N)) s.fly = true;
            else gpu_fail(gpu::last_error());
        };
        auto collect = [&](Stage& s) -> bool {          // true: s.eb holds wt * (down + bias) for each of the N rows
            bool ok = false;
            if (s.fly) {
                s.eb.resize((size_t) s.N * H);
                std::lock_guard<std::mutex> g(gmu);
                ok = gpu::expert_collect(gpu::kExpertCtxBatch, s.eb.data());
                if (!ok) gpu_fail(gpu::last_error());
            }
            if (s.sl) { std::lock_guard<std::mutex> g(gmu); s.sl->busy = false; }
            s.fly = false;
            return ok;
        };
        Stage* cur = &stg[0]; Stage* nxt = &stg[1];
        if (!act.empty()) { prep(*cur, act[0]); launch(*cur); }
        for (size_t a = 0; a < act.size(); ++a) {
            const bool more = a + 1 < act.size();
            if (more) prep(*nxt, act[a + 1]);                              // gather + PCIe upload while `cur` computes
            const bool done = collect(*cur);
            if (more) launch(*nxt);                                        // the device is busy again while the host scatters below
            if (done) {
                const auto& lst = by_exp[(size_t) cur->e];                 // a token has an expert at most once: no overlap on Y[t]
                for (int i = 0; i < cur->N; ++i) {
                    float* yt = &Y[(size_t) (lst[(size_t) i] / K) * H];
                    const float* eb = &cur->eb[(size_t) i * H];
                    for (int j = 0; j < H; ++j) yt[j] += eb[j];
                }
            } else sync_expert(cur->e);                                    // no slot / launch or collect failed: the old path
            std::swap(cur, nxt);
        }
    }
}

// One chunk of the prefill with the activations resident in VRAM: x goes up ONCE, then for every layer [norm, q/k/v, bias+RoPE+KV write, attention, o,
// residual, norm, router] run on the device; down come only the post-norm rows (the experts' input), the router logits and the K/V rows of the chunk (to keep
// the host KV, which decode / checkpoints / reuse_prefix use, identical). The routed experts go through moe_prefill_layer; their sum goes up and is added on
// the device. At the end only the last row of x comes back (the lm_head input). false: the device failed -> the caller redoes the chunk on the host
// (the host X was never touched and every KV slot of the chunk is rewritten by that path).
bool Model::Impl::chunk_dev(int T, int p0, const float* X0, float* Xlast) {
    const int H = c.hidden, E = c.n_exp, K = c.topk, QD = c.q_dim(), KD = c.kv_dim();
    auto bad = [&]() { if (gpu_on) gpu_fail(gpu::last_error()); return false; };
    std::vector<float> Hn((size_t) T * H), Gl((size_t) T * E), Y((size_t) T * H), WT((size_t) T * K);
    std::vector<int32_t> IDX((size_t) T * K);
    {
        std::lock_guard<std::mutex> g(gmu);
        if (!gpu_on || !kv_sync_to_dev(p0) || !gpu::h2d(a_X, X0, (size_t) T * H * 4)) return bad();
    }
    for (int l = 0; l < c.layers; ++l) {
        if (!gpu_on) return false;
        Layer& y = L[(size_t) l];
        gpu::AttnShape a;
        a.heads = c.heads; a.kv_heads = c.kv_heads; a.D = c.head_dim; a.window = c.window; a.sliding = y.sliding; a.cap = y.cap; a.rope_pairs = o.rope_pairs;
        a.k_fmt = dev_kfmt; a.v_fmt = dev_vfmt; a.dKinv = y.dKinv; a.dVinv = y.dVinv; a.kv_limit = 448.0f;
        {
            std::lock_guard<std::mutex> g(gmu);
            bool ok = gpu::rms_rows(a_X, y.d_in_ln, c.eps, T, H, a_Hn) && gpu::dmm_bf16(y.dq, QD, H, a_Hn, T, a_Q) && gpu::dmm_bf16(y.dk, KD, H, a_Hn, T, a_K) &&
                      gpu::dmm_bf16(y.dv, KD, H, a_Hn, T, a_V) && gpu::qk_rope_kv(a_Q, a_K, a_V, y.d_bq, y.d_bk, y.d_bv, d_cos, d_sin, T, p0, a, (float*)y.dKc, (float*)y.dVc) &&
                      gpu::attn_rows(a_Q, (const float*)y.dKc, (const float*)y.dVc, y.d_sinks, T, p0, a, a_Att) && gpu::dmm_bf16(y.dobj, H, QD, a_Att, T, a_Ob) && gpu::add_rows(a_X, a_Ob, y.d_bo, T, H) &&
                      gpu::rms_rows(a_X, y.d_post_ln, c.eps, T, H, a_Hn) && gpu::dmm_bf16(y.drt, E, H, a_Hn, T, a_Gl) &&
                      gpu::d2h(Hn.data(), a_Hn, (size_t) T * H * 4) && gpu::d2h(Gl.data(), a_Gl, (size_t) T * E * 4);
            for (int pos = p0; ok && pos < p0 + T;) {                          // K/V rows of the chunk -> host cache (one segment, two if the ring wraps)
                const size_t slot = y.slot(pos);
                const int run = y.sliding ? std::min(p0 + T - pos, y.cap - (int) slot) : p0 + T - pos;
                ok = gpu::kv_download_rows(dev_kfmt, &y.K[slot * KD], y.dKc, y.dKinv, run, pos, (int) KD, y.cap, y.sliding) &&
                     gpu::kv_download_rows(dev_vfmt, &y.V[slot * KD], y.dVc, y.dVinv, run, pos, (int) KD, y.cap, y.sliding);
                pos += run;
            }
            if (!ok) return bad();
        }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            float* gl = &Gl[(size_t) t * E];
            for (int e = 0; e < E; ++e) gl[e] += y.rbias[(size_t) e];
            route(gl, E, K, &IDX[(size_t) t * K], &WT[(size_t) t * K]);
            count_use(l, &IDX[(size_t) t * K]);
        }
        moe_prefill_layer(l, T, Hn.data(), IDX.data(), WT.data(), Y.data());
        {
            std::lock_guard<std::mutex> g(gmu);
            if (!gpu_on || !gpu::h2d(a_Y, Y.data(), (size_t) T * H * 4) || !gpu::add_rows(a_X, a_Y, nullptr, T, H)) return bad();
        }
    }
    {
        std::lock_guard<std::mutex> g(gmu);
        std::vector<float> last((size_t) H);
        if (!gpu_on || !gpu::d2h(last.data(), a_X + (size_t) (T - 1) * H, (size_t) H * 4)) return bad();
        std::memcpy(Xlast, last.data(), (size_t) H * 4);          // Xlast aliases the host X the caller would recompute from: written only on success
    }
    dev_kv_pos = p0 + T;
    return true;
}

// Batched prefill with the device: every projection and every expert is ONE pass over the weights for the whole chunk.
void Model::Impl::prefill_gpu(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    const int H = c.hidden, E = c.n_exp, K = c.topk, QD = c.q_dim(), KD = c.kv_dim(), D = c.head_dim, half = D / 2;
    std::vector<float> X, Hn, Qb, Kb, Vb, Ab, Ob, Gl, Y, Ks, Vs;
    std::vector<int32_t> IDX;
    std::vector<float> WT;
    for (int c0 = 0; c0 < n; c0 += chunk_max) {
        const int T = std::min(chunk_max, n - c0), p0 = start_pos + c0;
        X.assign((size_t) T * H, 0.0f); Hn.assign((size_t) T * H, 0.0f); Qb.assign((size_t) T * QD, 0.0f); Kb.assign((size_t) T * KD, 0.0f);
        Vb.assign((size_t) T * KD, 0.0f); Ab.assign((size_t) T * QD, 0.0f); Ob.assign((size_t) T * H, 0.0f); Gl.assign((size_t) T * E, 0.0f);
        Ks.assign((size_t) T * KD, 0.0f); Vs.assign((size_t) T * KD, 0.0f);
        KvStage st; st.K = Ks.data(); st.V = Vs.data(); st.base = p0;   // the chunk's own K/V stays out of the ring until its attention has read it
        Y.assign((size_t) T * H, 0.0f); IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) X[(size_t) t * H + i] = bf16_to_f32(((const uint16_t*) embed.p)[(size_t) tokens[c0 + t] * H + i]);
        bool dev_done = false;
        if (attn_dev && gpu_on) dev_done = chunk_dev(T, p0, X.data(), &X[(size_t) (T - 1) * H]);   // success: only X's last row is meaningful
        if (!dev_done) for (int l = 0; l < c.layers; ++l) {
            Layer& y = L[(size_t) l];
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) rmsnorm(&X[(size_t) t * H], y.in_ln.data(), c.eps, H, &Hn[(size_t) t * H]);
            mm(y.q, y.dq, Hn.data(), T, Qb.data());
            mm(y.k, y.dk, Hn.data(), T, Kb.data());
            mm(y.v, y.dv, Hn.data(), T, Vb.data());
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                const int pos = p0 + t;
                float* q = &Qb[(size_t) t * QD]; float* k = &Kb[(size_t) t * KD]; float* v = &Vb[(size_t) t * KD];
                for (int i = 0; i < QD; ++i) q[i] += y.bq[(size_t) i];
                for (int i = 0; i < KD; ++i) { k[i] += y.bk[(size_t) i]; v[i] += y.bv[(size_t) i]; }
                const float* cs = &rope_cos[(size_t) pos * half];
                const float* sn = &rope_sin[(size_t) pos * half];
                for (int h = 0; h < c.heads; ++h) rotary(q + (size_t) h * D, D, cs, sn, o.rope_pairs);
                for (int h = 0; h < c.kv_heads; ++h) rotary(k + (size_t) h * D, D, cs, sn, o.rope_pairs);
                std::memcpy(&Ks[(size_t) t * KD], k, sizeof(float) * (size_t) KD);
                std::memcpy(&Vs[(size_t) t * KD], v, sizeof(float) * (size_t) KD);
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // attention on the host: reads the stage, then the ring
                Work& w = my_work();
                std::copy(&Qb[(size_t) t * QD], &Qb[(size_t) (t + 1) * QD], w.q.begin());
                attn_heads(y, w, p0 + t, st);
                std::copy(w.att.begin(), w.att.end(), &Ab[(size_t) t * QD]);
            }
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {                                        // commit: only now the chunk's K/V goes into the ring
                const int pos = p0 + t;
                std::memcpy(&y.K[y.slot(pos) * KD], &Ks[(size_t) t * KD], sizeof(float) * (size_t) KD);
                std::memcpy(&y.V[y.slot(pos) * KD], &Vs[(size_t) t * KD], sizeof(float) * (size_t) KD);
            }
            mm(y.o, y.dobj, Ab.data(), T, Ob.data());
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < H; ++i) X[(size_t) t * H + i] += Ob[(size_t) t * H + i] + y.bo[(size_t) i];
                rmsnorm(&X[(size_t) t * H], y.post_ln.data(), c.eps, H, &Hn[(size_t) t * H]);
            }
            mm(y.router, y.drt, Hn.data(), T, Gl.data());
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                float* g = &Gl[(size_t) t * E];
                for (int e = 0; e < E; ++e) g[e] += y.rbias[(size_t) e];
                route(g, E, K, &IDX[(size_t) t * K], &WT[(size_t) t * K]);
                count_use(l, &IDX[(size_t) t * K]);
            }
            moe_prefill_layer(l, T, Hn.data(), IDX.data(), WT.data(), Y.data());
            for (size_t i = 0; i < X.size(); ++i) X[i] += Y[i];
        }
        if ((int) cached.size() > p0) cached.resize((size_t) p0);
        for (int t = 0; t < T; ++t) if ((int) cached.size() == p0 + t) cached.push_back(tokens[c0 + t]);
        if (logits_last && c0 + T == n) {
            rmsnorm(&X[(size_t) (T - 1) * H], final_norm.data(), c.eps, H, lastn.data());
            logits_last->resize((size_t) c.vocab);
            mv(lm_head, d_head, lastn.data(), logits_last->data());
        }
    }
}

pdev::Cfg Model::Impl::pdev_cfg() const {
    pdev::Cfg pc;
    pc.H = c.hidden; pc.I = c.moe_ff; pc.E = c.n_exp; pc.topk = c.topk; pc.layers = c.layers;
    pc.ctx = o.ctx; pc.window = c.window; pc.heads = c.heads; pc.kv_heads = c.kv_heads; pc.head_dim = c.head_dim;
    pc.eps = c.eps; pc.limit = c.swiglu_limit; pc.alpha = c.swiglu_alpha;
    pc.rope_pairs = o.rope_pairs; pc.glu_halves = o.glu_halves; pc.chunk_max = std::max(1, chunk_max);
    // Asymmetric KV quantization of the CARD cache: -k / -v choose the two sides independently.
    pc.k_fmt = (int) o.key_format; pc.v_fmt = (int) o.value_format;
    pc.kv_limit = 448.0f;   // e4m3 max; the row scale is the power of two that keeps the row inside it
    return pc;
}

// Device-resident prefill: the chunk's residual stream stays on the card for the whole layer loop. Per layer the host
// only routes (router logits down, routing tables up) and resolves the touched experts' slots; per chunk it crosses
// PCIe once more with X in/out and the new K/V rows back to the host cache (checkpoint + decode stay on the host KV).
int Model::Impl::prefill_pdev(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    const int H = c.hidden, E = c.n_exp, K = c.topk;
    std::vector<float> X, Gl;
    pdev::kv_trim(pdev_kv_pos);   // what the decode / a checkpoint wrote into the host KV since the last device prefill is not on the card
    preload_wait();   // the background preload holds every slot busy until it is done: starting now, no slot can take an expert and the device prefill gives up
    std::vector<int32_t> IDX;
    std::vector<float> WT;
    for (int c0 = 0; c0 < n && pdev_on; c0 += chunk_max) {
        const int T = std::min(chunk_max, n - c0), p0 = start_pos + c0;
        X.assign((size_t) T * H, 0.0f); Gl.assign((size_t) T * E, 0.0f);
        IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) X[(size_t) t * H + i] = bf16_to_f32(((const uint16_t*) embed.p)[(size_t) tokens[c0 + t] * H + i]);
        // A device failure in the middle of a chunk leaves the banded layers' rings half overwritten with this chunk's rows
        // (kv_store_back already ran for layers < l), and the host fallback recomputes the chunk from the OLD window in those
        // rings: it would read newer rows where it needs older ones. Snapshot the old window; restore it on any early return.
        struct RingSnap { Layer* y; std::vector<float> k, v; };
        std::vector<RingSnap> snap;
        const int w0 = std::max(0, p0 - c.window + 1);
        if (p0 > w0) for (auto& yy : L) if (yy.sliding) {
            RingSnap r; r.y = &yy;
            const size_t KDn = (size_t) c.kv_dim();
            r.k.resize((size_t) (p0 - w0) * KDn); r.v.resize(r.k.size());
            for (int u = w0; u < p0; ++u) {
                std::memcpy(&r.k[(size_t) (u - w0) * KDn], &yy.K[yy.slot(u) * KDn], sizeof(float) * KDn);
                std::memcpy(&r.v[(size_t) (u - w0) * KDn], &yy.V[yy.slot(u) * KDn], sizeof(float) * KDn);
            }
            snap.push_back(std::move(r));
        }
        struct SnapGuard {
            std::vector<RingSnap>& sn; int w0, p0, KD; bool ok = false;
            ~SnapGuard() {
                if (ok) return;
                for (auto& r : sn) for (int u = w0; u < p0; ++u) {
                    std::memcpy(&r.y->K[r.y->slot(u) * (size_t) KD], &r.k[(size_t) (u - w0) * (size_t) KD], sizeof(float) * (size_t) KD);
                    std::memcpy(&r.y->V[r.y->slot(u) * (size_t) KD], &r.v[(size_t) (u - w0) * (size_t) KD], sizeof(float) * (size_t) KD);
                }
            }
        } guard{snap, w0, p0, c.kv_dim()};
        if (!pdev::x_upload(X.data(), T)) { pdev_fail_("x upload"); return c0; }
        for (int l = 0; l < c.layers && pdev_on; ++l) {
            Layer& y = L[(size_t) l];
            pdev::kv_prefetch(l, y.K.data(), y.V.data(), y.cap, y.sliding, p0);
            pdev::LayerIO w{y.dq, y.dk, y.dv, y.dobj, y.drt};
            if (!pdev::run_layer(pdev_cfg(), l, w, T, p0, y.sliding, Gl.data())) { pdev_fail_("layer " + std::to_string(l) + " at position " + std::to_string(p0)); return c0; }
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                float* g = &Gl[(size_t) t * E];
                route(g, E, K, &IDX[(size_t) t * K], &WT[(size_t) t * K]);   // the router bias is already added on the card (k_router_bias)
                count_use(l, &IDX[(size_t) t * K]);
            }
            // group the routed rows by expert (index order), resolve every touched expert's slot, upload the tables
            std::vector<int> cnt((size_t) E, 0);
            for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) cnt[(size_t) IDX[(size_t) t * K + k]]++;
            std::vector<int> rowoff((size_t) E + 1, 0);
            for (int e = 0; e < E; ++e) rowoff[(size_t) e + 1] = rowoff[e] + cnt[e];
            const int R_tot = rowoff[E];
            std::vector<int> src((size_t) R_tot), pos_in_exp((size_t) E, 0);
            std::vector<float> WTR((size_t) R_tot, 0.0f);
            for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) {
                const int e = IDX[(size_t) t * K + k];
                const int r = rowoff[e] + pos_in_exp[e];
                src[r] = t;
                WTR[r] = WT[(size_t) t * K + k];
                IDX[(size_t) t * K + k] = r;   // reuse IDX as the global rowof table (t*K+k order, what run_experts wants)
                ++pos_in_exp[e];
            }
            std::vector<pdev::ExpertPtrs> per((size_t) E);
            {
                std::lock_guard<std::mutex> g(gmu);
                for (int e = 0; e < E && pdev_on; ++e) {
                    if (!cnt[e]) continue;
                    Slot* sl = resident(l, e);
                    if (!sl) { pdev_fail_("no slot can take this expert"); break; }   // no slot can take it: the old prefill does this chunk's session from here on
                    per[(size_t) e].gu_b = sl->gu_b; per[(size_t) e].dn_b = sl->dn_b;
                    per[(size_t) e].gu_s = sl->gu_s; per[(size_t) e].dn_s = sl->dn_s;
                    per[(size_t) e].gu_bias = sl->gu_bias; per[(size_t) e].dn_bias = sl->dn_bias;
                }
            }
            if (!pdev_on) return c0;
            if (!pdev::run_experts(pdev_cfg(), T, per.data(), cnt.data(), src.data(), WTR.data(), IDX.data())) { pdev_fail_("experts"); return c0; }
            pdev::kv_store_back(l, y.K.data(), y.V.data(), y.cap, y.sliding, T, p0);
        }
        if (!pdev_on) return c0;
        pdev::advance(T);
        if (!pdev::x_download(X.data(), T)) { pdev_fail_("x download"); return c0; }
        if (!pdev::sync_all()) { pdev_fail_("sync"); return c0; }   // the K/V rows are back in the host cache before anything reads it
        guard.ok = true;
        pdev_kv_pos = p0 + T;
        if ((int) cached.size() > p0) cached.resize((size_t) p0);
        for (int t = 0; t < T; ++t) if ((int) cached.size() == p0 + t) cached.push_back(tokens[c0 + t]);
        if (logits_last && c0 + T == n) {
            rmsnorm(&X[(size_t) (T - 1) * H], final_norm.data(), c.eps, H, lastn.data());
            logits_last->resize((size_t) c.vocab);
            mv(lm_head, d_head, lastn.data(), logits_last->data());
        }
    }
    return n;
}

void Model::forward(int token, int pos, std::vector<float>* logits_out) {
    Impl& s = *p_;
    const GptConfig& c = s.c;
    const int H = c.hidden;
    if (pos < 0 || pos >= s.o.ctx || token < 0 || token >= c.vocab) { if (logits_out) logits_out->assign((size_t) c.vocab, 0.0f); return; }
    s.counting = true;                                   // this token's expert lookups are counted (the per-request line)
    std::vector<float> x((size_t) H), acc((size_t) H);
    for (int i = 0; i < H; ++i) x[(size_t) i] = bf16_to_f32(((const uint16_t*) s.embed.p)[(size_t) token * H + i]);
    Work& w = s.works[0];
    if (s.pdev_kv_pos > pos) s.pdev_kv_pos = pos;
    if (s.dev_kv_pos > pos) s.dev_kv_pos = pos;            // decode writes the host KV only: the device copy is valid up to here
    for (int l = 0; l < c.layers; ++l) {
        Layer& L = s.L[(size_t) l];
        s.attn_in(L, w, x.data(), pos);
        s.attn_out(L, w, x.data(), pos);
        int32_t idx[64]; float wt[64];
        s.route_tok(L, w, x.data(), idx, wt);
        const std::vector<float> h = w.h;
        std::fill(acc.begin(), acc.end(), 0.0f);
        s.experts_decode(L, w, h.data(), idx, wt, acc.data());   // device experts in flight together; whatever has no slot runs on the host
        for (int i = 0; i < H; ++i) x[(size_t) i] += acc[(size_t) i];
    }
    if ((int) s.cached.size() > pos) s.cached.resize((size_t) pos);
    if ((int) s.cached.size() == pos) s.cached.push_back(token);
    if (logits_out) {
        rmsnorm(x.data(), s.final_norm.data(), c.eps, H, s.lastn.data());
        logits_out->resize((size_t) c.vocab);
        s.mv(s.lm_head, s.d_head, s.lastn.data(), logits_out->data());
    }
    s.counting = false;
}

void Model::prefill(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    Impl& s = *p_;
    const GptConfig& c = s.c;
    const int H = c.hidden, E = c.n_exp, K = c.topk, QD = c.q_dim(), KD = c.kv_dim();
    if (n <= 0) return;
    bool bad = start_pos < 0 || start_pos + n > s.o.ctx;
    for (int i = 0; i < n && !bad; ++i) bad = tokens[i] < 0 || tokens[i] >= c.vocab;
    if (bad) { if (logits_last) logits_last->assign((size_t) c.vocab, 0.0f); return; }
    static const bool chk = std::getenv("GPTOSS_CHECK_PREFILL") != nullptr;
    if (chk && s.gpu_on && !s.in_check) {
        // The device prefill as in production, then the plain CPU path over the same tokens. The CPU result is kept (logits and K/V rows),
        // the comparison goes to the log.
        s.in_check = true;
        std::vector<float> ld, lc;
        prefill(tokens, n, start_pos, &ld);
        const bool snap = n <= 1024;
        std::vector<std::vector<float>> sk, sv;
        if (snap) for (auto& y : s.L) {
            if (y.sliding && n > y.cap) { sk.emplace_back(); sv.emplace_back(); continue; }
            std::vector<float> a((size_t) n * KD), b((size_t) n * KD);
            for (int t = 0; t < n; ++t) {
                std::memcpy(&a[(size_t) t * KD], &y.K[y.slot(start_pos + t) * KD], sizeof(float) * (size_t) KD);
                std::memcpy(&b[(size_t) t * KD], &y.V[y.slot(start_pos + t) * KD], sizeof(float) * (size_t) KD);
            }
            sk.push_back(std::move(a)); sv.push_back(std::move(b));
        }
        const bool g = s.gpu_on;
        s.gpu_on = false;
        prefill(tokens, n, start_pos, &lc);
        s.gpu_on = g;
        s.dev_kv_pos = 0; s.pdev_kv_pos = 0;                // the CPU rewrote the host rows: the card's mirrors are stale
        s.in_check = false;
        double md = 0; size_t a1 = 0, a2 = 0;
        for (size_t i = 0; i < lc.size() && i < ld.size(); ++i) {
            md = std::max(md, (double) std::fabs(ld[i] - lc[i]));
            if (ld[i] > ld[a1]) a1 = i;
            if (lc[i] > lc[a2]) a2 = i;
        }
        std::fprintf(stderr, "check: prefill start %d n %d: logits max|dev-cpu| %.4g, top1 dev %zu cpu %zu -> %s\n", start_pos, n, md, a1, a2, a1 == a2 && md < 1.0 ? "ok" : "MISMATCH");
        if (snap) {
            int shown = 0, badl = 0;
            for (int l = 0; l < c.layers; ++l) {
                if (sk[(size_t) l].empty()) continue;
                Layer& y = s.L[(size_t) l];
                double dk = 0, dv = 0, mk = 0, mv2 = 0; int first_t = -1;
                for (int t = 0; t < n; ++t) for (int j = 0; j < KD; ++j) {
                    const float ck = y.K[y.slot(start_pos + t) * KD + j], cv = y.V[y.slot(start_pos + t) * KD + j];
                    const double ek = std::fabs(sk[(size_t) l][(size_t) t * KD + j] - ck), ev = std::fabs(sv[(size_t) l][(size_t) t * KD + j] - cv);
                    if (ek > dk) { dk = ek; if (first_t < 0 && ek > 0.05 * (mk + 1e-3)) first_t = t; }
                    dv = std::max(dv, ev); mk = std::max(mk, (double) std::fabs(ck)); mv2 = std::max(mv2, (double) std::fabs(cv));
                }
                const bool mis = dk > 0.05 * (mk + 1e-6) || dv > 0.05 * (mv2 + 1e-6);
                badl += mis ? 1 : 0;
                if (mis && shown < 6) { ++shown; std::fprintf(stderr, "check:   layer %d (%s): K max diff %.4g (of %.4g), V max diff %.4g (of %.4g), first bad token %d\n", l, y.sliding ? "banded" : "dense", dk, mk, dv, mv2, first_t); }
            }
            std::fprintf(stderr, "check: K/V rows of the chunk: %d of %d layers differ from the CPU's\n", badl, c.layers);
        }
        if (logits_last) *logits_last = lc;
        return;
    }
    if (s.gpu_on && s.pdev_on) {
        const int done = s.prefill_pdev(tokens, n, start_pos, logits_last);   // a device failure stops it: the host cache is consistent up to `done`
        if (done == n) return;
        s.prefill_gpu(tokens + done, n - done, start_pos + done, logits_last);   // the old path finishes the rest from the same host cache
        return;
    }
    if (s.gpu_on) { s.prefill_gpu(tokens, n, start_pos, logits_last); return; }
    std::vector<float> X, Hh, Y, Q, Ks, Vs;
    std::vector<int32_t> IDX; std::vector<float> WT;
    for (int c0 = 0; c0 < n; c0 += s.chunk_max) {
        const int T = std::min(s.chunk_max, n - c0), p0 = start_pos + c0;
        X.assign((size_t) T * H, 0.0f); Hh.assign((size_t) T * H, 0.0f); Y.assign((size_t) T * H, 0.0f); Q.assign((size_t) T * QD, 0.0f);
        Ks.assign((size_t) T * KD, 0.0f); Vs.assign((size_t) T * KD, 0.0f);
        KvStage st; st.K = Ks.data(); st.V = Vs.data(); st.base = p0;   // the chunk's own K/V stays out of the ring until its attention has read it
        IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) X[(size_t) t * H + i] = bf16_to_f32(((const uint16_t*) s.embed.p)[(size_t) tokens[c0 + t] * H + i]);
        for (int l = 0; l < c.layers; ++l) {
            Layer& L = s.L[(size_t) l];
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // K/V of the whole chunk first
                Work& w = s.my_work();
                s.attn_in(L, w, &X[(size_t) t * H], p0 + t, st);
                std::copy(w.q.begin(), w.q.end(), &Q[(size_t) t * QD]);            // the next loop may run this token on another thread
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // then every token attends to its window / past
                Work& w = s.my_work();
                std::copy(&Q[(size_t) t * QD], &Q[(size_t) (t + 1) * QD], w.q.begin());
                s.attn_out(L, w, &X[(size_t) t * H], p0 + t, st);
                s.route_tok(L, w, &X[(size_t) t * H], &IDX[(size_t) t * K], &WT[(size_t) t * K]);
                std::copy(w.h.begin(), w.h.end(), &Hh[(size_t) t * H]);
            }
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {                                        // commit: only after the chunk's attention has read it
                std::memcpy(&L.K[L.slot(p0 + t) * KD], &Ks[(size_t) t * KD], sizeof(float) * (size_t) KD);
                std::memcpy(&L.V[L.slot(p0 + t) * KD], &Vs[(size_t) t * KD], sizeof(float) * (size_t) KD);
            }
            std::fill(Y.begin(), Y.end(), 0.0f);
            std::vector<std::vector<int>> by_exp((size_t) E);
            for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) by_exp[(size_t) IDX[(size_t) t * K + k]].push_back(t * K + k);
            for (int e = 0; e < E; ++e) {                                          // one pass over each expert's weights per chunk
                const auto& lst = by_exp[(size_t) e];
#pragma omp parallel for schedule(dynamic, 1)
                for (int i = 0; i < (int) lst.size(); ++i) {
                    const int t = lst[(size_t) i] / K;                             // an expert appears once per token: no race on Y[t]
                    s.expert_add(L, s.my_work(), &Hh[(size_t) t * H], e, WT[(size_t) lst[(size_t) i]], &Y[(size_t) t * H]);
                }
            }
            for (size_t i = 0; i < X.size(); ++i) X[i] += Y[i];
        }
        if ((int) s.cached.size() > p0) s.cached.resize((size_t) p0);
        for (int t = 0; t < T; ++t) if ((int) s.cached.size() == p0 + t) s.cached.push_back(tokens[c0 + t]);
        if (logits_last && c0 + T == n) {
            rmsnorm(&X[(size_t) (T - 1) * H], s.final_norm.data(), c.eps, H, s.lastn.data());
            logits_last->resize((size_t) c.vocab);
            s.mv(s.lm_head, s.d_head, s.lastn.data(), logits_last->data());
        }
    }
}

int Model::reuse_prefix(const std::vector<int>& ids) {
    Impl& s = *p_;
    const int n = (int) s.cached.size();
    int k = 0;
    while (k < n && k < (int) ids.size() && s.cached[(size_t) k] == ids[(size_t) k]) ++k;
    if (!ids.empty() && k >= (int) ids.size()) k = (int) ids.size() - 1;      // the last prompt token is always recomputed: its logits start the answer
    for (const Layer& y : s.L) {                                              // a banded layer must still hold positions k-window+1 .. k-1
        if (!y.sliding) continue;
        const int need_lo = std::max(0, k - s.c.window + 1);
        if (n - need_lo > y.cap) {
            std::fprintf(stderr, "gptoss: cannot rewind to %d (cached %d): banded layers hold %d positions (raise --kv-rollback); recomputing the prompt\n", k, n, y.cap);
            k = 0; break;
        }
    }
    s.cached.resize((size_t) k);
    return k;
}

namespace {
struct KvHead { char magic[8]; uint32_t layers, kv_dim, fp_len; uint64_t n; };
}

bool Model::save_kv(const std::string& path, std::string& err) const {
    const Impl& s = *p_;
    const std::string tmp = path + ".tmp", fp = s.fingerprint();
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    const int n = (int) s.cached.size(), KD = s.c.kv_dim();
    KvHead h{{'G', 'P', 'K', 'V', '0', '0', '0', '1'}, (uint32_t) s.c.layers, (uint32_t) KD, (uint32_t) fp.size(), (uint64_t) n};
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(fp.data(), 1, fp.size(), f) == fp.size() &&
              std::fwrite(s.cached.data(), sizeof(int), s.cached.size(), f) == s.cached.size();
    for (const Layer& y : s.L) {
        const int lo = y.sliding ? std::max(0, n - y.cap) : 0;
        for (int p = lo; p < n && ok; ++p) ok = std::fwrite(&y.K[y.slot(p) * KD], sizeof(float), (size_t) KD, f) == (size_t) KD;
        for (int p = lo; p < n && ok; ++p) ok = std::fwrite(&y.V[y.slot(p) * KD], sizeof(float), (size_t) KD, f) == (size_t) KD;
    }
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { err = "write error on " + tmp; std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "rename failed"; return false; }
    return true;
}

bool Model::load_kv(const std::string& path, std::string& err) {
    Impl& s = *p_;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    KvHead h;
    if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, "GPKV0001", 8) != 0) { std::fclose(f); err = "not a KV checkpoint"; return false; }
    std::string fp(h.fp_len, '\0');
    if (h.fp_len > 4096 || std::fread(&fp[0], 1, h.fp_len, f) != h.fp_len || fp != s.fingerprint() || h.layers != (uint32_t) s.c.layers) {
        std::fclose(f); err = "checkpoint belongs to another model/options"; return false;
    }
    if (h.n > (uint64_t) s.o.ctx) { std::fclose(f); err = "checkpoint has more tokens than --ctx"; return false; }
    const int n = (int) h.n, KD = s.c.kv_dim();
    std::vector<int> toks((size_t) n);
    bool ok = std::fread(toks.data(), sizeof(int), (size_t) n, f) == (size_t) n;
    for (Layer& y : s.L) {
        const int lo = y.sliding ? std::max(0, n - y.cap) : 0;
        for (int p = lo; p < n && ok; ++p) ok = std::fread(&y.K[y.slot(p) * KD], sizeof(float), (size_t) KD, f) == (size_t) KD;
        for (int p = lo; p < n && ok; ++p) ok = std::fread(&y.V[y.slot(p) * KD], sizeof(float), (size_t) KD, f) == (size_t) KD;
    }
    std::fclose(f);
    if (!ok) { s.cached.clear(); err = "truncated checkpoint"; return false; }
    s.cached = std::move(toks);
    s.pdev_kv_pos = 0;
    s.dev_kv_pos = 0;                                  // the host KV was replaced: the device copy is re-uploaded at the next prefill
    return true;
}

bool Model::save_profile(const std::string& path, std::string& err) {
    Impl& s = *p_;
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    const uint32_t L = (uint32_t) s.c.layers, E = (uint32_t) s.c.n_exp;
    bool ok = std::fwrite("GPPRF001", 1, 8, f) == 8 && std::fwrite(&L, 4, 1, f) == 1 && std::fwrite(&E, 4, 1, f) == 1 &&
              std::fwrite(s.usage.data(), sizeof(uint64_t), s.usage.size(), f) == s.usage.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { err = "write error on " + tmp; std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "rename failed"; return false; }
    return true;
}

// --profile-decay: the profile is a sum over every past session, so after a few long runs its counts are
// dominated by old traffic and the pinning at the next start follows the wrong session. Halving the whole
// table once it passes the decay bound keeps the ranking (a monotone transform of every count) while
// bounding how much history still weighs. Off (0) leaves the file byte-identical to the last release.
void Model::decay_profile() {
    Impl& s = *p_;
    if (s.o.profile_decay <= 0 || s.usage.empty()) return;
    uint64_t total = 0;
    for (uint64_t v : s.usage) total += v;
    if ((long long) total <= s.o.profile_decay) return;
    for (uint64_t& v : s.usage) v /= 2;
    std::fprintf(stderr, "gptoss: profile decayed: %llu routed activations halved, the ranking is kept\n", (unsigned long long) total);
    std::fflush(stderr);
}

bool Model::load_profile(const std::string& path, std::string& err) {
    Impl& s = *p_;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[8]; uint32_t L = 0, E = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "GPPRF001", 8) == 0 && std::fread(&L, 4, 1, f) == 1 && std::fread(&E, 4, 1, f) == 1;
    if (!ok || L != (uint32_t) s.c.layers || E != (uint32_t) s.c.n_exp) { std::fclose(f); err = "not a profile of this model"; return false; }
    std::vector<uint64_t> u((size_t) L * E);
    ok = std::fread(u.data(), sizeof(uint64_t), u.size(), f) == u.size();
    std::fclose(f);
    if (!ok) { err = "truncated profile"; return false; }
    s.usage = std::move(u);
    return true;
}

void Model::begin_request_stats() {
    Impl& s = *p_;
    s.base_hits = s.hits; s.base_lookups = s.hits + s.host_served; s.base_pcie = s.pcie_served;
}

void Model::request_stats(unsigned long long* hits, unsigned long long* lookups, unsigned long long* pcie) const {
    const Impl& s = *p_;
    *hits = (unsigned long long) (s.hits - s.base_hits);
    *lookups = (unsigned long long) ((s.hits + s.host_served) - s.base_lookups);
    *pcie = (unsigned long long) (s.pcie_served - s.base_pcie);
}

std::string Model::stats() const {
    const Impl& s = *p_;
    if (s.pool.empty()) return "";
    size_t pinned = 0;
    for (const auto& sl : s.pool) pinned += sl.pinned ? 1 : 0;
    const unsigned long long lookups = (unsigned long long) (s.hits + s.host_served);
    const double tot = (double) lookups;
    char b[400];
    // The line stays exactly as in the last release unless --expert-promote is on: a new field would move the
    // fields the server window reads.
    if (s.o.expert_promote <= 0) {
        std::snprintf(b, sizeof b, "experts: %zu slots (%zu pinned), hits %llu (of which pinned %llu), lookups %llu, hit rate %.1f%%; %llu of the %llu routed experts read over PCIe, %llu on the host, oom fallbacks %llu",
                      s.pool.size(), pinned, (unsigned long long) s.hits, (unsigned long long) s.pin_hits, lookups,
                      tot > 0 ? 100.0 * (double) s.hits / tot : 0.0,
                      (unsigned long long) s.pcie_served, (unsigned long long) (lookups + s.pcie_served),
                      (unsigned long long) s.host_served, (unsigned long long) s.oom_fallbacks);
    } else {
        std::snprintf(b, sizeof b, "experts: %zu slots (%zu pinned), hits %llu (of which pinned %llu), lookups %llu, hit rate %.1f%%; %llu of the %llu routed experts read over PCIe, %llu on the host, oom fallbacks %llu, promotions %llu",
                      s.pool.size(), pinned, (unsigned long long) s.hits, (unsigned long long) s.pin_hits, lookups,
                      tot > 0 ? 100.0 * (double) s.hits / tot : 0.0,
                      (unsigned long long) s.pcie_served, (unsigned long long) (lookups + s.pcie_served),
                      (unsigned long long) s.host_served, (unsigned long long) s.oom_fallbacks,
                      (unsigned long long) s.promotions);
    }
    std::string r = b;
    if (s.pre_state.load() != 0) {
        const char* st[] = {"off", "running", "done", "stopped on an error"};
        std::snprintf(b, sizeof b, ", preload %zu/%zu (%s)", s.pre_done.load(), s.pre_targets.size(), st[s.pre_state.load()]);
        r += b;
    }
    if (s.ram_state.load() != 0) {
        const char* st[] = {"off", "running", "done", "stopped on an error"};
        std::snprintf(b, sizeof b, ", page-locked %zu layers %.1f GiB (%s)", s.ram_layers.load(), (double) s.ram_bytes.load() / 1073741824.0, st[s.ram_state.load()]);
        r += b;
    }
    return r;
}

}  // namespace gptoss

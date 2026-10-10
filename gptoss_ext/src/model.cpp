#include "gptoss/model.hpp"
#include "gptoss/safetensors.hpp"
#include "gptoss/weights.hpp"
#include "gptoss/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
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
    size_t slot(int pos) const { return sliding ? (size_t) pos % (size_t) cap : (size_t) pos; }
};
struct Work { std::vector<float> h, q, k, v, att, tmp, glog, y, gu, act, eb, sc; };
struct Slot {                                          // one expert in VRAM: [gate_up blocks | gate_up scales | down blocks | down scales]
    void* base = nullptr;
    uint8_t *gu_b = nullptr, *gu_s = nullptr, *dn_b = nullptr, *dn_s = nullptr;
    int layer = -1, exp = -1;
    uint64_t tick = 0;
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
    // --- device state (empty/false unless RunOpts::cuda)
    bool gpu_on = false;
    void* d_head = nullptr;                            // lm_head on the device
    std::vector<Slot> pool;                            // expert slots
    std::unordered_map<long long, int> where;          // layer*E+expert -> slot
    uint64_t tick = 0, hits = 0, misses = 0;
    std::mutex gmu;                                    // every device call is serialised

    void init_work(Work& w) const;
    Work& my_work();
    void attn_in(Layer& y, Work& w, const float* x, int pos);
    void attn_out(Layer& y, Work& w, float* x, int pos);
    void route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt);
    void expert_add(Layer& y, Work& w, const float* h, int e, float wt, float* acc);
    std::string fingerprint() const;
    void plan_vram() const;
    bool gpu_setup(std::string& err);
    void gpu_fail(const std::string& what);
    Slot* resident(int layer, int e);                  // expert -> slot (upload on miss, LRU eviction); nullptr when there is no pool
    void mv(const WView& w, const void* dW, const float* x, float* y);
    void mm(const WView& w, const void* dW, const float* x, int N, float* y);
    void attn_heads(Layer& y, Work& w, int pos);
    void prefill_gpu(const int* tokens, int n, int start_pos, std::vector<float>* logits_last);
};

Model::Model() : p_(new Impl) {}
Model::~Model() {
    if (!p_) return;
    Impl& s = *p_;
    if (s.hits + s.misses) std::fprintf(stderr, "gptoss: expert slots (%zu): %llu hits, %llu misses\n", s.pool.size(), (unsigned long long) s.hits, (unsigned long long) s.misses);
    for (auto& sl : s.pool) gpu::release(sl.base);
    for (auto& y : s.L) { gpu::release(y.dq); gpu::release(y.dk); gpu::release(y.dv); gpu::release(y.dobj); gpu::release(y.drt); }
    gpu::release(s.d_head);
}
const GptConfig& Model::config() const { return p_->c; }
uint64_t Model::weight_bytes() const { return p_->st.total_bytes(); }
void Model::reset() { p_->cached.clear(); }
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
    s.chunk_max = ring == opts.ctx ? std::max(1, opts.prefill_chunk) : std::max(1, std::min(opts.prefill_chunk, ring - c.window));   // a wrapping ring must not be overwritten mid-chunk
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
    w.act.assign(c.moe_ff, 0); w.eb.assign(c.hidden, 0); w.sc.assign((size_t) o.ctx + 1, 0);
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

void Model::Impl::attn_in(Layer& y, Work& w, const float* x, int pos) {
    const int H = c.hidden, D = c.head_dim, half = D / 2, QD = c.q_dim(), KD = c.kv_dim();
    rmsnorm(x, y.in_ln.data(), c.eps, H, w.h.data());
    mv(y.q, y.dq, w.h.data(), w.q.data());
    mv(y.k, y.dk, w.h.data(), w.k.data());
    mv(y.v, y.dv, w.h.data(), w.v.data());
    for (int i = 0; i < QD; ++i) w.q[(size_t) i] += y.bq[(size_t) i];
    for (int i = 0; i < KD; ++i) { w.k[(size_t) i] += y.bk[(size_t) i]; w.v[(size_t) i] += y.bv[(size_t) i]; }
    const float* cs = &rope_cos[(size_t) pos * half];
    const float* sn = &rope_sin[(size_t) pos * half];
    for (int h = 0; h < c.heads; ++h) rotary(&w.q[(size_t) h * D], D, cs, sn, o.rope_pairs);
    for (int h = 0; h < c.kv_heads; ++h) rotary(&w.k[(size_t) h * D], D, cs, sn, o.rope_pairs);
    std::memcpy(&y.K[y.slot(pos) * KD], w.k.data(), sizeof(float) * (size_t) KD);
    std::memcpy(&y.V[y.slot(pos) * KD], w.v.data(), sizeof(float) * (size_t) KD);
}

void Model::Impl::attn_heads(Layer& y, Work& w, int pos) {
    const int H = c.hidden, D = c.head_dim, KD = c.kv_dim(), rep = c.heads / c.kv_heads;
    const int lo = y.sliding ? std::max(0, pos - c.window + 1) : 0;
    const float scale = 1.0f / std::sqrt((float) D);
    auto one_head = [&](Work& ww, int hd) {
        const float* q = &w.q[(size_t) hd * D];
        const int g = hd / rep;
        float* sc = ww.sc.data();
        float mx = y.sinks[(size_t) hd];                       // the sink joins the softmax denominator only (no value)
        for (int u = lo; u <= pos; ++u) {
            const float* k = &y.K[y.slot(u) * KD + (size_t) g * D];
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
            const float* v = &y.V[y.slot(u) * KD + (size_t) g * D];
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

void Model::Impl::attn_out(Layer& y, Work& w, float* x, int pos) {
    attn_heads(y, w, pos);
    mv(y.o, y.dobj, w.att.data(), w.tmp.data());
    for (int i = 0; i < c.hidden; ++i) x[i] += w.tmp[(size_t) i] + y.bo[(size_t) i];
}

void Model::Impl::route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt) {
    rmsnorm(x, y.post_ln.data(), c.eps, c.hidden, w.h.data());
    mv(y.router, y.drt, w.h.data(), w.glog.data());
    for (int e = 0; e < c.n_exp; ++e) w.glog[(size_t) e] += y.rbias[(size_t) e];
    route(w.glog.data(), c.n_exp, c.topk, idx, wt);
}

void Model::Impl::gpu_fail(const std::string& what) {
    std::fprintf(stderr, "gptoss: CUDA call failed (%s): continuing on the CPU for the rest of the session\n", what.c_str());
    gpu_on = false;                                    // sticky errors poison every later call: stop using the device
}

bool Model::Impl::gpu_setup(std::string& err) {
    if (!gpu::init(o.cuda_device, err)) return false;
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
    auto al = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t s_gub = 2 * I * (H / 32) * 16, s_gus = 2 * I * (H / 32), s_dnb = H * (I / 32) * 16, s_dns = H * (I / 32);
    const size_t o2 = al(s_gub), o3 = o2 + al(s_gus), o4 = o3 + al(s_dnb), slot_bytes = o4 + al(s_dns);
    long long want = std::min<long long>((long long) c.layers * c.n_exp, budget > 0 ? (long long) (budget / (double) slot_bytes) : 0);
    if (o.gpu_slots >= 0) want = std::min<long long>(want, o.gpu_slots);
    for (long long i = 0; i < want; ++i) {
        void* b = gpu::alloc(slot_bytes);
        if (!b) break;
        Slot sl;
        uint8_t* p = (uint8_t*) b;
        sl.base = b; sl.gu_b = p; sl.gu_s = p + o2; sl.dn_b = p + o3; sl.dn_s = p + o4;
        pool.push_back(sl);
    }
    gpu_on = true;
    std::fprintf(stderr, "gptoss: CUDA device '%s' (%s): dense weights %.2f GiB on device%s, %zu expert slots of %.2f MiB (%.0f%% of the %d experts)\n",
                 gpu::device_name(), gpu::is_emulated() ? "EMULATED" : "real", dense_used / 1073741824.0, d_head ? "" : " (lm_head stays on the host: no room)",
                 pool.size(), (double) slot_bytes / 1048576.0, 100.0 * (double) pool.size() / ((double) c.layers * c.n_exp), c.layers * c.n_exp);
    return true;
}

Slot* Model::Impl::resident(int li, int e) {
    if (pool.empty()) return nullptr;
    const long long key = (long long) li * c.n_exp + e;
    ++tick;
    auto it = where.find(key);
    if (it != where.end()) { ++hits; pool[(size_t) it->second].tick = tick; return &pool[(size_t) it->second]; }
    ++misses;
    int v = -1;
    for (size_t i = 0; i < pool.size() && v < 0; ++i) if (pool[i].layer < 0) v = (int) i;
    if (v < 0) {                                       // LRU
        uint64_t best = ~(uint64_t) 0;
        for (size_t i = 0; i < pool.size(); ++i) if (pool[i].tick < best) { best = pool[i].tick; v = (int) i; }
    }
    Slot& sl = pool[(size_t) v];
    if (sl.layer >= 0) where.erase((long long) sl.layer * c.n_exp + sl.exp);
    sl.layer = -1;
    const Layer& y = L[(size_t) li];
    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff, G1 = H / 32, G2 = I / 32;
    const bool ok = gpu::h2d(sl.gu_b, y.gu.blocks + (size_t) e * 2 * I * G1 * 16, 2 * I * G1 * 16) &&
                    gpu::h2d(sl.gu_s, y.gu.scales + (size_t) e * 2 * I * G1, 2 * I * G1) &&
                    gpu::h2d(sl.dn_b, y.dn.blocks + (size_t) e * H * G2 * 16, H * G2 * 16) &&
                    gpu::h2d(sl.dn_s, y.dn.scales + (size_t) e * H * G2, H * G2);
    if (!ok) { gpu_fail(gpu::last_error()); return nullptr; }
    sl.layer = li; sl.exp = e; sl.tick = tick;
    where[key] = v;
    return &sl;
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

// Batched prefill with the device: every projection and every expert is ONE pass over the weights for the whole chunk.
void Model::Impl::prefill_gpu(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    const int H = c.hidden, E = c.n_exp, K = c.topk, I = c.moe_ff, QD = c.q_dim(), KD = c.kv_dim(), D = c.head_dim, half = D / 2;
    std::vector<float> X, Hn, Qb, Kb, Vb, Ab, Ob, Gl, Y, Xe, GU, ACT, EB;
    std::vector<int32_t> IDX;
    std::vector<float> WT;
    for (int c0 = 0; c0 < n; c0 += chunk_max) {
        const int T = std::min(chunk_max, n - c0), p0 = start_pos + c0;
        X.assign((size_t) T * H, 0.0f); Hn.assign((size_t) T * H, 0.0f); Qb.assign((size_t) T * QD, 0.0f); Kb.assign((size_t) T * KD, 0.0f);
        Vb.assign((size_t) T * KD, 0.0f); Ab.assign((size_t) T * QD, 0.0f); Ob.assign((size_t) T * H, 0.0f); Gl.assign((size_t) T * E, 0.0f);
        Y.assign((size_t) T * H, 0.0f); IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) X[(size_t) t * H + i] = bf16_to_f32(((const uint16_t*) embed.p)[(size_t) tokens[c0 + t] * H + i]);
        for (int l = 0; l < c.layers; ++l) {
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
                std::memcpy(&y.K[y.slot(pos) * KD], k, sizeof(float) * (size_t) KD);
                std::memcpy(&y.V[y.slot(pos) * KD], v, sizeof(float) * (size_t) KD);
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // attention on the host: the KV cache lives there
                Work& w = my_work();
                std::copy(&Qb[(size_t) t * QD], &Qb[(size_t) (t + 1) * QD], w.q.begin());
                attn_heads(y, w, p0 + t);
                std::copy(w.att.begin(), w.att.end(), &Ab[(size_t) t * QD]);
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
            }
            std::fill(Y.begin(), Y.end(), 0.0f);
            std::vector<std::vector<int>> by_exp((size_t) E);
            for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) by_exp[(size_t) IDX[(size_t) t * K + k]].push_back(t * K + k);
            for (int e = 0; e < E; ++e) {
                const auto& lst = by_exp[(size_t) e];
                const int N = (int) lst.size();
                if (N == 0) continue;
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
            }
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

void Model::forward(int token, int pos, std::vector<float>* logits_out) {
    Impl& s = *p_;
    const GptConfig& c = s.c;
    const int H = c.hidden;
    if (pos < 0 || pos >= s.o.ctx || token < 0 || token >= c.vocab) { if (logits_out) logits_out->assign((size_t) c.vocab, 0.0f); return; }
    std::vector<float> x((size_t) H), acc((size_t) H);
    for (int i = 0; i < H; ++i) x[(size_t) i] = bf16_to_f32(((const uint16_t*) s.embed.p)[(size_t) token * H + i]);
    Work& w = s.works[0];
    for (int l = 0; l < c.layers; ++l) {
        Layer& L = s.L[(size_t) l];
        s.attn_in(L, w, x.data(), pos);
        s.attn_out(L, w, x.data(), pos);
        int32_t idx[64]; float wt[64];
        s.route_tok(L, w, x.data(), idx, wt);
        const std::vector<float> h = w.h;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int k = 0; k < c.topk; ++k) s.expert_add(L, w, h.data(), idx[k], wt[k], acc.data());
        for (int i = 0; i < H; ++i) x[(size_t) i] += acc[(size_t) i];
    }
    if ((int) s.cached.size() > pos) s.cached.resize((size_t) pos);
    if ((int) s.cached.size() == pos) s.cached.push_back(token);
    if (logits_out) {
        rmsnorm(x.data(), s.final_norm.data(), c.eps, H, s.lastn.data());
        logits_out->resize((size_t) c.vocab);
        s.mv(s.lm_head, s.d_head, s.lastn.data(), logits_out->data());
    }
}

void Model::prefill(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    Impl& s = *p_;
    const GptConfig& c = s.c;
    const int H = c.hidden, E = c.n_exp, K = c.topk, QD = c.q_dim();
    if (n <= 0) return;
    bool bad = start_pos < 0 || start_pos + n > s.o.ctx;
    for (int i = 0; i < n && !bad; ++i) bad = tokens[i] < 0 || tokens[i] >= c.vocab;
    if (bad) { if (logits_last) logits_last->assign((size_t) c.vocab, 0.0f); return; }
    if (s.gpu_on) { s.prefill_gpu(tokens, n, start_pos, logits_last); return; }
    std::vector<float> X, Hh, Y, Q;
    std::vector<int32_t> IDX; std::vector<float> WT;
    for (int c0 = 0; c0 < n; c0 += s.chunk_max) {
        const int T = std::min(s.chunk_max, n - c0), p0 = start_pos + c0;
        X.assign((size_t) T * H, 0.0f); Hh.assign((size_t) T * H, 0.0f); Y.assign((size_t) T * H, 0.0f); Q.assign((size_t) T * QD, 0.0f);
        IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) X[(size_t) t * H + i] = bf16_to_f32(((const uint16_t*) s.embed.p)[(size_t) tokens[c0 + t] * H + i]);
        for (int l = 0; l < c.layers; ++l) {
            Layer& L = s.L[(size_t) l];
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // K/V of the whole chunk first
                Work& w = s.my_work();
                s.attn_in(L, w, &X[(size_t) t * H], p0 + t);
                std::copy(w.q.begin(), w.q.end(), &Q[(size_t) t * QD]);            // the next loop may run this token on another thread
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                          // then every token attends to its window / past
                Work& w = s.my_work();
                std::copy(&Q[(size_t) t * QD], &Q[(size_t) (t + 1) * QD], w.q.begin());
                s.attn_out(L, w, &X[(size_t) t * H], p0 + t);
                s.route_tok(L, w, &X[(size_t) t * H], &IDX[(size_t) t * K], &WT[(size_t) t * K]);
                std::copy(w.h.begin(), w.h.end(), &Hh[(size_t) t * H]);
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
    return true;
}

}  // namespace gptoss

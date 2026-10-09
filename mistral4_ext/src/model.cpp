#include "m4/model.hpp"
#include "m4/safetensors.hpp"
#include "m4/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace m4 {

namespace {
struct Layer {
    std::vector<float> in_ln, post_ln, qa_ln, kva_ln;
    WView q_a, q_b, kv_a, o, gate, sh_gate, sh_up, sh_down;
    std::vector<float> kv_b;                       // dequantised [heads*(nope+v) x kv_lora]: the matrices of the absorbed form
    const StTensor *gu = nullptr, *dn = nullptr;   // fused expert tensors (mmap): gate_up [E,2I,H], down [E,H,I]
    std::vector<float> gu_scale, dn_scale;         // effective scales after the mul/div switch
    int gu_parts = 1;                              // 1: one scale per expert; 2: one for gate and one for up
    int sh_rows = 0;
    std::vector<float> ckv, kpe;                   // KV cache: [ctx][kv_lora] and [ctx][rope] (rope already applied)
};
}  // namespace

struct Work {                                     // scratch of ONE thread (batched prefill runs one token per thread)
    std::vector<float> h, cq, q, ckv, att, tmp, glog, y, gbuf, ubuf, act, ebuf, qa, sc, cl;
};

struct Model::Impl {
    M4Config c;
    RunOpts o;
    StFiles st;
    std::vector<Layer> L;
    WView embed, lm_head;
    std::vector<float> final_norm, rope_cos, rope_sin;
    double sm_scale = 0;
    std::vector<Work> works;
    std::vector<int> cached;                      // token at every cached position (the KV cache holds exactly cached.size() positions)
    std::vector<float> lastn;

    void init_work(Work& w) const;
    Work& my_work();
    void attn_in(Layer& y, Work& w, const float* x, int pos);
    void attn_out(Layer& y, Work& w, float* x, int pos);
    void route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt);      // fills w.h (post-attn norm), routes
    void expert_add(Layer& y, Work& w, const float* h, int e, float wt, float* acc);
    void shared_add(Layer& y, Work& w, const float* h, float* acc);
    std::string fingerprint() const;
    void plan_vram() const;
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
const M4Config& Model::config() const { return p_->c; }
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
    M4Config& c = s.c;
    std::vector<std::string> shards;
    if (!discover_shards(dir, shards, err) || !s.st.open(shards, err)) return false;
    if (opts.verbose) std::fprintf(stderr, "m4: %s\nm4: %zu shard(s), %zu tensors, %.2f GiB mapped (not read yet)\n", config_summary(c).c_str(),
                                   shards.size(), s.st.tensors().size(), (double) s.st.total_bytes() / (1024.0 * 1024 * 1024));

    for (const char* p : {"language_model.model.", "model.language_model.", "model."})
        if (s.st.find(std::string(p) + "embed_tokens.weight")) { c.prefix = p; break; }
    if (c.prefix.empty()) { err = "no tensor named *embed_tokens.weight (unknown naming)"; return false; }

    auto find = [&](const std::string& n) -> const StTensor* { return s.st.find(n); };
    auto get = [&](const std::string& n, const StTensor*& t) { t = find(n); if (!t) err = "missing tensor " + n; return t != nullptr; };
    auto shape_is = [&](const StTensor* t, std::vector<int64_t> want) {
        if (t->shape != want) {
            err = "tensor " + t->name + " has shape [";
            for (size_t i = 0; i < t->shape.size(); ++i) err += (i ? "," : "") + std::to_string(t->shape[i]);
            err += "], expected [";
            for (size_t i = 0; i < want.size(); ++i) err += (i ? "," : "") + std::to_string(want[i]);
            err += "]";
            return false;
        }
        return true;
    };
    auto mat = [&](const std::string& base, int rows, int cols, WView& v) {
        const StTensor* w; if (!get(base + ".weight", w) || !shape_is(w, {rows, cols})) return false;
        if (w->dtype == DType::Unknown) { err = base + ": unsupported dtype " + w->dtype_str; return false; }
        v.p = w->data; v.rows = rows; v.cols = cols; v.dt = w->dtype; v.scale = 1.0f;
        if (w->dtype == DType::F8E4M3) {
            const StTensor* sc; if (!get(base + ".weight_scale_inv", sc)) return false;
            if (sc->numel() != 1) { err = base + ".weight_scale_inv has " + std::to_string(sc->numel()) + " values: only per-tensor scales are implemented"; return false; }
            v.scale = fp8_scale_apply(decode_all(*sc)[0], opts.fp8_scale_div);
        }
        return true;
    };
    auto vec = [&](const std::string& n, int len, std::vector<float>& out) {
        const StTensor* t; if (!get(n, t) || !shape_is(t, {len})) return false;
        out = decode_all(*t); return true;
    };
    auto escales = [&](const std::string& n, std::vector<float>& out) {
        const StTensor* sc; if (!get(n, sc)) return false;
        out = decode_all(*sc);
        for (auto& v : out) v = fp8_scale_apply(v, opts.fp8_scale_div);
        return true;
    };

    const std::string& P = c.prefix;
    const int H = c.hidden, nh = c.heads, E = c.n_exp, I = c.moe_ff;
    const int qk = c.qk_head(), bs = c.nope + c.v_head;
    const StTensor* t;
    if (!get(P + "embed_tokens.weight", t) || !shape_is(t, {c.vocab, H})) return false;
    s.embed = {t->data, c.vocab, H, t->dtype, 1.0f};
    const char* head_names[] = {"language_model.lm_head.weight", "lm_head.weight"};
    t = nullptr;
    for (const char* n : head_names) if ((t = find(n))) break;
    if (!t) { err = "missing lm_head.weight"; return false; }
    if (!shape_is(t, {c.vocab, H})) return false;
    s.lm_head = {t->data, c.vocab, H, t->dtype, 1.0f};
    if (!vec(P + "norm.weight", H, s.final_norm)) return false;

    s.L.resize((size_t) c.layers);
    for (int l = 0; l < c.layers; ++l) {
        Layer& y = s.L[(size_t) l];
        const std::string b = P + "layers." + std::to_string(l) + ".";
        if (!vec(b + "input_layernorm.weight", H, y.in_ln) || !vec(b + "post_attention_layernorm.weight", H, y.post_ln) ||
            !vec(b + "self_attn.q_a_layernorm.weight", c.q_lora, y.qa_ln) || !vec(b + "self_attn.kv_a_layernorm.weight", c.kv_lora, y.kva_ln)) return false;
        if (!mat(b + "self_attn.q_a_proj", c.q_lora, H, y.q_a) || !mat(b + "self_attn.q_b_proj", nh * qk, c.q_lora, y.q_b) ||
            !mat(b + "self_attn.kv_a_proj_with_mqa", c.kv_lora + c.rope, H, y.kv_a) || !mat(b + "self_attn.o_proj", H, nh * c.v_head, y.o)) return false;
        WView kvb;
        if (!mat(b + "self_attn.kv_b_proj", nh * bs, c.kv_lora, kvb)) return false;
        y.kv_b.resize((size_t) nh * bs * c.kv_lora);
        for (int r = 0; r < nh * bs; ++r) decode_row(kvb, r, &y.kv_b[(size_t) r * c.kv_lora]);
        if (!mat(b + "mlp.gate", E, H, y.gate)) { err = err.empty() ? "mlp.gate" : err; return false; }
        if (y.gate.dt == DType::F8E4M3) { err = b + "mlp.gate.weight is FP8: routing in FP8 is not implemented"; return false; }
        if (!get(b + "mlp.experts.gate_up_proj", y.gu) || !shape_is(y.gu, {E, 2 * I, H}) ||
            !get(b + "mlp.experts.down_proj", y.dn) || !shape_is(y.dn, {E, H, I})) return false;
        for (const StTensor* w : {y.gu, y.dn})
            if (w->dtype != DType::F8E4M3 && w->dtype != DType::BF16) { err = w->name + ": dtype " + w->dtype_str + " not implemented for experts"; return false; }
        if (y.gu->dtype == DType::F8E4M3) {
            if (!escales(b + "mlp.experts.gate_up_proj_scale_inv", y.gu_scale)) return false;
            const size_t n = y.gu_scale.size();
            if (n == 0 || n % (size_t) E || (n / (size_t) E != 1 && n / (size_t) E != 2)) { err = b + "mlp.experts.gate_up_proj_scale_inv has " + std::to_string(n) + " values: expected E or 2E"; return false; }
            y.gu_parts = (int) (n / (size_t) E);
        }
        if (y.dn->dtype == DType::F8E4M3) {
            if (!escales(b + "mlp.experts.down_proj_scale_inv", y.dn_scale)) return false;
            if (y.dn_scale.size() != (size_t) E && y.dn_scale.size() != 1) { err = b + "mlp.experts.down_proj_scale_inv has " + std::to_string(y.dn_scale.size()) + " values: expected E or 1"; return false; }
        }
        const StTensor* sg;
        if (!get(b + "mlp.shared_experts.gate_proj.weight", sg)) return false;
        y.sh_rows = (int) sg->shape[0];
        if (!mat(b + "mlp.shared_experts.gate_proj", y.sh_rows, H, y.sh_gate) || !mat(b + "mlp.shared_experts.up_proj", y.sh_rows, H, y.sh_up) ||
            !mat(b + "mlp.shared_experts.down_proj", H, y.sh_rows, y.sh_down)) return false;
        y.ckv.assign((size_t) opts.ctx * c.kv_lora, 0.0f);
        y.kpe.assign((size_t) opts.ctx * c.rope, 0.0f);
    }

    std::vector<double> f;
    yarn_freqs(c.rope, c.rope_theta, c.yarn_factor, c.yarn_orig, c.yarn_beta_fast, c.yarn_beta_slow, f);
    rope_table(f, opts.ctx, s.rope_cos, s.rope_sin);
    s.sm_scale = mla_softmax_scale(qk, c.yarn_factor, c.yarn_mscale_all_dim, opts.mscale_softmax);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    s.works.resize((size_t) nt);
    for (auto& w : s.works) s.init_work(w);
    s.lastn.assign(H, 0);
    s.cached.clear();
    if (opts.vram_pct >= 0) s.plan_vram();
    if (!opts.kv_checkpoint.empty()) {
        std::string kerr;
        if (FILE* f = std::fopen(opts.kv_checkpoint.c_str(), "rb")) {
            std::fclose(f);
            if (load_kv(opts.kv_checkpoint, kerr)) std::fprintf(stderr, "m4: KV checkpoint restored: %zu tokens from %s\n", s.cached.size(), opts.kv_checkpoint.c_str());
            else std::fprintf(stderr, "m4: KV checkpoint ignored (%s)\n", kerr.c_str());
        }
    }
    return true;
}

void Model::Impl::init_work(Work& w) const {
    const int sh = std::max(L[0].sh_rows, 1);
    w.h.assign(c.hidden, 0); w.cq.assign(c.q_lora, 0); w.q.assign((size_t) c.heads * c.qk_head(), 0); w.ckv.assign(c.kv_lora + c.rope, 0);
    w.att.assign((size_t) c.heads * c.v_head, 0); w.tmp.assign(c.hidden, 0); w.glog.assign(c.n_exp, 0); w.y.assign(c.hidden, 0);
    const size_t m = (size_t) std::max(c.moe_ff, sh);
    w.gbuf.assign(m, 0); w.ubuf.assign(m, 0); w.act.assign(m, 0); w.ebuf.assign(c.hidden, 0);
    w.qa.assign(c.kv_lora, 0); w.cl.assign(c.kv_lora, 0); w.sc.assign((size_t) o.ctx, 0);
}

Work& Model::Impl::my_work() {
#ifdef _OPENMP
    return works[(size_t) omp_get_thread_num() % works.size()];
#else
    return works[0];
#endif
}

void Model::Impl::plan_vram() const {
    // Plan only. Dense weights (attention, shared expert, gate, embeddings, lm_head) would live on the card; the rest of the budget
    // becomes expert slots. KV cache and activations stay on the host in this engine, as in dsv4_ext.
    uint64_t dense = 0;
    for (const auto& t : st.tensors())
        if (t.name.rfind("language_model.", 0) == 0 && t.name.find(".experts.") == std::string::npos) dense += t.nbytes();
    const Layer& l0 = L[0];
    const uint64_t slot = (l0.gu->nbytes() + l0.dn->nbytes()) / (uint64_t) c.n_exp;       // gate_up + down of ONE expert
    const double total = (double) o.vram_total_mib * 1048576.0, budget = total * o.vram_pct / 100.0 - (double) o.vram_reserve_mib * 1048576.0 - (double) dense;
    std::fprintf(stderr, "m4: VRAM plan (NOT executed: no CUDA backend in this build): card %lld MiB x %.0f%% - reserve %lld MiB - dense weights %.2f GiB\n",
                 o.vram_total_mib, o.vram_pct, o.vram_reserve_mib, (double) dense / 1073741824.0);
    if (o.vram_total_mib <= 0) { std::fprintf(stderr, "m4:   --vram-total-mib missing: cannot plan\n"); return; }
    if (budget < (double) slot) { std::fprintf(stderr, "m4:   the dense weights alone do not fit in that budget: 0 expert slots\n"); return; }
    const long long slots = (long long) (budget / (double) slot);
    const long long per_layer = std::min<long long>(slots / c.layers, c.n_exp);
    std::fprintf(stderr, "m4:   expert = %.1f MiB -> %lld slots, %lld per layer of %d (%.0f%% of the experts; hit rate >= that with a usage profile)\n",
                 (double) slot / 1048576.0, slots, per_layer, c.n_exp, 100.0 * (double) per_layer / c.n_exp);
}

std::string Model::Impl::fingerprint() const {
    char b[512];
    std::snprintf(b, sizeof b, "%s|ctx-independent|router=%d|fp8div=%d|bytes=%llu", config_summary(c).c_str(), (int) o.router, (int) o.fp8_scale_div,
                  (unsigned long long) st.total_bytes());
    return b;
}

// ---- building blocks shared by the single-token path and the batched prefill

void Model::Impl::attn_in(Layer& y, Work& w, const float* x, int pos) {
    const int H = c.hidden, kl = c.kv_lora, nope = c.nope, rope = c.rope, qk = nope + rope, half = rope / 2;
    rmsnorm(x, y.in_ln.data(), c.eps, H, w.h.data());
    matvec(y.q_a, w.h.data(), w.cq.data());
    rmsnorm(w.cq.data(), y.qa_ln.data(), c.eps, c.q_lora, w.cq.data());
    matvec(y.q_b, w.cq.data(), w.q.data());
    matvec(y.kv_a, w.h.data(), w.ckv.data());
    float* cc = &y.ckv[(size_t) pos * kl];
    float* kp = &y.kpe[(size_t) pos * rope];
    rmsnorm(w.ckv.data(), y.kva_ln.data(), c.eps, kl, cc);
    std::memcpy(kp, w.ckv.data() + kl, sizeof(float) * (size_t) rope);
    const float* cs = &rope_cos[(size_t) pos * half];
    const float* sn = &rope_sin[(size_t) pos * half];
    rotary_pairs(kp, rope, cs, sn);
    for (int hd = 0; hd < c.heads; ++hd) rotary_pairs(&w.q[(size_t) hd * qk + nope], rope, cs, sn);
}

void Model::Impl::attn_out(Layer& y, Work& w, float* x, int pos) {
    const int H = c.hidden, nh = c.heads, kl = c.kv_lora, nope = c.nope, rope = c.rope, vh = c.v_head, qk = nope + rope, bs = nope + vh, T = pos + 1;
    const double s4 = llama4_scale(pos, c.llama4_beta, c.yarn_orig);
    const float s_nope = o.l4_qpe_only ? 1.0f : (float) s4, s_pe = (float) s4;
    auto one_head = [&](Work& ww, int hd) {                  // q and att come from `w`, the scratch from `ww`
        const float* qh = &w.q[(size_t) hd * qk];
        float* qa = ww.qa.data(); float* sc = ww.sc.data(); float* cl = ww.cl.data();
        std::fill(qa, qa + kl, 0.0f);
        for (int d = 0; d < nope; ++d) {                       // qa = W_UK^T q_nope
            const float* row = &y.kv_b[(size_t) (hd * bs + d) * kl];
            const float qd = qh[d];
            for (int j = 0; j < kl; ++j) qa[j] += qd * row[j];
        }
        float mx = -INFINITY;
        for (int u = 0; u < T; ++u) {
            const float* cu = &y.ckv[(size_t) u * kl];
            const float* ku = &y.kpe[(size_t) u * rope];
            float a = 0, b = 0;
            for (int j = 0; j < kl; ++j) a += qa[j] * cu[j];
            for (int j = 0; j < rope; ++j) b += qh[nope + j] * ku[j];
            sc[u] = (float) (((double) a * s_nope + (double) b * s_pe) * sm_scale);
            mx = std::max(mx, sc[u]);
        }
        double z = 0;
        for (int u = 0; u < T; ++u) { sc[u] = std::exp(sc[u] - mx); z += sc[u]; }
        std::fill(cl, cl + kl, 0.0f);
        for (int u = 0; u < T; ++u) {                           // ctx_latent = sum p_u c_u
            const float p = (float) (sc[u] / z);
            const float* cu = &y.ckv[(size_t) u * kl];
            for (int j = 0; j < kl; ++j) cl[j] += p * cu[j];
        }
        for (int d = 0; d < vh; ++d) {                          // out = W_UV ctx_latent
            const float* row = &y.kv_b[(size_t) (hd * bs + nope + d) * kl];
            float a = 0;
            for (int j = 0; j < kl; ++j) a += row[j] * cl[j];
            w.att[(size_t) hd * vh + d] = a;
        }
    };
    bool in_par = false;
#ifdef _OPENMP
    in_par = omp_in_parallel();
#endif
    if (in_par) { for (int hd = 0; hd < nh; ++hd) one_head(w, hd); }     // batched prefill: the parallelism is over tokens
    else {
#pragma omp parallel for schedule(static)
        for (int hd = 0; hd < nh; ++hd) one_head(my_work(), hd);          // decode: split the heads across threads
    }
    matvec(y.o, w.att.data(), w.tmp.data());
    for (int i = 0; i < H; ++i) x[i] += w.tmp[i];
}

void Model::Impl::route_tok(Layer& y, Work& w, const float* x, int32_t* idx, float* wt) {
    rmsnorm(x, y.post_ln.data(), c.eps, c.hidden, w.h.data());
    matvec(y.gate, w.h.data(), w.glog.data());
    route(w.glog.data(), c.n_exp, c.topk, o.router, c.norm_topk, c.routed_scale, idx, wt);
}

void Model::Impl::expert_add(Layer& y, Work& w, const float* h, int e, float wt, float* acc) {
    const int H = c.hidden, I = c.moe_ff;
    const size_t esz = (size_t) dtype_bytes(y.gu->dtype);
    const uint8_t* g0 = y.gu->data + (size_t) e * 2 * I * H * esz;
    const float sg = y.gu_scale.empty() ? 1.0f : y.gu_scale[(size_t) e * y.gu_parts];
    const float su = y.gu_scale.empty() ? 1.0f : y.gu_scale[(size_t) e * y.gu_parts + (y.gu_parts - 1)];
    matvec({g0, I, H, y.gu->dtype, sg}, h, w.gbuf.data());
    matvec({g0 + (size_t) I * H * esz, I, H, y.gu->dtype, su}, h, w.ubuf.data());
    for (int i = 0; i < I; ++i) w.act[(size_t) i] = silu(w.gbuf[(size_t) i]) * w.ubuf[(size_t) i];
    const float sd = y.dn_scale.empty() ? 1.0f : (y.dn_scale.size() == 1 ? y.dn_scale[0] : y.dn_scale[(size_t) e]);
    matvec({y.dn->data + (size_t) e * H * I * (size_t) dtype_bytes(y.dn->dtype), H, I, y.dn->dtype, sd}, w.act.data(), w.ebuf.data());
    for (int i = 0; i < H; ++i) acc[i] += wt * w.ebuf[(size_t) i];
}

void Model::Impl::shared_add(Layer& y, Work& w, const float* h, float* acc) {
    matvec(y.sh_gate, h, w.gbuf.data());
    matvec(y.sh_up, h, w.ubuf.data());
    for (int i = 0; i < y.sh_rows; ++i) w.act[(size_t) i] = silu(w.gbuf[(size_t) i]) * w.ubuf[(size_t) i];
    matvec(y.sh_down, w.act.data(), w.ebuf.data());
    for (int i = 0; i < c.hidden; ++i) acc[i] += w.ebuf[(size_t) i];
}

void Model::forward(int token, int pos, std::vector<float>* logits_out) {
    Impl& s = *p_;
    const M4Config& c = s.c;
    const int H = c.hidden;
    if (pos < 0 || pos >= s.o.ctx || token < 0 || token >= c.vocab) { if (logits_out) logits_out->assign((size_t) c.vocab, 0.0f); return; }
    std::vector<float> x((size_t) H);
    decode_row(s.embed, token, x.data());
    Work& w = s.works[0];
    std::vector<float> y((size_t) H), acc((size_t) H);
    for (int l = 0; l < c.layers; ++l) {
        Layer& L = s.L[(size_t) l];
        s.attn_in(L, w, x.data(), pos);
        s.attn_out(L, w, x.data(), pos);
        int32_t idx[64]; float wt[64];
        s.route_tok(L, w, x.data(), idx, wt);
        const std::vector<float> h = w.h;                       // expert_add reuses w's buffers, h must survive
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int k = 0; k < c.topk; ++k) s.expert_add(L, w, h.data(), idx[k], wt[k], acc.data());
        s.shared_add(L, w, h.data(), acc.data());
        for (int i = 0; i < H; ++i) x[(size_t) i] += acc[(size_t) i];
    }
    if ((int) s.cached.size() > pos) s.cached.resize((size_t) pos);
    if ((int) s.cached.size() == pos) s.cached.push_back(token);
    if (logits_out) {
        rmsnorm(x.data(), s.final_norm.data(), c.eps, H, s.lastn.data());
        logits_out->resize((size_t) c.vocab);
        matvec(s.lm_head, s.lastn.data(), logits_out->data());
    }
}

void Model::prefill(const int* tokens, int n, int start_pos, std::vector<float>* logits_last) {
    Impl& s = *p_;
    const M4Config& c = s.c;
    const int H = c.hidden, E = c.n_exp, K = c.topk;
    if (n <= 0) return;
    if (start_pos < 0 || start_pos + n > s.o.ctx) { if (logits_last) logits_last->assign((size_t) c.vocab, 0.0f); return; }
    for (int i = 0; i < n; ++i) if (tokens[i] < 0 || tokens[i] >= c.vocab) { if (logits_last) logits_last->assign((size_t) c.vocab, 0.0f); return; }
    const int chunk = std::max(1, s.o.prefill_chunk);
    std::vector<float> X, Hh, Y, Q;
    std::vector<int32_t> IDX; std::vector<float> WT;
    for (int c0 = 0; c0 < n; c0 += chunk) {
        const int T = std::min(chunk, n - c0), p0 = start_pos + c0;
        const size_t qn = (size_t) c.heads * c.qk_head();
        Q.assign((size_t) T * qn, 0.0f);
        X.assign((size_t) T * H, 0.0f); Hh.assign((size_t) T * H, 0.0f); Y.assign((size_t) T * H, 0.0f);
        IDX.assign((size_t) T * K, 0); WT.assign((size_t) T * K, 0.0f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) decode_row(s.embed, tokens[c0 + t], &X[(size_t) t * H]);
        for (int l = 0; l < c.layers; ++l) {
            Layer& L = s.L[(size_t) l];
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                                                 // fills the cache of the whole chunk first
                Work& w = s.my_work();
                s.attn_in(L, w, &X[(size_t) t * H], p0 + t);
                std::copy(w.q.begin(), w.q.end(), &Q[(size_t) t * qn]);                                   // the next loop may run on another thread
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {                                                                 // then every token attends to its past
                Work& w = s.my_work();
                std::copy(&Q[(size_t) t * qn], &Q[(size_t) (t + 1) * qn], w.q.begin());
                s.attn_out(L, w, &X[(size_t) t * H], p0 + t);
                s.route_tok(L, w, &X[(size_t) t * H], &IDX[(size_t) t * K], &WT[(size_t) t * K]);
                std::copy(w.h.begin(), w.h.end(), &Hh[(size_t) t * H]);
            }
            std::fill(Y.begin(), Y.end(), 0.0f);
            std::vector<std::vector<int>> by_exp((size_t) E);                                            // token slots per expert
            for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) by_exp[(size_t) IDX[(size_t) t * K + k]].push_back(t * K + k);
            for (int e = 0; e < E; ++e) {                                                                // one pass over each expert's weights
                const auto& lst = by_exp[(size_t) e];
#pragma omp parallel for schedule(dynamic, 1)
                for (int i = 0; i < (int) lst.size(); ++i) {
                    const int t = lst[(size_t) i] / K;                                                   // a token has an expert at most once: no race on Y[t]
                    s.expert_add(L, s.my_work(), &Hh[(size_t) t * H], e, WT[(size_t) lst[(size_t) i]], &Y[(size_t) t * H]);
                }
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < T; ++t) {
                s.shared_add(L, s.my_work(), &Hh[(size_t) t * H], &Y[(size_t) t * H]);
                for (int i = 0; i < H; ++i) X[(size_t) t * H + i] += Y[(size_t) t * H + i];
            }
        }
        if ((int) s.cached.size() > p0) s.cached.resize((size_t) p0);
        for (int t = 0; t < T; ++t) if ((int) s.cached.size() == p0 + t) s.cached.push_back(tokens[c0 + t]);
        if (logits_last && c0 + T == n) {
            rmsnorm(&X[(size_t) (T - 1) * H], s.final_norm.data(), c.eps, H, s.lastn.data());
            logits_last->resize((size_t) c.vocab);
            matvec(s.lm_head, s.lastn.data(), logits_last->data());
        }
    }
}

int Model::reuse_prefix(const std::vector<int>& ids) {
    Impl& s = *p_;
    size_t k = 0;
    while (k < s.cached.size() && k < ids.size() && s.cached[k] == ids[k]) ++k;
    if (!ids.empty() && k >= ids.size()) k = ids.size() - 1;      // the last prompt token is always recomputed: its logits start the answer
    s.cached.resize(k);
    return (int) k;
}

// ---- KV checkpoint
namespace {
struct KvHead { char magic[8]; uint32_t layers, kv_lora, rope, fp_len; uint64_t n_pos; };
}

bool Model::save_kv(const std::string& path, std::string& err) const {
    const Impl& s = *p_;
    const std::string tmp = path + ".tmp", fp = s.fingerprint();
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    KvHead h{{'M', '4', 'K', 'V', '0', '0', '0', '1'}, (uint32_t) s.c.layers, (uint32_t) s.c.kv_lora, (uint32_t) s.c.rope, (uint32_t) fp.size(), (uint64_t) s.cached.size()};
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(fp.data(), 1, fp.size(), f) == fp.size() &&
              std::fwrite(s.cached.data(), sizeof(int), s.cached.size(), f) == s.cached.size();
    for (const Layer& L : s.L) {
        ok = ok && std::fwrite(L.ckv.data(), sizeof(float), s.cached.size() * (size_t) s.c.kv_lora, f) == s.cached.size() * (size_t) s.c.kv_lora;
        ok = ok && std::fwrite(L.kpe.data(), sizeof(float), s.cached.size() * (size_t) s.c.rope, f) == s.cached.size() * (size_t) s.c.rope;
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
    bool ok = std::fread(&h, sizeof h, 1, f) == 1 && std::memcmp(h.magic, "M4KV0001", 8) == 0;
    if (!ok) { std::fclose(f); err = "not a KV checkpoint"; return false; }
    std::string fp(h.fp_len, '\0');
    if (h.fp_len > 4096 || std::fread(&fp[0], 1, h.fp_len, f) != h.fp_len || fp != s.fingerprint() || h.layers != (uint32_t) s.c.layers ||
        h.kv_lora != (uint32_t) s.c.kv_lora || h.rope != (uint32_t) s.c.rope) { std::fclose(f); err = "checkpoint belongs to another model/options"; return false; }
    if (h.n_pos > (uint64_t) s.o.ctx) { std::fclose(f); err = "checkpoint has more tokens than --ctx"; return false; }
    const size_t n = (size_t) h.n_pos;
    std::vector<int> toks(n);
    ok = std::fread(toks.data(), sizeof(int), n, f) == n;
    for (Layer& L : s.L) {
        ok = ok && std::fread(L.ckv.data(), sizeof(float), n * (size_t) s.c.kv_lora, f) == n * (size_t) s.c.kv_lora;
        ok = ok && std::fread(L.kpe.data(), sizeof(float), n * (size_t) s.c.rope, f) == n * (size_t) s.c.rope;
    }
    std::fclose(f);
    if (!ok) { s.cached.clear(); err = "truncated checkpoint"; return false; }
    s.cached = std::move(toks);
    return true;
}

}  // namespace m4

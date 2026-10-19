// dsv4_forward.hpp - forward completo di DeepSeek-V4 in CPU, prefill su sequenza intera, senza cache.
// Port del grafo llama_model_deepseek4::graph::graph (righe 1222-1356 di deepseek4.cpp): embedding replicato su hc,
// per ogni layer: hc_pre -> attn_norm -> attenzione -> hc_post; hc_pre -> ffn_norm -> MoE + esperto condiviso -> hc_post;
// poi hc_head -> output_norm -> output (logits dell'ultimo token).
//
// Il routing MoE (build_moe_ffn) NON è nel file ricevuto: è scritto dalla semantica nota di llama.cpp e segnato come
// assunzione A14-A18. Va confermato contro llama-graph.cpp.
//
// ASSUNZIONI aggiuntive:
//  A14 expert_weights_norm: pesi del top-K divisi per la somma (senza eps), poi moltiplicati per expert_weights_scale.
//  A15 hc_fn, hc_scale, hc_base, attn_norm, ffn_norm, output_norm, hc_head_*: tensori F32 (il binding deve verificarlo).
//  A16 gating SQRT_SOFTPLUS: prob = sqrt(softplus(logit)), softplus stabile (x > 20 -> x).
//  A17 layer hash (il < dsv4_hash_layer_count): esperti = tid2eid[token][k], pesi dalle prob non biasate.
//  A18 layer non hash: selezione top-K su prob + exp_probs_b, pesi dalle prob senza bias.
//  A19 logits calcolati solo per l'ultimo token (prefill). Il layout di tokens è [T] di int32.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gptoss/attn_dsv4_comp.hpp"
#include "gptoss/cpu_matvec.hpp"
#include "gptoss/dequant_registry.hpp"
#include "gptoss/hc_ops.hpp"

namespace gptoss::dsv4 {

struct MoeCfg {
    int64_t n_embd = 4096;
    int64_t n_expert = 256;
    int64_t n_expert_used = 6;
    int64_t n_ff_exp = 2048;
    int64_t n_ff_shexp = 2048;     // DA LEGGERE dal tensore ffn_gate_shexp (dims[1])
    bool weights_norm = true;      // A14, hparams.expert_weights_norm
    float weights_scale = 1.0f;    // hparams.expert_weights_scale
};

struct MoeWeights {
    QuantMat gate_inp;             // [E -> NE]
    QuantMat up_exps;              // in=E, out=NFF per esperto, NE esperti impilati
    QuantMat gate_exps;            // come up_exps
    QuantMat down_exps;            // in=NFF, out=E per esperto
    const float* exp_probs_b = nullptr;  // [NE], null nei layer hash (A18)
    const int32_t* tid2eid = nullptr;    // [V][K], solo nei layer hash (A17)
    QuantMat gate_shexp;           // in=E, out=NFFsh
    QuantMat up_shexp;             // in=E, out=NFFsh
    QuantMat down_shexp;           // in=NFFsh, out=E
};

// Matvec su un singolo esperto di una matrice impilata (righe contigue per esperto).
inline void qmat_expert(const QuantMat& M, int64_t e, const float* x, float* y, std::vector<float>& buf) {
    const size_t rb = dequant_row_bytes(M.type, M.in);
    const uint8_t* W = M.data + static_cast<size_t>(e * M.out) * rb;
    matvec_quant(M.type, W, M.in, M.out, x, y, buf);
}

inline float silu_f(float v) { return v / (1.0f + std::exp(-v)); }

inline float softplus_stable(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

// MoE + esperto condiviso per un token. x: [E] normalizzato (ffn_norm). out: [E].
inline void moe_forward_token(const MoeCfg& c, const MoeWeights& w, const float* x, int32_t tok, bool hash_layer,
                              float* out, std::vector<float>& buf) {
    const int64_t NE = c.n_expert, K = c.n_expert_used, E = c.n_embd, NF = c.n_ff_exp, NS = c.n_ff_shexp;

    std::vector<float> logits(static_cast<size_t>(NE)), probs(static_cast<size_t>(NE));
    matvec_quant(w.gate_inp.type, w.gate_inp.data, E, NE, x, logits.data(), buf);
    for (int64_t e = 0; e < NE; ++e) probs[e] = std::sqrt(softplus_stable(logits[e]));  // A16

    std::vector<int64_t> sel(static_cast<size_t>(K));
    if (hash_layer) {  // A17
        for (int64_t k = 0; k < K; ++k) sel[k] = w.tid2eid[tok * K + k];
    } else {           // A18
        std::vector<int64_t> idx(static_cast<size_t>(NE));
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> sc(static_cast<size_t>(NE));
        for (int64_t e = 0; e < NE; ++e) sc[e] = probs[e] + (w.exp_probs_b ? w.exp_probs_b[e] : 0.0f);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int64_t a, int64_t b) { return sc[a] > sc[b]; });
        for (int64_t k = 0; k < K; ++k) sel[k] = idx[k];
    }

    std::vector<float> wgt(static_cast<size_t>(K));
    for (int64_t k = 0; k < K; ++k) wgt[k] = probs[sel[k]];
    if (c.weights_norm) {  // A14
        float s = 0.0f;
        for (float v : wgt) s += v;
        for (float& v : wgt) v /= s;
    }
    for (float& v : wgt) v *= c.weights_scale;

    for (int64_t i = 0; i < E; ++i) out[i] = 0.0f;
    std::vector<float> g(static_cast<size_t>(NF)), u(static_cast<size_t>(NF)), hbuf(static_cast<size_t>(NF));
    std::vector<float> y(static_cast<size_t>(E));
    for (int64_t k = 0; k < K; ++k) {
        const int64_t e = sel[k];
        qmat_expert(w.gate_exps, e, x, g.data(), buf);
        qmat_expert(w.up_exps, e, x, u.data(), buf);
        for (int64_t i = 0; i < NF; ++i) hbuf[i] = silu_f(g[i]) * u[i];
        qmat_expert(w.down_exps, e, hbuf.data(), y.data(), buf);
        for (int64_t i = 0; i < E; ++i) out[i] += wgt[k] * y[i];
    }

    // esperto condiviso (ffn_*_shexp, SILU in parallelo)
    std::vector<float> gs(static_cast<size_t>(NS)), us(static_cast<size_t>(NS)), hs(static_cast<size_t>(NS));
    matvec_quant(w.gate_shexp.type, w.gate_shexp.data, E, NS, x, gs.data(), buf);
    matvec_quant(w.up_shexp.type, w.up_shexp.data, E, NS, x, us.data(), buf);
    for (int64_t i = 0; i < NS; ++i) hs[i] = silu_f(gs[i]) * us[i];
    matvec_quant(w.down_shexp.type, w.down_shexp.data, NS, E, hs.data(), y.data(), buf);
    for (int64_t i = 0; i < E; ++i) out[i] += y[i];
}

// Un layer del modello con i pesi già legati (il binding dai GGUF è il passo successivo).
struct DsLayerW {
    AttnWeights attn;
    CompLayer cl;                  // cl.ratio = compress_ratios[il]
    const float* hc_attn_fn = nullptr;   // [hc*E ... ] A15
    const float* hc_attn_scale = nullptr;
    const float* hc_attn_base = nullptr;
    const float* attn_norm = nullptr;
    const float* hc_ffn_fn = nullptr;
    const float* hc_ffn_scale = nullptr;
    const float* hc_ffn_base = nullptr;
    const float* ffn_norm = nullptr;
    MoeWeights moe;
    bool hash_layer = false;       // il < dsv4_hash_layer_count
};

struct DsModel {
    AttnParams ap;                 // campi comuni (E, H, D, n_rot, n_swa, rms_eps, rope ...)
    MoeCfg moe_cfg;
    int64_t n_vocab = 129280;
    int64_t hc = 4;
    int sinkhorn_iters = 20;       // dsv4_hc_sinkhorn_iters  DA LEGGERE
    float hc_eps = 1e-6f;          // dsv4_hc_eps             DA LEGGERE
    QuantMat tok_embd;             // in=E, out=V: una riga per token
    std::vector<DsLayerW> layers;
    const float* hc_head_fn = nullptr;
    const float* hc_head_scale = nullptr;
    const float* hc_head_base = nullptr;
    const float* output_norm = nullptr;
    QuantMat output;               // in=E, out=V
};

// Forward su sequenza intera. tokens: [T]. logits: [V], dell'ultimo token (A19).
inline void forward_logits_last(const DsModel& m, const int32_t* tokens, int64_t T, float* logits) {
    const int64_t E = m.ap.n_embd, HC = m.hc, HCE = HC * E, V = m.n_vocab;
    const float eps = m.ap.rms_eps;
    std::vector<float> buf;

    // embedding, replicato su hc
    std::vector<float> X(static_cast<size_t>(T * HCE));
    std::vector<float> emb(static_cast<size_t>(E));
    const size_t erb = dequant_row_bytes(m.tok_embd.type, E);
    for (int64_t t = 0; t < T; ++t) {
        dequantize(m.tok_embd.type, m.tok_embd.data + static_cast<size_t>(tokens[t]) * erb, emb.data(), E);
        for (int64_t h = 0; h < HC; ++h)
            std::copy(emb.begin(), emb.end(), X.begin() + t * HCE + h * E);
    }

    std::vector<float> res(static_cast<size_t>(T * HCE)), post(static_cast<size_t>(T * HC)),
        comb(static_cast<size_t>(T * HC * HC)), cur(static_cast<size_t>(T * E)), sub(static_cast<size_t>(T * E));

    for (const DsLayerW& L : m.layers) {
        // ---- sotto-layer attenzione ----
        res = X;
        for (int64_t t = 0; t < T; ++t)
            hc_pre(X.data() + t * HCE, HC, E, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, eps, m.hc_eps,
                   m.sinkhorn_iters, cur.data() + t * E, post.data() + t * HC, comb.data() + t * HC * HC);
        for (int64_t t = 0; t < T; ++t) rms_norm_weighted(cur.data() + t * E, E, L.attn_norm, eps, cur.data() + t * E);
        attn_forward_layer(m.ap, L.attn, L.cl, cur.data(), T, sub.data());
        for (int64_t t = 0; t < T; ++t)
            hc_post(sub.data() + t * E, res.data() + t * HCE, post.data() + t * HC, comb.data() + t * HC * HC, HC, E,
                    X.data() + t * HCE);

        // ---- sotto-layer FFN ----
        res = X;
        for (int64_t t = 0; t < T; ++t)
            hc_pre(X.data() + t * HCE, HC, E, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, eps, m.hc_eps,
                   m.sinkhorn_iters, cur.data() + t * E, post.data() + t * HC, comb.data() + t * HC * HC);
        for (int64_t t = 0; t < T; ++t) rms_norm_weighted(cur.data() + t * E, E, L.ffn_norm, eps, cur.data() + t * E);
        for (int64_t t = 0; t < T; ++t)
            moe_forward_token(m.moe_cfg, L.moe, cur.data() + t * E, tokens[t], L.hash_layer, sub.data() + t * E, buf);
        for (int64_t t = 0; t < T; ++t)
            hc_post(sub.data() + t * E, res.data() + t * HCE, post.data() + t * HC, comb.data() + t * HC * HC, HC, E,
                    X.data() + t * HCE);
    }

    // head sull'ultimo token (A19)
    std::vector<float> hx(static_cast<size_t>(E));
    const int64_t t = T - 1;
    hc_head(X.data() + t * HCE, HC, E, m.hc_head_fn, m.hc_head_scale, m.hc_head_base, eps, m.hc_eps, hx.data());
    rms_norm_weighted(hx.data(), E, m.output_norm, eps, hx.data());
    matvec_quant(m.output.type, m.output.data, E, V, hx.data(), logits, buf);
}

}  // namespace gptoss::dsv4

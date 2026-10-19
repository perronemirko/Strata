// dsv4_bind.hpp - lega i tensori GGUF (uno o più shard) alle strutture DsModel / DsLayerW di dsv4_forward.hpp.
// Nomi e forme dei tensori: quelli verificati da deepseek4_contract.cpp sul file reale (CONTRATTO OK, 43 layer).
// Le dimensioni vengono lette dai tensori stessi. Le chiavi di metadati NON derivabili dai tensori vengono
// lette con i nomi sotto "deepseek4." e, se mancano, riportate in BindReport::missing_kv: vanno verificate
// (gguf_inspect non stampa i metadati; la lista dei nomi reali va presa dal GGUF).
//
// ASSUNZIONI di binding:
//  B1 tipo F32 per hc_*, attn_norm, ffn_norm, output_norm, attn_sinks, exp_probs_b, *_norm, ape (come da A15).
//  B2 tid2eid di tipo I32 (id GGML 26), forma [K, V].
//  B3 ratio di compressione letto da attn_compressor_ape.dims[1]; indexer presente solo se ratio == 4.
//  B4 n_layer contato da blk.i.attn_norm.weight finché esiste.
//  B5 G e o_lora_rank derivati da attn_output_a.dims: H*D = dims[0]*G, dims[1] = OL*G.
//  B6 weights_norm NON letto dai metadati (il reader scarta i bool): default true, da confermare.
//  B7 ext_factor: 1.0 se la chiave di scaling è presente, 0 altrimenti (da confermare con la config del modello).
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gptoss/dsv4_forward.hpp"
#include "gptoss/gguf_model.hpp"

namespace gptoss::dsv4 {

struct BindReport {
    std::vector<std::string> errors;      // tensori mancanti, forma o tipo errati
    std::vector<std::string> missing_kv;  // metadati non trovati (nome chiave da verificare)
    bool ok() const { return errors.empty() && missing_kv.empty(); }
};

namespace detail {

inline bool shape_ok(const GgufTensor& t, const std::vector<int64_t>& want) {
    if (t.dims.size() != want.size()) return false;
    for (size_t i = 0; i < want.size(); ++i)
        if ((int64_t) t.dims[i] != want[i]) return false;
    return true;
}

class Binder {
public:
    Binder(const GgufModel& m, BindReport& r) : m_(m), r_(r) {}

    bool has(const std::string& n) const { return m_.find(n) != nullptr; }

    int64_t dim(const std::string& n, size_t i) {
        const GgufRef* g = get(n);
        if (!g || g->t->dims.size() <= i) return 0;
        return (int64_t) g->t->dims[i];
    }

    const GgufRef* get(const std::string& n) {
        const GgufRef* g = m_.find(n);
        if (!g) r_.errors.push_back("manca tensore " + n);
        return g;
    }

    QuantMat qm(const std::string& n, int64_t in, int64_t out) {
        QuantMat q;
        const GgufRef* g = get(n);
        if (!g) return q;
        if (!shape_ok(*g->t, {in, out})) r_.errors.push_back("forma errata " + n);
        q.data = g->data();
        q.type = (int) g->t->type;
        q.in = in;
        q.out = out;
        return q;
    }

    // Esperti impilati: forma GGML [in, out, ne], out per esperto.
    QuantMat qm_exp(const std::string& n, int64_t in, int64_t out, int64_t ne) {
        QuantMat q;
        const GgufRef* g = get(n);
        if (!g) return q;
        if (!shape_ok(*g->t, {in, out, ne})) r_.errors.push_back("forma errata " + n);
        q.data = g->data();
        q.type = (int) g->t->type;
        q.in = in;
        q.out = out;
        return q;
    }

    const float* f32(const std::string& n, const std::vector<int64_t>& want) {
        const GgufRef* g = get(n);
        if (!g) return nullptr;
        if (g->t->type != 0) r_.errors.push_back("tipo non F32 " + n);
        if (!shape_ok(*g->t, want)) r_.errors.push_back("forma errata " + n);
        return reinterpret_cast<const float*>(g->data());
    }

    const int32_t* i32(const std::string& n, const std::vector<int64_t>& want) {
        const GgufRef* g = get(n);
        if (!g) return nullptr;
        if (g->t->type != 26) r_.errors.push_back("tipo non I32 " + n);
        if (!shape_ok(*g->t, want)) r_.errors.push_back("forma errata " + n);
        return reinterpret_cast<const int32_t*>(g->data());
    }

private:
    const GgufModel& m_;
    BindReport& r_;
};

inline double kvf(const GgufModel& m, const char* key, double def, BindReport& r) {
    const double v = m.files()[0]->meta_float(key, std::numeric_limits<double>::quiet_NaN());
    if (std::isnan(v)) {
        r.missing_kv.push_back(key);
        return def;
    }
    return v;
}

// Chiave opzionale: se assente usa il default di llama.cpp senza segnalarla come mancante.
inline double kvf_opt(const GgufModel& m, const char* key, double def) {
    const double v = m.files()[0]->meta_float(key, std::numeric_limits<double>::quiet_NaN());
    return std::isnan(v) ? def : v;
}

inline int64_t kvi(const GgufModel& m, const char* key, int64_t def, BindReport& r) {
    const uint64_t sentinel = std::numeric_limits<uint64_t>::max();
    const uint64_t v = m.files()[0]->meta_int(key, sentinel);
    if (v == sentinel) {
        r.missing_kv.push_back(key);
        return def;
    }
    return (int64_t) v;
}

}  // namespace detail

// Costruisce il modello legato ai tensori. Il GgufModel deve restare vivo finché il modello è in uso.
inline DsModel bind_dsv4(const GgufModel& m, BindReport& r) {
    using detail::Binder;
    Binder B(m, r);
    DsModel M;

    // dimensioni globali
    const int64_t E = B.dim("output_norm.weight", 0);
    const int64_t V = B.dim("token_embd.weight", 1);
    const int64_t H = B.dim("blk.0.attn_sinks.weight", 0);
    const int64_t D = B.dim("blk.0.attn_kv.weight", 1);
    const int64_t NQ = B.dim("blk.0.attn_q_a.weight", 1);
    const int64_t NE = B.dim("blk.0.ffn_gate_inp.weight", 1);
    const int64_t NFF = B.dim("blk.0.ffn_gate_exps.weight", 1);
    const int64_t NFFsh = B.dim("blk.0.ffn_gate_shexp.weight", 1);
    const int64_t mix_dim = B.dim("blk.0.hc_attn_fn.weight", 1);
    const int64_t hc_dim = B.dim("blk.0.hc_attn_fn.weight", 0);
    const int64_t HC = (E > 0) ? hc_dim / E : 0;
    const int64_t G = (B.dim("blk.0.attn_output_a.weight", 0) > 0) ? (H * D) / B.dim("blk.0.attn_output_a.weight", 0) : 0;
    const int64_t OL = (G > 0) ? B.dim("blk.0.attn_output_a.weight", 1) / G : 0;

    int64_t n_layer = 0;
    while (B.has("blk." + std::to_string(n_layer) + ".attn_norm.weight")) ++n_layer;  // B4

    if (HC > 0 && mix_dim != (2 + HC) * HC) r.errors.push_back("hc_mix_dim non coerente con hc");

    // hparams da metadati (chiavi da verificare, vedi missing_kv)
    M.ap.n_embd = E;
    M.ap.n_head = H;
    M.ap.head_dim = D;
    M.ap.n_lora_q = NQ;
    M.ap.n_groups = G;
    M.ap.o_lora_rank = OL;
    M.ap.n_rot = detail::kvi(m, "deepseek4.rope.dimension_count", 64, r);
    M.ap.n_swa = detail::kvi(m, "deepseek4.attention.sliding_window", 128, r);
    M.ap.rms_eps = (float) detail::kvf(m, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6, r);
    M.ap.freq_base = (float) detail::kvf(m, "deepseek4.rope.freq_base", 10000.0, r);
    M.ap.compress_rope_base = (float) detail::kvf(m, "deepseek4.attention.compress_rope_freq_base", 160000.0, r);
    const double factor = m.files()[0]->meta_float("deepseek4.rope.scaling.factor", std::numeric_limits<double>::quiet_NaN());
    if (std::isnan(factor)) {
        M.ap.freq_scale = 1.0f;
        M.ap.ext_factor = 0.0f;  // B7
        r.missing_kv.push_back("deepseek4.rope.scaling.factor");
    } else {
        M.ap.freq_scale = (float) (1.0 / factor);
        M.ap.ext_factor = 1.0f;  // B7
    }
    M.ap.beta_fast = (float) detail::kvf_opt(m, "deepseek4.rope.scaling.beta_fast", 32.0);  // default llama
    M.ap.beta_slow = (float) detail::kvf_opt(m, "deepseek4.rope.scaling.beta_slow", 1.0);  // default llama
    M.ap.n_ctx_orig = (float) detail::kvi(m, "deepseek4.rope.scaling.original_context_length", 0, r);

    M.hc = HC;
    M.hc_eps = (float) detail::kvf(m, "deepseek4.hyper_connection.epsilon", 1e-6, r);
    M.sinkhorn_iters = (int) detail::kvi(m, "deepseek4.hyper_connection.sinkhorn_iterations", 20, r);
    M.n_vocab = V;

    // tensori globali
    M.tok_embd = B.qm("token_embd.weight", E, V);
    M.output = B.qm("output.weight", E, V);
    M.output_norm = B.f32("output_norm.weight", {E});
    M.hc_head_fn = B.f32("output_hc_fn.weight", {hc_dim, HC});
    M.hc_head_base = B.f32("output_hc_base.weight", {HC});
    M.hc_head_scale = B.f32("output_hc_scale.weight", {1});

    // hparams MoE
    M.moe_cfg.n_embd = E;
    M.moe_cfg.n_expert = NE;
    M.moe_cfg.n_ff_exp = NFF;
    M.moe_cfg.n_ff_shexp = NFFsh;
    M.moe_cfg.n_expert_used = detail::kvi(m, "deepseek4.expert_used_count", 6, r);
    M.moe_cfg.weights_norm = true;  // B6
    M.moe_cfg.weights_scale = (float) detail::kvf(m, "deepseek4.expert_weights_scale", 1.0, r);

    const int64_t indexer_top_k = detail::kvi(m, "deepseek4.attention.indexer.top_k", 512, r);

    for (int64_t i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        DsLayerW L;

        // attenzione
        L.attn.wq_a = B.qm(p + "attn_q_a.weight", E, NQ);
        L.attn.wq_b = B.qm(p + "attn_q_b.weight", NQ, H * D);
        L.attn.wkv = B.qm(p + "attn_kv.weight", E, D);
        L.attn.wo_a = B.qm(p + "attn_output_a.weight", (H * D) / G, OL * G);
        L.attn.wo_b = B.qm(p + "attn_output_b.weight", G * OL, E);
        L.attn.q_a_norm = B.f32(p + "attn_q_a_norm.weight", {NQ});
        L.attn.kv_norm = B.f32(p + "attn_kv_a_norm.weight", {D});
        L.attn.sinks = B.f32(p + "attn_sinks.weight", {H});
        L.attn_norm = B.f32(p + "attn_norm.weight", {E});

        // hyper-connection
        L.hc_attn_fn = B.f32(p + "hc_attn_fn.weight", {hc_dim, mix_dim});
        L.hc_attn_base = B.f32(p + "hc_attn_base.weight", {mix_dim});
        L.hc_attn_scale = B.f32(p + "hc_attn_scale.weight", {3});
        L.hc_ffn_fn = B.f32(p + "hc_ffn_fn.weight", {hc_dim, mix_dim});
        L.hc_ffn_base = B.f32(p + "hc_ffn_base.weight", {mix_dim});
        L.hc_ffn_scale = B.f32(p + "hc_ffn_scale.weight", {3});

        // compressori (B3)
        int64_t ratio = 0;
        if (B.has(p + "attn_compressor_ape.weight")) ratio = B.dim(p + "attn_compressor_ape.weight", 1);
        L.cl.ratio = ratio;
        L.cl.indexer_top_k = indexer_top_k;
        if (ratio != 0) {
            const int64_t coff = (ratio == 4) ? 2 : 1;
            L.cl.comp.head = D;
            L.cl.comp.width = coff * D;
            L.cl.comp.wkv = B.qm(p + "attn_compressor_kv.weight", E, coff * D);
            L.cl.comp.wgate = B.qm(p + "attn_compressor_gate.weight", E, coff * D);
            L.cl.comp.ape = B.f32(p + "attn_compressor_ape.weight", {coff * D, ratio});
            L.cl.comp.norm = B.f32(p + "attn_compressor_norm.weight", {D});
            if (ratio == 4) {
                const int64_t NIH = B.dim(p + "indexer.proj.weight", 1);
                const int64_t IH = (NIH > 0) ? B.dim(p + "indexer.attn_q_b.weight", 1) / NIH : 0;
                L.cl.idx.n_idx_head = NIH;
                L.cl.idx.idx_head = IH;
                L.cl.idx.proj = B.qm(p + "indexer.proj.weight", E, NIH);
                L.cl.idx.wq_b = B.qm(p + "indexer.attn_q_b.weight", NQ, NIH * IH);
                L.cl.idx.comp.head = IH;
                L.cl.idx.comp.width = 2 * IH;
                L.cl.idx.comp.wkv = B.qm(p + "indexer_compressor_kv.weight", E, 2 * IH);
                L.cl.idx.comp.wgate = B.qm(p + "indexer_compressor_gate.weight", E, 2 * IH);
                L.cl.idx.comp.ape = B.f32(p + "indexer_compressor_ape.weight", {2 * IH, 4});
                L.cl.idx.comp.norm = B.f32(p + "indexer_compressor_norm.weight", {IH});
            }
        }

        // FFN e routing
        L.ffn_norm = B.f32(p + "ffn_norm.weight", {E});
        L.moe.gate_inp = B.qm(p + "ffn_gate_inp.weight", E, NE);
        L.hash_layer = B.has(p + "ffn_gate_tid2eid.weight");
        if (L.hash_layer) {  // B2
            const int64_t K = B.dim(p + "ffn_gate_tid2eid.weight", 0);
            L.moe.tid2eid = B.i32(p + "ffn_gate_tid2eid.weight", {K, V});
            if (M.moe_cfg.n_expert_used != K) r.errors.push_back("expert_used_count non coerente con tid2eid");
        } else {
            L.moe.exp_probs_b = B.f32(p + "exp_probs_b.bias", {NE});
        }
        L.moe.gate_exps = B.qm_exp(p + "ffn_gate_exps.weight", E, NFF, NE);
        L.moe.up_exps = B.qm_exp(p + "ffn_up_exps.weight", E, NFF, NE);
        L.moe.down_exps = B.qm_exp(p + "ffn_down_exps.weight", NFF, E, NE);
        L.moe.gate_shexp = B.qm(p + "ffn_gate_shexp.weight", E, NFFsh);
        L.moe.up_shexp = B.qm(p + "ffn_up_shexp.weight", E, NFFsh);
        L.moe.down_shexp = B.qm(p + "ffn_down_shexp.weight", NFFsh, E);

        M.layers.push_back(L);
    }

    // DsLayerW::hash_layer usato dal forward; l'indice hash è coerente con la presenza di tid2eid (B2)
    return M;
}

}  // namespace gptoss::dsv4

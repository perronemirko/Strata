// attn_dsv4_comp.hpp - attenzione DeepSeek-V4 in CPU per i layer compressi (compress_ratio 4 = CSA con indexer,
// 128 = HCA), in modalità prefill su sequenza intera, senza cache.
// Port di build_overlap_compressed_kv_from_state / build_hca_compressed_kv_from_state (righe 470-588),
// build_lid_top_k (590-676), build_top_k_mask (678-705), build_csa_lid_attention (707-767), build_hca_attention (769-822).
// Richiede attn_dsv4.hpp per il percorso raw, le proiezioni e l'output.
//
// Il percorso di decodifica incrementale (cache raw, stati dei compressori) NON è coperto: qui la compressione
// viene ricalcolata sull'intera sequenza, che è equivalente per il prefill.
//
// ASSUNZIONI aggiuntive da confermare (non deducibili dal file ricevuto, che non contiene la costruzione delle maschere
// né degli input):
//  A8  comp_pos del blocco b = b * ratio (primo token del blocco). Con questo valore il RoPE delle chiavi compresse è
//      definito solo se l'input builder usa la stessa convenzione.
//  A9  visibilità causale dei blocchi compressi: il blocco b è visibile al token t se (b+1)*ratio <= t+1 (blocco completo).
//      Vale sia per l'indexer sia per HCA.
//  A10 ape: la riga è indicizzata da (t mod ratio), cioè la posizione dentro il blocco.
//  A11 compressione overlap (CSA): per il blocco b, metà 0 dei token del blocco b-1 e metà 1 dei token del blocco b.
//      Per b = 0 la parte "precedente" è esclusa (score -inf nel sorgente).
//  A12 indexer: le rotazioni Hadamard (k_rot) di indexer_q e delle chiavi si annullano nel prodotto scalare se sono
//      ortonormali, quindi sono omesse. Il punteggio è sum_h w_h * relu(q_h · k) con w già scalato da 1/sqrt(IH*NH).
//  A13 indexer_top_k: DA LEGGERE dal GGUF (LLM_KV_ATTENTION_INDEXER_TOP_K).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "gptoss/attn_dsv4.hpp"

namespace gptoss::dsv4 {

// Pesi di un compressore (attn_comp_* oppure indexer_comp_*).
struct CompWeights {
    QuantMat wkv;               // [n_embd -> width]
    QuantMat wgate;             // [n_embd -> width]
    const float* ape = nullptr; // [ratio][width]  (A10)
    const float* norm = nullptr;// [head]
    int64_t head = 0;           // dimensione per testa del compressore
    int64_t width = 0;          // head (senza overlap) oppure 2*head (overlap CSA)
};

struct IdxWeights {
    QuantMat wq_b;              // indexer_attn_q_b [n_lora_q -> n_idx_head*idx_head]
    QuantMat proj;              // indexer_proj     [n_embd -> n_idx_head]
    CompWeights comp;           // indexer_comp_*   (head = idx_head, overlap)
    int64_t n_idx_head = 64;    // indexer_n_head
    int64_t idx_head = 128;     // indexer_head_size
};

struct CompLayer {
    int64_t ratio = 0;          // 0 raw, 4 CSA (con indexer), 128 HCA
    CompWeights comp;           // attn_comp_*  (usato se ratio != 0)
    IdxWeights idx;             // solo se ratio == 4
    int64_t indexer_top_k = 512;  // A13
};

inline AttnParams with_head_dim(const AttnParams& p, int64_t hd) {
    AttnParams q = p;
    q.head_dim = hd;
    return q;
}

// qr = rms_norm(wq_a · x): il latente q normalizzato, che alimenta sia wq_b sia indexer_attn_q_b.
inline void latent_q(const AttnParams& p, const AttnWeights& w, const float* x, float* qr_out, std::vector<float>& buf) {
    matvec_quant(w.wq_a.type, w.wq_a.data, p.n_embd, p.n_lora_q, x, qr_out, buf);
    rms_norm_weighted(qr_out, p.n_lora_q, w.q_a_norm, p.rms_eps, qr_out);
}

// Compressione a blocchi di tutta la sequenza. X: [T][n_embd] -> out: [T/ratio][head] (RoPE applicato, A8).
inline void compress_sequence(const AttnParams& p, const CompWeights& c, int64_t ratio, int64_t T, const float* X,
                              float* out, std::vector<float>& buf) {
    const bool overlap = c.width == 2 * c.head;
    const int64_t H = c.head, W = c.width, E = p.n_embd;
    const int64_t nb = T / ratio;
    const RopeCfg rc = rope_cfg_for_layer(p, true);
    const AttnParams pc = with_head_dim(p, H);

    std::vector<float> kvt(static_cast<size_t>(T * W)), sct(static_cast<size_t>(T * W));
    for (int64_t t = 0; t < T; ++t) {
        matvec_quant(c.wkv.type, c.wkv.data, E, W, X + t * E, kvt.data() + t * W, buf);
        matvec_quant(c.wgate.type, c.wgate.data, E, W, X + t * E, sct.data() + t * W, buf);
        const float* a = c.ape + (t % ratio) * W;  // A10
        for (int64_t j = 0; j < W; ++j) sct[t * W + j] += a[j];
    }

    std::vector<float> sc, vl;
    for (int64_t b = 0; b < nb; ++b) {
        float* cb = out + b * H;
        for (int64_t d = 0; d < H; ++d) {
            sc.clear();
            vl.clear();
            if (!overlap) {
                for (int64_t i = 0; i < ratio; ++i) {
                    const int64_t t = b * ratio + i;
                    sc.push_back(sct[t * W + d]);
                    vl.push_back(kvt[t * W + d]);
                }
            } else {
                if (b > 0) {  // A11: metà 0 del blocco precedente
                    for (int64_t i = 0; i < ratio; ++i) {
                        const int64_t t = (b - 1) * ratio + i;
                        sc.push_back(sct[t * W + d]);
                        vl.push_back(kvt[t * W + d]);
                    }
                }
                for (int64_t i = 0; i < ratio; ++i) {  // metà 1 del blocco corrente
                    const int64_t t = b * ratio + i;
                    sc.push_back(sct[t * W + H + d]);
                    vl.push_back(kvt[t * W + H + d]);
                }
            }
            float mx = -std::numeric_limits<float>::infinity();
            for (float s : sc) mx = std::max(mx, s);
            float den = 0.0f, num = 0.0f;
            for (size_t k = 0; k < sc.size(); ++k) {
                const float wgt = std::exp(sc[k] - mx);
                den += wgt;
                num += wgt * vl[k];
            }
            cb[d] = num / den;
        }
        rms_norm_weighted(cb, H, c.norm, p.rms_eps, cb);
        rope_apply(cb, pc, rc, b * ratio, false);  // A8
    }
}

// Visibilità causale di un blocco compresso (A9).
inline bool comp_block_visible(int64_t b, int64_t t, int64_t ratio) { return (b + 1) * ratio <= t + 1; }

// Layer compresso completo su T token (prefill). X: [T][n_embd] -> Y: [T][n_embd].
inline void attn_forward_layer(const AttnParams& p, const AttnWeights& w, const CompLayer& L, const float* X, int64_t T,
                               float* Y) {
    const int64_t H = p.n_head, D = p.head_dim, E = p.n_embd, NQ = p.n_lora_q;
    const bool compressed = L.ratio != 0;
    std::vector<float> buf;
    std::vector<float> qr(static_cast<size_t>(T * NQ)), q(static_cast<size_t>(T * H * D));
    std::vector<float> kv(static_cast<size_t>(T * D)), o(static_cast<size_t>(T * H * D));

    for (int64_t t = 0; t < T; ++t) {
        latent_q(p, w, X + t * E, qr.data() + t * NQ, buf);
        attn_project_qkv(p, w, X + t * E, t, compressed, q.data() + t * H * D, kv.data() + t * D, buf);
    }

    if (!compressed) {
        attn_core_raw(p, q.data(), kv.data(), T, w.sinks, o.data());
    } else {
        const int64_t nb = T / L.ratio;
        std::vector<float> comp(static_cast<size_t>(nb * D));
        compress_sequence(p, L.comp, L.ratio, T, X, comp.data(), buf);

        std::vector<std::vector<int64_t>> sel(static_cast<size_t>(T));
        if (L.ratio == 4) {  // CSA: indexer + top-k
            const AttnParams ip = with_head_dim(p, L.idx.idx_head);
            const int64_t NI = L.idx.n_idx_head, IH = L.idx.idx_head;
            std::vector<float> idk(static_cast<size_t>(nb * IH));
            compress_sequence(ip, L.idx.comp, 4, T, X, idk.data(), buf);
            const RopeCfg rci = rope_cfg_for_layer(p, true);
            const float wscale = 1.0f / std::sqrt((float) (IH * NI));
            std::vector<float> qi(static_cast<size_t>(NI * IH)), wt(static_cast<size_t>(NI));
            for (int64_t t = 0; t < T; ++t) {
                matvec_quant(L.idx.wq_b.type, L.idx.wq_b.data, NQ, NI * IH, qr.data() + t * NQ, qi.data(), buf);
                for (int64_t h = 0; h < NI; ++h) rope_apply(qi.data() + h * IH, ip, rci, t, false);
                matvec_quant(L.idx.proj.type, L.idx.proj.data, E, NI, X + t * E, wt.data(), buf);
                for (int64_t h = 0; h < NI; ++h) wt[h] *= wscale;

                std::vector<std::pair<float, int64_t>> cand;
                for (int64_t b = 0; b < nb; ++b) {
                    if (!comp_block_visible(b, t, 4)) continue;
                    float s = 0.0f;
                    for (int64_t h = 0; h < NI; ++h) {
                        float dot = 0.0f;
                        for (int64_t d = 0; d < IH; ++d) dot += qi[h * IH + d] * idk[b * IH + d];
                        if (dot > 0.0f) s += wt[h] * dot;  // relu
                    }
                    cand.emplace_back(s, b);
                }
                const int64_t K = std::min<int64_t>((int64_t) cand.size(), L.indexer_top_k);
                std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                                  [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                                      return a.first > b.first;
                                  });
                for (int64_t k = 0; k < K; ++k) sel[t].push_back(cand[k].second);
            }
        } else {  // HCA (ratio 128): tutti i blocchi visibili
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t b = 0; b < nb; ++b) {
                    if (comp_block_visible(b, t, L.ratio)) sel[t].push_back(b);
                }
            }
        }

        // Attenzione: chiavi = finestra raw SWA + blocchi compressi selezionati. v = k. Sink come nel percorso raw.
        const float scale = 1.0f / std::sqrt((float) D);
        std::vector<const float*> rows;
        std::vector<float> sc;
        for (int64_t t = 0; t < T; ++t) {
            rows.clear();
            const int64_t j0 = std::max<int64_t>(0, t - p.n_swa + 1);
            for (int64_t j = j0; j <= t; ++j) rows.push_back(kv.data() + j * D);
            for (int64_t b : sel[t]) rows.push_back(comp.data() + b * D);
            for (int64_t h = 0; h < H; ++h) {
                const float* qh = q.data() + (t * H + h) * D;
                sc.clear();
                float mx = w.sinks[h];
                for (const float* kr : rows) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) dot += qh[d] * kr[d];
                    dot *= scale;
                    sc.push_back(dot);
                    mx = std::max(mx, dot);
                }
                float den = std::exp(w.sinks[h] - mx);
                float* oh = o.data() + (t * H + h) * D;
                for (int64_t d = 0; d < D; ++d) oh[d] = 0.0f;
                for (size_t k = 0; k < rows.size(); ++k) {
                    const float wgt = std::exp(sc[k] - mx);
                    den += wgt;
                    for (int64_t d = 0; d < D; ++d) oh[d] += wgt * rows[k][d];
                }
                const float inv = 1.0f / den;
                for (int64_t d = 0; d < D; ++d) oh[d] *= inv;
            }
        }
    }

    for (int64_t t = 0; t < T; ++t) {
        attn_project_out(p, w, o.data() + t * H * D, t, compressed, Y + t * E, buf);
    }
}

}  // namespace gptoss::dsv4

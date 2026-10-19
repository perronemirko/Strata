// dsv4_decode.hpp - decodifica incrementale di DeepSeek-V4 in CPU con cache per layer.
// Un token alla volta attraversa tutti i layer aggiornando lo stato:
//  - rawk: chiavi raw già ruotate, una per token (finestra SWA letta da qui);
//  - comp: per-token kv e score del compressore (W), blocchi compressi completati (H ciascuno);
//  - idx:  come comp, per l'indexer (solo ratio 4).
// Un token t vede solo il passato (causale), quindi lo stato incrementale è equivalente al ricalcolo su tutta
// la sequenza (dsv4_forward.hpp). Questa equivalenza è un'argomentazione, non un test: non è verificata.
//
// Assunzioni ereditate: A1-A19 di attn_dsv4*.hpp e dsv4_forward.hpp. Specifiche di questo file:
//  C1 un blocco compresso b è disponibile al token t quando (b+1)*ratio <= t+1, cioè dopo il suo ultimo token.
//  C2 la cache non ha limite: cresce con T. Nessuna potatura della finestra SWA nella cache raw (si legge solo la finestra).
//  C3 prefill = step ripetuti, uno per token del prompt; i logit servono solo per l'ultimo (want_logits).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "gptoss/attn_dsv4_comp.hpp"
#include "gptoss/dsv4_forward.hpp"

namespace gptoss::dsv4 {

struct CompState {
    int64_t ratio = 0;          // 0 = non attivo
    int64_t W = 0;              // larghezza del compressore per token (2*H con overlap)
    int64_t H = 0;              // head del compressore
    std::vector<float> kvt;     // [t][W]
    std::vector<float> sct;     // [t][W], score già con ape
    std::vector<float> blocks;  // [b][H], RoPE già applicato
    int64_t n_blocks() const { return H ? (int64_t) blocks.size() / H : 0; }
};

struct LayerCache {
    std::vector<float> rawk;    // [t][D], RoPE già applicato
    CompState comp;
    CompState idx;
};

// Proiezioni per-token del compressore: kv e score (+ape alla riga t % ratio).
inline void comp_push_token(const AttnParams& p, const CompWeights& c, CompState& st, const float* x, int64_t t,
                            std::vector<float>& buf) {
    const int64_t W = c.width, E = p.n_embd;
    std::vector<float> kv(static_cast<size_t>(W)), sc(static_cast<size_t>(W));
    matvec_quant(c.wkv.type, c.wkv.data, E, W, x, kv.data(), buf);
    matvec_quant(c.wgate.type, c.wgate.data, E, W, x, sc.data(), buf);
    const float* a = c.ape + (t % st.ratio) * W;
    for (int64_t j = 0; j < W; ++j) sc[j] += a[j];
    st.kvt.insert(st.kvt.end(), kv.begin(), kv.end());
    st.sct.insert(st.sct.end(), sc.begin(), sc.end());
}

// Blocco b, dal solo stato già disponibile (token del blocco b e, con overlap, del blocco b-1). Stessa regola
// di compress_sequence (A8, A11). Restituisce H valori con RoPE applicato.
inline std::vector<float> compress_block_inc(const AttnParams& p, const CompWeights& c, const CompState& st, int64_t b) {
    const bool overlap = c.width == 2 * c.head;
    const int64_t H = c.head, W = c.width, ratio = st.ratio;
    const RopeCfg rc = rope_cfg_for_layer(p, true);
    const AttnParams pc = with_head_dim(p, H);
    std::vector<float> comp(static_cast<size_t>(H));
    std::vector<float> sc, vl;
    for (int64_t d = 0; d < H; ++d) {
        sc.clear();
        vl.clear();
        if (!overlap) {
            for (int64_t i = 0; i < ratio; ++i) {
                const int64_t t = b * ratio + i;
                sc.push_back(st.sct[t * W + d]);
                vl.push_back(st.kvt[t * W + d]);
            }
        } else {
            if (b > 0) {
                for (int64_t i = 0; i < ratio; ++i) {
                    const int64_t t = (b - 1) * ratio + i;
                    sc.push_back(st.sct[t * W + d]);
                    vl.push_back(st.kvt[t * W + d]);
                }
            }
            for (int64_t i = 0; i < ratio; ++i) {
                const int64_t t = b * ratio + i;
                sc.push_back(st.sct[t * W + H + d]);
                vl.push_back(st.kvt[t * W + H + d]);
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
        comp[d] = num / den;
    }
    rms_norm_weighted(comp.data(), H, c.norm, p.rms_eps, comp.data());
    rope_apply(comp.data(), pc, rc, b * ratio, false);
    return comp;
}

class DsDecoder {
public:
    explicit DsDecoder(const DsModel& m) : M_(m), cache_(m.layers.size()) {
        for (size_t i = 0; i < cache_.size(); ++i) {
            const DsLayerW& L = m.layers[i];
            LayerCache& c = cache_[i];
            if (L.cl.ratio != 0) {
                c.comp.ratio = L.cl.ratio;
                c.comp.W = L.cl.comp.width;
                c.comp.H = L.cl.comp.head;
            }
            if (L.cl.ratio == 4) {
                c.idx.ratio = 4;
                c.idx.W = L.cl.idx.comp.width;
                c.idx.H = L.cl.idx.comp.head;
            }
        }
    }

    int64_t pos() const { return t_; }
    bool debug = false;  // stampa norme per layer (diagnostica)

    // Elabora un token alla posizione pos(). Se want_logits, scrive V logit in logits.
    void step(int32_t tok, bool want_logits, float* logits) {
        const AttnParams& p = M_.ap;
        const int64_t E = p.n_embd, HC = M_.hc, HCE = HC * E, H = p.n_head, D = p.head_dim, NQ = p.n_lora_q;
        const float eps = p.rms_eps;
        std::vector<float> buf;

        std::vector<float> X(static_cast<size_t>(HCE));
        const size_t erb = dequant_row_bytes(M_.tok_embd.type, E);
        std::vector<float> emb(static_cast<size_t>(E));
        dequantize(M_.tok_embd.type, M_.tok_embd.data + static_cast<size_t>(tok) * erb, emb.data(), E);
        for (int64_t h = 0; h < HC; ++h) std::copy(emb.begin(), emb.end(), X.begin() + h * E);
        if (debug) dbg_vec("embd", emb.data(), E);

        for (size_t il = 0; il < M_.layers.size(); ++il) {
            const DsLayerW& L = M_.layers[il];
            LayerCache& C = cache_[il];
            const bool compressed = L.cl.ratio != 0;
            const int64_t ratio = L.cl.ratio;

            // ---- attenzione ----
            std::vector<float> res = X;
            std::vector<float> cur(static_cast<size_t>(E)), post(static_cast<size_t>(HC)), comb(static_cast<size_t>(HC * HC));
            hc_pre(X.data(), HC, E, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, eps, M_.hc_eps, M_.sinkhorn_iters,
                   cur.data(), post.data(), comb.data());
            rms_norm_weighted(cur.data(), E, L.attn_norm, eps, cur.data());
            if (debug && il == 0) {
                dbg_vec("attn_norm-0", cur.data(), E);
                std::vector<float> qa(static_cast<size_t>(NQ)), qn(static_cast<size_t>(NQ)), qp(static_cast<size_t>(H * D)),
                    kvp(static_cast<size_t>(D));
                if (FILE* fd = std::fopen("attn_norm_l0.bin", "wb")) {  // dump per l'oracolo ggml
                    std::fwrite(cur.data(), sizeof(float), static_cast<size_t>(E), fd);
                    std::fclose(fd);
                }
                matvec_quant(L.attn.wq_a.type, L.attn.wq_a.data, E, NQ, cur.data(), qa.data(), buf);
                dbg_vec("qr-0", qa.data(), NQ);
                qn = qa;
                rms_norm_weighted(qn.data(), NQ, L.attn.q_a_norm, eps, qn.data());
                dbg_vec("qr_norm-0", qn.data(), NQ);
                matvec_quant(L.attn.wq_b.type, L.attn.wq_b.data, NQ, H * D, qn.data(), qp.data(), buf);
                dbg_vec("q-0 pre-head-norm", qp.data(), H * D);
                matvec_quant(L.attn.wkv.type, L.attn.wkv.data, E, D, cur.data(), kvp.data(), buf);
                dbg_vec("node_23 kv pre-norm", kvp.data(), D);
            }

            std::vector<float> q(static_cast<size_t>(H * D)), kv(static_cast<size_t>(D));
            attn_project_qkv(p, L.attn, cur.data(), t_, compressed, q.data(), kv.data(), buf);
            // llama.cpp salva la cache KV in f16 per default: arrotondiamo kv a fp16 prima di metterlo in cache.
            // DSV4_KVF16=0 la disattiva (solo per A/B).
            static const bool kvf16 = !std::getenv("DSV4_KVF16") || std::atoi(std::getenv("DSV4_KVF16")) != 0;
            if (kvf16) {
                for (float& v : kv) v = round_fp16(v);
            }
            C.rawk.insert(C.rawk.end(), kv.begin(), kv.end());

            std::vector<int64_t> sel;  // blocchi compressi visibili e selezionati
            if (compressed) {
                comp_push_token(p, L.cl.comp, C.comp, cur.data(), t_, buf);
                if ((t_ + 1) % ratio == 0) {  // C1
                    const int64_t b = (t_ + 1) / ratio - 1;
                    std::vector<float> blk = compress_block_inc(p, L.cl.comp, C.comp, b);
                    C.comp.blocks.insert(C.comp.blocks.end(), blk.begin(), blk.end());
                }
                if (ratio == 4) {  // CSA: indexer + top-k
                    comp_push_token(p, L.cl.idx.comp, C.idx, cur.data(), t_, buf);
                    if ((t_ + 1) % ratio == 0) {
                        const int64_t b = (t_ + 1) / ratio - 1;
                        std::vector<float> blk = compress_block_inc(with_head_dim(p, L.cl.idx.idx_head), L.cl.idx.comp, C.idx, b);
                        C.idx.blocks.insert(C.idx.blocks.end(), blk.begin(), blk.end());
                    }
                    const AttnParams ip = with_head_dim(p, L.cl.idx.idx_head);
                    const int64_t NI = L.cl.idx.n_idx_head, IH = L.cl.idx.idx_head;
                    std::vector<float> qr(static_cast<size_t>(NQ));
                    latent_q(p, L.attn, cur.data(), qr.data(), buf);
                    std::vector<float> qi(static_cast<size_t>(NI * IH)), wt(static_cast<size_t>(NI));
                    matvec_quant(L.cl.idx.wq_b.type, L.cl.idx.wq_b.data, NQ, NI * IH, qr.data(), qi.data(), buf);
                    const RopeCfg rci = rope_cfg_for_layer(p, true);
                    for (int64_t h = 0; h < NI; ++h) rope_apply(qi.data() + h * IH, ip, rci, t_, false);
                    matvec_quant(L.cl.idx.proj.type, L.cl.idx.proj.data, E, NI, cur.data(), wt.data(), buf);
                    const float wscale = 1.0f / std::sqrt((float) (IH * NI));
                    for (int64_t h = 0; h < NI; ++h) wt[h] *= wscale;

                    const int64_t nb = C.idx.n_blocks();
                    std::vector<std::pair<float, int64_t>> cand;
                    for (int64_t b = 0; b < nb; ++b) {
                        float s = 0.0f;
                        for (int64_t h = 0; h < NI; ++h) {
                            float dot = 0.0f;
                            for (int64_t d = 0; d < IH; ++d) dot += qi[h * IH + d] * C.idx.blocks[b * IH + d];
                            if (dot > 0.0f) s += wt[h] * dot;
                        }
                        cand.emplace_back(s, b);
                    }
                    const int64_t K = std::min<int64_t>((int64_t) cand.size(), L.cl.indexer_top_k);
                    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                                      [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                                          return a.first > b.first;
                                      });
                    for (int64_t k = 0; k < K; ++k) sel.push_back(cand[k].second);
                } else {  // HCA: tutti i blocchi completati
                    for (int64_t b = 0; b < C.comp.n_blocks(); ++b) sel.push_back(b);
                }
            }

            // attenzione: finestra raw SWA + blocchi selezionati, sink, v = k
            std::vector<const float*> rows;
            const int64_t j0 = std::max<int64_t>(0, t_ - p.n_swa + 1);
            for (int64_t j = j0; j <= t_; ++j) rows.push_back(C.rawk.data() + j * D);
            for (int64_t b : sel) rows.push_back(C.comp.blocks.data() + b * D);

            const float scale = 1.0f / std::sqrt((float) D);
            // Diagnostica: moltiplicatore della scala degli score (default 1). Solo per A/B della scala.
            static const float kscale_mult = std::getenv("DSV4_KSCALE") ? (float) std::atof(std::getenv("DSV4_KSCALE")) : 1.0f;
            std::vector<float> o(static_cast<size_t>(H * D)), sc;
            for (int64_t h = 0; h < H; ++h) {
                const float* qh = q.data() + h * D;
                sc.clear();
                const int sink_mode = std::getenv("DSV4_SINK_MODE") ? std::atoi(std::getenv("DSV4_SINK_MODE")) : 0;
                float mx = sink_mode == 1 ? -std::numeric_limits<float>::infinity() : L.attn.sinks[h];
                for (const float* kr : rows) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) dot += qh[d] * kr[d];
                    dot *= scale * kscale_mult;
                    sc.push_back(dot);
                    mx = std::max(mx, dot);
                }
                if (debug && il == 0 && h == 0) {
                    std::printf("      rope: freq_base=%g n_rot=%lld neox=%d head_dim=%lld\n", (double) p.freq_base,
                                (long long) p.n_rot, (int) p.neox, (long long) p.head_dim);
                    std::printf("      head0 sink=%.6g scores(scalati):", L.attn.sinks[h]);
                    for (float s : sc) std::printf(" %.6g", s);
                    std::printf("\n");
                }
                float den = sink_mode == 1 ? 0.0f : std::exp(L.attn.sinks[h] - mx);  // sink_mode 1: senza termine del sink (A/B)
                float* oh = o.data() + h * D;
                for (int64_t d = 0; d < D; ++d) oh[d] = 0.0f;
                for (size_t k = 0; k < rows.size(); ++k) {
                    const float wgt = std::exp(sc[k] - mx);
                    den += wgt;
                    for (int64_t d = 0; d < D; ++d) oh[d] += wgt * rows[k][d];
                }
                const float inv = 1.0f / den;
                for (int64_t d = 0; d < D; ++d) oh[d] *= inv;
            }

            if (debug && il == 0) {
                dbg_vec("q-0", q.data(), H * D);
                dbg_vec("kv-0", kv.data(), D);
                dbg_vec("attn_raw-0 (o)", o.data(), H * D);
            }
            std::vector<float> y(static_cast<size_t>(E));
            attn_project_out(p, L.attn, o.data(), t_, compressed, y.data(), buf);
            hc_post(y.data(), res.data(), post.data(), comb.data(), HC, E, X.data());
            if (debug) dbg_line(il, "attn", X, y);
            if (debug && il == 0) {
                dbg_vec("attn_out-0", y.data(), E);
                // wo_a per gruppi (come attn_project_out, a pos 0 il derope è identità) + dump per l'oracolo di wo_b
                const int64_t G = p.n_groups, og = (H / G) * D;
                std::vector<float> oa(static_cast<size_t>(p.o_lora_rank * G));
                const size_t rba = dequant_row_bytes(L.attn.wo_a.type, og);
                for (int64_t g = 0; g < G; ++g) {
                    matvec_quant(L.attn.wo_a.type, L.attn.wo_a.data + static_cast<size_t>(g * p.o_lora_rank) * rba, og,
                                 p.o_lora_rank, o.data() + g * og, oa.data() + g * p.o_lora_rank, buf);
                }
                dbg_vec("attn_wo_a-0 (gruppo 0)", oa.data(), p.o_lora_rank);
                if (FILE* fd = std::fopen("oa_l0.bin", "wb")) {
                    std::fwrite(oa.data(), sizeof(float), oa.size(), fd);
                    std::fclose(fd);
                }
            }

            // ---- FFN ----
            res = X;
            hc_pre(X.data(), HC, E, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, eps, M_.hc_eps, M_.sinkhorn_iters,
                   cur.data(), post.data(), comb.data());
            rms_norm_weighted(cur.data(), E, L.ffn_norm, eps, cur.data());
            if (debug && il == 0) dbg_vec("ffn_norm-0", cur.data(), E);
            std::vector<float> f(static_cast<size_t>(E));
            moe_debug() = debug && il == 0;
            moe_forward_token(M_.moe_cfg, L.moe, cur.data(), tok, L.hash_layer, f.data(), buf);
            moe_debug() = false;
            if (debug && il == 0) dbg_vec("ffn_moe_out-0", f.data(), E);
            hc_post(f.data(), res.data(), post.data(), comb.data(), HC, E, X.data());
            if (debug) dbg_line(il, "ffn ", X, f);
        }

        ++t_;
        if (want_logits) {
            std::vector<float> buf2;
            std::vector<float> hx(static_cast<size_t>(E));
            hc_head(X.data(), HC, E, M_.hc_head_fn, M_.hc_head_scale, M_.hc_head_base, eps, M_.hc_eps, hx.data());
            rms_norm_weighted(hx.data(), E, M_.output_norm, eps, hx.data());
            matvec_quant(M_.output.type, M_.output.data, E, M_.n_vocab, hx.data(), logits, buf2);
        }
    }

private:
    static void dbg_line(size_t il, const char* tag, const std::vector<float>& X, const std::vector<float>& sub) {
        double sx = 0, sy = 0;
        float mx = 0;
        int bad = 0;
        for (float v : X) { if (!std::isfinite(v)) ++bad; sx += (double) v * v; mx = std::max(mx, std::fabs(v)); }
        for (float v : sub) { if (!std::isfinite(v)) ++bad; sy += (double) v * v; }
        std::printf("  L%02zu %s |X|=%.3e max=%.3e |sub|=%.3e nonfinite=%d\n", il, tag, std::sqrt(sx), mx, std::sqrt(sy), bad);
        std::printf("      %s[0:4] = %.6g %.6g %.6g %.6g\n", tag, sub[0], sub[1], sub[2], sub[3]);
    }

    const DsModel& M_;
    std::vector<LayerCache> cache_;
    int64_t t_ = 0;
};

}  // namespace gptoss::dsv4

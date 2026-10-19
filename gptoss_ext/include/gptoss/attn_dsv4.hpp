// attn_dsv4.hpp - attenzione DeepSeek-V4 in CPU, percorso RAW (layer con compress_ratio == 0).
// Port di build_attention_impl (righe 878-1220) e build_raw_attention (righe 824-858) di deepseek4.cpp.
// Copre: proiezione q (q-LoRA), proiezione kv, RoPE parziale, attenzione con sink e finestra SWA, output a gruppi.
// NON coperti (blocco successivo): compressori CSA/HCA, indexer lightning e top-k, stato delle cache.
//
// ASSUNZIONI da confermare (il sorgente non le dichiara direttamente nel file ricevuto):
//  A1 rope_type: NEOX, cioè coppie (i, i + n_rot/2) dentro la porzione di rope. Parametro: AttnParams::neox
//     (oggi il ramo non-NEOX non è implementato: lo stop esplicito evita di indovinarlo in silenzio).
//  A2 attn_sinks: un logit per testa nel softmax, il sink non contribuisce al valore (build_attn_mha non è nel file).
//  A3 finestra SWA: chiavi j con t - n_swa < j <= t.
//  A4 Hadamard k_rot del percorso raw: omesso. Se la matrice è ortonormale, q·k e l'output si annullano.
//  A5 wo_a con layout GGML [o_group_dim, o_lora_rank, n_groups]: riga r + g*o_lora_rank, colonne contigue.
//  A6 attn_sinks tipo F32.
//  A7 rope: mscale = attn_factor, come calcolato da dsv4_rope_attn_factor nel sorgente.
//
// Layout: head-major. Testa h del token t inizia a (t*n_head + h)*head_dim. Il RoPE agisce sugli ultimi n_rot
// valori di ogni testa (offset nope = head_dim - n_rot), come ggml_rope_set_offset.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gptoss/cpu_matvec.hpp"
#include "gptoss/dequant_registry.hpp"
#include "gptoss/hc_ops.hpp"  // gptoss::rms_norm_plain

namespace gptoss::dsv4 {

struct AttnParams {
    int64_t n_embd = 4096;
    int64_t n_head = 64;
    int64_t head_dim = 512;        // n_embd_head_k (= n_embd_head_v)
    int64_t n_rot = 64;            // hparams.n_rot()          DA LEGGERE dal GGUF
    int64_t n_lora_q = 1024;       // LLM_KV_ATTENTION_Q_LORA_RANK
    int64_t n_groups = 8;          // dsv4_o_group_count
    int64_t o_lora_rank = 1024;    // dsv4_o_lora_rank
    int64_t n_swa = 128;           // n_swa                    DA LEGGERE dal GGUF
    float rms_eps = 1e-6f;         // f_norm_rms_eps           DA LEGGERE dal GGUF
    // RoPE globale (layer non compressi usano freq_base, compressi usano compress_rope_base)
    float freq_base = 10000.0f;    // DA LEGGERE
    float compress_rope_base = 160000.0f;  // dsv4_compress_rope_base  DA LEGGERE
    float freq_scale = 1.0f;       // DA LEGGERE (scala YaRN)
    float ext_factor = 0.0f;       // DA LEGGERE (YaRN)
    float beta_fast = 32.0f;       // DA LEGGERE (YaRN)
    float beta_slow = 1.0f;        // DA LEGGERE (YaRN)
    float n_ctx_orig = 0.0f;       // n_ctx_orig               DA LEGGERE
    bool neox = true;              // A1
};

// Pesi quantizzati di un matrice in GGML: in = ne[0], out = ne[1], righe contigue.
struct QuantMat {
    const uint8_t* data = nullptr;
    int type = 0;
    int64_t in = 0;
    int64_t out = 0;
};

struct AttnWeights {
    QuantMat wq_a;                 // [n_embd -> n_lora_q]
    QuantMat wq_b;                 // [n_lora_q -> n_head*head_dim]
    QuantMat wkv;                  // [n_embd -> head_dim]
    QuantMat wo_a;                 // [o_group_dim -> o_lora_rank*n_groups] (A5)
    QuantMat wo_b;                 // [o_lora_rank*n_groups -> n_embd]
    const float* q_a_norm = nullptr;   // attn_q_a_norm, peso di dimensione n_lora_q
    const float* kv_norm = nullptr;    // attn_kv_norm,  peso di dimensione head_dim
    const float* sinks = nullptr;      // attn_sinks,    dimensione n_head (A6)
};

// Configurazione RoPE per layer (come nelle righe 902-909 del sorgente).
struct RopeCfg {
    float base = 10000.0f;
    float scale = 1.0f;
    float ext = 0.0f;
    float attn = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    float n_ctx_orig = 0.0f;
};

inline float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) return 1.0f;
    return 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
}

inline RopeCfg rope_cfg_for_layer(const AttnParams& p, bool compressed) {
    RopeCfg c;
    if (compressed) {
        c.base = p.compress_rope_base;
        c.scale = p.freq_scale;
        c.ext = p.ext_factor;
        c.attn = dsv4_rope_attn_factor(c.scale, c.ext);
        c.beta_fast = p.beta_fast;
        c.beta_slow = p.beta_slow;
        c.n_ctx_orig = p.n_ctx_orig;
    } else {
        c.base = p.freq_base;
        c.scale = 1.0f;
        c.ext = 0.0f;
        c.attn = 1.0f;
        c.beta_fast = 0.0f;
        c.beta_slow = 0.0f;
        c.n_ctx_orig = 0.0f;
    }
    return c;
}

// RMS norm con peso per elemento. y può coincidere con x.
inline void rms_norm_weighted(const float* x, int64_t n, const float* w, float eps, float* y) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) x[i] * x[i];
    const float inv = 1.0f / std::sqrt((float) (s / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

// Rampa YaRN (ggml_rope_yarn_ramp). i = indice della coppia (i0/2 in ggml).
inline float yarn_ramp(float low, float high, int64_t i) {
    const float y = ((float) i - low) / std::fmax(0.001f, high - low);
    return 1.0f - std::fmin(1.0f, std::fmax(0.0f, y));
}

inline float yarn_corr_dim(float n_dims, float n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log(n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * std::log(base));
}

// RoPE parziale su v (lunghezza head_dim): ruota gli ultimi n_rot valori. inverse = ggml_rope_ext_back.
inline void rope_apply(float* v, const AttnParams& p, const RopeCfg& c, int64_t pos, bool inverse) {
    const int64_t n_rot = p.n_rot;
    const int64_t off = p.head_dim - n_rot;
    const int64_t half = n_rot / 2;

    if (!p.neox) return;  // A1: il ramo non-NEOX non è implementato

    float corr_lo = 0.0f, corr_hi = 0.0f;
    const bool yarn = c.ext != 0.0f;
    if (yarn) {
        corr_lo = std::fmax(0.0f, std::floor(yarn_corr_dim((float) n_rot, c.n_ctx_orig, c.beta_fast, c.base)));
        corr_hi = std::fmin((float) (n_rot - 1), std::ceil(yarn_corr_dim((float) n_rot, c.n_ctx_orig, c.beta_slow, c.base)));
    }

    for (int64_t i = 0; i < half; ++i) {
        const float theta_extrap = (float) pos * std::pow(c.base, -2.0f * (float) i / (float) n_rot);
        float theta = c.scale * theta_extrap;
        if (yarn) {
            const float mix = yarn_ramp(corr_lo, corr_hi, i) * c.ext;
            theta = theta * (1.0f - mix) + theta_extrap * mix;
        }
        const float cs = std::cos(theta) * c.attn;
        float sn = std::sin(theta) * c.attn;
        if (inverse) sn = -sn;

        float& a = v[off + i];
        float& b = v[off + i + half];
        const float na = a * cs - b * sn;
        const float nb = a * sn + b * cs;
        a = na;
        b = nb;
    }
}

// Attenzione raw con sink e finestra SWA. q: [T][n_head][head_dim], k: [T][head_dim] (k == v, MQA a una testa).
// out: [T][n_head][head_dim]. Scala = 1/sqrt(head_dim) (come nel sorgente).
inline void attn_core_raw(const AttnParams& p, const float* q, const float* k, int64_t T, const float* sinks, float* out) {
    const int64_t H = p.n_head, D = p.head_dim;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> s;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t j0 = std::max<int64_t>(0, t - p.n_swa + 1);
        for (int64_t h = 0; h < H; ++h) {
            const float* qh = q + (t * H + h) * D;
            s.clear();
            float mx = sinks[h];
            for (int64_t j = j0; j <= t; ++j) {
                const float* kj = k + j * D;
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) dot += qh[d] * kj[d];
                dot *= scale;
                s.push_back(dot);
                if (dot > mx) mx = dot;
            }
            float denom = std::exp(sinks[h] - mx);
            float* oh = out + (t * H + h) * D;
            for (int64_t d = 0; d < D; ++d) oh[d] = 0.0f;
            for (size_t idx = 0; idx < s.size(); ++idx) {
                const int64_t j = j0 + (int64_t) idx;
                const float w = std::exp(s[idx] - mx);
                denom += w;
                const float* vj = k + j * D;  // v == k
                for (int64_t d = 0; d < D; ++d) oh[d] += w * vj[d];
            }
            const float inv = 1.0f / denom;
            for (int64_t d = 0; d < D; ++d) oh[d] *= inv;
        }
    }
}

// Proiezioni q e kv per un token. x: [n_embd]. q: [n_head*head_dim], kv: [head_dim].
inline void attn_project_qkv(const AttnParams& p, const AttnWeights& w, const float* x, int64_t pos, bool compressed,
                             float* q, float* kv, std::vector<float>& buf) {
    const RopeCfg rc = rope_cfg_for_layer(p, compressed);
    const int64_t H = p.n_head, D = p.head_dim;

    std::vector<float> qr(static_cast<size_t>(p.n_lora_q));
    matvec_quant(w.wq_a.type, w.wq_a.data, p.n_embd, p.n_lora_q, x, qr.data(), buf);
    rms_norm_weighted(qr.data(), p.n_lora_q, w.q_a_norm, p.rms_eps, qr.data());

    matvec_quant(w.wq_b.type, w.wq_b.data, p.n_lora_q, H * D, qr.data(), q, buf);
    for (int64_t h = 0; h < H; ++h) {
        float* qh = q + h * D;
        gptoss::rms_norm_plain(qh, D, p.rms_eps, qh);  // norma per testa, senza peso
        rope_apply(qh, p, rc, pos, false);
    }

    matvec_quant(w.wkv.type, w.wkv.data, p.n_embd, D, x, kv, buf);
    rms_norm_weighted(kv, D, w.kv_norm, p.rms_eps, kv);
    rope_apply(kv, p, rc, pos, false);
}

// Output di un token: derope inverso, wo_a per gruppi, wo_b. o: [n_head*head_dim] (modificato in copia interna).
inline void attn_project_out(const AttnParams& p, const AttnWeights& w, const float* o, int64_t pos, bool compressed,
                             float* out_embd, std::vector<float>& buf) {
    const RopeCfg rc = rope_cfg_for_layer(p, compressed);
    const int64_t H = p.n_head, D = p.head_dim, G = p.n_groups;
    const int64_t nhg = H / G;
    const int64_t og = nhg * D;  // o_group_dim

    std::vector<float> ob(o, o + H * D);
    for (int64_t h = 0; h < H; ++h) rope_apply(ob.data() + h * D, p, rc, pos, true);

    std::vector<float> oa(static_cast<size_t>(p.o_lora_rank * G));
    const size_t rb = dequant_row_bytes(w.wo_a.type, og);
    for (int64_t g = 0; g < G; ++g) {
        matvec_quant(w.wo_a.type, w.wo_a.data + static_cast<size_t>(g * p.o_lora_rank) * rb, og, p.o_lora_rank,
                     ob.data() + g * og, oa.data() + g * p.o_lora_rank, buf);
    }
    matvec_quant(w.wo_b.type, w.wo_b.data, p.o_lora_rank * G, p.n_embd, oa.data(), out_embd, buf);
}

// Layer raw completo su T token (prefill senza cache). X: [T][n_embd] -> Y: [T][n_embd].
inline void attn_forward_raw(const AttnParams& p, const AttnWeights& w, const float* X, int64_t T, float* Y) {
    const int64_t H = p.n_head, D = p.head_dim;
    std::vector<float> q(static_cast<size_t>(T * H * D)), kv(static_cast<size_t>(T * D));
    std::vector<float> o(static_cast<size_t>(T * H * D));
    std::vector<float> buf;
    for (int64_t t = 0; t < T; ++t) {
        attn_project_qkv(p, w, X + t * p.n_embd, t, false, q.data() + t * H * D, kv.data() + t * D, buf);
    }
    attn_core_raw(p, q.data(), kv.data(), T, w.sinks, o.data());
    for (int64_t t = 0; t < T; ++t) {
        attn_project_out(p, w, o.data() + t * H * D, t, false, Y + t * p.n_embd, buf);
    }
}

}  // namespace gptoss::dsv4

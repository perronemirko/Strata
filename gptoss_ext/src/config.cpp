#include "gptoss/config.hpp"
#include "gptoss/json.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace gptoss {

bool config_from_file(const std::string& path, GptConfig& c, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    Json t; std::string jerr;
    if (!parse_json(ss.str(), t, jerr)) { err = path + ": " + jerr; return false; }
    if (t.str("model_type") != "gpt_oss") { err = "model_type is '" + t.str("model_type") + "', expected 'gpt_oss'"; return false; }
    c.arch = t.str("model_type");                 // the components above keep their gpt-oss defaults
    c.hidden = (int) t.num("hidden_size", 0);
    c.layers = (int) t.num("num_hidden_layers", 0);
    c.heads = (int) t.num("num_attention_heads", 0);
    c.kv_heads = (int) t.num("num_key_value_heads", 0);
    c.head_dim = (int) t.num("head_dim", 0);
    c.vocab = (int) t.num("vocab_size", 0);
    c.n_exp = (int) t.num("num_local_experts", 0);
    c.topk = (int) t.num("num_experts_per_tok", (int) t.num("experts_per_token", 0));
    c.moe_ff = (int) t.num("intermediate_size", 0);
    c.window = (int) t.num("sliding_window", 0);
    c.eps = (float) t.num("rms_norm_eps", 1e-5);
    c.swiglu_limit = (float) t.num("swiglu_limit", 7.0);
    c.rope_theta = (float) t.num("rope_theta", 150000.0);
    c.max_pos = (int64_t) t.num("max_position_embeddings", 0);
    c.attention_bias = t.boolean("attention_bias", true);
    c.pad = (int) t.num("pad_token_id", -1);
    c.bos = (int) t.num("bos_token_id", -1);
    if (const Json* e = t.get("eos_token_id")) c.eos = e->t == Json::Arr && !e->a.empty() ? (int) e->a[0].n : (int) e->n;
    const Json* rs = t.get("rope_scaling");
    if (!rs) rs = t.get("rope_parameters");
    if (!rs || rs->str("rope_type", rs->str("type")) != "yarn") { err = "only rope_scaling.rope_type=yarn is implemented"; return false; }
    c.yarn_factor = rs->num("factor", 1.0);
    c.yarn_orig = (int64_t) rs->num("original_max_position_embeddings", t.num("initial_context_length", 0));
    c.yarn_beta_fast = rs->num("beta_fast", 32.0);
    c.yarn_beta_slow = rs->num("beta_slow", 1.0);
    c.yarn_truncate = rs->boolean("truncate", true);
    c.sliding.assign((size_t) c.layers, 0);
    const Json* lt = t.get("layer_types");
    for (int l = 0; l < c.layers; ++l)
        c.sliding[(size_t) l] = lt && (size_t) l < lt->a.size() ? (lt->a[(size_t) l].s == "sliding_attention") : (l % 2 == 0);   // fallback: even layers slide
    if (c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.kv_heads <= 0 || c.head_dim <= 0 || c.vocab <= 0 || c.n_exp <= 0 || c.topk <= 0 ||
        c.moe_ff <= 0 || c.window <= 0 || c.yarn_orig <= 0) { err = "config.json lacks a required field"; return false; }
    if (c.heads % c.kv_heads) { err = "num_attention_heads must be a multiple of num_key_value_heads"; return false; }
    if (c.head_dim % 2) { err = "head_dim must be even"; return false; }
    if (c.hidden % 32 || c.moe_ff % 32) { err = "hidden_size and intermediate_size must be multiples of 32 (MXFP4 blocks)"; return false; }
    if (c.topk > c.n_exp || c.topk > 64) { err = "num_experts_per_tok out of range"; return false; }
    // This build implements the components of gpt-oss only (see GptConfig). Anything else would run with the wrong math.
    if (c.arch != "gpt_oss" || c.attn != "gqa_sink" || c.rope != "yarn" || c.router != "softmax_topk_bias" || c.act != "swiglu_clamped" || c.expert_format != "mxfp4") {
        err = "this build implements the components of gpt-oss only (arch=" + c.arch + ", attn=" + c.attn + ", experts=" + c.expert_format + ")";
        return false;
    }
    return true;
}

std::string config_summary(const GptConfig& c) {
    int ns = 0; for (char s : c.sliding) ns += s;
    char b[640];
    std::snprintf(b, sizeof b, "gpt-oss: %d layers (%d sliding/%d dense, window %d), hidden %d, %d heads / %d kv x %d | MoE %d experts top-%d ff %d limit %g | "
                  "yarn x%g orig %lld theta %g trunc %d | vocab %d eos %d",
                  c.layers, ns, c.layers - ns, c.window, c.hidden, c.heads, c.kv_heads, c.head_dim, c.n_exp, c.topk, c.moe_ff, (double) c.swiglu_limit,
                  c.yarn_factor, (long long) c.yarn_orig, (double) c.rope_theta, (int) c.yarn_truncate, c.vocab, c.eos);
    return b;
}

}  // namespace gptoss

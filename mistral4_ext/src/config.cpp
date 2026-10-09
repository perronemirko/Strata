#include "m4/config.hpp"
#include "m4/json.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace m4 {

bool config_from_file(const std::string& path, M4Config& c, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    Json root; std::string jerr;
    if (!parse_json(ss.str(), root, jerr)) { err = path + ": " + jerr; return false; }
    const Json* tc = root.get("text_config");
    const Json& t = tc ? *tc : root;
    const std::string mt = t.str("model_type");
    if (mt != "mistral4") { err = "text model_type is '" + mt + "', expected 'mistral4'"; return false; }
    c.hidden = (int) t.num("hidden_size", 0);
    c.layers = (int) t.num("num_hidden_layers", 0);
    c.heads = (int) t.num("num_attention_heads", 0);
    c.vocab = (int) t.num("vocab_size", 0);
    c.q_lora = (int) t.num("q_lora_rank", 0);
    c.kv_lora = (int) t.num("kv_lora_rank", 0);
    c.nope = (int) t.num("qk_nope_head_dim", 0);
    c.rope = (int) t.num("qk_rope_head_dim", 0);
    c.v_head = (int) t.num("v_head_dim", 0);
    c.n_exp = (int) t.num("n_routed_experts", 0);
    c.topk = (int) t.num("num_experts_per_tok", 0);
    c.n_shared = (int) t.num("n_shared_experts", 0);
    c.moe_ff = (int) t.num("moe_intermediate_size", 0);
    c.n_group = (int) t.num("n_group", 1);
    c.topk_group = (int) t.num("topk_group", 1);
    c.norm_topk = t.boolean("norm_topk_prob", true);
    c.rope_interleave = t.boolean("rope_interleave", true);
    c.eps = (float) t.num("rms_norm_eps", 1e-6);
    c.routed_scale = (float) t.num("routed_scaling_factor", 1.0);
    c.max_pos = (int64_t) t.num("max_position_embeddings", 0);
    c.bos = (int) t.num("bos_token_id", -1);
    c.eos = (int) t.num("eos_token_id", -1);
    c.pad = (int) t.num("pad_token_id", -1);
    const Json* rp = t.get("rope_parameters");
    if (!rp) { err = "rope_parameters missing"; return false; }
    if (rp->str("rope_type", rp->str("type")) != "yarn") { err = "only rope_type=yarn is implemented"; return false; }
    c.rope_theta = (float) rp->num("rope_theta", 10000.0);
    c.yarn_factor = rp->num("factor", 1.0);
    c.yarn_orig = (int64_t) rp->num("original_max_position_embeddings", 0);
    c.yarn_beta_fast = rp->num("beta_fast", 32.0);
    c.yarn_beta_slow = rp->num("beta_slow", 1.0);
    c.yarn_mscale = rp->num("mscale", 1.0);
    c.yarn_mscale_all_dim = rp->num("mscale_all_dim", 1.0);
    c.llama4_beta = rp->num("llama_4_scaling_beta", 0.0);

    if (c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.vocab <= 0 || c.kv_lora <= 0 || c.nope <= 0 || c.rope <= 0 ||
        c.v_head <= 0 || c.n_exp <= 0 || c.topk <= 0 || c.moe_ff <= 0 || c.q_lora <= 0 || c.yarn_orig <= 0) {
        err = "config.json lacks a required field (see config_summary)"; return false;
    }
    if (c.rope % 2) { err = "qk_rope_head_dim must be even"; return false; }
    if (!c.rope_interleave) { err = "rope_interleave=false (half-split rotary) is not implemented"; return false; }
    if (t.num("first_k_dense_replace", 0) != 0) { err = "first_k_dense_replace != 0 (dense leading layers) is not implemented"; return false; }
    if (c.n_group != 1 || c.topk_group != 1) { err = "group-limited routing (n_group/topk_group != 1) is not implemented"; return false; }
    if (c.topk > c.n_exp) { err = "num_experts_per_tok > n_routed_experts"; return false; }
    return true;
}

std::string config_summary(const M4Config& c) {
    char b[768];
    std::snprintf(b, sizeof b,
        "mistral4: %d layers, hidden %d, %d heads | MLA q_lora %d kv_lora %d nope %d rope %d v %d | MoE %d experts top-%d (+%d shared, ff %d)%s | "
        "yarn x%g orig %lld theta %g | llama4 beta %g | vocab %d bos %d eos %d",
        c.layers, c.hidden, c.heads, c.q_lora, c.kv_lora, c.nope, c.rope, c.v_head, c.n_exp, c.topk, c.n_shared, c.moe_ff,
        c.norm_topk ? " norm" : "", c.yarn_factor, (long long) c.yarn_orig, (double) c.rope_theta, c.llama4_beta, c.vocab, c.bos, c.eos);
    return b;
}

}  // namespace m4

// m4/config.hpp - Mistral Small 4 (model_type "mistral4") configuration, read from the HF config.json.
#pragma once
#include <cstdint>
#include <string>

namespace m4 {

struct M4Config {
    int hidden = 0, layers = 0, heads = 0, vocab = 0;
    int q_lora = 0, kv_lora = 0, nope = 0, rope = 0, v_head = 0;     // MLA
    int n_exp = 0, topk = 0, n_shared = 0, moe_ff = 0, n_group = 1, topk_group = 1;
    bool norm_topk = true, rope_interleave = true;
    float eps = 1e-6f, routed_scale = 1.0f, rope_theta = 10000.0f;
    double yarn_factor = 1, yarn_beta_fast = 32, yarn_beta_slow = 1, yarn_mscale = 1, yarn_mscale_all_dim = 1;
    int64_t yarn_orig = 0;               // original_max_position_embeddings (YaRN and the Llama-4 scale share it)
    double llama4_beta = 0;
    int64_t max_pos = 0;
    int bos = -1, eos = -1, pad = -1;
    std::string prefix;                  // tensor-name prefix, filled by the loader ("language_model.model.")
    int qk_head() const { return nope + rope; }
};

/// Accepts the multimodal wrapper (text_config nested) or a bare text config. Refuses what the engine does not implement
/// (dense leading layers, group-limited routing) instead of running it wrongly.
bool config_from_file(const std::string& config_json_path, M4Config& c, std::string& err);
std::string config_summary(const M4Config& c);

}  // namespace m4

// gptoss/config.hpp - gpt-oss (model_type "gpt_oss") configuration from the HF config.json.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gptoss {

struct GptConfig {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
    int n_exp = 0, topk = 0, moe_ff = 0;           // moe_ff = intermediate_size (also the per-expert width)
    int window = 0;                                // sliding-window width (keys visible, current token included)
    float eps = 1e-5f, swiglu_limit = 7.0f, swiglu_alpha = 1.702f, rope_theta = 150000.0f;
    double yarn_factor = 1, yarn_beta_fast = 32, yarn_beta_slow = 1;
    int64_t yarn_orig = 0, max_pos = 0;
    bool yarn_truncate = false, attention_bias = true;
    int bos = -1, eos = -1, pad = -1;
    std::vector<char> sliding;                     // per layer: 1 = banded window, 0 = dense (from layer_types)
    std::string prefix = "model.";
    // Architecture and components (phase 0 of the generalization). Filled from config.json; only the gpt-oss values exist so far.
    std::string arch = "gpt_oss";                // model_type: selects the model file
    std::string attn = "gqa_sink";               // gqa_sink: grouped-query attention with a sink in the softmax denominator (MLA for other models)
    std::string rope = "yarn";                   // yarn
    std::string router = "softmax_topk_bias";    // softmax over the router logits, top-k, learned bias added before the top-k
    std::string act = "swiglu_clamped";          // SwiGLU with the limit / alpha above
    std::string expert_format = "mxfp4";         // the routed experts' weight format
    int q_dim() const { return heads * head_dim; }
    int kv_dim() const { return kv_heads * head_dim; }
};

bool config_from_file(const std::string& path, GptConfig& c, std::string& err);
std::string config_summary(const GptConfig& c);

}  // namespace gptoss

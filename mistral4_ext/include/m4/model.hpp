// m4/model.hpp - Mistral Small 4 on CPU: MLA (latent KV cache, absorbed form) + 128-expert top-4 MoE + shared expert.
#pragma once
#include "m4/config.hpp"
#include "m4/ops.hpp"

#include <memory>
#include <string>
#include <vector>

namespace m4 {

struct RunOpts {
    int ctx = 4096;               // KV cache positions allocated (the model supports far more; memory is 46 KiB/token fp32 at full size)
    int threads = 0;              // 0 = every core
    // --- facts the sources read for this port do NOT settle (docs/mistral_small4_porting.md, "Non noto"). Defaults = best reading.
    Router router = Router::Softmax;   // softmax (HF doc: "post-softmax") vs sigmoid (DeepSeek-V3 style)
    bool mscale_softmax = false;       // params.json says yarn "apply_scale": false; DeepSeek-V3's HF code multiplies the scale by mscale^2
    bool l4_qpe_only = false;          // llama-4 scale on q_pe only (NeMo's description) vs on the whole q (vLLM)
    bool fp8_scale_div = false;        // *_scale_inv multiplies (DeepSeek convention) vs divides
    bool verbose = true;
    // --- prefill / KV (serve)
    int prefill_chunk = 512;           // tokens per batched prefill step (layer by layer, experts grouped); 1 = old token-by-token behaviour
    std::string kv_checkpoint;         // file: loaded at start if present, rewritten after every request ("" = off)
    // --- VRAM budget. PLAN ONLY: there is no CUDA backend in this tree yet, so nothing is uploaded; the numbers are what the plan WOULD hold.
    double vram_pct = -1;              // percent of the card to use (-1 = no plan)
    long long vram_total_mib = 0;      // size of the card (no driver query without a CUDA backend)
    long long vram_reserve_mib = 1024;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// `dir` = the HF snapshot directory (config.json + model-*.safetensors) or a single .safetensors next to a config.json.
    bool load(const std::string& dir, const RunOpts& opts, std::string& err);
    void reset();                                              // forget the KV cache (new request)
    /// One token at position `pos` (0, 1, 2, ... without gaps). `logits_out` may be null (prompt tokens whose logits nobody reads).
    void forward(int token, int pos, std::vector<float>* logits_out);
    const M4Config& config() const;
    uint64_t weight_bytes() const;

    /// Batched prefill of tokens[0..n) at positions start_pos.. : every layer is run over the whole chunk, the routed experts are grouped
    /// (each expert's weights are walked once per chunk instead of once per token). `logits_last` (may be null) gets the logits of the last token.
    void prefill(const int* tokens, int n, int start_pos, std::vector<float>* logits_last);

    /// The KV cache stays between requests. Returns how many leading tokens of `ids` are already cached (never all of them: the last
    /// one is always recomputed so its logits exist) and drops everything after that point. The next position to feed is the return value.
    int reuse_prefix(const std::vector<int>& ids);
    int cached_len() const;

    /// KV checkpoint: tokens + latent cache of every layer, with a fingerprint of config/options/weights. Written atomically (tmp + rename).
    bool save_kv(const std::string& path, std::string& err) const;
    bool load_kv(const std::string& path, std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace m4

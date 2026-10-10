// gptoss/model.hpp - gpt-oss-120b on CPU: GQA with attention sinks, alternating banded/dense layers, 128-expert top-4 MXFP4 MoE.
#pragma once
#include "gptoss/config.hpp"
#include "gptoss/ops.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gptoss {

struct RunOpts {
    int ctx = 8192;                    // positions of the dense layers' KV cache (the model supports 131072; fp32 KV = 4 KiB/layer/token)
    int threads = 0;                   // 0 = every core
    int prefill_chunk = 512;           // tokens per batched prefill step; clamped so a banded layer's ring is never overwritten mid-chunk
    int kv_rollback = 8192;            // extra positions kept by every banded layer's ring: how far a new prompt may rewind into the cache
    std::string kv_checkpoint;         // loaded at start if present, rewritten after EVERY request ("" = off)
    // --- facts the model card / config.json do not settle (docs/gptoss_porting.md, "Non noto"). Defaults = best reading.
    bool glu_halves = false;           // gate_up rows: interleaved (gate = even rows) vs gate = first half, up = second half
    bool rope_pairs = false;           // RoPE on adjacent pairs instead of the half-split layout
    bool no_concentration = false;     // drop the YaRN factor 0.1*ln(factor)+1 on cos/sin
    bool verbose = true;
    // --- CUDA. Needs the binary built with `sh build.sh --cuda` (the plain binary carries the EMULATED device and refuses --cuda).
    bool cuda = false;
    int cuda_device = 0;
    int gpu_slots = -1;                // cap on the expert slots in VRAM (-1 = as many as the budget allows)
    // --- VRAM budget: --vram-pct of the card (or of --vram-total-mib when given) minus the reserve. With cuda=false it is only a plan.
    double vram_pct = -1;
    long long vram_total_mib = 0, vram_reserve_mib = 1024;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const std::string& dir, const RunOpts& opts, std::string& err);
    void reset();
    void forward(int token, int pos, std::vector<float>* logits_out);
    /// Batched prefill of tokens[0..n) at start_pos..: layer by layer over the chunk, routed experts grouped so each expert is walked once per chunk.
    void prefill(const int* tokens, int n, int start_pos, std::vector<float>* logits_last);
    /// Keeps the KV cache between requests: returns how many leading tokens of `ids` are already cached (the last prompt token is always
    /// recomputed). Banded layers only hold `window + kv_rollback` positions: if the rewind would need overwritten ones, returns 0.
    int reuse_prefix(const std::vector<int>& ids);
    int cached_len() const;
    bool save_kv(const std::string& path, std::string& err) const;
    bool load_kv(const std::string& path, std::string& err);
    const GptConfig& config() const;
    uint64_t weight_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace gptoss

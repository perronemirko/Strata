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
    std::string expert_profile;        // expert-usage counts: loaded at start, rewritten after EVERY request; decides which experts get PINNED in VRAM
    std::string expert_profile_save;   // where the profile is written (default: the --expert-profile path)
    double expert_profile_save_every = 0;   // seconds between periodic saves while generating (0 = only after each request and at exit)
    double pin_pct = 80;               // % of the expert slots reserved for the most-used experts of the profile (the rest is LRU for the others)
    int gpu_slots = -1;                // cap on the expert slots in VRAM (-1 = as many as the budget allows)
    // --- expert cache: three opt-in knobs that only move WHICH experts hold the slots, never how many slots there are.
    //     0 / false keeps the previous release's behaviour byte-for-byte.
    int expert_promote = 0;          // 0 = off: an LRU expert counted this many hits becomes pinned, and the least-used pinned expert becomes LRU (no PCIe traffic, same slot count)
    bool expert_lru_layered = false; // evict inside the requesting layer first, so a cold expert of one layer cannot push out a hot expert of another
    long long profile_decay = 0;     // 0 = off: halve every usage count after a save while their sum is above this, so the pinning follows the recent sessions
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
    /// Expert usage profile (how many times each (layer, expert) was routed to). Binary: "GPPRF001", u32 layers, u32 experts, u64[layers*experts].
    bool save_profile(const std::string& path, std::string& err);   // with --profile-decay it also ages the in-memory counts it just wrote
    bool load_profile(const std::string& path, std::string& err);
    void decay_profile();             // call after every profile save: with --profile-decay 0 it does nothing
    std::string stats() const;          // slots / pinned / hits / misses / PCIe ("" without a device pool)
    /// Per-request decode counters, same fields as Strata's DONE line (hits, lookups, offloaded over PCIe).
    /// Call begin_request_stats() before the request; request_stats() after it returns the request's own counts.
    /// A routed expert is either a VRAM hit, uploaded over PCIe, or computed on the host: lookups = hits + host-served,
    /// offloaded = PCIe-served, so lookups + offloaded = every expert the decode routed. Prefill lookups are not counted.
    void begin_request_stats();
    void request_stats(unsigned long long* hits, unsigned long long* lookups, unsigned long long* pcie) const;
    const GptConfig& config() const;
    uint64_t weight_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace gptoss

/**
 * @file model.hpp
 * @brief DeepSeek-V4-Flash model inference engine: load, forward pass, routing, and statistics.
 *
 * This module implements the complete DeepSeek-V4-Flash transformer model in C++, including:
 *   - GGUF multi-shard loading with memory-mapped tensor data
 *   - Full forward pass: hyper-connections, CSA/HCA attention, router, hash experts, SwiGLU
 *   - HIT/MISS expert tiering with VRAM slots and LRU RAM arena
 *   - Prefill batching (chunked token processing)
 *   - YaRN context extension, sliding window attention, sparse attention
 *   - Expert profiling and routing frequency tracking
 *
 * @par Architecture overview
 *   The model uses the Pimpl idiom (`struct Impl`) to hide ~1500 lines of implementation details.
 *   Public API is three methods: `load()`, `forward()`, and `forward_chunk()`. All internal state
 *   (tensors, KV cache, expert VRAM slots, scratch buffers) lives in `Impl`.
 *
 * @par Expert residency pipeline
 *   @code
 *   MISS → RAM arena (LRU) → mmap'd GGUF shard
 *   HIT  → VRAM slot (resident, evaluated on device)
 *   @endcode
 *   The LRU RAM arena (`ram_cache_mib`, default 8192 MiB) caches MISS experts that are neither in
 *   the profile's RAM copy nor in a VRAM slot. Without it, every miss re-reads ~6.8 MiB from disk.
 *
 * @par Prefill batching
 *   `forward_chunk()` processes N tokens at once (default 256). Each expert is evaluated ONCE for all
 *   its routed tokens within the chunk, saving the weight decode cost. The sliding window attention
 *   requires a ring buffer snapshot to prevent future-token reads.
 */
#pragma once

#include "dsv4/config.hpp"
#include "dsv4/mem_plan.hpp"
#include "dsv4/kv_quant.hpp"
#include "dsv4/sampler.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace dsv4 {

/**
 * @struct RunOpts
 * @brief Runtime options for model loading and inference.
 *
 * Controls VRAM allocation, expert residency, profiling, prefill batching, and output verbosity.
 * Most options map directly to command-line flags in `dsv4_run`.
 */
struct RunOpts {
    int ctx = 2048;                     ///< Context window size (number of tokens for KV cache).
    bool gpu = true;                    ///< Enable GPU/device acceleration (false = CPU-only).
    int threads = 0;                    ///< OpenMP thread count: 0 = all cores, >0 = specific count.
    bool qat_sim = true;                ///< Simulate QAT (Quantization-Aware Training) activation clipping.
    int max_layers = -1;                ///< Max layers to load: -1 = all, >=0 = partial for testing.
    double vram_pct = -1.0;             ///< VRAM percentage for experts: -1=auto, 0-100=explicit.
    long long abs_slots = -1;           ///< --expert-cache N: absolute slot count (-1 = use pct).
    long long reserve_mib = 1024;       ///< VRAM safety reserve in MiB (default 1 GiB).
    long long vram_free_mib = -1;       ///< Override free VRAM query (-1 = auto-detect from device).
    long long vram_total_mib = -1;      ///< Override total VRAM (-1 = auto-detect).
    std::string profile_in;             ///< Input expert routing profile file path.
    std::string profile_out;            ///< Output expert routing profile file path (saved on exit).
    bool ram_all = false;               ///< --expert-ram all: copy ALL experts into RAM at load time.
    long long ram_cache_mib = 8192;     ///< LRU RAM arena size in MiB for MISS experts (0 = disabled).
    long long host_ram_reserve_mib = 0; ///< Host RAM this run must not touch: what the sibling engines already hold.
    int adapt_swaps = 0;                ///< Adaptive expert swap count tracking.
    bool verify_slots = false;          ///< Verify VRAM slot assignments at load time.
    bool verbose = true;                ///< Print startup report: memory plan, profile, RAM/VRAM counts.
    int prefill_chunk = 256;            ///< Prefill batch size: 0 = disabled (one token at a time).
    int spec_n = 0;                     ///< --spec N: speculative decoding with up to N draft tokens per verify window (0 = off).
    float spec_min_p = 1.0f;            ///< --spec-min-p P: acceptance threshold in sampling mode (1.0 = exact rejection sampling).
    int suffix_draft = 0;               ///< --suffix-draft N: prompt-lookup drafter (the divergence the paper's spec has and this engine lacked): proposes up to N tokens continuing an earlier repeat of the context; the adaptive DraftPolicy decides per round whether an MTP or a lookup window pays. 0 = off.
    int lookup_chain = 0;               ///< --lookup-chain K: append up to K lookup tokens after a full MTP window (the chain rule of the same policy). Needs --suffix-draft. 0 = off.
    bool spec_adaptive = false;         ///< --spec-adaptive: the DraftPolicy chooses the window size/source per round (Strata's verified policy). Auto-on when --suffix-draft is used with --spec; this flag also enables it for MTP-only runs.
    std::string mtp_path;               ///< --mtp FILE|auto: separate GGUF holding the MTP tensors ("auto" = detect them in the main shards).
    std::string kv_checkpoint;          ///< --kv-checkpoint FILE: restored at load, rewritten after every request.
    KVFormat key_format = KVFormat::F32;   ///< -k / --key-format: format of the cached (fused) KV rows (e4m3/f16/f32).
    KVFormat value_format = KVFormat::F32; ///< -v / --value-format: must match key_format - DSv4 caches one fused row per token.
};

/**
 * @struct Stats
 * @brief Runtime statistics collected during inference.
 *
 * All counters are accumulated across all tokens processed since the last `reset()`.
 * Read by `dsv4_run` for the final report and by the serve loop for progress output.
 */
struct Stats {
    uint64_t h2d_bytes = 0;             ///< Total bytes uploaded to VRAM (device memory bandwidth).
    uint64_t admits = 0;                ///< Experts loaded into a free VRAM slot.
    uint64_t swaps = 0;                 ///< Adaptive evict+load decisions (expert cache management).
    uint64_t hits = 0;                  ///< Routed expert evaluations served from VRAM slots.
    uint64_t misses = 0;                ///< Routed expert evaluations served from RAM/disk.
    uint64_t cache_hits = 0;            ///< MISS experts already in the RAM LRU arena (no disk read).
    uint64_t cache_loads = 0;           ///< MISS experts copied into arena from mmap'd GGUF.
    uint64_t cache_evictions = 0;       ///< Arena slots reused for a different expert (LRU eviction).
    uint64_t cache_bytes = 0;           ///< Total bytes copied into the RAM arena.
    uint64_t tokens = 0;                ///< Total tokens processed (prompt + generated).
    uint64_t prefill_tokens = 0;        ///< Tokens processed inside batched chunks.
    uint64_t prefill_chunks = 0;        ///< Number of batched prefill chunks run.
    double total_s = 0;                 ///< Total forward pass time in seconds.
    double cpu_miss_s = 0;              ///< Time spent computing MISS experts on CPU/host.
    double gpu_hit_s = 0;               ///< Time spent waiting for HIT experts on GPU/device.
    uint64_t spec_steps = 0;            ///< Speculative steps executed (draft + verify rounds).
    uint64_t spec_drafts = 0;           ///< Draft tokens proposed by the MTP module.
    uint64_t spec_accepted = 0;         ///< Draft tokens accepted by the main model's verification.
};

/**
 * @class Model
 * @brief DeepSeek-V4-Flash transformer model: load, forward pass, and expert management.
 *
 * The main inference engine class. Handles GGUF multi-shard loading, tensor mapping, expert VRAM
 * residency planning, forward pass (including all DeepSeek-V4 innovations: hyper-connections,
 * sparse attention, YaRN, hash experts), and statistics collection.
 *
 * @par Thread safety
 *   Not thread-safe: each Model instance serves one inference session. Use multiple instances
 *   for concurrent requests.
 *
 * @par Usage
 *   @code
 *   Model model;
 *   RunOpts opts;
 *   opts.vram_pct = 75.0;
 *   opts.prefill_chunk = 256;
 *   std::string err;
 *   if (!model.load({"shard-1.gguf", "shard-2.gguf"}, opts, err)) {
 *       fprintf(stderr, "Load failed: %s\n", err.c_str());
 *   }
 *   model.forward(token_id, position, &logits);
 *   @endcode
 */
class Model {
public:
    /**
     * @brief Construct an empty Model instance.
     *
     * No resources are allocated at construction time. Call `load()` to initialize.
     */
    Model();

    /**
     * @brief Destroy the Model, freeing all allocated resources.
     *
     * Unmaps GGUF shards, frees device scratch buffers, releases VRAM allocations.
     */
    ~Model();

    /// Disable copy construction and assignment (unique ownership of internal state).
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /**
     * @brief Load the model from GGUF shard files.
     *
     * Parses the GGUF header, extracts config, builds expert inventory, plans VRAM residency,
     * memory-maps all tensor data, and optionally loads a routing profile for expert caching.
     *
     * @param shards  Vector of GGUF shard file paths in order (1..N).
     * @param opts    Runtime options controlling loading behavior.
     * @param err_out Output: error message on failure (empty on success).
     * @return true if model loaded successfully; false otherwise (err_out contains details).
     */
    bool load(const std::vector<std::string>& shards, const RunOpts& opts, std::string& err_out);

    /**
     * @brief Reset the model state: clear KV cache, reset position counter, zero statistics.
     *
     * Call between independent inference sessions to avoid cross-contamination of context.
     */
    void reset();

    /**
     * @brief Forward pass for a single token (decode phase).
     *
     * Processes one token at the given position through all layers, including:
     *   1. Embedding lookup and positional encoding (YaRN rope)
     *   2. For each layer: RMSNorm → hyper-connections → CSA/HCA attention → router → MoE experts
     *   3. Final RMSNorm → output projection → logit computation
     *
     * Expert residency: HIT experts are evaluated from VRAM slots; MISS experts are computed on
     * the host (from RAM arena or mmap'd GGUF). Routing statistics are accumulated.
     *
     * @param token         Input token id for this position.
     * @param pos           Current position in the sequence (for positional encoding).
     * @param logits_out    Output: vocabulary-sized logit vector (replaced on each call).
     */
    void forward(int token, int pos, std::vector<float>* logits_out);

    /**
     * @brief Forward pass for a chunk of consecutive tokens (prefill phase).
     *
     * Processes N consecutive tokens starting at position `pos0` through all layers in parallel.
     * Only the LAST token's logit vector is computed and returned (that's the one that starts generation).
     * The intermediate tokens' expert evaluations are accumulated but their logits are discarded.
     *
     * @par Prefill batching benefits
     *   Each expert is evaluated ONCE for all its routed tokens within the chunk, saving the weight
     *   decode cost which dominates MoE traffic at ~1.8 GB/token. The sliding window attention uses
     *   a ring buffer snapshot to prevent future-token reads.
     *
     * @param tokens      Array of token ids (N consecutive tokens).
     * @param n           Number of tokens in the array.
     * @param pos0        Starting position in the sequence.
     * @param logits_out  Output: logit vector for the LAST token only.
     * @return Number of tokens actually processed (may be less than N if chunk exceeds limits).
     */
    int forward_chunk(const int* tokens, int n, int pos0, std::vector<float>* logits_out);

    /**
     * @brief Batched forward that returns the logit vector of EVERY position (speculative verify).
     *
     * Same arithmetic as forward_chunk(), but all_logits[t] receives the distribution produced after
     * processing token t at position pos0 + t. The MTP layers are NOT run here: their state was already
     * advanced by the drafting phase, and verification must only touch the main model's cache.
     */
    int forward_chunk_all(const int* tokens, int n, int pos0, std::vector<std::vector<float>>& all_logits);

    /**
     * @brief One speculative decoding step (requires MTP layers: mtp_layers() > 0).
     *
     * Drafts up to RunOpts::spec_n tokens with the MTP module, verifies them against the main model in a
     * single batched forward, and returns the accepted token sequence - always at least one token (the
     * current distribution's own sample/argmax when every draft is rejected). The KV cache (main AND MTP
     * layers) is left exactly consistent with the returned tokens: on partial acceptance the compressor
     * rolling buffers are rolled back and the history truncated.
     *
     * @param cur_logits  Distribution produced by the last forward()/forward_chunk() - it predicts the token
     *                    at position cached_tokens().size(), i.e. the first draft's claim.
     * @param so          Sampling options of the request (temperature, penalties, ...).
     * @param sampler     The request's sampler: its RNG stream is consumed for every random decision, so a
     *                    fixed seed replays identically with or without speculation.
     */
    /// @param next_logits Receives the distribution of the NEXT position (from the verify chunk) - feed it back in.
    std::vector<int> spec_step(const float* cur_logits, const SampleOpts& so, Sampler* sampler, std::vector<float>* next_logits);

    /** @brief Number of MTP (extra) layers loaded and executed: 0 when the model has none or --spec is off. */
    int mtp_layers() const;

    /** @brief The final hidden state [hc * dim] left by the last forward()/forward_chunk() - feeds draft step 0. */
    std::vector<float> last_hidden_state() const;

    /**
     * @brief Save the KV cache and token history to a checkpoint file (magic DSKV0002).
     *
     * The file carries a fingerprint of model + options; load_kv() refuses it when they differ.
     * Written atomically (tmp + rename); after the first save, saves that extend the same history
     * append only the new segment instead of rewriting the whole file.
     */
    bool save_kv(const std::string& path, std::string& err) const;

    /**
     * @brief Restore the KV cache and token history from a checkpoint file (see save_kv()).
     *
     * After a successful call the engine is exactly where the saving run left off: forward() may
     * continue at position cached_tokens().size(). The host cache comes back f32 whatever -k/-v were.
     */
    bool load_kv(const std::string& path, std::string& err);

    /**
     * @brief Reuse the leading part of `ids` already held in the KV cache (prefix reuse).
     *
     * Returns k: ids[0..k) are cached and must NOT be prefilled again; the caller starts at position k.
     * The last prompt token is always recomputed (its logits start the answer), so k <= ids.size()-1.
     * A rewind deeper than the window ring cannot be served (the rows are gone): then 0 is returned
     * and the whole prompt must run again. Truncates the internal history to k on success.
     */
    int reuse_prefix(const std::vector<int>& ids);

    /**
     * @brief Copy of the token history currently held in the KV cache (positions 0..size()-1).
     */
    std::vector<int> cached_tokens() const;

    /**
     * @brief Get routing information for all tokens of the last `forward_chunk()` call.
     *
     * Returns the full routing matrix: [t][layer*K + k] where t is token index within chunk,
     * layer is the transformer layer, K is top-k (n_expert_used), and k is the expert rank.
     * Empty after a plain `forward()` call — only populated after `forward_chunk()`.
     *
     * @return Const reference to the routing matrix vector.
     */
    const std::vector<int>& chunk_routing() const { return chunk_routing_; }

    /**
     * @brief Get the number of routing entries per token in `chunk_routing_`.
     * @return Number of integers per token's routing entry (layer_count × top_k).
     */
    int chunk_routing_n() const { return chunk_routing_n_; }

    /**
     * @brief Get the maximum prefill chunk size this build can support.
     * @return Maximum tokens per chunk (0 = prefill batching is disabled in this build).
     */
    int max_chunk() const;

    /**
     * @brief Signal end of prompt: finalize any pending chunk processing.
     *
     * Called when the prompt ends but generation hasn't started yet. Ensures the sliding window
     * ring buffer and KV cache are in a consistent state before decode begins.
     */
    void end_token();

    /**
     * @brief Signal end of a partial chunk: process remaining tokens.
     *
     * Called when fewer than `prefill_chunk` tokens remain at the end of a prompt. Processes
     * the remaining N tokens and finalizes their forward pass.
     *
     * @param n  Number of remaining tokens to finalize.
     */
    void end_chunk(int n);

    /**
     * @brief Save the current expert routing profile to `opts.profile_out`.
     *
     * The profile records which experts are accessed most frequently, enabling faster startup
     * on subsequent runs by pre-loading hot experts into RAM/VRAM. Written atomically when the
     * engine stops (also in --serve mode).
     *
     * @param err  Output: error message if saving failed (empty on success).
     * @return true if profile was saved successfully.
     */
    bool save_profile(std::string& err) const;

    /**
     * @brief Run a self-check: verify dequantization and forward pass correctness.
     *
     * Tests all quantization types present in the model, checking for finite values, mean ≈ 0,
     * and expected standard deviations. Prints results to stdout. Used by `dsv4_run --selfcheck`.
     */
    void selfcheck() const;

    /**
     * @brief Get a hash of the loaded model (based on tensor data).
     * @return FNV-1a 64-bit hash of all mapped tensor bytes.
     */
    uint64_t model_hash() const;

    /**
     * @brief Get a reference to the parsed model configuration.
     * @return Const reference to Dsv4Config (architecture, dimensions, hyperparameters).
     */
    const Dsv4Config& config() const;

    /**
     * @brief Get the token strings for vocabulary decoding.
     * @return Vector of byte-encoded token strings (for `decode_token_text()`).
     */
    const std::vector<std::string>& tokens() const;

    /**
     * @var last_routing
     * @brief Routing for the last `forward()` call: [top_k] expert indices with weights.
     *
     * Populated after each single-token forward pass. For chunked forwarding, use `chunk_routing()`.
     */
    std::vector<int> last_routing;

    /**
     * @var chunk_routing_
     * @brief Internal storage for chunk routing data (see `chunk_routing()`).
     */
    std::vector<int> chunk_routing_;

    /**
     * @var chunk_routing_n_
     * @brief Number of routing entries per token in `chunk_routing_`.
     */
    int chunk_routing_n_ = 0;

    /**
     * @var stats
     * @brief Runtime statistics, accumulated during inference.
     *
     * Read by `dsv4_run` for the final report and by the serve loop for progress output.
     * Reset on each call to `reset()`.
     */
    Stats stats;  // p_->st points here (set in the Model constructor)

private:
    struct Impl;                          ///< Pimpl implementation detail (defined in model.cpp).
    std::unique_ptr<Impl> p_;             ///< Owned implementation instance.
};

/**
 * @brief Decode a byte-level token string back to text.
 *
 * Maps the model's byte-level token representation (used by BPE tokenizers) back to readable text.
 * Handles multi-byte UTF-8 sequences and the inverse mapping used by DeepSeek-V4's tokenizer.
 *
 * @param tok  Byte-encoded token string from the model's vocabulary.
 * @return Decoded text string, or partial text if the token represents a sub-word fragment.
 */
std::string decode_token_text(const std::string& tok);

} // namespace dsv4

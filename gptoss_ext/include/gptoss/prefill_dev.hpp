// gptoss/prefill_dev.hpp - Strata-style device-resident prefill for gpt-oss (CUDA build only).
//
// The old prefill (Model::Impl::prefill_gpu) round-trips every projection and every expert over PCIe with a full
// stream sync in between, re-uploading the chunk's activations on every call: that is the 8-16 tok/s. This path keeps
// the chunk's residual stream on the device for the whole layer loop (one H2D at the chunk start, one D2H at the end),
// runs the dense projections with bf16 tensor-core GEMM without per-call syncs, does attention on the card against an
// fp32 KV window (sinks + sliding window included), and computes the routed experts in grouped passes: one launch per
// projection reads each touched expert's MXFP4 blob where it already is (a resident pool slot or a streamed ring slot)
// with bf16 tensor cores, decoding the 4-bit weights to shared memory per 32-value group. The host crosses PCIe only for:
// the chunk's X in/out, each layer's router logits down + routing tables up (tens of KB), and the new K/V rows back to
// the host cache (the checkpoint and the decode path stay on the host KV).
//
// C++14. The emulated build links a stub where available() is false: Model keeps the old prefill then.
#pragma once
#include <cstddef>

namespace gptoss {
namespace pdev {

/// This binary carries the CUDA implementation (sh build.sh --cuda). The CPU/emulated build does not.
bool available();

struct Cfg {
    int H = 0, I = 0;                 // hidden, moe_ff
    int E = 0, topk = 0;              // experts, routed per token (topk <= 8: the combine kernel unrolls it)
    int layers = 0;
    int ctx = 0;                      // dense layers' KV positions on the device (fp32)
    int window = 0;                   // sliding-window width (keys visible, current included); banded ring size on the device
    int heads = 0, kv_heads = 0, head_dim = 0;
    float eps = 1e-5f, limit = 7.0f, alpha = 1.702f;
    bool rope_pairs = false;          // RoPE layout (half-split default)
    bool glu_halves = false;          // gate_up rows: interleaved (gate = even) vs first half / second half
    int chunk_max = 1;                // tokens per prefill chunk: sizes the activation buffers
    int q_dim() const { return heads * head_dim; }
    int kv_dim() const { return kv_heads * head_dim; }
};

/// Allocates the device KV windows (fp32), the chunk activation buffers, the expert scratch and the pinned staging
/// inside `budget_bytes`. `sliding` gives each layer's type (nonzero = banded) - it decides the ring size. Returns false
/// when it does not fit: the caller keeps the old prefill for the whole session.
bool init(const Cfg& c, const char* sliding, size_t budget_bytes);
void shutdown();

/// The YaRN cos/sin table [len][head_dim/2], uploaded once (host fp32).
void set_rope(const float* cos_t, const float* sin_t, int len);

/// One layer's small host data + its device BF16 dense weights. Called for every layer at start; the five weight
/// pointers are what Model already uploaded (any nullptr = pdev refuses that layer and init fails).
bool upload_layer(int l, bool sliding, const float* in_ln, const float* post_ln, const float* sinks,
                  const float* bq, const float* bk, const float* bv, const float* bo, const float* rbias,
                  const void* dq, const void* dk, const void* dv, const void* dobj, const void* drt);

/// The chunk's residual stream on the device ([chunk_max][H] fp32). x_upload is async (compute stream), x_download
/// syncs. Both return false on a CUDA error.
float* x_dev();
bool x_upload(const float* host_x, int T);
bool x_download(float* host_x, int T);

/// The host K/V cache rows of one layer ([cap][kv_dim] fp32) for the positions a chunk at pos0 needs: banded layers
/// take the `window - 1` past rows (ring front), dense layers take only the gap [fill, pos0). Enqueued on the copy
/// stream; run_layer waits. Async: no host sync here.
void kv_prefetch(int l, const float* k_host, const float* v_host, int cap_host, bool sliding, int pos0);

// ---- one layer of a chunk of T tokens at positions pos0..pos0+T-1 (all buffers device-side except where noted)
struct LayerIO {
    const void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dobj = nullptr, *drt = nullptr;   // BF16 dense weights
};

/// rmsnorm + q/k/v GEMM + bias/RoPE/KV write + attention (sinks, sliding window) + o GEMM + residual add + post-norm
/// + router GEMM. The router logits come back on the host ([T][E] fp32 in `gl_host`, preallocated by Model) - this is
/// the layer's only sync point. ok = false: a CUDA error, the caller falls back to the old path (the device stays usable).
bool run_layer(const Cfg& c, int l, const LayerIO& w, int T, int pos0, bool sliding, float* gl_host);

/// One routed expert's blobs where they already are (a pool slot or a ring slot). All six must be non-null for the
/// expert to run on the device. gu/dn: MXFP4 blocks + scales; biases: BF16 rows [2I] / [H].
struct ExpertPtrs {
    const void *gu_b = nullptr, *dn_b = nullptr, *gu_s = nullptr, *dn_s = nullptr;
    const void *gu_bias = nullptr, *dn_bias = nullptr;
};

/// The routed experts of the layer, grouped by the host. `per_expert` and `cnt_host` are indexed by expert (0..E-1):
/// an untouched expert has a null ExpertPtrs and cnt 0. Rows are grouped by expert in index order:
///   src[row]    = the token of routed row `row`
///   wt[row]     = the router weight of that (token, k) pair
///   rowof[t][k] = the global row of token t's k-th chosen expert (the combine sums in k order, as the host path does)
/// X [T][H] fp32 gets the experts' contribution added. No sync: ordered on the compute stream after run_layer.
bool run_experts(const Cfg& c, int T, const ExpertPtrs* per_expert, const int* cnt_host,
                 const int* src_host, const float* wt_host, const int* rowof_host);

/// The new K/V rows of the chunk back to the host cache (device fp32 -> host fp32), ring slots for banded layers.
/// Async on the copy stream; sync_all() waits for it and scatters the rows into the host rings before returning.
void kv_store_back(int l, float* k_host, float* v_host, int cap_host, bool sliding, int T, int pos0);

/// Waits for every pending pdev transfer (the chunk's K/V store-backs) and finishes their host-side scatter. Call once
/// at the end of a chunk, before the host touches the cache again (checkpoint save, decode).
bool sync_all();

/// The device KV fill advanced by T positions: call once per chunk after its last layer so the next kv_prefetch knows
/// which dense rows are already on the card.
void advance(int T);

}  // namespace pdev
}  // namespace gptoss

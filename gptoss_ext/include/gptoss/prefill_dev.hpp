// gptoss/prefill_dev.hpp - DEVICE-RESIDENT PREFILL for gpt-oss.
// The batched prefill of src/model.cpp crosses PCIe once per projection and per expert: with a 512-token chunk that is
// ~9000 device round-trips per chunk, and the launch latency, not the math, is what caps prefill at 8-16 tok/s.
// This path keeps the chunk's residual stream ON THE CARD for the whole layer loop: per chunk the host touches the
// device only for X in/out, the router logits down and the new K/V rows back (the decode and the checkpoint stay
// on the host KV, so nothing else has to move). Per layer the host only routes and resolves the experts' slots.
//
// Two implementations behind this header, chosen by the binary that links them (same pattern as gptoss/gpu.hpp):
//   src/prefill_dev_stub.cpp  no device prefill: available() == false, every call is a no-op. The CPU build links this.
//   src/prefill_dev.cu        the CUDA implementation (sh build.sh --cuda). available() == true.
// The Model never calls this namespace unless available() said true, so the stub only has to link, not to work.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace gptoss {
namespace pdev {

bool available();  // false in the stub: the Model keeps its own batched prefill

struct Cfg {
    int H = 0, I = 0, E = 0, topk = 0;              // hidden, expert intermediate, experts per layer, routed per token
    int layers = 0, ctx = 0;                        // layer count, positions of the KV cache
    int window = 0, heads = 0, kv_heads = 0, head_dim = 0;
    float eps = 1e-5f, limit = 7.0f, alpha = 1.702f;
    bool rope_pairs = false, glu_halves = false;    // the same layout switches the Model was built with
    int chunk_max = 1;                              // tokens per chunk; init() sizes the scratch for this
    // Asymmetric KV quantization of the CARD cache (-k / -v on the command line). The host KV cache and the
    // KV checkpoint stay fp32: the Model, the decode and the rollback read them. Only the card's copy is
    // packed, and the attention kernels dequantize it into registers.
    int k_fmt = 2, v_fmt = 2;                       // gptoss::gpu::KVFormat: E4M3 = 0, F16 = 1, F32 = 2
    float kv_limit = 448.0f;                        // e4m3 saturation; the row scale is the power of two below
};

struct LayerIO {                                    // the layer's BF16 projections already on the device (null = on the host)
    const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *router = nullptr;
};

struct ExpertPtrs {                                 // one expert's MXFP4 payload, already resident in its VRAM slot
    const void *gu_b = nullptr, *gu_s = nullptr, *gu_bias = nullptr;
    const void *dn_b = nullptr, *dn_s = nullptr, *dn_bias = nullptr;
};

bool init(const Cfg& cfg, const char* layer_sliding, size_t budget_bytes);   // allocate the chunk scratch + the KV windows; false = it does not fit
int chosen_chunk();                                 // the chunk_max init() really reserved (<= Cfg::chunk_max): the Model must use it
size_t reserved_bytes();                            // what init() actually took: the caller subtracts it from the VRAM budget
void shutdown();                                    // free everything init() took (idempotent)
bool upload_layer(int layer, bool sliding, const float* in_ln, const float* post_ln, const float* sinks,
                  const float* bq, const float* bk, const float* bv, const float* bo, const float* router_bias,
                  const void* dq, const void* dk, const void* dv, const void* dobj, const void* drt);
void set_rope(const float* cos_, const float* sin_, int seqlen);            // [seqlen][head_dim/2] each, uploaded once

bool x_upload(const float* hx, int T);              // chunk's residual stream up (T x H)
bool x_download(float* hx, int T);                  // and back down, after the layer loop

bool kv_prefetch(int layer, const float* host_K, const float* host_V, int cap, bool sliding, int pos0);
void kv_trim(int pos);                              // the host KV changed from `pos` on (decode, checkpoint): the card's mirror is stale from there
bool run_layer(const Cfg& cfg, int layer, const LayerIO& w, int T, int pos0, bool sliding, float* router_logits);
void kv_store_back(int layer, float* host_K, float* host_V, int cap, bool sliding, int T, int pos0);

bool run_experts(const Cfg& cfg, int T, const ExpertPtrs* per_expert, const int* cnt, const int* src_row,
                 const float* row_weight, const int32_t* idx_tok);
void advance(int T);                                // the chunk's positions become history for the next chunk
bool sync_all();                                    // every pending KV copy is back in the host cache
std::string last_error();                           // why the last call refused (empty = no CUDA call failed): the Model prints it when it turns this path off

}  // namespace pdev
}  // namespace gptoss

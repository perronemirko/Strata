// include/strata/kernels/dense_kernels.hpp - the kernels a DENSE qwen35 model (Qwen3.8-27B) needs and the MoE
// engine does not have.
//
// The GDN recurrence, the quantized GEMVs (native_mmvq), RoPE, the per-head RMSNorm and the attention output gate
// already exist and are reused by src/core/dense_model.cpp.  What was missing, because the MoE model routes its
// attention through the QSA indexer and has no plain FFN:
//
//   * an RMSNorm that writes to a second buffer          (dense_rms_norm)
//   * SwiGLU                                             (dense_swiglu)
//   * an F32 GEMV for the two tiny GDN projections       (dense_gemv_f32)   ssm_alpha / ssm_beta, n_out = 48
//   * a K/V cache and full causal attention over it      (dense_kv_append, dense_attn_decode)
//
// The attention is a split-K flash decode: one block per (head, chunk of 256 cells) keeps an online softmax
// (m, l, acc) per warp, the warps are merged inside the block, and a second kernel merges the chunks.  Every
// query head reads the K/V head `h / (n_head / n_kv)` (GQA: contiguous groups, the same map qsa.hpp pins).
//
// KV layout: [n_kv][max_ctx][head_dim] IEEE half, so one head's cells are contiguous.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w        (w NOT shifted: the GGUF already holds 1 + w)
void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream);

/// The GDN output norm of a DENSE qwen35 model: y[h, :] = rms_norm(o[h, :]) * w * SiLU(z[h, :]) for each of `heads`
/// value heads of `cols` (= state_size) values.  llama.cpp's build_norm_gated for qwen35 uses SiLU; the MoE
/// engine's native_gdn_out_norm gates with a sigmoid, which is wrong here (it was the cause of the garbage text).
void dense_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int heads, int cols, float eps,
                        void* stream);

/// Greedy pick over one row of `n` logits: *out_id = the index of the largest (the lowest on a tie; NaN ignored) and
/// *out_p = its softmax probability.  Both are DEVICE pointers.  This is what an MTP draft needs (the token and how
/// sure the head is of it, for the --draft-p-min cut-off).
void dense_argmax_prob(const float* logits, int n, int* out_id, float* out_p, void* stream);

/// out[i] = silu(gate[i]) * up[i]
void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream);

/// y[r] = sum_i W[r * n_in + i] * x[i]           W is a GGUF matrix stored as F32: ne0 = n_in is the contiguous axis
void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);

/// dst[0..n) = value   (the RoPE position of every row)
void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream);

/// Writes this token's K and V (n_kv, head_dim) f32 into cell `pos` of every head's cache as half.
void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream);

/// Bytes of scratch dense_attn_decode needs for a cache of `max_ctx` cells.
uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx);

/// out (n_head, head_dim) = softmax(q . K^T * scale) V over cells [0, n_ctx), n_ctx >= 1 (the current token's
/// cell is already appended).  q is (n_head, head_dim) f32, already normalised and rotated.
void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream);

// ============================ the batched prompt path (T consecutive tokens at a time) ============================
//
// Same arithmetic as the per-token kernels above, with the token as the slow index: every buffer is [T, ...] and the
// order-dependent steps (the GDN conv and delta-rule state, the KV append, the causal mask) walk the tokens inside
// one launch.  Feeding a chunk therefore has to land on the same logits as feeding those tokens one at a time.

/// q[r, :] = q_full[r, 0:head_dim] for r in [0, rows): the query half of every head's [q | gate] block.  q_full is
/// [rows, 2*head_dim] and q is [rows, head_dim] (rows = T * n_head).
void dense_split_q(const float* q_full, float* q, int rows, int head_dim, void* stream);

/// out[i] = attn[i] * sigmoid(gate) with the gate the second half of every head's q_full block (rows = T * n_head).
void dense_gate_apply(const float* attn, const float* q_full, float* out, int rows, int n_head, int head_dim,
                      void* stream);

/// dst[i] = pos0 + i / heads   (the position of every row of a [T, heads, head_dim] tensor)
void dense_positions_i32(int32_t* dst, int rows, int heads, int32_t pos0, void* stream);

/// K and V of T consecutive cells (k/v are [T, n_kv, head_dim] f32) into cells pos0..pos0+T-1 of every cache.
void dense_kv_append_rows(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int T, int pos0,
                          int n_kv, int head_dim, int max_ctx, void* stream);

/// Causal attention for T consecutive queries at positions pos0..pos0+T-1: one block per (head, token), the same
/// warp-per-cell online softmax as dense_attn_decode, restricted to cells [0, pos0 + t].  q is [T, n_head, head_dim]
/// f32 (normalised and rotated), out is the same shape.
void dense_attn_chunk(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int T, int pos0,
                      int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream);

/// y[r, :] = W x[r, :] for r in [0, rows): the batched form of dense_gemv_f32 (one warp per output row per input row).
void dense_gemv_f32_rows(const float* W, const float* X, float* y, int n_in, int n_out, int rows, void* stream);

/// The 4-tap causal conv + SiLU over a chunk: history is [channels, 3] (oldest first, updated in place to the
/// chunk's last three inputs), qkv and h are [T, channels].  Only the SiLU output is written (the decode path keeps
/// the raw conv output in its own buffer; the prompt path does not need it).
void dense_gdn_conv_chunk(float* history, const float* qkv, const float* conv_w, float* h, int channels, int T,
                          void* stream);

/// In-place L2 norm of the q and k heads of every token of a chunk: row (t, r) of the norm lives at
/// h + t*channels + base + r*cols, cols = 128 (the heads of one token are contiguous inside the token's C channels).
/// The same scaling as native_gdn_l2_norm.
void dense_gdn_l2_norm(float* h, int T, int channels, int base, int rows_per_token, int cols, float eps, void* stream);

/// gate[t, h] = softplus(alpha[t, h] + dt[h]) * ssm_a[h];  beta[t, h] = sigmoid(beta[t, h])   (heads = 48)
void dense_gdn_gates(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int heads,
                     int rows, void* stream);

/// The delta-rule recurrence over a chunk, state in registers, block per (value head, 32 state columns), walking the
/// T tokens in order, then the output norm y = rms_norm(o) * gamma * SiLU(z) - SiLU, not the MoE path's sigmoid.
/// h is [T, 2*S*k_heads + S*v_heads], gate/beta [T, v_heads], z and y [T, v_heads * S], state [S, v_heads, S].
void dense_gdn_rec_chunk(float* state, const float* h, const float* gate, const float* beta, const float* z,
                         const float* gamma, float eps, float* y, int k_heads, int v_heads, int T, void* stream);

}  // namespace strata::kernels

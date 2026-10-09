// m4/ops.hpp - CPU reference math of Mistral Small 4 (float32 on the fast path, double where it is cheap).
#pragma once
#include <cstdint>
#include <vector>

namespace m4 {

/// y = x * rsqrt(mean(x^2) + eps) * w   (the checkpoint's RMSNorm: weight multiplies, no "+1")
void rmsnorm(const float* x, const float* w, float eps, int n, float* y);

/// YaRN inverse frequencies for a rotary of `dim` elements (dim/2 values): HF _compute_yarn_parameters with truncate=true.
void yarn_freqs(int dim, double base, double factor, int64_t orig, double beta_fast, double beta_slow, std::vector<double>& f);

/// cos/sin [seqlen][dim/2] from the frequencies (attention_factor is 1 when mscale == mscale_all_dim, which is the case here).
void rope_table(const std::vector<double>& f, int seqlen, std::vector<float>& cos_t, std::vector<float>& sin_t);

/// Rotation of adjacent pairs (2i, 2i+1) of x[0..dim) -- rope_interleave=true.  (HF de-interleaves and rotates halves; the layouts
/// differ by a fixed permutation of q_pe/k_pe, which cancels in q_pe . k_pe.)
void rotary_pairs(float* x, int dim, const float* cos_row, const float* sin_row);

enum class Router { Softmax, Sigmoid };

/// Top-k router for ONE token. Scores per expert; selection by score (ties -> lower index); weights are the selected scores,
/// normalised to sum 1 when `norm` (norm_topk_prob), times `scale` (routed_scaling_factor). No bias: the checkpoint has none.
void route(const float* logits, int n, int topk, Router fn, bool norm, float scale, int32_t* idx, float* w);

/// Llama-4 style position scale: 1 + beta * ln(1 + floor(pos / orig)).  Exactly 1 for pos < orig (8192 in the real model).
double llama4_scale(int64_t pos, double beta, int64_t orig);

/// MLA softmax scale: qk_head_dim^-0.5, times mscale^2 (mscale = 0.1 * mscale_all_dim * ln(factor) + 1) when `with_mscale`.
double mla_softmax_scale(int qk_head, double factor, double mscale_all_dim, bool with_mscale);

inline float silu(float v) { return v / (1.0f + __builtin_expf(-v)); }

}  // namespace m4

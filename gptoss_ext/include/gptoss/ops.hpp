// gptoss/ops.hpp - CPU math of gpt-oss.
#pragma once
#include <cstdint>
#include <vector>

namespace gptoss {

void rmsnorm(const float* x, const float* w, float eps, int n, float* y);

/// YaRN inverse frequencies exactly as OpenAI's reference: low/high from beta_fast/beta_slow, linear ramp, optional floor/ceil (truncate).
void yarn_freqs(int dim, double base, double factor, int64_t orig, double beta_fast, double beta_slow, bool truncate, std::vector<double>& f);
/// cos/sin [seqlen][dim/2], both multiplied by `concentration` (0.1*ln(factor)+1 in the reference; 1 disables it).
void rope_table(const std::vector<double>& f, int seqlen, double concentration, std::vector<float>& c, std::vector<float>& s);
double yarn_concentration(double factor);

/// RoPE on x[0..dim): half-split layout (x[i], x[i+dim/2]) -- what OpenAI's reference and HF use; `pairs` rotates (2i, 2i+1) instead.
void rotary(float* x, int dim, const float* c, const float* s, bool pairs);

/// Top-k by router logit (ties -> lower index), then softmax over ONLY the selected logits.
void route(const float* logits, int n, int topk, int32_t* idx, float* w);

/// gpt-oss SwiGLU: gate <- min(gate, limit); up <- clamp(up, -limit, limit); (up + 1) * gate * sigmoid(alpha * gate).
float swiglu(float gate, float up, float limit, float alpha);

}  // namespace gptoss

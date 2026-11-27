// The auto Host KV Offload + elastic growth + DSpark ctx growth, exactly as the engine computes them,
// fed with the numbers measured on the RTX 4090 (CC 8.9) on 2026-11-25 (models/inkling-2-mini/serve.log):
//   --ctx 532768 --vram-pct 90 --kv fp16 -> CUDA error 2 (out of memory) at inklin2ristrata_runner.cuh:1013
//   --ctx 532768 --vram-pct 90 --kv k8v4 -> engine stopped (exit code -11) at 43,008 of 43,649 prompt tokens
// The crash position is the SAME for both --kv values, so the fault is NOT in --kv: the shared path
// (chunk auto-sizing, init_glob, the K/V grow ledger, the DSpark ctx growth, the position mapping) is what runs.
// The models below are the engine's own arithmetic (inklin2ristrata_cuda_backend.cu lines 203-311,
// inklin2ristrata_runner.cuh lines 1966-2196 and 2458-2636, inklin2ristrata_dspark.cuh lines 1400-1435), no GPU needed.
#include "strata/core/ink_kv_grow.hpp"
#include "strata/core/ink_chunk.hpp"

#include <algorithm>
#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr size_t MiB = 1ull << 20;

// strata::ink constants (inklin2ristrata_cuda.cuh:167-169)
constexpr int kInkKVHeads = 8;
constexpr int kInkHeadDim = 128;
constexpr int kInkKVDim   = kInkKVHeads * kInkHeadDim;   // 1024
constexpr int kInkVisits  = 16;                          // 8 global + 8 SWA visits

// InkCudaRunner::kv_bytes_per_visit_pos (inklin2ristrata_runner.cuh:1966-1970)
size_t per_visit_pos(int kv_fmt) {  // 0=fp16, 1=fp8, 2=k8v8, 3=k8v4
    if (kv_fmt == 0) return (size_t)kInkKVDim * 4u;
    if (kv_fmt == 3) return (size_t)kInkKVDim + ((size_t)kInkKVDim / 2u) + (size_t)(2 * kInkKVHeads * 4);
    return (size_t)kInkKVDim * 2u + (size_t)(2 * kInkKVHeads * 4);
}

// inklin2ristrata_cuda_backend.cu:207-211 init_glob for --serve (kv-offload auto, one slot)
int init_glob(long long total_vram_bytes, int kv_offload_mode, int max_ctx, int slots) {
    if (kv_offload_mode == 1) return 0;
    if (kv_offload_mode == 0 && total_vram_bytes <= (13ll << 30))
        return std::min(std::max(256, max_ctx), 2048);
    return std::max(256, std::min(std::max(256, max_ctx), (slots == 1) ? 8192 : 4096));
}

// inklin2ristrata_cuda_backend.cu:305-306 kv_grow_mib, the amount printed in the chunk line
double kv_grow_mib(int kv_offload_mode, int slots, int max_ctx, int init, size_t per_pos) {
    if (kv_offload_mode == 1) return 0.0;
    return (double)8 * slots * std::max(0, std::max(256, max_ctx) - init) * (double)per_pos / 1048576.0;
}

// inklin2ristrata_dspark.cuh:1400-1402 the DSpark ctx growth target (kv_fmt-independent: its own fp16 KV)
int dspark_ctx_target(int max_ctx, int needed_pos) {
    return std::min(max_ctx, std::max(needed_pos, ((needed_pos + 8191) / 8192) * 8192));
}
}  // namespace

int main() {
    // ---- the log's request: ctx 532768, 1 slot, chunk 512, prompt 43,649 tokens, crash at 43,008 ----
    const int max_ctx = 532768, slots = 1, chunk = 512;
    const int prompt = 43649, crash_at = 43008;

    // ---- per-visit K/V size, all four formats (the sizing the whole path shares) ----
    CHECK(per_visit_pos(0) == 4096u);   // fp16: K+V, 2 bytes each
    CHECK(per_visit_pos(1) == 2112u);   // fp8:  K+V, 1 byte each + 64 B of scales
    CHECK(per_visit_pos(2) == 2112u);   // k8v8: same bytes as fp8
    CHECK(per_visit_pos(3) == 1600u);   // k8v4: K 1 B, V 4-bit, + 64 B of scales

    // ---- init_glob on the 4090 (24 GiB card, auto offload, 1 slot): 8192 ----
    CHECK(init_glob(24ll << 30, 0, max_ctx, 1) == 8192);
    // the small-card rule and the --kv-offload on/off rules, same function
    CHECK(init_glob(12ll << 30, 0, max_ctx, 1) == 2048);
    CHECK(init_glob(24ll << 30, 1, max_ctx, 1) == 0);
    CHECK(init_glob(24ll << 30, -1, max_ctx, 1) == 8192);
    CHECK(init_glob(24ll << 30, 0, max_ctx, 2) == 4096);

    // ---- the chunk line's ledger, exactly as the log printed it ----
    // "[InkCUDA] prefill chunk 512 (auto: 4488 MiB free after the experts, 16393 MiB of K/V still to grow" (fp16)
    // "... 6404 MiB of K/V still to grow" (k8v4)
    const double grow_fp16 = kv_grow_mib(0, slots, max_ctx, 8192, per_visit_pos(0));
    const double grow_k8v4 = kv_grow_mib(0, slots, max_ctx, 8192, per_visit_pos(3));
    CHECK((long long)grow_fp16 == 16393);   // %.0f of 16393.046875
    CHECK((long long)(grow_k8v4 + 0.5) == 6404);
    // the auto chunk the log picked: 512 both times (the K/V still to grow stays free of the chunk)
    CHECK(ink_auto_chunk(4488.0, slots, grow_fp16) == 512);
    CHECK(ink_auto_chunk(4488.0, slots, grow_k8v4) == 512);

    // ---- the grow ledger the SAME for both --kv runs: the steps from 8192 ----
    int cap = 8192;
    for (int i = 0; i < 5; ++i) cap = ink_kv_next_cap(cap, cap + 1, max_ctx);
    CHECK(cap == 49152);   // 8192 -> 16384 -> 24576 -> 32768 -> 40960 -> 49152
    // the crash position falls inside the 40960 -> 49152 step, for BOTH formats:
    CHECK(crash_at > 40960 && crash_at < 49152);
    // the prompt's end forces that step: the chunk starting at 40960 needs 41472 > 40960
    CHECK(prompt > 40960);

    // ---- the DSpark ctx growth (kv_fmt-independent): the same 8K steps, its own guard ----
    CHECK(dspark_ctx_target(max_ctx, crash_at + 1) == 49152);
    CHECK(dspark_ctx_target(max_ctx, 1) == 8192);
    CHECK(dspark_ctx_target(49152, 49153) == 49152);   // never past its own max_ctx
    // inklin2ristrata_cuda_backend.cu:1045/1080 guard: a draft at the prompt's end is refused, not run
    CHECK(crash_at + 12 < max_ctx);   // the guard passes here: the draft path DOES run at 43k

    // ---- the auto Host KV Offload split the log printed, and its invariants ----
    // fp16: 60160 VRAM + 472608 host; k8v4: 138752 VRAM + 394016 host (max_ctx=532768 both)
    const int dev_fp16 = 60160, host_fp16 = 472608;
    const int dev_k8v4 = 138752, host_k8v4 = 394016;
    CHECK(dev_fp16 + host_fp16 == max_ctx);
    CHECK(dev_k8v4 + host_k8v4 == max_ctx);
    CHECK((dev_fp16 & 255) == 0 && (dev_k8v4 & 255) == 0);   // dev_part is 256-aligned (runner:2173)
    // the host MiB the log printed: 14769.0 (fp16), 4809.8 (k8v4)
    CHECK((long long)((double)host_fp16 * 8 * per_visit_pos(0) / 1048576.0 * 10.0 + 0.5) == 147690);
    CHECK((long long)((double)host_k8v4 * 8 * per_visit_pos(3) / 1048576.0 * 10.0 + 0.5) == 48098);
    // the window is 3/4 of what fits, so the smaller the K/V, the deeper the window: k8v4 > fp16
    CHECK(dev_k8v4 > dev_fp16);

    // ---- the position -> (VRAM | host) mapping (inklin2ristrata_attn.cuh:34-35) ----
    // global positions [0..dev_cap-1] live in VRAM, [dev_cap..] in mapped host RAM
    auto in_vram = [](int pos, int dev_cap) { return pos < dev_cap; };
    CHECK(in_vram(crash_at, dev_fp16));
    CHECK(in_vram(crash_at, dev_k8v4));
    CHECK(in_vram(prompt - 1, dev_fp16));
    CHECK(!in_vram(dev_fp16, dev_fp16));
    CHECK(!in_vram(max_ctx - 1, dev_fp16));
    // the crash is BELOW both dev windows: the fault never touched the host tail -> not a --kv fault
    CHECK(crash_at < dev_fp16 && crash_at < dev_k8v4);

    // ---- the chunk boundaries of the crashed request ----
    CHECK(crash_at % chunk == 0);                       // 43,008 = 84 x 512: a chunk edge
    CHECK(prompt - ((prompt / chunk) * chunk) == 129);  // the last chunk is partial: 129 tokens
    CHECK((prompt / chunk) == 85);                      // 85 chunks: 84 whole + 1 partial

    // ---- the host tail rules the spill path uses (ink_kv_grow.hpp) ----
    CHECK(ink_kv_host_tail_full(dev_fp16, max_ctx) == host_fp16);
    CHECK(ink_kv_host_tail_full(dev_k8v4, max_ctx) == host_k8v4);
    CHECK(ink_kv_host_tail_need(dev_fp16, 49152) == 0);   // 49152 already inside the VRAM window

    if (failed) {
        std::fprintf(stderr, "ink_kv_offload_test: %d failed\n", failed);
        return 1;
    }
    std::printf("ink_kv_offload_test: ok\n");
    return 0;
}

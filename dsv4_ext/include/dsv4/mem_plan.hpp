/**
 * @file mem_plan.hpp
 * @brief Expert VRAM residency planning: how many experts live in GPU memory.
 *
 * Pure arithmetic module (no CUDA dependency), fully testable on any machine. Computes the optimal
 * distribution of expert slots across layers given a VRAM budget, then reports the plan as a
 * human-readable summary.
 *
 * @par Algorithm
 *   @code
 *   free = vram_free - reserve
 *   budget_expert = max(0, free - fixed)     // fixed = weights + kv + act + prefill + mtp + scratch
 *   expert_budget = P/100 * budget_expert    // P is a share of what's LEFT, not physical VRAM
 *   @endcode
 *
 * Slots are handed out per layer (never global): layers may have different bytes-per-expert
 * (dynamic quants), so the plan works on real per-layer sizes. When `pct` is negative (auto mode),
 * it allocates 100% if all experts fit, otherwise falls back to 90%.
 *
 * @par Two allocation modes
 *   - **Percentage** (`--expert-vram-pct P`): allocate P% of the expert budget, distribute evenly
 *     across layers, then spend remainder one slot at a time (last layers first).
 *   - **Absolute** (`--expert-cache N`): exactly N total slots, distributed as evenly as possible
 *     with remainder going to last layers. Overrides percentage mode.
 *
 * @par Critical fix: zero fixed costs at load time
 *   Both backends report free VRAM *already net of what this model has put on the card* (`cudaMemGetInfo`
 *   for CUDA, pool counter for emulated). So `plan_expert_memory()` is called with `weights = kv = act =
 *   prefill = mtp = scratch = 0`: charging them again double-counts and used to abort with
 *   `fixed costs (8742 MiB) exceed the usable VRAM (-883 MiB after the reserve)` on a 24 GiB card.
 *
 * @par The host-RAM side of the same budget
 *   The expert tiers are host memory, not device memory, and the machine that hosts them is shared: a
 *   sibling engine already resident, the page cache of the model itself and the desktop all compete for
 *   it. `host_mem_available` is what the kernel reports as still allocatable (`/proc/meminfo`
 *   MemAvailable) and `host_ram_reserve` is what the caller refuses to touch; `host_budget` is the
 *   difference. Nothing here fails when the tiers exceed it: over-subscribing RAM is not an error, it
 *   is a promise the machine breaks by swapping, and a load that swaps for 900 s is what made
 *   `serve/server.py` report `EngineDied: the engine did not report READY within 900 s`. The caller
 *   uses `host_budget` to shrink the arena it can shrink (the LRU cache) and to report the rest.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

/**
 * @struct MemPlanInput
 * @brief Input parameters for expert memory planning.
 *
 * All values in bytes unless noted otherwise. The planner computes `slots_layer[]` from these inputs,
 * distributing slots across layers proportionally to their bytes-per-expert and the available budget.
 */
struct MemPlanInput {
    int64_t vram_total = 0;               ///< Total physical VRAM size in MiB (from device query).
    int64_t vram_free = 0;                ///< Free VRAM after this model's other allocations (MiB).
    int64_t reserve = 1024LL << 20;       ///< Safety reserve: --expert-vram-reserve-mib, default 1 GiB.
    int64_t weights = 0, kv = 0, act = 0, prefill = 0, mtp = 0, scratch = 0;  ///< Fixed costs (bytes).
    int n_expert = 0;                     ///< Total experts per layer (from config: expert_count).
    std::vector<uint64_t> bytes_per_expert;  ///< Per-layer expert size [n_layer]: 0 if no routed experts.
    double pct = -1.0;                    ///< VRAM percentage for experts: 0..100, < 0 = auto mode.
    int64_t abs_slots = -1;               ///< --expert-cache N: >= 0 overrides pct with absolute count.
    uint64_t ram_expert_bytes = 0;        ///< Host RAM copy of all experts (report only, not counted in VRAM).
    int64_t host_mem_available = -1;      ///< Host RAM the machine can still hand out (bytes); < 0 = unknown.
    int64_t host_ram_reserve = 0;         ///< Host RAM the caller refuses to touch (bytes): the sibling engines.
    int64_t ram_cache_bytes = 0;          ///< The LRU RAM arena the caller asked for (bytes); 0 = none.
};

/**
 * @struct MemPlan
 * @brief Result of expert memory planning: slots per layer and budget summary.
 *
 * All fields are populated after `plan_expert_memory()` returns successfully (`ok == true`).
 */
struct MemPlan {
    bool ok = false;                      ///< true if the plan was computed without errors.
    std::string err;                      ///< Error message if ok == false.
    int64_t libera = 0, fixed = 0, budget_expert = 0, expert_budget = 0;  ///< Budget breakdown (bytes).
    double pct_used = 0;                  ///< Percentage of expert budget consumed (for reporting).
    std::vector<int> slots_layer;         ///< Per-layer slot count [n_layer]: <= n_expert.
    int64_t slots_total = 0;              ///< Sum of all slots across layers.
    uint64_t slot_bytes_total = 0;        ///< Total bytes for all slots: sum(slots_layer[l] * bpe[l]).
    bool all_resident = false;            ///< true if every layer holds ALL its experts (fast path).
    uint64_t all_expert_bytes = 0;        ///< Total bytes for ALL experts across all layers.
    int64_t host_budget = 0;              ///< Host RAM the tiers may use: host_mem_available - host_ram_reserve.
    int64_t host_demand = 0;              ///< Host RAM the tiers want: ram_expert_bytes + ram_cache_bytes.
    bool host_over = false;               ///< true if host_demand exceeds host_budget: the load will swap.
};

/**
 * @brief Compute the expert VRAM residency plan from input parameters.
 *
 * Distributes expert slots across layers given a VRAM budget, using either percentage-based or
 * absolute slot allocation mode. Validates that the resulting plan fits within the budget.
 *
 * @par Algorithm steps
 *   1. Calculate usable VRAM: `free - reserve`
 *   2. Subtract fixed costs (weights, KV cache, activations, etc.) → expert budget
 *   3. Apply percentage or absolute slot constraint
 *   4. Distribute slots evenly across active layers, remainder to last layers
 *   5. Validate total bytes ≤ budget_expert
 *
 * @param in  Input parameters (VRAM size, fixed costs, per-layer expert sizes, allocation mode).
 * @return MemPlan with `ok == true` on success, or `ok == false` with error details.
 */
MemPlan plan_expert_memory(const MemPlanInput& in);

/**
 * @brief Host RAM the machine can still hand out, in bytes, or -1 when it cannot be read.
 *
 * MemAvailable from /proc/meminfo, which is the only figure that already accounts for the sibling
 * engines, the page cache and the desktop: MemTotal - used - the reclaimable cache. Linux only;
 * anything else returns -1 and the caller keeps its own default.
 */
int64_t host_mem_available_bytes();

/**
 * @brief Generate a human-readable report of the memory plan.
 *
 * Outputs VRAM totals, fixed costs breakdown, expert budget, slot counts, per-layer range,
 * and RAM expert copy size in a formatted multi-line string suitable for logging or CLI output.
 *
 * @param in  Input parameters (for reporting total/allocated values).
 * @param p   Computed plan (for reporting results).
 * @return Formatted report string with MiB values and slot counts.
 */
std::string plan_report(const MemPlanInput& in, const MemPlan& p);

}  // namespace dsv4

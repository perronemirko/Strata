// tests/test_sampler.cpp - TDD spec for dsv4/sampler.cpp (dsv4/sampler.hpp).
//
// Behavioural specification of the sampling pipeline (llama.cpp sample.cpp semantics):
//   1. penalties first (on raw logits)  2. temperature  3. top-k  4. top-p/min-p cut  5. roulette
//   temperature == 0 -> plain argmax. RNG: xorshift64*, reproducible from a seed; next_uniform()
//   shares the stream with sample() (the speculative acceptance contract).
//
// Built and run by build.sh; needs no model file.

#include "dsv4/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

int main() {
    // ---------------------------------------------------------------- greedy: argmax, no RNG
    {
        const float lg[] = {0.1f, 5.0f, -2.0f, 4.9f};
        SampleOpts so;   // temperature 0
        Sampler smp(so);
        CHECK(smp.sample(lg, 4, {}) == 1);
        // NaN logits: the max walk skips NaN comparisons, first finite stays (documented edge case)
        const float lgn[] = {std::nanf("1"), 1.0f};
        CHECK(smp.sample(lgn, 2, {}) == 0);   // NaN comparisons are false: the walk keeps index 0
    }
    // ---------------------------------------------------------------- determinism: same seed replays
    {
        const float lg[] = {1.0f, 2.0f, 3.0f};
        SampleOpts so; so.temperature = 0.7f; so.seed = 42;
        std::vector<int> hist;
        Sampler a(so), b(so);
        bool same = true;
        for (int i = 0; i < 50; ++i) if (a.sample(lg, 3, hist) != b.sample(lg, 3, hist)) same = false;
        CHECK(same);
    }
    // ---------------------------------------------------------------- top-k excludes the rest
    {
        float lg[10];
        for (int i = 0; i < 10; ++i) lg[i] = (float) i;   // id 9 is the best
        SampleOpts so; so.temperature = 1.0f; so.top_k = 3; so.seed = 7;
        Sampler smp(so);
        bool in_top = true;
        for (int i = 0; i < 200; ++i) if (smp.sample(lg, 10, {}) < 7) in_top = false;
        CHECK(in_top);   // only ids 7,8,9 can ever be picked
    }
    // ---------------------------------------------------------------- top-p nucleus cuts the tail
    {
        float lg[6];
        for (int i = 0; i < 6; ++i) lg[i] = i * 2.0f;   // softmax concentrates on the top entries
        SampleOpts so; so.temperature = 1.0f; so.top_p = 0.9f; so.seed = 11;
        Sampler smp(so);
        // The nucleus of this distribution is {5,4,3}: id 0..2 must essentially never appear.
        int low = 0;
        for (int i = 0; i < 500; ++i) if (smp.sample(lg, 6, {}) < 3) ++low;
        CHECK(low == 0);
    }
    // ---------------------------------------------------------------- min_p relative to the best
    {
        float lg[6];
        for (int i = 0; i < 6; ++i) lg[i] = i * 2.0f;
        SampleOpts so; so.temperature = 1.0f; so.min_p = 0.1f; so.seed = 13;
        Sampler smp(so);
        int low = 0;
        for (int i = 0; i < 500; ++i) if (smp.sample(lg, 6, {}) < 3) ++low;
        CHECK(low == 0);   // p < 0.1 * p_max pruned
    }
    // ---------------------------------------------------------------- penalties change the distribution
    {
        float lg[4] = {1.0f, 1.0f, 1.0f, 1.0f};   // uniform: penalties decide
        const std::vector<int> hist = {0, 0, 0};
        SampleOpts so; so.temperature = 1.0f; so.penalty_present = 5.0f; so.penalty_last_n = 8; so.seed = 3;
        Sampler smp(so);
        int zero_picked = 0;
        for (int i = 0; i < 200; ++i) if (smp.sample(lg, 4, hist) == 0) ++zero_picked;
        CHECK(zero_picked == 0);   // id 0 is penalised -5: never chosen
        // penalty_freq subtracts once per occurrence in the window
        SampleOpts so2; so2.temperature = 1.0f; so2.penalty_freq = 50.0f; so2.penalty_last_n = 8; so2.seed = 3;
        Sampler smp2(so2);
        const float lg2[3] = {2.0f, 2.0f, 2.0f};
        int one_picked = 0;
        for (int i = 0; i < 200; ++i) if (smp2.sample(lg2, 3, {1, 1}) == 1) ++one_picked;
        CHECK(one_picked == 0);   // id 1's logit becomes 2-100: never chosen
        // penalty_repeat divides positive logits (llama.cpp convention): a weaker but visible effect
        SampleOpts so4; so4.temperature = 1.0f; so4.penalty_repeat = 100.0f; so4.penalty_last_n = 8; so4.seed = 5;
        Sampler smp4(so4);
        const float lg4[3] = {5.0f, 5.0f, 5.0f};
        int one4 = 0;
        for (int i = 0; i < 200; ++i) if (smp4.sample(lg4, 3, {1}) == 1) ++one4;
        CHECK(one4 == 0);   // 5/100 = 0.05 vs 5.0: pruned in practice
        // penalty_last_n = 0 disables penalties
        SampleOpts so3; so3.temperature = 1.0f; so3.penalty_present = 5.0f; so3.penalty_last_n = 0; so3.seed = 3;
        Sampler smp3(so3);
        int seen[4] = {};
        for (int i = 0; i < 100; ++i) ++seen[smp3.sample(lg, 4, hist)];
        CHECK(seen[0] > 0);   // no penalty applied: id 0 reachable again
    }
    // ---------------------------------------------------------------- shared RNG stream (the spec contract)
    {
        SampleOpts so; so.temperature = 1.0f; so.seed = 99;
        Sampler smp(so);
        const float lg[3] = {0.f, 0.f, 0.f};
        // sample() and next_uniform() must draw from the SAME stream: interleaving them with a
        // second sampler that only calls next_uniform must produce identical sequences of draws.
        const float a1 = smp.next_uniform();
        const int pick = smp.sample(lg, 3, {});
        const float a2 = smp.next_uniform();
        Sampler ref(so);
        const float b1 = ref.next_uniform();
        const float b2 = ref.next_uniform();
        const float b3 = ref.next_uniform();
        CHECK(a1 == b1);
        CHECK(a2 == b3);   // sample() consumed exactly one draw (uniform over a uniform distribution)
        (int) pick;
    }
    // ---------------------------------------------------------------- edge cases
    {
        SampleOpts so; so.temperature = 1.0f; so.seed = 1;
        Sampler smp(so);
        const float lg[] = {std::nanf("1"), std::nanf("1")};   // softmax is NaN: fall back to idx[0]
        CHECK(smp.sample(lg, 2, {}) == 0);
        CHECK(smp.sample(lg, 0, {}) == 0);   // n <= 0 -> 0
    }

    std::printf("\nsampler tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

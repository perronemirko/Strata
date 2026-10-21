// dsv4 test: KV checkpoint (DSKV0002) + prefix reuse, ported from gptoss_ext.
//
//   python3 tools/tiny_model.py /tmp/dsv4_tiny
//   ./build-out/test_kv_checkpoint --model /tmp/dsv4_tiny/tiny.gguf
//
// What it proves:
//  * save_kv/load_kv round-trip: a fresh engine restored from the file continues decoding with
//    bit-identical logits to the run that saved it (the whole per-layer state is captured).
//  * reuse_prefix(): a prompt sharing its leading tokens skips re-prefilling them, and the result
//    matches a full recompute; a rewind deeper than the window ring is refused.
//  * the fingerprint refuses checkpoints of another model/options (ctx or -k/-v changed).
#include "dsv4/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while (0)

static double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

static int argmax(const std::vector<float>& v) {
    size_t bi = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[bi]) bi = i;
    return (int) bi;
}

struct Run {   // one engine + its token history, decoding greedily
    Model m;
    std::vector<int> toks;
    std::vector<std::vector<float>> logits;   // distribution produced after each processed position
    int pos = 0;
};

static void step(Run& r, int tok) {
    std::vector<float> lg;
    r.m.forward(tok, r.pos, &lg);
    r.m.end_token();
    r.toks.push_back(tok);
    r.logits.push_back(lg);
    ++r.pos;
}

static void init_run(Run& r, const std::string& model, int ctx, KVFormat kf = KVFormat::F32) {
    RunOpts o;
    o.ctx = ctx; o.gpu = false; o.qat_sim = false; o.verbose = false; o.prefill_chunk = 0;
    o.key_format = kf; o.value_format = kf;
    std::string err;
    if (!r.m.load({model}, o, err)) { std::printf("FAIL load: %s\n", err.c_str()); std::exit(1); }
    r.m.reset();
}

int main(int argc, char** argv) {
    std::string model = "/tmp/dsv4_tiny/tiny.gguf";
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], "--model")) model = argv[++i];

    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::vector<int> P = {5, 9, 17, 23, 31, 7, 11, 3};   // the same prompt test_prefill_batch uses
    const int GEN = 6;                                          // greedy continuation length for the checkpoint

    // ------------------------------------------------------------------ T1: save/restore round-trip
    {
        Run a; init_run(a, model, 64);
        for (int t : P) step(a, t);
        std::vector<int> gen;
        for (int i = 0; i < GEN; ++i) { int g = argmax(a.logits.back()); gen.push_back(g); step(a, g); }
        CHECK((int) a.m.cached_tokens().size() == (int) P.size() + GEN);

        std::string err;
        CHECK(a.m.save_kv("/tmp/dsv4_ck_t1.bin", err));
        const std::vector<int> saved_toks = a.toks;           // the history as of the save

        // A keeps going: the reference logits for "what comes next"
        const int nxt = argmax(a.logits.back());
        step(a, nxt);
        const std::vector<float> ref_next = a.logits.back();

        Run b; init_run(b, model, 64);
        CHECK(b.m.load_kv("/tmp/dsv4_ck_t1.bin", err));
        CHECK(b.m.cached_tokens() == saved_toks);             // the whole history came back
        std::vector<float> lg;
        b.m.forward(nxt, (int) P.size() + GEN, &lg);            // continue exactly where A saved
        const double d = max_diff(ref_next, lg);
        CHECK(d < 1e-5);                                        // same state -> same math
        std::printf("T1 round-trip: %zu tokens restored, next-step logits max|diff| = %.2e\n", b.m.cached_tokens().size(), d);

        // T1b: append mode - a second save extends the file instead of rewriting it from scratch.
        // Each appended segment carries its ring rows plus BOTH compressors' rolling-buffer snapshot
        // (O(segment + ratio^2 * hd), not O(file)), so no size threshold is asserted: the proof that
        // this was an append, not a rewrite, is that every previous byte survives verbatim.
        std::vector<char> old_bytes; {
            FILE* f0 = std::fopen("/tmp/dsv4_ck_t1.bin", "rb");
            if (f0) { fseek(f0, 0, SEEK_END); const long sz = ftell(f0); fseek(f0, 0, SEEK_SET); old_bytes.resize((size_t) sz); CHECK(sz > 0 && std::fread(old_bytes.data(), 1, old_bytes.size(), f0) == old_bytes.size()); fclose(f0); }
        }
        CHECK(a.m.save_kv("/tmp/dsv4_ck_t1.bin", err));         // one more token was processed: a new segment appends
        long sz_after = -1; std::vector<char> head(old_bytes.size(), '\xff'); {
            FILE* f0 = std::fopen("/tmp/dsv4_ck_t1.bin", "rb");
            if (f0) { fseek(f0, 0, SEEK_END); sz_after = ftell(f0); fseek(f0, 0, SEEK_SET); CHECK(sz_after >= (long) old_bytes.size() && std::fread(head.data(), 1, head.size(), f0) == head.size()); fclose(f0); }
        }
        constexpr size_t kDskvHead = 8 + 6 * sizeof(uint32_t) + sizeof(uint64_t);   // DskvHead in model.cpp: the append rewrites ONLY this header (nseg/total bump)
        CHECK(sz_after > (long) old_bytes.size());              // grew: a new segment was appended
        CHECK(std::memcmp(head.data(), "DSKV0002", 8) == 0 && std::equal(old_bytes.begin() + kDskvHead, old_bytes.end(), head.begin() + kDskvHead));   // body byte-identical: an append, not a rewrite
        std::printf("T1b append: file grew %zu -> %ld bytes (body preserved, header updated)\n", old_bytes.size(), sz_after);

        Run c; init_run(c, model, 64);
        CHECK(c.m.load_kv("/tmp/dsv4_ck_t1.bin", err));         // the multi-segment file loads back whole
        CHECK((int) c.m.cached_tokens().size() == (int) P.size() + GEN + 1);
    }

    // ------------------------------------------------------------------ T2: prefix reuse
    {
        Run a; init_run(a, model, 64);
        for (size_t i = 0; i < 5; ++i) step(a, P[i]);           // cache the first five prompt tokens
        std::string err;
        CHECK(a.m.save_kv("/tmp/dsv4_ck_t2.bin", err));

        const int V = (int) a.m.tokens().size();              // vocab size of the tiny model
        CHECK(V > 13);                                        // room for the synthetic in-vocab tail tokens below
        const std::vector<int> Q = {P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], 13};   // shared prefix + new tail
        Run d; init_run(d, model, 64);
        CHECK(d.m.load_kv("/tmp/dsv4_ck_t2.bin", err));
        const int k = d.m.reuse_prefix(Q);
        CHECK(k == 5);                                          // five cached tokens are reusable (last prompt token always recomputed)

        std::vector<float> lg;
        for (size_t i = (size_t) k; i < Q.size(); ++i) {        // prefill only the delta, from position k
            d.m.forward(Q[i], (int) i, i + 1 == Q.size() ? &lg : nullptr);
            d.m.end_token();
        }

        Run e2; init_run(e2, model, 64);                           // control: the same prompt computed from scratch
        std::vector<float> ref;
        for (size_t i = 0; i < Q.size(); ++i) { e2.m.forward(Q[i], (int) i, i + 1 == Q.size() ? &ref : nullptr); e2.m.end_token(); }
        const double dmax = max_diff(ref, lg);
        CHECK(dmax < 1e-5);                                     // reuse == full recompute
        std::printf("T2 prefix reuse: k=%d of %zu, delta prefill matches full recompute (max|diff| %.2e)\n", k, Q.size(), dmax);

        // T2b: a rewind deeper than the window ring is refused
        Run f; init_run(f, model, 64);
        CHECK(f.m.load_kv("/tmp/dsv4_ck_t1.bin", err));         // 14 cached tokens (T1's file was re-saved with 15; either way > win)
        const std::vector<int> R = {P[0], P[1], 13};            // shares only two leading tokens (in-vocab)
        CHECK(f.m.reuse_prefix(R) == 0);                        // cannot rewind that far: the rows are gone
    }

    // ------------------------------------------------------------------ T3: fingerprint refusal
    {
        Run a; init_run(a, model, 64);
        for (int t : P) step(a, t);
        std::string err;
        CHECK(a.m.save_kv("/tmp/dsv4_ck_t3.bin", err));

        Run b; init_run(b, model, 32);                            // different --ctx: the math layout differs
        CHECK(!b.m.load_kv("/tmp/dsv4_ck_t3.bin", err));
        std::printf("T3 fingerprint: ctx mismatch refused (%s)\n", err.c_str());

        Run c; init_run(c, model, 64, KVFormat::E4M3);            // different -k/-v: the cache format differs
        CHECK(!c.m.load_kv("/tmp/dsv4_ck_t3.bin", err));
        std::printf("T3 fingerprint: kv-format mismatch refused (%s)\n", err.c_str());

        Run d; init_run(d, model, 64);                            // identical options: accepted
        CHECK(d.m.load_kv("/tmp/dsv4_ck_t3.bin", err));
    }

    // ------------------------------------------------------------------ T4: quantized cache (-k/-v) stays coherent across checkpoint + reuse
    {
        Run a; init_run(a, model, 64, KVFormat::E4M3);            // the cached rows live packed as FP8
        for (int t : P) step(a, t);
        std::vector<int> gen;
        for (int i = 0; i < GEN; ++i) { int g = argmax(a.logits.back()); gen.push_back(g); step(a, g); }
        const int nxt = argmax(a.logits.back());
        std::string err;
        CHECK(a.m.save_kv("/tmp/dsv4_ck_t4.bin", err));

        Run b; init_run(b, model, 64, KVFormat::E4M3);            // same format: the fingerprint matches
        CHECK(b.m.load_kv("/tmp/dsv4_ck_t4.bin", err));
        std::vector<float> lg;
        b.m.forward(nxt, (int) P.size() + GEN, &lg);
        step(a, nxt);                                           // A continues on its own quantized cache
        const double d = max_diff(a.logits.back(), lg);
        CHECK(d < 1e-5);                                        // restore is exact: the checkpoint stores f32 rows
        std::printf("T4 e4m3 cache + checkpoint: next-step logits max|diff| = %.2e\n", d);

        Run c; init_run(c, model, 64, KVFormat::F16);             // a different format must NOT load the file
        CHECK(!c.m.load_kv("/tmp/dsv4_ck_t4.bin", err));
    }

    std::remove("/tmp/dsv4_ck_t1.bin"); std::remove("/tmp/dsv4_ck_t2.bin");
    std::remove("/tmp/dsv4_ck_t3.bin"); std::remove("/tmp/dsv4_ck_t4.bin");
    if (g_fail) { std::printf("test_kv_checkpoint: %d FAILURES\n", g_fail); return 1; }
    std::printf("test_kv_checkpoint: all checks passed\n");
    return 0;
}

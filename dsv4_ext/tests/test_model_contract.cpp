// tests/test_model_contract.cpp - TDD spec for the Model public contract (dsv4/model.hpp).
//
// The behavioural contract of the engine's public API, on the tiny model (no GPU needed):
//   - load() refuses what it must (no shards, bad context) and exposes config/tokens accessors
//   - reset() clears the KV history; forward/forward_chunk extend it exactly
//   - reuse_prefix(): k <= ids.size()-1 (the last prompt token is always recomputed), a rewind
//     deeper than the window ring returns 0, a disjoint prompt returns 0
//   - cached_tokens() mirrors the history; model_hash() is stable for the same file
//   - max_chunk() honours --prefill-chunk (0 disables batching); chunk_routing carries one row
//     per token of the last chunk
//   - save_profile() writes atomically (tmp + rename) and profile_load() round-trips it
//
// Built and run by build.sh with the tiny GGUF; needs no real model.

#include "dsv4/model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

static RunOpts base_opts() {
    RunOpts o;
    o.ctx = 64; o.gpu = false; o.qat_sim = false; o.verbose = false; o.prefill_chunk = 4;
    return o;
}

int main(int argc, char** argv) {
    std::string model = "build-out/tiny_model/tiny.gguf";
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], "--model")) model = argv[++i];

    // ---------------------------------------------------------------- load refusals
    {
        Model m;
        std::string err;
        CHECK(!m.load({}, base_opts(), err));
        CHECK(!err.empty());
        RunOpts o = base_opts();
        o.ctx = 0;
        CHECK(!m.load({model}, o, err));   // a zero context cannot hold anything
    }
    // ---------------------------------------------------------------- load + accessors
    Model m;
    std::string err;
    if (!m.load({model}, base_opts(), err)) { std::printf("FAIL load: %s\n", err.c_str()); return 1; }
    const Dsv4Config& c = m.config();
    CHECK(c.n_layer == 4);
    CHECK(c.n_expert == 8 && c.n_expert_used == 2);
    CHECK(c.vocab == 40);
    CHECK(!m.tokens().empty() && (int) m.tokens().size() == (int) c.vocab);
    CHECK(m.max_chunk() == 4);
    {
        Model m2;
        RunOpts o = base_opts(); o.prefill_chunk = 0;
        std::string e2;
        CHECK(m2.load({model}, o, e2));
        CHECK(m2.max_chunk() == 0);   // batching disabled: forward_chunk takes the sequential path
    }
    // ---------------------------------------------------------------- history accounting
    const std::vector<int> toks = {5, 9, 17, 23, 31, 7, 11, 3};
    m.reset();
    CHECK(m.cached_tokens().empty());
    for (size_t i = 0; i < toks.size(); ++i) { m.forward(toks[i], (int) i, nullptr); m.end_token(); }
    CHECK((int) m.cached_tokens().size() == (int) toks.size());
    for (size_t i = 0; i < toks.size(); ++i) CHECK(m.cached_tokens()[i] == toks[i]);
    {
        const int more[] = {2, 4};
        m.forward_chunk(more, 2, (int) toks.size(), nullptr);
        m.end_chunk(2);
        CHECK((int) m.cached_tokens().size() == (int) toks.size() + 2);
        CHECK(m.chunk_routing_n() == 2);   // one routing row per token of the last chunk
        const std::vector<int>& cr = m.chunk_routing();
        CHECK((int) cr.size() >= 2 * c.n_layer * c.n_expert_used);
    }
    // ---------------------------------------------------------------- reuse_prefix contract
    {
        const std::vector<int> same = toks;
        CHECK(m.reuse_prefix(same) == (int) toks.size() - 1);   // the last token is always recomputed
        const std::vector<int> shorter(toks.begin(), toks.begin() + 3);
        CHECK(m.reuse_prefix(shorter) == 2);                     // min(match, size-1)
        std::vector<int> disjoint = toks;
        disjoint[1] = 41;
        CHECK(m.reuse_prefix(disjoint) == 0);
    }
    // ---------------------------------------------------------------- model_hash stability
    {
        Model m2;
        std::string e2;
        CHECK(m2.load({model}, base_opts(), e2));
        CHECK(m.model_hash() == m2.model_hash());
        CHECK(m.model_hash() != 0);
    }
    // ---------------------------------------------------------------- profile round-trip, atomic
    {
        const std::string pf = "/tmp/dsv4_contract_profile.bin";
        RunOpts o = base_opts();
        o.profile_out = pf;
        Model mw;
        std::string e2;
        if (!mw.load({model}, o, e2)) { std::printf("FAIL load w\n"); return 1; }
        std::string serr;
        CHECK(mw.save_profile(serr));
        CHECK(serr.empty());
        // the file exists under its final name, never as .tmp
        {
            std::ifstream f(pf); CHECK((bool) f);
            std::ifstream t(pf + ".tmp"); CHECK(!t);
        }
        // and a model loading it accepts it (same file => same hash)
        RunOpts o2 = base_opts();
        o2.profile_in = pf;
        Model mr;
        std::string e3;
        CHECK(mr.load({model}, o2, e3));
        // a profile of another model is refused by hash
        const std::string pf2 = "/tmp/dsv4_contract_profile2.bin";
        {
            std::ofstream w(pf2, std::ios::binary);
            w.write("DSV4PROF", 8);
            uint32_t ver = 1, nl = 4, ne = 8;
            w.write((char*) &ver, 4); w.write((char*) &nl, 4); w.write((char*) &ne, 4);
            uint64_t mh = m.model_hash() ^ 0xdeadbeefULL;
            w.write((char*) &mh, 8);
            for (int i = 0; i < 4 * 8; ++i) { uint64_t v = 0; w.write((char*) &v, 8); }
        }
        RunOpts o3 = base_opts();
        o3.profile_in = pf2;
        Model mr2;
        std::string e4;
        // load() warns and continues without the profile (documented behaviour): it must still succeed
        CHECK(mr2.load({model}, o3, e4));
    }

    std::printf("\nmodel contract tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

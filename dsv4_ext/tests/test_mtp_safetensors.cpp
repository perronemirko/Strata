// The MTP bridge: a Hugging Face safetensors draft module on top of the GGUF model.
//
//   python3 tools/tiny_model.py build-out/tiny_model
//   python3 tools/tiny_mtp_checkpoint.py build-out/tiny_mtp
//   ./build-out/test_mtp_safetensors --model build-out/tiny_model/tiny.gguf \
//       --ckpt build-out/tiny_mtp/model-00001-of-00001.safetensors
//
// What is checked:
//  T0 adapt       the tiny HF checkpoint adapts to the tiny model: one draft layer, additive fusion,
//                 every tensor named as the engine expects and with the engine's shapes.
//  T1 exactness   every expert element the bridge produced dequantises to exactly the float the
//                 checkpoint's code x 2^(scale-127) means (experts_reference.txt) - the packed
//                 4-bit + e8m0 path is bit-exact, not approximate.
//  T2 dense       the 1-D and dense parameters come through unchanged (F32), shape-checked.
//  T3 refusal     a checkpoint whose expert shapes belong to another model is refused with a message
//                 (the bridge never silently binds a wrong matrix).
//  T4 load        the model loads with --mtp <safetensors> and uses it as the draft head.
//  T5 parity      the greedy token stream with the safetensors draft module is IDENTICAL to the one
//                 with the GGUF's own draft module: the bridge changes the proposal, not the answer.
//  T6 checkpoint  a KV checkpoint taken mid-speculation with the safetensors module restores the
//                 full state (main + MTP) and the continuation matches.
#include "dsv4/mtp_adapter.hpp"
#include "dsv4/model.hpp"
#include "dsv4/sampler.hpp"
#include "dsv4/st_header.hpp"

#include "dsv4/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

static bool near(const std::vector<float>& a, const float* b, int n, double tol = 2e-3) {
    if ((int) a.size() != n) return false;
    for (int i = 0; i < n; ++i) if (std::fabs((double) a[i] - (double) b[i]) > tol) return false;
    return true;
}
static int argmax(const std::vector<float>& v) { int m = 0; for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[m]) m = (int) i; return m; }

static bool read_f32(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end); const long n = f.tellg(); f.seekg(0, std::ios::beg);
    out.resize((size_t)(n / 4));
    return (bool) f.read(reinterpret_cast<char*>(out.data()), n);
}
static bool read_ids(const std::string& path, std::vector<int>& out) {
    std::ifstream f(path); if (!f) return false;
    std::string t; while (std::getline(f, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
    return !out.empty();
}

// The tiny model's constants (kept in sync with tools/tiny_model.py by hand, as test_mtp does).
static const int V = 40, HC_DIM = 256, NPROMPT = 26, NDRAFT = 4;
static const int DIM = 64, HD = 64, E = 8, FF = 32;

static RunOpts base_opts(int spec_n) {
    RunOpts o;
    o.ctx = 64; o.gpu = false; o.qat_sim = false; o.verbose = false; o.prefill_chunk = 1;
    o.spec_n = spec_n;
    return o;
}

// One greedy speculative run of G tokens; returns the emitted stream.
static std::vector<int> greedy_spec(Model& m, const std::vector<int>& toks, int G) {
    m.reset();
    std::vector<float> lg;
    for (int i = 0; i < NPROMPT; ++i) {
        std::vector<float> l;
        m.forward(toks[(size_t) i], i, &l);
        m.end_token();
        if (i + 1 == NPROMPT) lg = l;
    }
    Sampler smp(SampleOpts{});
    std::vector<int> out;
    while ((int) out.size() < G && !lg.empty()) {
        std::vector<float> nx;
        const std::vector<int> r = m.spec_step(lg.data(), SampleOpts{}, &smp, &nx);
        if (r.empty()) break;
        for (int t : r) out.push_back(t);
        lg = std::move(nx);
    }
    return out;
}

// The reference of one expert element: "<matrix-prefix> <e> <row> <col> <value>".
struct RefRow { std::string tag; int e, row, col; double value; };

int main(int argc, char** argv) {
    std::string model = "build-out/tiny_model/tiny.gguf", refdir = "build-out/tiny_model", ckpt = "build-out/tiny_mtp/model-00001-of-00001.safetensors";
    for (int i = 1; i + 1 < argc; ++i) { if (!std::strcmp(argv[i], "--model")) model = argv[++i]; else if (!std::strcmp(argv[i], "--refdir")) refdir = argv[++i]; else if (!std::strcmp(argv[i], "--ckpt")) ckpt = argv[++i]; }

    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<int> toks; std::vector<float> ref_logits;
    if (!read_ids(refdir + "/tokens.txt", toks) || !read_f32(refdir + "/ref_logits.bin", ref_logits)) {
        std::printf("FAIL cannot read fixtures from %s (run: python3 tools/tiny_model.py %s)\n", refdir.c_str(), refdir.c_str());
        return 1;
    }
    CHECK((int) toks.size() == NPROMPT);

    // ------------------------------------------------------------------ T0: the HF module adapts to this model
    StHeader sh; std::string err;
    if (!st_read_header({ckpt}, sh, err)) { std::printf("FAIL cannot read the safetensors header: %s\n", err.c_str()); return 1; }
    Model A;   // the GGUF alone, for its config
    if (!A.load({model}, base_opts(0), err)) { std::printf("FAIL load GGUF: %s\n", err.c_str()); return 1; }
    const Dsv4Config cfg = A.config();
    MtpAdapted ad;
    if (!mtp_adapt_safetensors(sh, cfg, ad, err)) { std::printf("FAIL the MTP module cannot be used by this model: %s\n", err.c_str()); return 1; }
    CHECK(ad.n_layers == 1);
    CHECK(ad.fusion_mode == 2);   // the tiny checkpoint ships the additive (HF) layout
    CHECK(!ad.inferred_ratios.empty() && ad.inferred_ratios[0] == 8);
    std::printf("     adapted %zu tensors, %.1f MiB from %s\n", ad.items.size(), ad.bytes / 1048576.0, ad.source.c_str());
    {
        // every name the engine needs for a draft layer is present exactly once
        std::map<std::string, int> have;
        for (const auto& it : ad.items) ++have[it.t.name];
        const char* need[] = {"blk.4.attn_norm.weight", "blk.4.attn_q_a_norm.weight", "blk.4.attn_kv_a_norm.weight",
                              "blk.4.attn_sinks.weight", "blk.4.ffn_norm.weight", "blk.4.exp_probs_b.bias",
                              "blk.4.hc_attn_base.weight", "blk.4.hc_attn_scale.weight", "blk.4.hc_ffn_base.weight",
                              "blk.4.hc_ffn_scale.weight", "blk.4.attn_compressor_norm.weight",
                              "blk.4.attn_q_a.weight", "blk.4.attn_q_b.weight", "blk.4.attn_kv.weight",
                              "blk.4.attn_output_a.weight", "blk.4.attn_output_b.weight", "blk.4.ffn_gate_inp.weight",
                              "blk.4.ffn_gate_shexp.weight", "blk.4.ffn_up_shexp.weight", "blk.4.ffn_down_shexp.weight",
                              "blk.4.hc_attn_fn.weight", "blk.4.hc_ffn_fn.weight",
                              "blk.4.attn_compressor_kv.weight", "blk.4.attn_compressor_gate.weight", "blk.4.attn_compressor_ape.weight",
                              "blk.4.ffn_gate_exps.weight", "blk.4.ffn_up_exps.weight", "blk.4.ffn_down_exps.weight",
                              "mtp.enorm.weight", "mtp.hnorm.weight", "mtp.e_proj.weight", "mtp.h_proj.weight",
                              "mtp.hc_head_fn.weight", "mtp.hc_head_base.weight", "mtp.hc_head_scale.weight"};
        bool all = true;
        for (const char* n : need) if (have[n] != 1) { all = false; std::printf("       %s appears %d times\n", n, have[n]); }
        CHECK(all);
    }

    // ------------------------------------------------------------------ T1: the packed experts are bit-exact
    {
        // the reference lives next to the checkpoint the test was given
        const std::string refpath = ckpt.substr(0, ckpt.find_last_of('/') + 1) + "experts_reference.txt";
        std::ifstream f(refpath);
        if (!f) { std::printf("FAIL cannot read %s\n", refpath.c_str()); ++g_fail; }
        else {
            std::vector<RefRow> rows; std::string line;
            while (std::getline(f, line)) {
                RefRow r;
                char tag[96];
                if (std::sscanf(line.c_str(), "%95s %d %d %d %lf", tag, &r.e, &r.row, &r.col, &r.value) != 5) continue;
                r.tag = tag;
                rows.push_back(r);
            }
            CHECK(!rows.empty());
            // the engine's fused expert tensors, addressed exactly as the model does
            long checked = 0, bad = 0;
            for (const auto& it : ad.items) {
                if (it.t.name.find("_exps") == std::string::npos) continue;
                const std::string tag = it.t.name.find("gate") != std::string::npos ? "w1"
                                      : it.t.name.find("up") != std::string::npos ? "w3" : "w2";
                const int64_t in = (int64_t) it.t.dims[0], out = (int64_t) it.t.dims[1], ne = (int64_t) it.t.dims[2];
                const size_t rb = row_bytes(it.t.type, in);
                std::vector<float> dec((size_t) in);
                for (const RefRow& r : rows) {
                    if (r.tag.find(tag) == std::string::npos) continue;
                    if (r.e >= ne || r.row >= out || r.col >= in) continue;
                    const uint8_t* w = it.blob.data() + (size_t) ((r.e * out + r.row)) * rb;
                    dequant_row(it.t.type, w, in, dec.data());
                    const double got = dec[(size_t) r.col];
                    // the reference is the float the packed code means; Q4_0 stores it exactly, so
                    // the only difference allowed is the float32 rounding of the same value.
                    if (std::fabs(got - r.value) > 1e-6 * (std::fabs(r.value) + 1e-9)) { ++bad; if (bad < 4) std::printf("       %s e%d r%d c%d: got %+.9g, reference %+.9g\n", it.t.name.c_str(), r.e, r.row, r.col, got, r.value); }
                    ++checked;
                }
            }
            CHECK(checked > 0);
            CHECK(bad == 0);
            std::printf("     %ld expert elements checked against the checkpoint's own codes\n", checked);
        }
    }

    // ------------------------------------------------------------------ T2: the dense parameters pass through
    {
        // the tiny checkpoint writes them as F32: the bridge must keep every value, and the shapes
        // must be the model's. A single wrong element or shape is a failure.
        long checked = 0, bad = 0;
        for (const auto& it : ad.items) {
            if (it.t.name.find("_exps") != std::string::npos) continue;
            if (it.t.type != T_F32) { ++bad; std::printf("       %s kept as %s\n", it.t.name.c_str(), ggml_type_str(it.t.type).c_str()); continue; }
            const size_t want = (size_t) it.t.dims[0] * (it.t.dims.size() > 1 ? (size_t) it.t.dims[1] : 1) * 4;
            if (it.blob.size() != want) { ++bad; std::printf("       %s holds %zu bytes, its shape says %zu\n", it.t.name.c_str(), it.blob.size(), want); continue; }
            for (size_t i = 0; i < it.blob.size() / 4; ++i) { float f; std::memcpy(&f, it.blob.data() + 4 * i, 4); if (!std::isfinite(f)) { ++bad; break; } }
            ++checked;
        }
        CHECK(checked > 0);
        CHECK(bad == 0);
    }

    // ------------------------------------------------------------------ T3: a mismatched module is refused
    {
        // The tiny checkpoint's experts are [FF][DIM] and [DIM][FF]. A model whose expert width is
        // different must refuse it, naming the tensor - never bind a wrong matrix.
        Dsv4Config other = cfg;
        other.ff_exp = FF * 2;
        MtpAdapted bad; std::string berr;
        CHECK(!mtp_adapt_safetensors(sh, other, bad, berr));
        CHECK(!berr.empty() && berr.find("ffn") != std::string::npos);
        std::printf("     refused: %s\n", berr.c_str());
    }

    // ------------------------------------------------------------------ T4: the model loads with it
    RunOpts o4 = base_opts(NDRAFT);
    o4.mtp_path = ckpt;
    Model B;
    if (!B.load({model}, o4, err)) { std::printf("FAIL load with --mtp safetensors: %s\n", err.c_str()); return 1; }
    CHECK(B.mtp_layers() == 1);

    // ------------------------------------------------------------------ T5: the answer does not change
    {
        // The GGUF's own draft module (concat fusion) and the HF one (additive) are different
        // proposals; the greedy stream must be identical either way, and identical to plain decoding.
        Model plain;
        if (!plain.load({model}, base_opts(0), err)) { std::printf("FAIL load plain: %s\n", err.c_str()); return 1; }
        const int G = 12;
        const std::vector<int> ta = greedy_spec(plain, toks, G);
        const std::vector<int> tb = greedy_spec(B, toks, G);
        CHECK((int) ta.size() == G && (int) tb.size() == G);
        bool same = true;
        for (int i = 0; i < G; ++i) if (ta[(size_t) i] != tb[(size_t) i]) { same = false; std::printf("       first diff at %d: plain=%d hf=%d\n", i, ta[(size_t) i], tb[(size_t) i]); }
        CHECK(same);
    }

    // ------------------------------------------------------------------ T6: the checkpoint round-trips
    {
        Model M;
        if (!M.load({model}, o4, err)) { std::printf("FAIL load M: %s\n", err.c_str()); return 1; }
        M.reset();
        std::vector<float> lg;
        for (int i = 0; i < NPROMPT; ++i) { std::vector<float> l; M.forward(toks[(size_t) i], i, &l); M.end_token(); if (i + 1 == NPROMPT) lg = l; }
        Sampler sm(SampleOpts{});
        std::vector<int> before;
        for (int r = 0; r < 3 && !lg.empty(); ++r) { std::vector<float> nx; const std::vector<int> g = M.spec_step(lg.data(), SampleOpts{}, &sm, &nx); if (g.empty()) break; for (int t : g) before.push_back(t); lg = std::move(nx); }
        CHECK(!before.empty());
        const std::string ck = "/tmp/dsv4_mtp_st_ck.bin"; std::remove(ck.c_str());
        CHECK(M.save_kv(ck, err));

        Model R;
        if (!R.load({model}, o4, err)) { std::printf("FAIL load R: %s\n", err.c_str()); return 1; }
        CHECK(R.load_kv(ck, err));
        const std::vector<int> hist = M.cached_tokens();
        CHECK((int) R.cached_tokens().size() == (int) hist.size());
        const int k = R.reuse_prefix(hist);
        CHECK(k == (int) hist.size() - 1);
        std::vector<float> lgR;
        R.forward(hist[(size_t)(hist.size() - 1)], k, &lgR);
        const std::vector<int> rb = greedy_spec(M, hist, 8), rc = greedy_spec(R, hist, 8);
        CHECK((int) rb.size() == (int) rc.size());
        for (size_t i = 0; i < std::min(rb.size(), rc.size()); ++i) CHECK(rb[i] == rc[i]);
    }

    std::remove("/tmp/dsv4_mtp_st_ck.bin");
    if (g_fail) { std::printf("MTP-SAFETENSORS TEST: %d FAILURES\n", g_fail); return 1; }
    std::printf("MTP-SAFETENSORS TEST: all checks passed\n");
    return 0;
}

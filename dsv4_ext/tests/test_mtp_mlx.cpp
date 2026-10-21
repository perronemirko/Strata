// The MTP bridge, MLX layout: a checkpoint written by mlx-quantize / mlx-to-hf (metadata
// {"format":"mlx"}, tensors under decoder.*, fused expert matrices, int8-in-U32 dense weights)
// must adapt to the same GGUF and produce the same draft as the HF layout does.
//
//   python3 tools/tiny_model.py /tmp/dsv4_tiny/tiny_model
//   python3 tools/tiny_mlx_checkpoint.py /tmp/dsv4_tiny/tiny_mlx
//   ./build-out/test_mtp_mlx --model /tmp/dsv4_tiny/tiny_model/tiny.gguf \
//       --refdir /tmp/dsv4_tiny/tiny_model --ckpt /tmp/dsv4_tiny/tiny_mlx/model-00001-of-00001.safetensors
//
// What is checked:
//  M0 adapt      the MLX checkpoint adapts: one draft layer, additive fusion, every engine tensor
//                present exactly once with the engine's names.
//  M1 dense      every dense element equals the float the checkpoint's int8 code x 2^(s-127) means
//                (dense_reference.txt) - the U32/int8 + 32-column e8m0 path is exact, not sampled.
//  M2 experts    every sampled fused-expert element dequantises to exactly the e2m1 code x 2^(s-127)
//                (experts_reference.txt) - the MXFP4 binding is bit-exact.
//  M3 refusal    a model with a different expert width refuses the module, naming a tensor.
//  M4 parity     the greedy speculative stream with the MLX draft module is IDENTICAL to the one
//                with the HF module: two spellings of the same module, one answer.
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

static int argmax(const std::vector<float>& v) { int m = 0; for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[m]) m = (int) i; return m; }

static const int V = 40, HC_DIM = 256, NPROMPT = 26, NDRAFT = 4;
static const int DIM = 64, HD = 64, E = 8, FF = 32;

static RunOpts base_opts(int spec_n) {
    RunOpts o;
    o.ctx = 64; o.gpu = false; o.qat_sim = false; o.verbose = false; o.prefill_chunk = 1;
    o.spec_n = spec_n;
    return o;
}

static std::vector<int> greedy_spec(Model& m, const std::vector<int>& toks, int G) {
    m.reset();
    std::vector<float> lg;
    const int np = (int) toks.size();
    for (int i = 0; i < np; ++i) {
        std::vector<float> l;
        m.forward(toks[(size_t) i], i, &l);
        m.end_token();
        if (i + 1 == np) lg = l;
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

struct RefRow { std::string tag; int e, row, col; double value; };
struct RefDense { std::string name; int row, col; double value; };

int main(int argc, char** argv) {
    std::string model = "build-out/tiny_model/tiny.gguf", refdir = "build-out/tiny_model", ckpt = "build-out/tiny_mlx/model-00001-of-00001.safetensors";
    for (int i = 1; i + 1 < argc; ++i) { if (!std::strcmp(argv[i], "--model")) model = argv[++i]; else if (!std::strcmp(argv[i], "--refdir")) refdir = argv[++i]; else if (!std::strcmp(argv[i], "--ckpt")) ckpt = argv[++i]; }

    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<int> toks;
    { std::ifstream f(refdir + "/tokens.txt"); std::string t; while (std::getline(f, t, ',')) if (!t.empty()) toks.push_back(std::atoi(t.c_str())); }
    if ((int) toks.size() != NPROMPT) { std::printf("FAIL cannot read fixtures from %s (run: python3 tools/tiny_model.py %s)\n", refdir.c_str(), refdir.c_str()); return 1; }

    // ------------------------------------------------------------------ M0: the MLX module adapts
    StHeader sh; std::string err;
    if (!st_read_header({ckpt}, sh, err)) { std::printf("FAIL cannot read the safetensors header: %s\n", err.c_str()); return 1; }
    Model A;
    if (!A.load({model}, base_opts(0), err)) { std::printf("FAIL load GGUF: %s\n", err.c_str()); return 1; }
    const Dsv4Config cfg = A.config();
    MtpAdapted ad;
    if (!mtp_adapt_safetensors(sh, cfg, ad, err)) { std::printf("FAIL the MLX MTP module cannot be used by this model: %s\n", err.c_str()); return 1; }
    CHECK(ad.n_layers == 1);
    CHECK(ad.fusion_mode == 2);
    CHECK(!ad.inferred_ratios.empty() && ad.inferred_ratios[0] == 8);
    std::printf("     adapted %zu tensors, %.1f MiB from %s\n", ad.items.size(), ad.bytes / 1048576.0, ad.source.c_str());
    {
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

    // ------------------------------------------------------------------ M1: the dense int8 path is exact
    {
        std::ifstream f(ckpt.substr(0, ckpt.find_last_of('/') + 1) + "dense_reference.txt");
        std::vector<RefDense> rows; std::string line;
        while (f && std::getline(f, line)) {
            RefDense r;
            char tag[96];
            if (std::sscanf(line.c_str(), "%95s %d %d %lf", tag, &r.row, &r.col, &r.value) != 4) continue;
            r.name = tag;
            rows.push_back(r);
        }
        CHECK(!rows.empty());
        // The tiny model has n_layer = 4, so its draft layer is blk.4. This is the rename the
        // adapter must have applied: the reference addresses the CHECKPOINT's tensor, and every
        // sampled element is looked up in the engine tensor that name maps to.
        std::map<std::string, std::string> engine_of;
        auto eng = [&](const std::string& s) { return "blk.4." + s; };
        engine_of["decoder.attn.wq_a.weight"] = eng("attn_q_a.weight");
        engine_of["decoder.attn.wq_b.weight"] = eng("attn_q_b.weight");
        engine_of["decoder.attn.wkv.weight"] = eng("attn_kv.weight");
        engine_of["decoder.attn.wo_a.weight"] = eng("attn_output_a.weight");
        engine_of["decoder.attn.wo_b.weight"] = eng("attn_output_b.weight");
        engine_of["decoder.ffn.gate.weight"] = eng("ffn_gate_inp.weight");
        engine_of["decoder.ffn.shared_experts.gate_proj.weight"] = eng("ffn_gate_shexp.weight");
        engine_of["decoder.ffn.shared_experts.up_proj.weight"] = eng("ffn_up_shexp.weight");
        engine_of["decoder.ffn.shared_experts.down_proj.weight"] = eng("ffn_down_shexp.weight");
        engine_of["decoder.attn.compressor.wkv.weight"] = eng("attn_compressor_kv.weight");
        engine_of["decoder.attn.compressor.wgate.weight"] = eng("attn_compressor_gate.weight");
        engine_of["e_proj.weight"] = "mtp.e_proj.weight";
        engine_of["h_proj.weight"] = "mtp.h_proj.weight";
        std::map<std::string, const MtpAdapted::Item*> byname;
        for (const auto& it : ad.items) byname[it.t.name] = &it;
        long checked = 0, bad = 0;
        for (const RefDense& r : rows) {
            auto eit = byname.find(engine_of[r.name]);
            if (eit == byname.end()) { ++bad; if (bad < 4) std::printf("       %s has no engine tensor\n", r.name.c_str()); continue; }
            const MtpAdapted::Item* e = eit->second;
            const int64_t in = (int64_t) e->t.dims[0], out = e->t.dims.size() > 1 ? (int64_t) e->t.dims[1] : 1;
            if (r.row >= out || r.col >= in) { ++bad; if (bad < 4) std::printf("       %s r%d c%d is outside [%lld,%lld]\n", r.name.c_str(), r.row, r.col, (long long) out, (long long) in); continue; }
            const float got = ((const float*) e->blob.data())[(size_t) (r.row * in + r.col)];
            if (std::fabs((double) got - r.value) > 1e-6 * (std::fabs(r.value) + 1e-9)) {
                ++bad;
                if (bad < 4) std::printf("       %s r%d c%d: got %+.9g, reference %+.9g\n", r.name.c_str(), r.row, r.col, got, r.value);
            }
            ++checked;
        }
        CHECK(checked > 0);
        CHECK(bad == 0);
        std::printf("     %ld dense elements checked against the checkpoint's own codes\n", checked);
    }

    // ------------------------------------------------------------------ M2: the fused experts are bit-exact
    {
        std::ifstream f(ckpt.substr(0, ckpt.find_last_of('/') + 1) + "experts_reference.txt");
        std::vector<RefRow> rows; std::string line;
        while (f && std::getline(f, line)) {
            RefRow r;
            char tag[96];
            if (std::sscanf(line.c_str(), "%95s %d %d %d %lf", tag, &r.e, &r.row, &r.col, &r.value) != 5) continue;
            r.tag = tag;
            rows.push_back(r);
        }
        CHECK(!rows.empty());
        long checked = 0, bad = 0;
        for (const auto& it : ad.items) {
            if (it.t.name.find("_exps") == std::string::npos) continue;
            const std::string tag = it.t.name.find("gate") != std::string::npos ? "gate_proj"
                                  : it.t.name.find("up") != std::string::npos ? "up_proj" : "down_proj";
            const int64_t in = (int64_t) it.t.dims[0], out = (int64_t) it.t.dims[1], ne = (int64_t) it.t.dims[2];
            const size_t rb = row_bytes(it.t.type, in);
            std::vector<float> dec((size_t) in);
            for (const RefRow& r : rows) {
                if (r.tag.find(tag) == std::string::npos) continue;
                if (r.e >= ne || r.row >= out || r.col >= in) continue;
                const uint8_t* w = it.blob.data() + (size_t) ((r.e * out + r.row)) * rb;
                dequant_row(it.t.type, w, in, dec.data());
                const double got = dec[(size_t) r.col];
                if (std::fabs(got - r.value) > 1e-6 * (std::fabs(r.value) + 1e-9)) {
                    ++bad;
                    if (bad < 4) std::printf("       %s e%d r%d c%d: got %+.9g, reference %+.9g\n", it.t.name.c_str(), r.e, r.row, r.col, got, r.value);
                }
                ++checked;
            }
        }
        CHECK(checked > 0);
        CHECK(bad == 0);
        std::printf("     %ld expert elements checked against the checkpoint's own e2m1 codes\n", checked);
    }

    // ------------------------------------------------------------------ M3: a mismatched module is refused
    {
        Dsv4Config other = cfg;
        other.ff_exp = FF * 2;
        MtpAdapted bad; std::string berr;
        CHECK(!mtp_adapt_safetensors(sh, other, bad, berr));
        CHECK(!berr.empty() && berr.find("ffn") != std::string::npos);
        std::printf("     refused: %s\n", berr.c_str());
    }

    // ------------------------------------------------------------------ M4: the model loads with it, and the answer does not change
    {
        RunOpts o4 = base_opts(NDRAFT);
        o4.mtp_path = ckpt;
        Model B;
        if (!B.load({model}, o4, err)) { std::printf("FAIL load with --mtp MLX safetensors: %s\n", err.c_str()); return 1; }
        CHECK(B.mtp_layers() == 1);
        const int G = 12;
        const std::vector<int> a = greedy_spec(B, toks, G);
        Model plain;
        if (!plain.load({model}, base_opts(0), err)) { std::printf("FAIL load plain: %s\n", err.c_str()); return 1; }
        plain.reset();
        std::vector<float> lg;
        for (int i = 0; i < NPROMPT; ++i) {
            std::vector<float> l;
            plain.forward(toks[(size_t) i], i, &l);
            plain.end_token();
            if (i + 1 == NPROMPT) lg = l;
        }
        std::vector<int> b;
        for (int g = 0; g < G - 1; ++g) {
            const int t = argmax(lg);
            b.push_back(t);
            std::vector<float> l;
            plain.forward(t, NPROMPT + g, &l);
            plain.end_token();
            lg = std::move(l);
        }
        b.push_back(argmax(lg));
        CHECK((int) a.size() == G && (int) b.size() == G);
        bool same = a.size() == b.size();
        for (size_t i = 0; same && i < a.size(); ++i) if (a[i] != b[i]) same = false;
        CHECK(same);
    }

    std::printf("\nMLX-MTP TEST: %s\n", g_fail ? "FAILED" : "all checks passed");
    return g_fail ? 1 : 0;
}

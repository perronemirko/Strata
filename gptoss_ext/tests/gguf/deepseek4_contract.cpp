// deepseek4_contract: verifica che un modello DeepSeek-V4 (shard GGUF) abbia tutti i tensori che il forward usa,
// con le forme attese dal codice di riferimento (load_arch_tensors di deepseek4.cpp).
// Uso: deepseek4_contract shard1.gguf shard2.gguf shard3.gguf
#include <cstdio>
#include <string>
#include <vector>
#include "gptoss/gguf_model.hpp"

namespace {

constexpr int kN_LAYER = 43;          // layer trunk (il nextn/MTP non è verificato qui)
constexpr uint64_t kN_EMBD = 4096, kN_VOCAB = 129280, kN_HEAD_DIM = 512, kN_HEAD = 64;
constexpr uint64_t kQ_LORA = 1024, kHC = 4, kHC_DIM = 16384, kHC_MIX = 24;
constexpr uint64_t kN_EXPERT = 256, kN_EXPERT_USED = 6, kN_FF_EXP = 2048;
constexpr uint64_t kO_GROUPS = 8, kO_LORA = 1024, kIDX_HEAD = 128, kIDX_N_HEAD = 64;
constexpr int kTYPE_F32 = 0, kTYPE_I32 = 26;

struct Ctx {
    const gptoss::GgufModel& m;
    int problems = 0;
};

std::string dims_str(const std::vector<uint64_t>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); ++i) s += (i ? " " : "") + std::to_string(d[i]);
    return s + "]";
}

void need(Ctx& c, const std::string& name, std::vector<uint64_t> dims, int type = -1) {
    const gptoss::GgufRef* r = c.m.find(name);
    if (!r) { std::printf("MANCA      %s\n", name.c_str()); c.problems++; return; }
    if (r->t->dims != dims) {
        std::printf("FORMA      %s: attesa %s, trovata %s\n", name.c_str(), dims_str(dims).c_str(), dims_str(r->t->dims).c_str());
        c.problems++;
    }
    if (type >= 0 && (int) r->t->type != type) {
        std::printf("TIPO       %s: atteso id %d, trovato id %u\n", name.c_str(), type, r->t->type);
        c.problems++;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("uso: deepseek4_contract shard1.gguf [shard2.gguf ...]\n"); return 2; }
    std::vector<std::string> paths(argv + 1, argv + argc);
    gptoss::GgufModel m(paths);
    Ctx c{m};
    std::printf("tensori nel modello: %zu (dichiarati: %llu)\n", m.size(), (unsigned long long) m.declared());

    // globali
    need(c, "token_embd.weight", {kN_EMBD, kN_VOCAB});
    need(c, "output_norm.weight", {kN_EMBD});
    need(c, "output.weight", {kN_EMBD, kN_VOCAB});
    need(c, "output_hc_fn.weight", {kHC_DIM, kHC});
    need(c, "output_hc_base.weight", {kHC});
    need(c, "output_hc_scale.weight", {1});

    int layers_ok = 0;
    for (int i = 0; i < kN_LAYER; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const int before = c.problems;

        need(c, p + "attn_norm.weight", {kN_EMBD}, kTYPE_F32);
        need(c, p + "attn_sinks.weight", {kN_HEAD}, kTYPE_F32);
        need(c, p + "attn_q_a.weight", {kN_EMBD, kQ_LORA});
        need(c, p + "attn_q_a_norm.weight", {kQ_LORA}, kTYPE_F32);
        need(c, p + "attn_q_b.weight", {kQ_LORA, kN_HEAD * kN_HEAD_DIM});
        need(c, p + "attn_kv.weight", {kN_EMBD, kN_HEAD_DIM});
        need(c, p + "attn_kv_a_norm.weight", {kN_HEAD_DIM}, kTYPE_F32);
        need(c, p + "attn_output_a.weight", {kN_HEAD * kN_HEAD_DIM / kO_GROUPS, kO_LORA * kO_GROUPS});
        need(c, p + "attn_output_b.weight", {kO_GROUPS * kO_LORA, kN_EMBD});
        need(c, p + "hc_attn_fn.weight", {kHC_DIM, kHC_MIX});
        need(c, p + "hc_attn_base.weight", {kHC_MIX}, kTYPE_F32);
        need(c, p + "hc_attn_scale.weight", {3}, kTYPE_F32);
        need(c, p + "hc_ffn_fn.weight", {kHC_DIM, kHC_MIX});
        need(c, p + "hc_ffn_base.weight", {kHC_MIX}, kTYPE_F32);
        need(c, p + "hc_ffn_scale.weight", {3}, kTYPE_F32);

        // compressore: presente solo se il layer ha rapporto di compressione diverso da 0
        const gptoss::GgufRef* ape = m.find(p + "attn_compressor_ape.weight");
        const uint64_t ratio = (ape && ape->t->dims.size() >= 2) ? ape->t->dims[1] : 0;
        if (ratio != 0) {
            const uint64_t coff = ratio == 4 ? 2 : 1;
            if (ratio != 4 && ratio != 128) { std::printf("RAPPORTO   layer %d: %llu non supportato\n", i, (unsigned long long) ratio); c.problems++; }
            need(c, p + "attn_compressor_kv.weight", {kN_EMBD, coff * kN_HEAD_DIM});
            need(c, p + "attn_compressor_gate.weight", {kN_EMBD, coff * kN_HEAD_DIM});
            need(c, p + "attn_compressor_ape.weight", {coff * kN_HEAD_DIM, ratio});
            need(c, p + "attn_compressor_norm.weight", {kN_HEAD_DIM}, kTYPE_F32);
            if (ratio == 4) {
                need(c, p + "indexer.proj.weight", {kN_EMBD, kIDX_N_HEAD});
                need(c, p + "indexer.attn_q_b.weight", {kQ_LORA, kIDX_N_HEAD * kIDX_HEAD});
                need(c, p + "indexer_compressor_kv.weight", {kN_EMBD, 2 * kIDX_HEAD});
                need(c, p + "indexer_compressor_gate.weight", {kN_EMBD, 2 * kIDX_HEAD});
                need(c, p + "indexer_compressor_ape.weight", {2 * kIDX_HEAD, 4});
                need(c, p + "indexer_compressor_norm.weight", {kIDX_HEAD}, kTYPE_F32);
            }
        }

        need(c, p + "ffn_norm.weight", {kN_EMBD}, kTYPE_F32);
        need(c, p + "ffn_gate_inp.weight", {kN_EMBD, kN_EXPERT});
        if (i < 3) need(c, p + "ffn_gate_tid2eid.weight", {kN_EXPERT_USED, kN_VOCAB}, kTYPE_I32);
        else need(c, p + "exp_probs_b.bias", {kN_EXPERT}, kTYPE_F32);
        need(c, p + "ffn_gate_exps.weight", {kN_EMBD, kN_FF_EXP, kN_EXPERT});
        need(c, p + "ffn_up_exps.weight", {kN_EMBD, kN_FF_EXP, kN_EXPERT});
        need(c, p + "ffn_down_exps.weight", {kN_FF_EXP, kN_EMBD, kN_EXPERT});
        need(c, p + "ffn_gate_shexp.weight", {kN_EMBD, kN_FF_EXP});
        need(c, p + "ffn_up_shexp.weight", {kN_EMBD, kN_FF_EXP});
        need(c, p + "ffn_down_shexp.weight", {kN_FF_EXP, kN_EMBD});

        if (c.problems == before) layers_ok++;
    }

    std::printf("\nlayer verificati: %d di %d senza problemi\n", layers_ok, kN_LAYER);
    std::printf(c.problems ? "PROBLEMI: %d\n" : "CONTRATTO OK\n", c.problems);
    return c.problems ? 1 : 0;
}

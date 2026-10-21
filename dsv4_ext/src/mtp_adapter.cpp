// dsv4 MTP adapter - see dsv4/mtp_adapter.hpp.
//
// The bridge between a Hugging Face DeepSeek-V4 checkpoint (safetensors) and this engine, for the
// draft module ONLY: the main model stays a GGUF. The adapter reads the `mtp.<J>.*` tensors,
// converts their dtypes to ggml types the engine already has kernels for, and renames them to the
// engine's names. Every step is confirmed by shape: a name that would fill the wrong matrix is a
// load error with a message, never a silently wrong draft.
//
// The names and dtypes below are what DeepSeek-V4-Flash ships (read from its model.safetensors.index
// and one shard header on 2026-10-21; see docs/mtp_safetensors.md). The dense weights are F8_E4M3
// with F8_E8M0 scales on a 128-element block grid; the routed experts are 4-bit signed codes packed
// two per byte with one e8m0 scale per 32 columns - exactly a ggml Q4_0 block, so that conversion
// is bit-exact.
#include "dsv4/mtp_adapter.hpp"

#include "dsv4/dequant.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dsv4 {
namespace {

// ------------------------------------------------------------------ dtype helpers
inline uint16_t ld16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline float bf16_to_f32(uint16_t u) { uint32_t x = (uint32_t) u << 16; float f; std::memcpy(&f, &x, 4); return f; }
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) { if (!man) bits = sign; else { uint32_t e = 0x7F; while (!(man & 0x400)) { man <<= 1; --e; } bits = (e << 23) | ((man & 0x3FF) << 13) | sign; } }
    else if (exp == 0x1F) bits = sign | 0x7F800000u | (man << 13);
    else bits = ((exp + 0x70u) << 23) | (man << 13) | sign;
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline float e4m3_to_f32(uint8_t b) {
    const uint32_t sign = (uint32_t) (b >> 7) << 31;
    const uint32_t exp = (b >> 3) & 0xF, man = b & 7;
    uint32_t bits;
    if (exp == 0) { if (!man) bits = sign; else { uint32_t e = 0x7F; uint32_t m = man << 20; while (!(m & 0x800000)) { m <<= 1; --e; } bits = (e << 23) | (m & 0x7FFFFFu) | sign; } }
    else bits = ((exp + 0x70u) << 23) | (man << 20) | sign;
    float f; std::memcpy(&f, &bits, 4); return f;
}
// fp16 bits of the exact power of two 2^e (an e8m0 byte s means 2^(s-127)).
inline uint16_t pow2_f16_bits(int e) { return (uint16_t) ((e + 15) << 10); }
// one 4-bit code of a packed byte as its signed value in [-8, 7]
inline int s4(uint8_t byte, bool high) { const int v = high ? (byte >> 4) : (byte & 0xF); return v >= 8 ? v - 16 : v; }

// read one tensor's bytes straight out of its file (no whole-shard copy)
bool read_bytes(const std::string& path, uint64_t off, uint64_t n, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    if (std::fseek(f, (long) off, SEEK_SET) != 0) { std::fclose(f); return false; }
    out.resize((size_t) n);
    const bool ok = n == 0 || std::fread(out.data(), 1, n, f) == n;
    std::fclose(f);
    return ok;
}

// ------------------------------------------------------------------ one source tensor, bytes read
struct Src {
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
    const uint8_t* data() const { return bytes.data(); }
    int64_t r0() const { return shape.empty() ? -1 : (int64_t) shape[0]; }
    int64_t r1() const { return shape.size() < 2 ? -1 : (int64_t) shape[1]; }
};

// ------------------------------------------------------------------ one engine tensor
struct Eng {
    std::string name;
    std::vector<uint64_t> dims;   // ggml order: dims[0] = contiguous input length
    uint32_t type = 0;
    std::vector<uint8_t> blob;
};

// 1-D float parameter (norm weights, biases, sinks, hyper-connection base/scale).
bool conv_1d(const Src& w, int64_t n, const std::string& name, Eng& e, std::string& err) {
    if (w.shape.size() != 1 || w.shape[0] != (uint64_t) n) {
        err = name + ": the checkpoint's tensor for it is " + w.dtype + " [" + std::to_string(w.r0()) + "], expected [" +
              std::to_string(n) + "]";
        return false;
    }
    e.name = name; e.dims = {(uint64_t) n}; e.type = T_F32;
    e.blob.resize((size_t) n * 4);
    if (w.dtype == "F32") {
        if (w.bytes.size() != e.blob.size()) { err = name + ": F32 byte count does not match its shape"; return false; }
        std::memcpy(e.blob.data(), w.data(), e.blob.size());
    } else if (w.dtype == "BF16" || w.dtype == "F16") {
        if (w.bytes.size() != (size_t) n * 2) { err = name + ": 16-bit byte count does not match its shape"; return false; }
        for (int64_t i = 0; i < n; ++i) {
            const float v = w.dtype == "BF16" ? bf16_to_f32(ld16(w.data() + 2 * i)) : f16_to_f32(ld16(w.data() + 2 * i));
            std::memcpy(&e.blob[(size_t) (4 * i)], &v, 4);
        }
    } else { err = name + " is read as float by the engine: dtype " + w.dtype + " cannot be"; return false; }
    return true;
}

// An MLX dense weight: int8 codes four per U32 element, one e8m0 scale byte per 32 columns.
// The header's shape counts U32 elements, so the logical [rows][cols] is folded from it: a
// [8,1024,1024] U32 tensor is 8192 rows of 4096 int8 codes. The value is code * 2^(scale-127),
// exact in float32 (an e8m0 scale is a power of two, an int8 code is an integer).
static bool conv_2d_int8_f32(const Src& w, const Src* sc, int64_t rows, int64_t cols, const std::string& name, Eng& e, std::string& err) {
    // The header counts U32 elements; the logical matrix is four int8 codes per element, and any
    // leading dimensions (an MLX wo_a is [8][1024][1024]) fold into the row count.
    int64_t lr = -1;
    if (w.shape.size() >= 2) { lr = 1; for (size_t i = 0; i + 1 < w.shape.size(); ++i) lr *= (int64_t) w.shape[i]; }
    const int64_t lc = w.shape.empty() ? 0 : (int64_t) w.shape.back() * 4;
    if (w.dtype != "U32" || lr != rows || lc != cols) {
        err = name + ": the checkpoint's tensor for it is " + w.dtype + " [" + std::to_string(lr) + "," +
              std::to_string(lc) + " int8 codes], expected [" + std::to_string(rows) + "," + std::to_string(cols) + "]";
        return false;
    }
    // The scale grid folds the same way: its last axis counts 32-column blocks.
    int64_t sr = -1, scols = -1;
    if (sc && sc->shape.size() >= 2) {
        sr = 1; for (size_t i = 0; i + 1 < sc->shape.size(); ++i) sr *= (int64_t) sc->shape[i];
        scols = (int64_t) sc->shape.back();
    }
    if (!sc || sc->dtype != "U8" || sr != rows || scols * 32 != cols) {
        err = name + ": scale grid [" + std::to_string(sr) + "," + std::to_string(scols * (scols > 0 ? 32 : 0)) +
              "] is not a 32-element tiling of [" + std::to_string(rows) + "," + std::to_string(cols) + "]";
        return false;
    }
    if (w.bytes.size() != (size_t) rows * cols || sc->bytes.size() != (size_t) rows * (cols / 32)) {
        err = name + ": byte count does not match its shape"; return false;
    }
    e.name = name; e.dims = {(uint64_t) cols, (uint64_t) rows}; e.type = T_F32;
    e.blob.resize((size_t) rows * cols * 4);
    const int64_t sb = scols;
    for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < cols; ++c) {
        const int s = sc->bytes[(size_t) (r * sb + c / 32)];
        const float v = (float) (int) (int8_t) w.bytes[(size_t) (r * cols + c)] * (float) std::ldexp(1.0, s - 127);
        std::memcpy(&e.blob[(size_t) (4 * (r * cols + c))], &v, 4);
    }
    return true;
}

// A dense 2-D weight, row-major [rows][cols] in the checkpoint, as F32 for the engine.
// Plain F32/BF16/F16 convert element by element; F8_E4M3 + F8_E8M0 (a 128-element block grid)
// converts exactly, because an e8m0 scale is a power of two. An MLX checkpoint reaches this
// through conv_2d_int8_f32 (int8 codes in U32 elements, 32-column scales).
bool conv_2d_f32(const Src& w, const Src* sc, int64_t rows, int64_t cols, const std::string& name, Eng& e, std::string& err) {
    if (w.dtype == "U32") return conv_2d_int8_f32(w, sc, rows, cols, name, e, err);
    if (w.shape.size() != 2 || w.r0() != rows || w.r1() != cols) {
        err = name + ": the checkpoint's tensor for it is " + w.dtype + " [" + std::to_string(w.r0()) + "," +
              std::to_string(w.r1()) + "], expected [" + std::to_string(rows) + "," + std::to_string(cols) + "]";
        return false;
    }
    e.name = name; e.dims = {(uint64_t) cols, (uint64_t) rows}; e.type = T_F32;
    e.blob.resize((size_t) rows * cols * 4);
    auto f_at = [&](const Src& s, int64_t i) -> float {
        if (s.dtype == "F32") { float f; std::memcpy(&f, s.data() + 4 * i, 4); return f; }
        if (s.dtype == "BF16") return bf16_to_f32(ld16(s.data() + 2 * i));
        return f16_to_f32(ld16(s.data() + 2 * i));
    };
    if (!sc) {
        if (w.dtype != "F32" && w.dtype != "BF16" && w.dtype != "F16") { err = name + " has dtype " + w.dtype + " with no scales: only F32, BF16 and F16 carry without them"; return false; }
        if (w.bytes.size() != (size_t) rows * cols * st_dtype_bytes(w.dtype)) { err = name + ": byte count does not match its shape"; return false; }
        for (int64_t i = 0; i < rows * cols; ++i) { const float v = f_at(w, i); std::memcpy(&e.blob[(size_t) (4 * i)], &v, 4); }
        return true;
    }
    if (w.dtype != "F8_E4M3" || sc->dtype != "F8_E8M0") { err = name + ": dtype pair " + w.dtype + "+" + sc->dtype + " is not F8_E4M3+F8_E8M0"; return false; }
    if (sc->shape.size() != 2 || sc->r0() * 128 != rows || sc->r1() * 128 != cols) {
        err = name + ": scale grid [" + std::to_string(sc->r0()) + "," + std::to_string(sc->r1()) + "] is not a 128-element tiling of [" +
              std::to_string(rows) + "," + std::to_string(cols) + "]";
        return false;
    }
    if (w.bytes.size() != (size_t) rows * cols || sc->bytes.size() != (size_t) sc->r0() * sc->r1()) { err = name + ": byte count does not match its shape"; return false; }
    const int64_t sb = sc->r1();
    for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < cols; ++c) {
        const int s = sc->bytes[(size_t) ((r / 128) * sb + c / 128)];
        const float v = e4m3_to_f32(w.bytes[(size_t) (r * cols + c)]) * (float) std::ldexp(1.0, s - 127);
        std::memcpy(&e.blob[(size_t) (4 * (r * cols + c))], &v, 4);
    }
    return true;
}

// The routed experts: the checkpoint stores one packed tensor per expert (4-bit signed codes, two
// per byte, one e8m0 scale per 32 columns). The engine wants one fused 3-D Q4_0 block. The
// conversion is bit-exact: Q4_0's fp16 delta is exactly 2^(s-127), and its codes are the source
// codes biased by +8.
bool conv_experts(const std::vector<Src>& w, const std::vector<Src>& sc, int64_t E, int64_t rows, int64_t cols,
                  const std::string& name, Eng& e, std::string& err) {
    if ((int64_t) w.size() != E) { err = name + ": the checkpoint has " + std::to_string(w.size()) + " of " + std::to_string(E) + " experts"; return false; }
    if (cols % 32) { err = name + ": " + std::to_string(cols) + " columns is not a whole number of 32-element blocks"; return false; }
    const int64_t nblk = cols / 32;
    e.name = name; e.dims = {(uint64_t) cols, (uint64_t) rows, (uint64_t) E}; e.type = T_Q4_0;
    e.blob.assign((size_t) E * rows * (nblk * 18), 0);
    for (int64_t ex = 0; ex < E; ++ex) {
        const Src& sw = w[(size_t) ex]; const Src& ss = sc[(size_t) ex];
        if (sw.shape.size() != 2 || sw.r0() != rows || sw.r1() * 2 != cols) {
            err = name + " expert " + std::to_string(ex) + ": packed shape [" + std::to_string(sw.r0()) + "," +
                  std::to_string(sw.r1()) + "] is not [" + std::to_string(rows) + "," + std::to_string(cols / 2) + "]";
            return false;
        }
        if (ss.shape.size() != 2 || ss.r0() != rows || ss.r1() != nblk) {
            err = name + " expert " + std::to_string(ex) + ": scale grid [" + std::to_string(ss.r0()) + "," +
                  std::to_string(ss.r1()) + "] is not [" + std::to_string(rows) + "," + std::to_string(nblk) + "]";
            return false;
        }
        if (sw.bytes.size() != (size_t) rows * (cols / 2) || ss.bytes.size() != (size_t) rows * nblk) {
            err = name + " expert " + std::to_string(ex) + ": byte count does not match its shape"; return false;
        }
        for (int64_t r = 0; r < rows; ++r) for (int64_t b = 0; b < nblk; ++b) {
            const int s = ss.bytes[(size_t) (r * nblk + b)];
            if (s - 127 < -24 || s - 127 > 15) { err = name + " expert " + std::to_string(ex) + " row " + std::to_string(r) +
                  " block " + std::to_string(b) + ": e8m0 scale 2^" + std::to_string(s - 127) + " is not representable as fp16 (Q4_0's scale)"; return false; }
            uint8_t* dst = &e.blob[(size_t) ((ex * rows + r) * nblk + b) * 18];
            const uint16_t d = pow2_f16_bits(s - 127);
            std::memcpy(dst, &d, 2);
            // HF packs element 2k in the low nibble and 2k+1 in the high one of byte k; ggml's Q4_0
            // puts element k in the low nibble of qs[k] and element k+16 in its high nibble.
            const uint8_t* src = sw.bytes.data() + (size_t) (r * (cols / 2) + b * 16);
            uint8_t* q = dst + 2;
            for (int k = 0; k < 16; ++k) {
                const int lo = s4(src[k >> 1], (k & 1) != 0) + 8;              // element k
                const int hi = s4(src[(k >> 1) + 8], (k & 1) != 0) + 8;       // element k+16
                q[k] = (uint8_t) (lo | (hi << 4));
            }
        }
    }
    return true;
}

// The routed experts of an MLX checkpoint: ONE fused tensor per matrix, [E, rows, cols/8] of U32
// (eight 4-bit codes per element, element 2k in the low nibble of byte k) plus one e8m0 scale
// byte per 32 columns. The codes are e2m1 (bit 3 = sign, magnitudes 0,.5,1,1.5,2,3,4,6), which
// is exactly ggml's MXFP4: its table stores the values doubled and its scale helper halves them,
// so value = e2m1(code) * 2^(s-127) = 0.5*2^(s-127) * (2*e2m1(code)) holds bit for bit.
bool conv_experts_fused(const Src& w, const Src& sc, int64_t E, int64_t rows, int64_t cols,
                        const std::string& name, Eng& e, std::string& err) {
    const int64_t lr = w.shape.size() >= 2 ? (int64_t) w.shape[1] : -1;
    const int64_t lc = w.shape.size() >= 3 ? (int64_t) w.shape[2] * 8 : -1;
    if (w.dtype != "U32" || w.shape.size() != 3 || (int64_t) w.shape[0] != E || lr != rows || lc != cols) {
        err = name + ": the checkpoint's expert tensor is " + w.dtype + " [" +
              std::to_string(w.shape.size() >= 1 ? (int64_t) w.shape[0] : -1) + "," +
              std::to_string(lr) + "," + std::to_string(lc) + " codes], expected [" +
              std::to_string(E) + "," + std::to_string(rows) + "," + std::to_string(cols) + "]";
        return false;
    }
    if (sc.dtype != "U8" || sc.shape.size() != 3 || (int64_t) sc.shape[0] != E || (int64_t) sc.shape[1] != rows ||
        (int64_t) sc.shape[2] * 32 != cols) {
        err = name + ": scale grid " + (sc.shape.size() >= 1 ? std::to_string(sc.r0()) : std::string("?")) +
              " x " + (sc.shape.size() >= 3 ? std::to_string(sc.r1() * 32) : std::string("?")) +
              " is not a 32-column tiling of the experts";
        return false;
    }
    if (cols % 32) { err = name + ": " + std::to_string(cols) + " columns is not a whole number of 32-element blocks"; return false; }
    const int64_t nblk = cols / 32;
    if (w.bytes.size() != (size_t) E * rows * (cols / 2) || sc.bytes.size() != (size_t) E * rows * nblk) {
        err = name + ": expert byte count does not match its shape"; return false;
    }
    e.name = name; e.dims = {(uint64_t) cols, (uint64_t) rows, (uint64_t) E}; e.type = T_MXFP4;
    e.blob.assign((size_t) E * rows * (nblk * 17), 0);
    for (int64_t i = 0; i < E * rows * nblk; ++i) {
        const uint8_t* src = w.bytes.data() + (size_t) i * 16;
        uint8_t* q = &e.blob[(size_t) i * 17];
        q[0] = sc.bytes[(size_t) i];
        // The source packs element 2k in the low nibble and 2k+1 in the high one of byte k;
        // ggml's MXFP4 puts element j in the low nibble of qs[j] and element j+16 in its high one.
        for (int j = 0; j < 16; ++j) {
            const uint8_t a = src[j >> 1], b = src[8 + (j >> 1)];
            q[1 + j] = (uint8_t) ((j & 1) ? ((a >> 4) | (b & 0xF0)) : ((a & 0x0F) | ((b & 0x0F) << 4)));
        }
    }
    return true;
}

// ------------------------------------------------------------------ the slot table
// The name table. The logical shapes are computed from the model config where the call happens
// (a shape that does not match is refused there), so this table only carries the names.
// 1-D parameters: `req` says whether a draft layer may omit it.
struct Slot1 { const char* hf; const char* eng; bool req; };
const Slot1 k_1d[] = {
    {"attn_norm.weight", "attn_norm.weight", true}, {"attn.q_norm.weight", "attn_q_a_norm.weight", true},
    {"attn.kv_norm.weight", "attn_kv_a_norm.weight", true}, {"attn.attn_sink", "attn_sinks.weight", true},
    {"ffn_norm.weight", "ffn_norm.weight", true}, {"ffn.gate.bias", "exp_probs_b.bias", false},
    {"hc_attn_base", "hc_attn_base.weight", true}, {"hc_attn_scale", "hc_attn_scale.weight", true},
    {"hc_ffn_base", "hc_ffn_base.weight", true}, {"hc_ffn_scale", "hc_ffn_scale.weight", true},
    {"attn.compressor.norm.weight", "attn_compressor_norm.weight", false},
    {"indexer.compressor.norm.weight", "indexer_compressor_norm.weight", false},
};

// An MLX checkpoint (mlx-quantize / mlx-to-hf) spells the same module differently: one draft
// layer under `decoder.*`, fused expert matrices under `decoder.ffn.switch_mlp.*`, and the
// fusion tensors at the top level. Rather than a second adapter, this table renames those
// tensors into the HF suffixes the code below already groups by; every mapping is still
// confirmed by shape, so a wrong rename is still a load error, never a wrong draft.
// A name mapped to "" is read and dropped on purpose: the module's own final head is not used,
// because draft logits come from the main model's head.
struct SlotMlx { const char* mlx; const char* hf; };
const SlotMlx k_mlx[] = {
    {"decoder.attn.attn_sink", "attn.attn_sink"},
    {"decoder.attn.q_norm.weight", "attn.q_norm.weight"},
    {"decoder.attn.kv_norm.weight", "attn.kv_norm.weight"},
    {"decoder.attn_norm.weight", "attn_norm.weight"},
    {"decoder.ffn_norm.weight", "ffn_norm.weight"},
    {"decoder.ffn.gate.e_score_correction_bias", "ffn.gate.bias"},
    {"decoder.ffn.gate.weight", "ffn.gate.weight"},
    // A matrix is spelled "<name>" or "<name>.weight" depending on the exporter (mlx-to-hf keeps
    // the mlx nn.Parameter dict key, mlx 0.x writes the module path); both are accepted, and the
    // scale tensor is always "<name>.scales". The shape check below is what decides.
    {"decoder.attn.wq_a", "attn.wq_a"}, {"decoder.attn.wq_a.weight", "attn.wq_a"}, {"decoder.attn.wq_a.scales", "attn.wq_a.scale"},
    {"decoder.attn.wq_b", "attn.wq_b"}, {"decoder.attn.wq_b.weight", "attn.wq_b"}, {"decoder.attn.wq_b.scales", "attn.wq_b.scale"},
    {"decoder.attn.wkv", "attn.wkv"}, {"decoder.attn.wkv.weight", "attn.wkv"}, {"decoder.attn.wkv.scales", "attn.wkv.scale"},
    {"decoder.attn.wo_a", "attn.wo_a"}, {"decoder.attn.wo_a.weight", "attn.wo_a"}, {"decoder.attn.wo_a.scales", "attn.wo_a.scale"},
    {"decoder.attn.wo_b", "attn.wo_b"}, {"decoder.attn.wo_b.weight", "attn.wo_b"}, {"decoder.attn.wo_b.scales", "attn.wo_b.scale"},
    {"decoder.attn.compressor.norm.weight", "attn.compressor.norm.weight"},
    {"decoder.attn.compressor.wkv", "attn.compressor.wkv"}, {"decoder.attn.compressor.wkv.weight", "attn.compressor.wkv"},
    {"decoder.attn.compressor.wkv.scales", "attn.compressor.wkv.scale"},
    {"decoder.attn.compressor.wgate", "attn.compressor.wgate"}, {"decoder.attn.compressor.wgate.weight", "attn.compressor.wgate"},
    {"decoder.attn.compressor.wgate.scales", "attn.compressor.wgate.scale"},
    {"decoder.attn.compressor.ape", "attn.compressor.ape"}, {"decoder.attn.compressor.ape.weight", "attn.compressor.ape"},
    {"decoder.indexer.wq_b", "indexer.wq_b"}, {"decoder.indexer.wq_b.weight", "indexer.wq_b"}, {"decoder.indexer.wq_b.scales", "indexer.wq_b.scale"},
    {"decoder.indexer.weights_proj", "indexer.weights_proj"}, {"decoder.indexer.weights_proj.weight", "indexer.weights_proj"},
    {"decoder.indexer.compressor.norm.weight", "indexer.compressor.norm.weight"},
    {"decoder.indexer.compressor.wkv", "indexer.compressor.wkv"}, {"decoder.indexer.compressor.wkv.weight", "indexer.compressor.wkv"},
    {"decoder.indexer.compressor.wkv.scales", "indexer.compressor.wkv.scale"},
    {"decoder.indexer.compressor.wgate", "indexer.compressor.wgate"}, {"decoder.indexer.compressor.wgate.weight", "indexer.compressor.wgate"},
    {"decoder.indexer.compressor.wgate.scales", "indexer.compressor.wgate.scale"},
    {"decoder.indexer.compressor.ape", "indexer.compressor.ape"}, {"decoder.indexer.compressor.ape.weight", "indexer.compressor.ape"},
    {"decoder.ffn.shared_experts.gate_proj", "ffn.shared_experts.w1"}, {"decoder.ffn.shared_experts.gate_proj.weight", "ffn.shared_experts.w1"},
    {"decoder.ffn.shared_experts.gate_proj.scales", "ffn.shared_experts.w1.scale"},
    {"decoder.ffn.shared_experts.up_proj", "ffn.shared_experts.w3"}, {"decoder.ffn.shared_experts.up_proj.weight", "ffn.shared_experts.w3"},
    {"decoder.ffn.shared_experts.up_proj.scales", "ffn.shared_experts.w3.scale"},
    {"decoder.ffn.shared_experts.down_proj", "ffn.shared_experts.w2"}, {"decoder.ffn.shared_experts.down_proj.weight", "ffn.shared_experts.w2"},
    {"decoder.ffn.shared_experts.down_proj.scales", "ffn.shared_experts.w2.scale"},
    {"decoder.attn_hc.base", "hc_attn_base"}, {"decoder.attn_hc.scale", "hc_attn_scale"}, {"decoder.attn_hc.fn", "hc_attn_fn"},
    {"decoder.ffn_hc.base", "hc_ffn_base"}, {"decoder.ffn_hc.scale", "hc_ffn_scale"}, {"decoder.ffn_hc.fn", "hc_ffn_fn"},
    {"decoder.ffn.switch_mlp.gate_proj", "ffn.experts.w1.fused"}, {"decoder.ffn.switch_mlp.gate_proj.weight", "ffn.experts.w1.fused"},
    {"decoder.ffn.switch_mlp.gate_proj.scales", "ffn.experts.w1.fusedscale"},
    {"decoder.ffn.switch_mlp.up_proj", "ffn.experts.w3.fused"}, {"decoder.ffn.switch_mlp.up_proj.weight", "ffn.experts.w3.fused"},
    {"decoder.ffn.switch_mlp.up_proj.scales", "ffn.experts.w3.fusedscale"},
    {"decoder.ffn.switch_mlp.down_proj", "ffn.experts.w2.fused"}, {"decoder.ffn.switch_mlp.down_proj.weight", "ffn.experts.w2.fused"},
    {"decoder.ffn.switch_mlp.down_proj.scales", "ffn.experts.w2.fusedscale"},
    {"enorm.weight", "enorm.weight"}, {"hnorm.weight", "hnorm.weight"},
    {"e_proj", "e_proj.weight"}, {"e_proj.weight", "e_proj.weight"}, {"e_proj.scales", "e_proj.scale"},
    {"h_proj", "h_proj.weight"}, {"h_proj.weight", "h_proj.weight"}, {"h_proj.scales", "h_proj.scale"},
    {"hc_head.base", "hc_head_base"}, {"hc_head.scale", "hc_head_scale"}, {"hc_head.fn", "hc_head_fn"},
    // the module's own final head: draft logits come from the main model's head, so it is dropped
    {"decoder.norm.weight", ""}, {"norm.weight", ""},
};

// Group an MLX checkpoint's tensors as the HF path would: one draft layer (index 0), keyed by the
// HF suffix. Returns false when the file holds no draft module at all (no decoder.* tensors).
bool mlx_group_layers(const StHeader& sh, std::map<int, std::map<std::string, const StTensor*>>& layers) {
    for (const StTensor& t : sh.tensors) {
        const std::string n = t.name;
        const char* hf = nullptr;
        for (const SlotMlx& s : k_mlx) if (s.mlx == n) { hf = s.hf; break; }
        if (!hf || !*hf) continue;
        layers[0][hf] = &t;
    }
    return !layers.empty();
}

}  // namespace

// ------------------------------------------------------------------ the public entry point
bool mtp_adapt_safetensors(const StHeader& sh, const Dsv4Config& c, MtpAdapted& out, std::string& err) {
    out = MtpAdapted();
    const int64_t dim = c.n_embd, hd = c.key_len, nh = c.n_head, q_lora = c.q_lora, o_lora = c.o_lora, o_groups = c.o_groups;
    const int64_t E = c.n_expert, ff = c.ff_exp, ih = c.idx_heads, ik = c.idx_key_len, hc = c.hc_count;
    if (dim <= 0 || E <= 0 || ff <= 0 || hc <= 0) { err = "model config is incomplete: cannot adapt an MTP module"; return false; }
    const int64_t gin = nh * hd / (o_groups > 0 ? o_groups : 1);
    const int64_t mix = (2 + hc) * hc;

    // ---- group the checkpoint's tensors by draft layer, then by suffix
    std::map<int, std::map<std::string, const StTensor*>> layers;
    for (const StTensor& t : sh.tensors) {
        if (t.name.rfind("mtp.", 0) != 0) continue;
        const std::string rest = t.name.substr(4);
        const size_t dot = rest.find('.');
        if (dot == std::string::npos) continue;
        layers[std::atoi(rest.c_str())][rest.substr(dot + 1)] = &t;
    }
    const bool mlx_layout = layers.empty() && mlx_group_layers(sh, layers);
    if (layers.empty()) {
        err = "no draft module (no mtp.<J>.* and no decoder.* tensors) in " + (sh.files.empty() ? std::string() : sh.files[0]);
        return false;
    }
    out.n_layers = (int) layers.size();
    out.source = sh.files[0];

    // bytes of one tensor, read once and cached by name
    std::map<std::string, Src> cache;
    auto get_src = [&](const StTensor* t) -> Src* {
        if (!t) return nullptr;
        auto it = cache.find(t->name);
        if (it != cache.end()) return &it->second;
        Src s; s.dtype = t->dtype; s.shape = t->shape;
        if (!read_bytes(sh.files[(size_t) t->shard], t->abs_offset, t->nbytes, s.bytes)) return nullptr;
        auto res = cache.emplace(t->name, std::move(s));
        return &res.first->second;
    };

    auto emit = [&](const Eng& e) {
        MtpAdapted::Item it;
        it.t.name = e.name; it.t.dims = e.dims; it.t.type = e.type; it.t.nbytes = e.blob.size(); it.t.known_type = true;
        it.blob = e.blob;
        out.bytes += it.blob.size();          // before the move: a moved-from vector is empty
        out.items.push_back(std::move(it));
    };

    for (const auto& kv : layers) {
        const int L = c.n_layer + kv.first;
        const std::string P = "blk." + std::to_string(L) + ".";
        const auto& by_suffix = kv.second;
        auto fnd = [&](const std::string& n) { auto it = by_suffix.find(n); return it == by_suffix.end() ? nullptr : it->second; };

        // The draft layer's compression ratio is not in the main GGUF when the checkpoint declares
        // none: read it back from the compressor tensors' own shapes (ape has `ratio` rows).
        int ratio = L < (int) c.compress_ratios.size() ? c.compress_ratios[(size_t) L] : 0;
        if (const StTensor* ape = fnd("attn.compressor.ape")) {
            const int64_t r = ape->shape.size() >= 1 ? (int64_t) ape->shape[0] : 0;
            if (r > 0) ratio = (int) r;
        }
        const int64_t coff = 1 + (ratio == 4 ? 1 : 0);
        out.inferred_ratios.push_back(ratio);

        // ---- 1-D parameters
        for (const Slot1& s : k_1d) {
            const std::string hf = s.hf;
            const int64_t n = hf == "attn_norm.weight" || hf == "ffn_norm.weight" ? dim
                            : hf == "attn.q_norm.weight" ? q_lora
                            : hf == "attn.kv_norm.weight" || hf == "attn.compressor.norm.weight" ? hd
                            : hf == "indexer.compressor.norm.weight" ? ik
                            : hf == "attn.attn_sink" ? nh
                            : hf == "ffn.gate.bias" ? E
                            : hf == "hc_attn_base" || hf == "hc_ffn_base" ? mix
                            : hf == "hc_attn_scale" || hf == "hc_ffn_scale" ? 3
                            : 0;
            if (n <= 0) continue;
            const StTensor* t = fnd(hf);
            if (!t) { if (s.req) { err = "draft layer " + std::to_string(kv.first) + " is missing " + hf; return false; } continue; }
            const Src* w = get_src(t);
            if (!w) { err = "cannot read " + t->name + " (" + std::to_string(t->abs_offset) + " + " + std::to_string(t->nbytes) +
                  " bytes) from " + sh.files[(size_t) t->shard]; return false; }
            Eng e;
            if (!conv_1d(*w, n, P + s.eng, e, err)) return false;
            emit(e);
        }

        // ---- 2-D weights, each with its exact logical shape
        struct Want { const char* hf; const char* eng; int64_t rows; int64_t cols; };
        const Want wants[] = {
            {"attn.wq_a", "attn_q_a.weight", q_lora, dim},
            {"attn.wq_b", "attn_q_b.weight", nh * hd, q_lora},
            {"attn.wkv", "attn_kv.weight", hd, dim},
            {"attn.wo_a", "attn_output_a.weight", o_groups * o_lora, gin},
            {"attn.wo_b", "attn_output_b.weight", dim, o_groups * o_lora},
            {"ffn.gate.weight", "ffn_gate_inp.weight", E, dim},
            {"ffn.shared_experts.w1", "ffn_gate_shexp.weight", ff, dim},
            {"ffn.shared_experts.w3", "ffn_up_shexp.weight", ff, dim},
            {"ffn.shared_experts.w2", "ffn_down_shexp.weight", dim, ff},
            {"hc_attn_fn", "hc_attn_fn.weight", mix, hc * dim},
            {"hc_ffn_fn", "hc_ffn_fn.weight", mix, hc * dim},
            {"attn.compressor.wkv", "attn_compressor_kv.weight", ratio ? coff * hd : 0, dim},
            {"attn.compressor.wgate", "attn_compressor_gate.weight", ratio ? coff * hd : 0, dim},
            {"attn.compressor.ape", "attn_compressor_ape.weight", ratio ? (int64_t) ratio : 0, ratio ? coff * hd : 0},
            {"indexer.compressor.wkv", "indexer_compressor_kv.weight", ratio == 4 ? 2 * ik : 0, dim},
            {"indexer.compressor.wgate", "indexer_compressor_gate.weight", ratio == 4 ? 2 * ik : 0, dim},
            {"indexer.compressor.ape", "indexer_compressor_ape.weight", ratio == 4 ? (int64_t) ratio : 0, ratio == 4 ? 2 * ik : 0},
            {"indexer.wq_b", "indexer.attn_q_b.weight", ratio == 4 ? ih * ik : 0, q_lora},
            {"indexer.weights_proj", "indexer.proj.weight", ratio == 4 ? ih : 0, dim},
        };
        for (const Want& wv : wants) {
            if (!wv.rows || !wv.cols) continue;   // this layer has no compressor / indexer
            const StTensor* t = fnd(wv.hf);
            if (!t) { err = "draft layer " + std::to_string(kv.first) + " is missing " + wv.hf; return false; }
            const StTensor* st = fnd(std::string(wv.hf) + ".scale");
            const Src* w = get_src(t);
            if (!w) { err = "cannot read " + t->name; return false; }
            const Src* sc = nullptr;
            if (st) { sc = get_src(st); if (!sc) { err = "cannot read " + st->name; return false; } }
            Eng e;
            if (!conv_2d_f32(*w, sc, wv.rows, wv.cols, P + wv.eng, e, err)) return false;
            emit(e);
        }

        // ---- the routed experts, split by shape: w1/w3 are [ff][dim/2] packed, w2 is [dim][ff/2]
        std::map<int, const StTensor*> g, u, d, gs, us, ds;
        for (const auto& ex : by_suffix) {
            const std::string& sfx = ex.first;
            if (sfx.rfind("ffn.experts.", 0) != 0) continue;
            // names are "ffn.experts.<e>.<w1|w2|w3>.<weight|scale>"
            const int eidx = std::atoi(sfx.c_str() + 12);
            const size_t last = sfx.rfind('.');
            if (last == std::string::npos) continue;
            const std::string tail = sfx.substr(last + 1);
            const bool is_scale = tail == "scale";
            if (tail != "weight" && !is_scale) continue;
            const size_t prev = last == 0 ? 0 : sfx.rfind('.', last - 1);
            if (prev == std::string::npos) continue;
            const std::string tag = sfx.substr(prev + 1, last - prev - 1);   // "w1" / "w2" / "w3"
            if (tag != "w1" && tag != "w2" && tag != "w3") continue;
            std::map<int, const StTensor*>* dst = nullptr;
            if (tag == "w2") dst = is_scale ? &ds : &d;
            else if (tag == "w1") dst = is_scale ? &gs : &g;
            else dst = is_scale ? &us : &u;
            (*dst)[eidx] = ex.second;
        }
        auto experts_ok = [&](std::map<int, const StTensor*>& w, std::map<int, const StTensor*>& sc, const std::string& eng) {
            if (w.empty()) { err = "draft layer " + std::to_string(kv.first) + " has no " + eng; return false; }
            if (sc.size() != w.size()) { err = eng + ": " + std::to_string(sc.size()) + " scale tensors for " + std::to_string(w.size()) + " weight tensors"; return false; }
            std::vector<Src> wl, sl;
            for (auto& kv2 : w) {
                auto sit = sc.find(kv2.first);
                if (sit == sc.end()) { err = eng + ": expert " + std::to_string(kv2.first) + " has no scale tensor"; return false; }
                const Src* a = get_src(kv2.second), *b = get_src(sit->second);
                if (!a || !b) { err = "cannot read an expert tensor of " + eng; return false; }
                wl.push_back(*a); sl.push_back(*b);
            }
            Eng e;
            const bool gate_or_up = eng == "ffn_gate_exps.weight" || eng == "ffn_up_exps.weight";
            if (!conv_experts(wl, sl, E, gate_or_up ? ff : dim, gate_or_up ? dim : ff, P + eng, e, err)) return false;
            emit(e);
            return true;
        };
        // The routed experts, in whichever layout the checkpoint used: HF ships one packed tensor
        // per expert; an MLX checkpoint ships one fused tensor per matrix, already e2m1 + e8m0.
        if (mlx_layout) {
            // gate/up/down are told apart by their logical shape, exactly as the HF path does.
            struct Want2 { const char* hf; const char* eng; bool down; };
            const Want2 fused[] = {
                {"ffn.experts.w1.fused", "ffn_gate_exps.weight", false},
                {"ffn.experts.w3.fused", "ffn_up_exps.weight", false},
                {"ffn.experts.w2.fused", "ffn_down_exps.weight", true},
            };
            for (const Want2& wv : fused) {
                const StTensor* t = fnd(wv.hf);
                const StTensor* st = fnd(std::string(wv.hf) + "scale");
                if (!t) { err = "draft layer " + std::to_string(kv.first) + " has no " + wv.hf; return false; }
                if (!st) { err = std::string(wv.hf) + ": no scale tensor for the fused experts"; return false; }
                const Src* w = get_src(t), *sc = get_src(st);
                if (!w || !sc) { err = "cannot read an expert tensor of " + std::string(wv.eng); return false; }
                Eng e;
                if (!conv_experts_fused(*w, *sc, E, wv.down ? dim : ff, wv.down ? ff : dim, P + wv.eng, e, err)) return false;
                emit(e);
            }
        } else {
            if (!experts_ok(g, gs, "ffn_gate_exps.weight")) return false;
            if (!experts_ok(u, us, "ffn_up_exps.weight")) return false;
            if (!experts_ok(d, ds, "ffn_down_exps.weight")) return false;
        }

        // ---- the fusion module: two layouts, chosen by which tensors exist
        const Src* enorm = get_src(fnd("enorm.weight")), *hnorm = get_src(fnd("hnorm.weight"));
        if (!enorm || !hnorm) { err = "draft layer " + std::to_string(kv.first) + " has no enorm/hnorm: no draft input"; return false; }
        Eng e;
        if (fnd("eh_proj.weight")) {
            // concat fusion (DeepSeek-V3 style, what the engine and the GGUF fixtures use):
            // enorm [dim], hnorm [hc*dim], eh_proj [dim+hc*dim] -> dim
            if (out.fusion_mode == 0) out.fusion_mode = 1;
            if (!conv_1d(*enorm, dim, "mtp.enorm.weight", e, err)) return false;
            emit(e);
            if (!conv_1d(*hnorm, hc * dim, "mtp.hnorm.weight", e, err)) return false;
            emit(e);
            const StTensor* eh = fnd("eh_proj.weight");
            const Src* w = get_src(eh);
            if (!w) { err = "cannot read eh_proj"; return false; }
            const Src* sc = nullptr;
            if (const StTensor* st = fnd("eh_proj.scale")) { sc = get_src(st); if (!sc) { err = "cannot read eh_proj.scale"; return false; } }
            if (!conv_2d_f32(*w, sc, dim, dim + hc * dim, "mtp.eh_proj.weight", e, err)) return false;
            emit(e);
        } else {
            // additive fusion (what the DeepSeek-V4 HF checkpoint ships): the module combines its own
            // residual streams with hc_head_*, normalises by hnorm [dim], projects with h_proj, and
            // ADDS e_proj(enorm(embedding)). Only shape-consistent: the checkpoint has no
            // [dim+hc*dim] projection, so a sum is the only fusion those shapes allow.
            const StTensor *ep = fnd("e_proj.weight"), *hp = fnd("h_proj.weight"), *hfn = fnd("hc_head_fn"), *hbs = fnd("hc_head_base"), *hsc = fnd("hc_head_scale");
            if (!ep || !hp || !hfn || !hbs || !hsc) { err = "draft layer " + std::to_string(kv.first) + " has no fusion module (need enorm + hnorm + eh_proj, or enorm + hnorm + e_proj + h_proj + hc_head_*)"; return false; }
            if (out.fusion_mode == 0) out.fusion_mode = 2;
            if (!conv_1d(*enorm, dim, "mtp.enorm.weight", e, err)) return false;
            emit(e);
            if (!conv_1d(*hnorm, dim, "mtp.hnorm.weight", e, err)) return false;
            emit(e);
            const Src *w_ep = get_src(ep), *w_hp = get_src(hp), *w_fn = get_src(hfn), *w_bs = get_src(hbs), *w_sc = get_src(hsc);
            if (!w_ep || !w_hp || !w_fn || !w_bs || !w_sc) { err = "cannot read e_proj / h_proj / hc_head_*"; return false; }
            const StTensor* s_ep = fnd("e_proj.scale"), *s_hp = fnd("h_proj.scale");
            const Src *sc_ep = s_ep ? get_src(s_ep) : nullptr, *sc_hp = s_hp ? get_src(s_hp) : nullptr;
            if ((s_ep && !sc_ep) || (s_hp && !sc_hp)) { err = "cannot read an e_proj/h_proj scale tensor"; return false; }
            if (!conv_2d_f32(*w_ep, sc_ep, dim, dim, "mtp.e_proj.weight", e, err)) return false;
            emit(e);
            if (!conv_2d_f32(*w_hp, sc_hp, dim, dim, "mtp.h_proj.weight", e, err)) return false;
            emit(e);
            Eng hb, hs, hf;
            if (!conv_1d(*w_bs, hc, "mtp.hc_head_base.weight", hb, err)) return false;
            if (!conv_1d(*w_sc, 1, "mtp.hc_head_scale.weight", hs, err)) return false;
            if (!conv_2d_f32(*w_fn, nullptr, hc, hc * dim, "mtp.hc_head_fn.weight", hf, err)) return false;
            emit(hb); emit(hs); emit(hf);
        }
    }
    return true;
}

}  // namespace dsv4

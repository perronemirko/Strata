/**
 * @file mtp_adapter.hpp
 * @brief The MTP (speculative draft) module from a Hugging Face checkpoint, as engine tensors.
 *
 * dsv4_ext's model is a GGUF; the draft head of an HF checkpoint is safetensors. This adapter is
 * the only bridge: it reads the `mtp.<J>.*` tensors of a safetensors checkpoint and produces the
 * tensor table the engine already consumes, in the engine's own names and ggml types. The main
 * model is never touched - only the draft module comes from safetensors.
 *
 * @par Why shape-driven
 *   The HF names of DeepSeek-V4 (`mtp.0.attn.wq_a`, `ffn.experts.7.w1`, ...) and the engine's
 *   (`blk.43.attn_q_a`, `ffn_gate_exps`) are different spellings of the same matrix. Rather than
 *   trusting a name table alone, every mapping is CONFIRMED by shape against the model config:
 *   a tensor whose rows/cols do not match the layer it would fill is refused with a message.
 *   A wrong name mapping is therefore a load error, never a silently wrong draft.
 *
 * @par What it converts
 *   - F32 / BF16 / F16 copy straight to the matching ggml type.
 *   - I8 or F8_E4M3 weights with F8_E8M0 block scales dequantise to F32 (scale = 2^(byte-127)).
 *   - Per-expert HF tensors (`experts.<e>.w1/w2/w3`) are concatenated into the engine's fused
 *     3D `ffn_*_exps` layout; w1/w2/w3 are assigned to gate/up/down BY SHAPE, not by name.
 *
 * @par Fusion module (the draft input)
 *   Two layouts are recognised, chosen by which tensors exist:
 *   - concat (DeepSeek-V3 style, what the engine and the GGUF fixtures use):
 *     `enorm`, `hnorm` over [dim] and [hc*dim], one `eh_proj` of [dim+hc*dim] -> [dim].
 *   - additive (what the DeepSeek-V4 HF checkpoint ships): the main hidden state is first
 *     combined by the module's own `hc_head_*`, normalised by `hnorm` [dim], projected by
 *     `h_proj` [dim]->[dim], and ADDED to `e_proj`(enorm(embedding)). Only shape-consistent:
 *     the checkpoint has no [dim+hc*dim] projection, so a sum is the only fusion those shapes
 *     allow. Verified by shape, not against the reference model.py - see docs/mtp_safetensors.md.
 *
 * @par Not taken from the checkpoint
 *   The module's own final head (`mtp.<J>.norm.weight`) is ignored: draft logits come from the
 *   main model's head. A draft is only a proposal - greedy acceptance and rejection sampling keep
 *   the emitted distribution exact whatever head produced it - so this changes acceptance rate,
 *   never correctness.
 *
 * @par What the caller does with the result
 *   The items are appended to the model's tensor table as one synthetic extra "shard" whose bytes
 *   live in a host arena (see Model::load): from there the engine treats the draft layers exactly
 *   like GGUF ones - same names, same residency plan, same KV checkpoint.
 */
#pragma once

#include "dsv4/config.hpp"
#include "dsv4/gguf_header.hpp"
#include "dsv4/st_header.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

/**
 * @struct MtpAdapted
 * @brief Tensors of the adapted draft module, in the engine's names and ggml types.
 *
 * Each entry is a GgufTensor exactly as gguf_read_header() would have produced it (dims in ggml
 * order, `type` a ggml id, `nbytes` the size of `blob`), plus the bytes themselves. `shard` is
 * left at 0: the caller assigns it when it appends these to the model's shard list.
 */
struct MtpAdapted {
    struct Item {
        GgufTensor t;
        std::vector<uint8_t> blob;      ///< The tensor's data, contiguous, at abs_offset 0.
    };
    std::vector<Item> items;

    int fusion_mode = 0;   ///< 0 = none, 1 = concat (eh_proj), 2 = additive (e_proj + h_proj)
    int n_layers = 0;      ///< Draft layers adapted (the `mtp.<J>` indices found).
    uint64_t bytes = 0;    ///< Total converted bytes, for the load report.
    std::string source;    ///< The safetensors file the module came from.
    std::vector<int> inferred_ratios;  ///< Per draft layer, the compression ratio read back from the
                                       ///< checkpoint's `attn.compressor.ape` rows (0 = none). The main
                                       ///< GGUF's compress_ratios may stop at n_layer, so the adapter
                                       ///< recovers it and the caller extends the config.
};

/**
 * @brief Adapt every `mtp.<J>.*` tensor of a safetensors checkpoint to engine tensors.
 *
 * @param sh     The parsed safetensors header (st_read_header).
 * @param c      The model's config, read from its GGUF: supplies dim, ff, expert count, ratios.
 * @param out    The adapted module.
 * @param err    A precise reason on failure: an unusable dtype, a shape that does not match the
 *              layer it would fill, or a missing half of the fusion module.
 * @return true when the module is complete and consistent with the model.
 */
bool mtp_adapt_safetensors(const StHeader& sh, const Dsv4Config& c, MtpAdapted& out, std::string& err);

}  // namespace dsv4

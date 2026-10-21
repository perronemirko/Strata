/**
 * @file st_header.hpp
 * @brief Header-only safetensors reader: metadata + tensor infos (never the weights).
 *
 * The safetensors analogue of gguf_header.hpp, built for one purpose: letting the MTP module
 * (the speculative draft head) come from a Hugging Face checkpoint while the main model stays
 * a GGUF. A safetensors shard is [u64 header length][JSON header][raw tensor data]: this reads
 * the header, indexes every tensor by name, and records the exact byte range of every tensor.
 * The reader never touches the weights: the MTP adapter reads only the ranges it needs.
 *
 * @par Design decisions
 *   - Little-endian hosts only, like the GGUF reader (all modern x86/ARM).
 *   - No JSON dependency: a minimal recursive-descent parser for exactly the header grammar
 *     (objects, arrays of numbers, strings). The header is at most 2^31 bytes by spec.
 *   - The dtype is kept as a string: the adapter, not this reader, decides what it can convert.
 *   - Shapes are stored in safetensors order (row-major, outermost first) — the OPPOSITE of
 *     ggml's dims order. The adapter reverses them; nothing here does.
 *   - `abs_offset` is the absolute byte offset from the start of the file, so `mmap + abs_offset`
 *     is the tensor's first byte, matching how the model layer already addresses GGUF tensors.
 *
 * @par Usage
 *   @code
 *   StHeader h;
 *   std::string err;
 *   st_read_header({"model-00046-of-00046.safetensors"}, h, err);
 *   const StTensor* t = h.find("mtp.0.e_proj.weight");
 *   if (t) printf("%s %s %zu dims\n", t->name.c_str(), t->dtype.c_str(), t->shape.size());
 *   @endcode
 */
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dsv4 {

/**
 * @struct StTensor
 * @brief One tensor of a safetensors shard: name, dtype, shape (row-major), byte range.
 */
struct StTensor {
    std::string name;               ///< Tensor name as in the header (e.g. "mtp.0.e_proj.weight").
    std::string dtype;              ///< Header dtype string verbatim ("BF16", "F32", "F8_E4M3", "I8", ...).
    std::vector<uint64_t> shape;    ///< Shape in safetensors order: outermost dimension first.
    uint64_t abs_offset = 0;        ///< Absolute byte offset in the file where the data starts.
    uint64_t nbytes = 0;            ///< Byte size of the tensor data.
    int shard = 0;                  ///< Index of the file this tensor lives in (0-based).
    bool known_dtype = false;       ///< true when the dtype string is one of the recognised names.

    int64_t numel() const { int64_t n = 1; for (uint64_t d : shape) n *= (int64_t) d; return n; }
};

/**
 * @struct StHeader
 * @brief Parsed headers of one or more safetensors shards: files, metadata, tensor table.
 */
struct StHeader {
    std::vector<std::string> files;                  ///< Shard file paths in order.
    std::vector<uint64_t> file_size;                 ///< Size in bytes of each file.
    std::string metadata;                            ///< The "__metadata__" JSON string of the last shard seen (may be empty).
    std::vector<StTensor> tensors;                   ///< All tensors of all shards, in file order.

    const StTensor* find(const std::string& name) const {
        for (const StTensor& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

/**
 * @brief Parse the headers of one or more safetensors files.
 *
 * Reads the u64 header length, the JSON header, and indexes every tensor. The weights are never
 * read: the shards are mmap'ed read-only and the caller touches only the pages it needs. A name
 * that appears in two shards is an error, as in the GGUF reader.
 *
 * @param paths  Shard files (a single file is the usual case for an MTP-only checkpoint).
 * @param out    The parsed header.
 * @param err    Filled with a human-readable reason on failure.
 * @return true when every file parsed cleanly.
 */
bool st_read_header(const std::vector<std::string>& paths, StHeader& out, std::string& err);

/**
 * @brief Map a safetensors dtype string to its element size in bytes, or 0 when unknown.
 * @param dtype Header dtype string verbatim.
 * @return Bytes per element: 4 for F32 and I32, 8 for F64 and I64, 2 for BF16 and F16, 1 for the
 *   8-bit families (F8_E4M3, F8_E8M0, I8, U8), 0 for anything else.
 */
int st_dtype_bytes(const std::string& dtype);

}  // namespace dsv4

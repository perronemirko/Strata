// dequant_registry.hpp - registro dei dequantizzatori per tipo GGML.
// Dato il tipo (id GGML), restituisce la funzione giusta, la dimensione del blocco e quanti elementi contiene.
// Un tipo non registrato solleva un errore esplicito con il nome: il motore segnala cosa manca, non lo indovina.
// Header-only. Gli id sono quelli di ggml.h verificati in precedenza.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "dequant_iq.hpp"
#include "dequant_kquants.hpp"
#include "dequant_fp4.hpp"
#include "dequant_q5k.hpp"
#include "dequant_float.hpp"

namespace gptoss {

using DequantFn = void (*)(const void* src, float* dst, int64_t k);

struct DequantType {
    int         ggml_id;
    const char* name;
    int64_t     block_elems;  // elementi per blocco (QK)
    size_t      block_bytes;  // byte per blocco (sizeof della struct)
    DequantFn   fn;           // nullptr se non ancora portato
};

// wrapper: converte il puntatore void nella struct del tipo e chiama il port
#define GPT_DQ_WRAP(fname, BlockT) \
    static void dq_##fname(const void* s, float* d, int64_t k) { \
        fname(static_cast<const BlockT*>(s), d, k); }

GPT_DQ_WRAP(dequantize_row_q8_0, block_q8_0)
GPT_DQ_WRAP(dequantize_row_q4_K, block_q4_K)
GPT_DQ_WRAP(dequantize_row_q6_K, block_q6_K)
GPT_DQ_WRAP(dequantize_row_q2_0, block_q2_0)
GPT_DQ_WRAP(dequantize_row_mxfp4, block_mxfp4)
GPT_DQ_WRAP(dequantize_row_iq2_xxs, block_iq2_xxs)
GPT_DQ_WRAP(dequantize_row_iq2_xs, block_iq2_xs)
GPT_DQ_WRAP(dequantize_row_iq2_s, block_iq2_s)
GPT_DQ_WRAP(dequantize_row_iq3_xxs, block_iq3_xxs)
GPT_DQ_WRAP(dequantize_row_iq3_s, block_iq3_s)
GPT_DQ_WRAP(dequantize_row_iq1_s, block_iq1_s)
GPT_DQ_WRAP(dequantize_row_iq1_m, block_iq1_m)
GPT_DQ_WRAP(dequantize_row_iq4_nl, block_iq4_nl)
GPT_DQ_WRAP(dequantize_row_iq4_xs, block_iq4_xs)
GPT_DQ_WRAP(dequantize_row_q5_K, block_q5_K)
GPT_DQ_WRAP(dequantize_row_f32, float)
GPT_DQ_WRAP(dequantize_row_f16, uint16_t)
GPT_DQ_WRAP(dequantize_row_bf16, uint16_t)

#undef GPT_DQ_WRAP

// Tipi GGML (ggml.h): 8 Q8_0, 12 Q4_K, 14 Q6_K, 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 19 IQ1_S,
// 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 29 IQ1_M, 39 MXFP4, 42 Q2_0.
// Non ancora portati (fn == nullptr): Q2_K 10, Q3_K 11, Q5_K 13, Q5_0/Q5_1, Q4_0/Q4_1, F16, BF16, NVFP4 40, Q1_0 41, TQ*.
inline const DequantType* dequant_lookup(int ggml_id) {
    static const DequantType table[] = {
        {8,  "Q8_0",     32,  34,  dq_dequantize_row_q8_0},
        {12, "Q4_K",     256, 144, dq_dequantize_row_q4_K},
        {14, "Q6_K",     256, 210, dq_dequantize_row_q6_K},
        {42, "Q2_0",     64,  18,  dq_dequantize_row_q2_0},
        {39, "MXFP4",    32,  17,  dq_dequantize_row_mxfp4},
        {16, "IQ2_XXS",  256, 66,  dq_dequantize_row_iq2_xxs},
        {17, "IQ2_XS",   256, 74,  dq_dequantize_row_iq2_xs},
        {22, "IQ2_S",    256, 82,  dq_dequantize_row_iq2_s},
        {18, "IQ3_XXS",  256, 98,  dq_dequantize_row_iq3_xxs},
        {21, "IQ3_S",    256, 110, dq_dequantize_row_iq3_s},
        {19, "IQ1_S",    256, 50,  dq_dequantize_row_iq1_s},
        {29, "IQ1_M",    256, 56,  dq_dequantize_row_iq1_m},
        {20, "IQ4_NL",   32,  18,  dq_dequantize_row_iq4_nl},
        {23, "IQ4_XS",   256, 136, dq_dequantize_row_iq4_xs},
        {13, "Q5_K",     256, 176, dq_dequantize_row_q5_K},
        {0,  "F32",      1,   4,   dq_dequantize_row_f32},
        {1,  "F16",      1,   2,   dq_dequantize_row_f16},
        {30, "BF16",     1,   2,   dq_dequantize_row_bf16},
    };
    for (const auto& t : table)
        if (t.ggml_id == ggml_id) return &t;
    return nullptr;
}

// Dimensione in byte di una riga di k elementi (k multiplo del blocco).
inline size_t dequant_row_bytes(int ggml_id, int64_t k) {
    const DequantType* t = dequant_lookup(ggml_id);
    if (!t) throw std::runtime_error("tipo GGML " + std::to_string(ggml_id) + " non registrato nel motore");
    if (k % t->block_elems) throw std::invalid_argument(std::string(t->name) + ": k non multiplo del blocco");
    return (size_t)(k / t->block_elems) * t->block_bytes;
}

// Dequantizza k elementi di tipo ggml_id da src a dst (float). Tipo non registrato: errore con il nome.
inline void dequantize(int ggml_id, const void* src, float* dst, int64_t k) {
    const DequantType* t = dequant_lookup(ggml_id);
    if (!t) throw std::runtime_error("tipo GGML " + std::to_string(ggml_id) + " non registrato nel motore");
    if (!t->fn) throw std::runtime_error(std::string("tipo ") + t->name + " registrato ma non ancora portato");
    if (k % t->block_elems) throw std::invalid_argument(std::string(t->name) + ": k non multiplo del blocco");
    t->fn(src, dst, k);
}

// Controllo a compile-time: le dimensioni dichiarate devono coincidere con le struct.
static_assert(sizeof(block_q8_0) == 34, "q8_0");
static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q6_K) == 210, "q6_K");
static_assert(sizeof(block_q2_0) == 18, "q2_0");
static_assert(sizeof(block_mxfp4) == 17, "mxfp4");
static_assert(sizeof(block_iq2_xxs) == 66, "iq2_xxs");
static_assert(sizeof(block_iq2_xs) == 74, "iq2_xs");
static_assert(sizeof(block_iq2_s) == 82, "iq2_s");
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs");
static_assert(sizeof(block_iq3_s) == 110, "iq3_s");
static_assert(sizeof(block_iq1_s) == 50, "iq1_s");
static_assert(sizeof(block_iq1_m) == 56, "iq1_m");
static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs");
static_assert(sizeof(block_q5_K) == 176, "q5_K");

}  // namespace gptoss

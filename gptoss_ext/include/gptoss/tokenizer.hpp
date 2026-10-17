// gptoss/tokenizer.hpp - the o200k_harmony tokenizer (GPT-2 byte-level BPE) in C++14, no dependencies beyond PCRE2.
//
// Reads the Hugging Face `tokenizer.json` of a gpt-oss snapshot directly: the BPE model (vocab + merges),
// the pre_tokenizer Split regex (compiled with PCRE2 UTF-8 mode - it carries \p{L}/\p{N} classes and inline
// (?i:'s|'t|...) groups) and the added_tokens.  The byte layer is the GPT-2 one: every BYTE maps to a
// printable unicode character, so any UTF-8 input round-trips with no UNK token - decode(encode(s)) == s
// is the check that catches a wrong offset or a merge applied in the wrong order.
//
// Merges are applied LOWEST RANK FIRST (a heap keyed (rank, position), exactly the reference algorithm of
// Strata's tools/strata_tokenizer.py _bpe_heap), not left-to-right: applying them in list order is the
// classic BPE bug that yields a plausible but wrong segmentation.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gptoss {

struct ChatMsg {
    std::string role;      // "user" | "assistant" (the system prompt is build_chat_prompt's own argument)
    std::string content;   // the final-channel text
    std::string thinking;  // the analysis channel ("" = none); kept for assistant turns only
};

class Tokenizer {
public:
    /// Loads <dir>/tokenizer.json. Fails with a clear message on anything it does not recognise
    /// (non-BPE model, no Split regex in pre_tokenizer, an added token outside the vocab).
    static bool load(const std::string& dir, Tokenizer& out, std::string& err);

    int encode(const std::string& text, std::vector<int>& ids) const;   // 0 on success
    /// decode(encode(s)) == s for any UTF-8 input (the byte layer is lossless).
    bool decode(const int* ids, int n, std::string& out) const;
    /// One token -> its raw bytes. A multi-byte character can span several tokens (an emoji is four
    /// single-byte tokens), so a streaming caller must buffer and print only complete code points.
    bool decode_token(int id, std::string& out) const;

    int vocab_size() const { return (int) tok_.size(); }
    const char* name() const { return "o200k_harmony"; }

    Tokenizer();   // default-construct, then load(): the byte maps are set up here

private:
    void init_byte_maps();
    /// One pre-tokenizer piece -> ids. The piece is UTF-8; it is byte-mapped to the GPT-2 alphabet and merged.
    std::vector<int> bpe(const std::string& piece) const;

    std::vector<std::string> tok_;                  // id -> string (the byte-level spelling: 'Ġ' for a space byte)
    std::unordered_map<std::string, int> tok2id_;   // the inverse of tok_
    std::unordered_map<long long, int> rank_;       // merge ((left_id << 32) | right_id) -> list position; lowest first
    std::vector<int> special_ids_;                  // the added_tokens (the Harmony specials), longest content first
    uint16_t b2u_[256];                             // byte -> GPT-2 alphabet code point (up to 323: needs 16 bits)
    unsigned char u2b_[0x1100];                     // alphabet code point -> byte (0 = not in the alphabet)
    std::string pre_pattern_;                       // the Split regex, as read from tokenizer.json
};

/// The Harmony chat template of gpt-oss, rendered for a conversation: the system message (identity + date +
/// reasoning effort), an optional developer block carrying `system`, every user/assistant turn with its
/// channels, and the generation prompt.  `reasoning` is "low" | "medium" | "high".  The output is plain text
/// ready to be encoded: the <|start|>/<|message|> tags are whole vocab tokens the tokenizer maps as specials.
std::string build_chat_prompt(const std::vector<ChatMsg>& msgs, const std::string& system, const std::string& reasoning);

/// The longest prefix of `buf` made of complete UTF-8 code points.  Streaming decode uses it to print per token
/// without ever emitting half an emoji: a multi-byte character arrives as several single-byte tokens, so one
/// token's bytes are not always a valid UTF-8 prefix - print up to here, keep the tail for the next token.
size_t utf8_printable_prefix(const std::string& buf);

/// Splits a raw assistant completion into its channels (the analysis text and the final text).  A reply that
/// carries no <|channel|> tags at all is taken as plain final content.  Feeding the parsed parts back through
/// build_chat_prompt reproduces exactly what the model emitted, which is what the template requires: a message
/// whose content field still contains <|channel|> tags is rejected by it.
bool parse_assistant_reply(const std::string& raw, std::string& thinking, std::string& final_text);

}  // namespace gptoss

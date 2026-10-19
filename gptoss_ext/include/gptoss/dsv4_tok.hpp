// dsv4_tok.hpp - tokenizer di DeepSeek-V4 letto dai metadati GGUF.
// Tipo "gpt2" (BPE a livello di byte) con pre-tokenizer "joyai-llm".
// Regex di pre-tokenizzazione: llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM (letto dalla copia del repo, 3 espressioni).
// Il regex è implementato a mano sui codepoint, con le classi Unicode di uni_cats.hpp, perché std::regex non supporta \p{..}.
//
// ASSUNZIONI:
//  T1 token speciali = token con tipo 3 (CONTROL) o 4 (USER_DEFINED): spezzati dal testo prima del pre-tokenizer.
//  T2 add_bos: il GGUF non ha tokenizer.ggml.add_bos_token. Il default qui è false. Forzabile con add_bos = true.
//  T3 decode: i token di tipo 3 (CONTROL) non producono testo; quelli di tipo 4 producono il loro testo letterale.
//  T4 la lettura in input è UTF-8 valido. Byte non validi diventano U+FFFD.
#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gptoss/gguf_reader.hpp"
#include "gptoss/uni_cats.hpp"

namespace gptoss::tok {

// ---------- UTF-8 ----------
inline std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + n > s.size()) { out.push_back(0xFFFD); break; }
        for (int k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

inline void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char) cp;
    } else if (cp < 0x800) {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char) (0xE0 | (cp >> 12));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    } else {
        s += (char) (0xF0 | (cp >> 18));
        s += (char) (0x80 | ((cp >> 12) & 0x3F));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    }
}

// ---------- mappa byte <-> carattere gpt2 ----------
struct ByteMap {
    uint32_t enc[256];
    std::unordered_map<uint32_t, uint8_t> dec;
};

inline ByteMap make_byte_map() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    bool used[256] = {false};
    for (int b : bs) used[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!used[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    ByteMap m;
    for (size_t i = 0; i < bs.size(); ++i) {
        m.enc[bs[i]] = (uint32_t) cs[i];
        m.dec[(uint32_t) cs[i]] = (uint8_t) bs[i];
    }
    return m;
}

// ---------- pre-tokenizer joyai-llm ----------
inline bool u_is_ws(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}
inline bool u_is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
inline bool u_is_L(uint32_t c) { return uni::is_L(c); }
inline bool u_is_M(uint32_t c) { return uni::is_M(c); }
inline bool u_is_N(uint32_t c) { return uni::is_N(c); }
inline bool u_is_P(uint32_t c) { return uni::is_P(c); }
inline bool u_is_S(uint32_t c) { return uni::is_S(c); }
inline bool u_is_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}
inline bool u_is_ascii_alpha(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool u_is_punct1(uint32_t c) {
    static const char* set = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    if (c >= 128) return false;
    for (const char* p = set; *p; ++p)
        if ((uint32_t) (unsigned char) *p == c) return true;
    return false;
}

// Alternativa 'big' (regex 3 di joyai-llm), in ordine. Restituisce la lunghezza del match in i, o 0.
inline size_t big_match(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    // a) [punct1][A-Za-z]+
    if (u_is_punct1(c[i])) {
        size_t j = i + 1;
        while (j < n && u_is_ascii_alpha(c[j])) ++j;
        if (j > i + 1) return j - i;
    }
    // b) [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    auto isLM = [](uint32_t x) { return u_is_L(x) || u_is_M(x); };
    if (!u_is_nl(c[i]) && !u_is_L(c[i]) && !u_is_P(c[i]) && !u_is_S(c[i]) && i + 1 < n && isLM(c[i + 1])) {
        size_t j = i + 1;
        while (j < n && isLM(c[j])) ++j;
        return j - i;
    }
    if (isLM(c[i])) {
        size_t j = i;
        while (j < n && isLM(c[j])) ++j;
        return j - i;
    }
    // c) ` ?[\p{P}\p{S}]+[\r\n]*`
    {
        size_t j = i;
        if (c[j] == ' ') ++j;
        size_t k = j;
        while (k < n && (u_is_P(c[k]) || u_is_S(c[k]))) ++k;
        if (k > j) {
            while (k < n && u_is_nl(c[k])) ++k;
            return k - i;
        }
    }
    // d) \s*[\r\n]+
    {
        size_t e = i;
        while (e < n && u_is_ws(c[e])) ++e;
        for (size_t q = e; q-- > i;) {
            if (u_is_nl(c[q])) {
                size_t k = q;
                while (k < n && u_is_nl(c[k])) ++k;
                return k - i;
            }
        }
    }
    // e) \s+(?!\S)  ed f) \s+
    if (u_is_ws(c[i])) {
        size_t e = i;
        while (e < n && u_is_ws(c[e])) ++e;
        if (e == n) return e - i;
        if (e - i >= 2) return e - i - 1;
        return e - i;
    }
    return 0;
}

// Applica un matcher a un frammento: i match diventano pezzi, i tratti non coperti diventano pezzi a parte.
template <typename Matcher>
inline void split_stage(const std::vector<std::vector<uint32_t>>& in, Matcher m,
                        std::vector<std::vector<uint32_t>>& out) {
    for (const auto& frag : in) {
        const size_t n = frag.size();
        size_t i = 0, gap = 0;
        while (i < n) {
            const size_t len = m(frag, i);
            if (len > 0) {
                if (gap < i) out.emplace_back(frag.begin() + gap, frag.begin() + i);
                out.emplace_back(frag.begin() + i, frag.begin() + i + len);
                i += len;
                gap = i;
            } else {
                ++i;
            }
        }
        if (gap < n) out.emplace_back(frag.begin() + gap, frag.end());
    }
}

inline std::vector<std::vector<uint32_t>> pretok_joyai(const std::vector<uint32_t>& text) {
    std::vector<std::vector<uint32_t>> s0{text}, s1, s2, s3;
    split_stage(s0, [](const std::vector<uint32_t>& c, size_t i) -> size_t {  // \p{N}{1,3}
        if (!u_is_N(c[i])) return 0;
        size_t len = 1;
        while (len < 3 && i + len < c.size() && u_is_N(c[i + len])) ++len;
        return len;
    }, s1);
    split_stage(s1, [](const std::vector<uint32_t>& c, size_t i) -> size_t {  // [一-龥぀-ゟ゠-ヿ]+
        if (!u_is_cjk(c[i])) return 0;
        size_t j = i;
        while (j < c.size() && u_is_cjk(c[j])) ++j;
        return j - i;
    }, s2);
    split_stage(s2, [](const std::vector<uint32_t>& c, size_t i) -> size_t { return big_match(c, i); }, s3);
    return s3;
}

// ---------- tokenizer ----------
class Tokenizer {
public:
    static Tokenizer from_gguf(const gptoss::GgufFile& f) {
        Tokenizer T;
        const std::string model = f.meta_str("tokenizer.ggml.model", "");
        const std::string pre = f.meta_str("tokenizer.ggml.pre", "");
        if (model != "gpt2") throw std::runtime_error("tokenizer: tipo non supportato '" + model + "' (atteso gpt2)");
        if (pre != "joyai-llm") throw std::runtime_error("tokenizer: pre-tokenizer non supportato '" + pre + "'");
        T.tokens_ = f.meta_strs("tokenizer.ggml.tokens");
        if (T.tokens_.empty()) throw std::runtime_error("tokenizer: tokenizer.ggml.tokens assente");
        for (int64_t v : f.meta_ints("tokenizer.ggml.token_type")) T.ttype_.push_back((int32_t) v);
        if (T.ttype_.size() != T.tokens_.size()) throw std::runtime_error("tokenizer: token_type e tokens di lunghezza diversa");
        for (size_t i = 0; i < T.tokens_.size(); ++i) T.id_of_.emplace(T.tokens_[i], (int32_t) i);
        const auto& merges = f.meta_strs("tokenizer.ggml.merges");
        for (size_t i = 0; i < merges.size(); ++i) T.merge_rank_.emplace(merges[i], (int) i);
        for (size_t i = 0; i < T.tokens_.size(); ++i) {
            if ((T.ttype_[i] == 3 || T.ttype_[i] == 4) && !T.tokens_[i].empty())
                T.specials_.emplace_back(T.tokens_[i], (int32_t) i);
        }
        std::sort(T.specials_.begin(), T.specials_.end(),
                  [](const std::pair<std::string, int32_t>& a, const std::pair<std::string, int32_t>& b) {
                      return a.first.size() > b.first.size();
                  });
        const uint64_t none = ~0ull;
        const uint64_t bos = f.meta_int("tokenizer.ggml.bos_token_id", none);
        const uint64_t eos = f.meta_int("tokenizer.ggml.eos_token_id", none);
        T.bos_ = bos == none ? -1 : (int32_t) bos;
        T.eos_ = eos == none ? -1 : (int32_t) eos;
        T.bm_ = make_byte_map();
        return T;
    }

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    size_t vocab_size() const { return tokens_.size(); }
    bool add_bos = false;  // T2

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> out;
        if (add_bos && bos_ >= 0) out.push_back(bos_);
        std::string buf;
        size_t pos = 0;
        while (pos < text.size()) {
            bool hit = false;
            for (const auto& sp : specials_) {
                if (!sp.first.empty() && text.compare(pos, sp.first.size(), sp.first) == 0) {
                    encode_plain(buf, out);
                    buf.clear();
                    out.push_back(sp.second);
                    pos += sp.first.size();
                    hit = true;
                    break;
                }
            }
            if (!hit) buf += text[pos++];
        }
        encode_plain(buf, out);
        return out;
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        std::string out;
        for (int32_t id : ids) {
            if (id < 0 || (size_t) id >= tokens_.size()) continue;
            const int32_t t = ttype_[id];
            if (t == 3) continue;                       // T3
            if (t == 4) { out += tokens_[id]; continue; }
            for (uint32_t cp : utf8_decode(tokens_[id])) {
                auto it = bm_.dec.find(cp);
                if (it != bm_.dec.end()) out += (char) it->second;
            }
        }
        return out;
    }

private:
    void encode_plain(const std::string& text, std::vector<int32_t>& out) const {
        if (text.empty()) return;
        for (const auto& piece : pretok_joyai(utf8_decode(text))) {
            std::vector<std::string> sym;
            std::string raw;
            for (uint32_t cp : piece) utf8_append(raw, cp);
            for (char ch : raw) {
                std::string s;
                utf8_append(s, bm_.enc[(unsigned char) ch]);
                sym.push_back(s);
            }
            sym = bpe(sym);
            for (const auto& s : sym) {
                auto it = id_of_.find(s);
                if (it != id_of_.end()) out.push_back(it->second);
            }
        }
    }

    std::vector<std::string> bpe(std::vector<std::string> sym) const {
        while (true) {
            int best = INT_MAX;
            size_t bi = std::string::npos;
            for (size_t k = 0; k + 1 < sym.size(); ++k) {
                auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
                if (it != merge_rank_.end() && it->second < best) {
                    best = it->second;
                    bi = k;
                }
            }
            if (bi == std::string::npos) break;
            sym[bi] += sym[bi + 1];
            sym.erase(sym.begin() + (long) bi + 1);
        }
        return sym;
    }

    std::vector<std::string> tokens_;
    std::vector<int32_t> ttype_;
    std::unordered_map<std::string, int32_t> id_of_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> specials_;
    int32_t bos_ = -1, eos_ = -1;
    ByteMap bm_{};
};

}  // namespace gptoss::tok

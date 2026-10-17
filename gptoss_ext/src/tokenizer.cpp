// gptoss/tokenizer.cpp - see include/gptoss/tokenizer.hpp.
//
// The o200k_harmony tokenizer of gpt-oss, self-contained: it reads the snapshot's own HF `tokenizer.json`
// (BPE vocab + merges, the pre_tokenizer Split regex, added_tokens) and needs nothing but PCRE2 for the
// unicode-aware split.  The byte layer is GPT-2's: every BYTE maps to a printable code point, so any UTF-8
// input round-trips with no UNK token - decode(encode(s)) == s is the check that catches a wrong offset or
// a merge applied in the wrong order (tests/test_tokenizer.cpp asserts it against transformers).
#include "gptoss/tokenizer.hpp"

#include "gptoss/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <queue>
#include <sstream>
#include <utility>

#define PCRE2_CODE_UNIT_WIDTH 8   // UTF-8 text, one byte per code unit (must precede pcre2.h)
#include <pcre2.h>

namespace gptoss {

namespace {

// ------------------------------------------------------------------ GPT-2 byte <-> unicode (the whole trick)
// Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0 controls
// and the high range that would be invisible) are shifted into 256+n so no byte is unprintable.  The ORDER of
// the shift is what a wrong tokenizer gets silently wrong: it still round-trips per piece but emits different ids.
void build_byte_maps(uint16_t (&b2u)[256], unsigned char (&u2b)[0x1100]) {
    std::vector<int> bytes, codes;
    for (int b = 0x21; b < 0x7F; ++b) { bytes.push_back(b); codes.push_back(b); }
    for (int b = 0xA1; b < 0xAD; ++b) { bytes.push_back(b); codes.push_back(b); }
    for (int b = 0xAE; b < 0x100; ++b) { bytes.push_back(b); codes.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) { bytes.push_back(b); codes.push_back(256 + n++); }
    }
    std::memset(u2b, 0, sizeof u2b[0] * 0x1100);
    for (size_t i = 0; i < bytes.size(); ++i) { b2u[(unsigned char) bytes[i]] = (uint16_t) codes[i]; u2b[codes[i]] = (unsigned char) bytes[i]; }
}

void utf8_push(std::string& out, int cp) {
    if (cp < 0x80) out += (char) cp;
    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 63)); }
    else if (cp < 0x10000) { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 63)); out += (char) (0x80 | (cp & 63)); }
    else { out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 63)); out += (char) (0x80 | ((cp >> 6) & 63)); out += (char) (0x80 | (cp & 63)); }
}

// ------------------------------------------------------------------ the BPE merge, lowest rank first
// The symbols of one pre-tokenizer piece as a doubly linked list; every adjacent pair that has a merge rule
// sits in a heap keyed (rank, position).  Popping the lowest rank and then the lowest position is exactly the
// order the reference scan picks (its strict < keeps the leftmost of equal ranks); a popped pair whose two
// symbols are no longer the ones it was pushed with is stale and skipped.  O(n log n) in the piece length -
// the naive rescan after every merge takes seconds on an 8K-character CJK run, which one prompt can carry.
struct BpeNode { std::string s; int prv = -1, nxt = -1; bool dead = false; };

std::vector<int> bpe_merge(const std::vector<std::string>& sym, const std::unordered_map<long long, int>& rank,
                           const std::unordered_map<std::string, int>& tok2id) {
    if (sym.empty()) return {};
    struct Item { int rank; int pos; std::string left, right; };
    struct Cmp { bool operator()(const Item& a, const Item& b) const {   // min-heap: the "larger" item has lower priority
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.pos > b.pos;
    } };
    std::vector<BpeNode> nd(sym.size());
    for (size_t i = 0; i < sym.size(); ++i) {
        nd[i].s = sym[i];
        if (i + 1 < sym.size()) nd[i].nxt = (int) (i + 1);
        if (i > 0) nd[i].prv = (int) (i - 1);
    }
    std::priority_queue<Item, std::vector<Item>, Cmp> heap;
    auto id_of = [&tok2id](const std::string& s) -> int {
        std::unordered_map<std::string, int>::const_iterator it = tok2id.find(s);
        return it == tok2id.end() ? -1 : it->second;
    };
    for (size_t i = 0; i + 1 < nd.size(); ++i) {
        const int li = id_of(nd[i].s), ri = id_of(nd[(size_t) i + 1].s);
        if (li < 0 || ri < 0) continue;   // a symbol outside the vocab cannot merge: it stays atomic
        std::unordered_map<long long, int>::const_iterator it = rank.find(((long long) li << 32) | (unsigned) ri);
        if (it != rank.end()) heap.push(Item{it->second, (int) i, nd[i].s, nd[(size_t) i + 1].s});
    }
    while (!heap.empty()) {
        Item it = heap.top(); heap.pop();
        const int i = it.pos;
        if (i < 0 || i >= (int) nd.size() - 1) continue;
        BpeNode& a = nd[(size_t) i];
        const int j = a.nxt;
        // Stale in two ways: its symbols changed, or node i was absorbed by its LEFT neighbour (it is dead and
        // its old fields still match - applying the merge there would splice a live symbol into a dead one).
        if (a.dead || j < 0 || a.s != it.left || nd[(size_t) j].s != it.right) continue;
        a.s += nd[(size_t) j].s;
        const int k = nd[(size_t) j].nxt;
        nd[(size_t) j].dead = true;
        a.nxt = k;
        if (k >= 0) nd[(size_t) k].prv = i;
        // The two new pairs that border the merged symbol.
        const int la = id_of(a.s);
        if (la < 0) return {};   // impossible by construction: a merge only produces vocab tokens
        if (a.prv >= 0) {
            const long long key = ((long long) id_of(nd[(size_t) a.prv].s) << 32) | (unsigned) la;
            std::unordered_map<long long, int>::const_iterator r = rank.find(key);
            if (r != rank.end()) heap.push(Item{r->second, a.prv, nd[(size_t) a.prv].s, a.s});
        }
        if (k >= 0) {
            const long long key = ((long long) la << 32) | (unsigned) id_of(nd[(size_t) k].s);
            std::unordered_map<long long, int>::const_iterator r = rank.find(key);
            if (r != rank.end()) heap.push(Item{r->second, i, a.s, nd[(size_t) k].s});
        }
    }
    std::vector<int> out;
    for (size_t i = 0; i < nd.size(); ++i) {
        if (nd[i].dead) continue;
        const int id = id_of(nd[i].s);
        if (id < 0) return {};   // a symbol outside the vocab: impossible by construction, guard anyway
        out.push_back(id);
    }
    return out;
}

}  // namespace

Tokenizer::Tokenizer() { init_byte_maps(); }

void Tokenizer::init_byte_maps() { build_byte_maps(b2u_, u2b_); }

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
    // Byte-map the UTF-8 piece into the GPT-2 alphabet (one symbol per BYTE), then merge lowest rank first.
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (size_t i = 0; i < piece.size(); ++i) {
        const int cp = b2u_[(unsigned char) piece[i]];
        std::string s; utf8_push(s, cp);
        sym.push_back(std::move(s));
    }
    return bpe_merge(sym, rank_, tok2id_);
}

int Tokenizer::encode(const std::string& text, std::vector<int>& ids) const {
    if (tok_.empty()) { ids.clear(); return 1; }
    // The specials are matched whole, longest first, anywhere in the text (HF added_tokens with single_word=false);
    // the gaps between them go through the pre-tokenizer + BPE.
    // This PCRE2 build refuses NULL error out-parameters (it writes them unconditionally), so pass real ones.
    int ecode = 0; PCRE2_SIZE eoff = 0;
    pcre2_code* re = pcre2_compile((PCRE2_SPTR) pre_pattern_.c_str(), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_ANCHORED, &ecode, &eoff, nullptr);
    if (!re) { ids.clear(); return 1; }
    pcre2_match_data* md = pcre2_match_data_create(32, nullptr);

    size_t pos = 0;
    while (pos < text.size()) {
        // A special token starting here? Longest first.
        bool hit = false;
        for (size_t s = 0; s < special_ids_.size(); ++s) {
            const std::string& t = tok_[(size_t) special_ids_[s]];
            if (text.compare(pos, t.size(), t) == 0) { ids.push_back(special_ids_[s]); pos += t.size(); hit = true; break; }
        }
        if (hit) continue;
        // Otherwise: the next special token anywhere ahead bounds this plain segment.
        size_t bound = text.size();
        for (size_t s = 0; s < special_ids_.size(); ++s) {
            const std::string& t = tok_[(size_t) special_ids_[s]];
            const size_t f = text.find(t, pos);
            if (f != std::string::npos && f < bound) bound = f;
        }
        // Pre-tokenize the plain segment and BPE every piece.  The pattern is total over its alphabet: every
        // character belongs to some alternative, so each match consumes at least one byte - still guard against
        // a zero-length match so a pathological regex cannot loop forever.
        PCRE2_SIZE off = 0;
        const size_t seg_len = bound - pos;
        for (;;) {
            if (off >= seg_len) break;
            // The subject is the segment ONCE; `off` is the start offset inside it.  Passing both an advanced
            // pointer and a non-zero startoffset would skip twice as far (the bug this loop used to have).
            const int rc = pcre2_match(re, (PCRE2_SPTR) text.data() + pos, (PCRE2_SIZE) seg_len, off, 0, md, nullptr);
            if (rc < 0) { pcre2_code_free(re); pcre2_match_data_free(md); ids.clear(); return 1; }
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            const size_t mstart = (size_t) ov[0], mlen = (size_t) (ov[1] - ov[0]);
            if (mlen == 0) { ++off; continue; }   // cannot happen with this pattern; the guard is the loop bound
            const std::vector<int> got = bpe(text.substr(pos + mstart, mlen));
            if (got.empty()) { pcre2_code_free(re); pcre2_match_data_free(md); ids.clear(); return 1; }
            for (size_t g = 0; g < got.size(); ++g) ids.push_back(got[g]);
            off = mstart + mlen;   // ov[0] is already absolute in the segment: do NOT add to the old offset
        }
        pos = bound;
    }
    pcre2_code_free(re);
    pcre2_match_data_free(md);
    return 0;
}

bool Tokenizer::decode_token(int id, std::string& out) const {
    if (id < 0 || id >= (int) tok_.size()) return false;
    const std::string& t = tok_[(size_t) id];
    // Walk the token's code points back to bytes through u2b_.  A vocab token is either a byte-level string
    // (every code point in the alphabet) or an added special (plain ASCII), so every code point must map.
    out.clear();
    size_t p = 0;
    while (p < t.size()) {
        const unsigned char c = (unsigned char) t[p];
        int cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6 && p + 1 < t.size()) { cp = ((c & 31) << 6) | (t[p + 1] & 63); len = 2; }
        else if ((c >> 4) == 14 && p + 2 < t.size()) { cp = ((c & 15) << 12) | (((unsigned char) t[p + 1] & 63) << 6) | ((unsigned char) t[p + 2] & 63); len = 3; }
        else if ((c >> 3) == 30 && p + 3 < t.size()) { cp = ((c & 7) << 18) | (((unsigned char) t[p + 1] & 63) << 12) | (((unsigned char) t[p + 2] & 63) << 6) | ((unsigned char) t[p + 3] & 63); len = 4; }
        else return false;   // malformed token string: not from this vocab
        const unsigned char b = u2b_[cp];
        if (!b && cp >= 0x80) return false;   // a code point outside the byte alphabet cannot come from encode
        out += (char) b;
        p += (size_t) len;
    }
    return true;
}

bool Tokenizer::decode(const int* ids, int n, std::string& out) const {
    out.clear();
    for (int i = 0; i < n; ++i) {
        std::string one;
        if (!decode_token(ids[i], one)) return false;
        out += one;
    }
    return true;
}

size_t utf8_printable_prefix(const std::string& buf) {
    // The longest prefix of `buf` that is a whole number of complete code points: the first offset at which an
    // incomplete multi-byte sequence (or a stray continuation byte) starts.  A streaming decoder prints up to
    // here and keeps the tail for the next token - an emoji arrives as four single-byte tokens, so printing per
    // token would emit three invalid bytes before the fourth completes it.
    size_t i = 0;
    while (i < buf.size()) {
        const unsigned char c = (unsigned char) buf[i];
        int len;
        if (c < 0x80) len = 1;
        else if ((c >> 5) == 6) len = 2;
        else if ((c >> 4) == 14) len = 3;
        else if ((c >> 3) == 30) len = 4;
        else return i;   // a stray continuation byte: print up to here, drop it on flush
        if (i + (size_t) len > buf.size()) return i;            // incomplete at the tail: hold back
        i += (size_t) len;
    }
    return buf.size();
}

bool Tokenizer::load(const std::string& dir, Tokenizer& out, std::string& err) {
    const std::string path = dir + "/tokenizer.json";
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) { err = "no " + path; return false; }
    std::ostringstream ss; ss << f.rdbuf();
    Json root;
    if (!parse_json(ss.str(), root, err)) { err = path + ": " + err; return false; }

    const Json* m = root.get("model");
    if (!m || m->t != Json::Obj) { err = path + ": no model object"; return false; }
    if (m->str("type") != "BPE") { err = path + ": model.type is not BPE"; return false; }
    const Json* vocab = m->get("vocab");
    const Json* merges = m->get("merges");
    // HF writes the vocab as an OBJECT (token -> id); a bare array of strings is tolerated too.
    if (!vocab || (vocab->t != Json::Obj && vocab->t != Json::Arr) || !merges || merges->t != Json::Arr) { err = path + ": model.vocab / model.merges missing"; return false; }

    out.init_byte_maps();

    // The BPE vocab and the added_tokens (the Harmony specials) are DISJOINT ranges: model.vocab holds ids
    // 0..N-1, the added tokens sit above it.  The full vocabulary is their union.
    struct Added { int id; std::string content; };
    std::vector<Added> added;
    const Json* aj = root.get("added_tokens");
    if (aj && aj->t == Json::Arr) {
        for (size_t i = 0; i < aj->a.size(); ++i) {
            const Json& a = aj->a[i];
            if (a.t != Json::Obj || !a.get("content") || !a.get("id")) continue;
            Added x; x.id = (int) a.num("id", -1); x.content = a.str("content");
            if (x.id < 0 || x.content.empty()) { err = path + ": added token " + std::to_string(i) + " is malformed"; return false; }
            added.push_back(x);
        }
    }
    // The full vocabulary is the union of the two ranges: the BPE ids and the added-token ids.  Either may be empty.
    size_t ntok = 0;
    if (vocab->t == Json::Obj) {
        for (size_t i = 0; i < vocab->o.size(); ++i) {
            if (vocab->o[i].second.t != Json::Num || vocab->o[i].second.n < 0) { err = path + ": vocab entry " + vocab->o[i].first + " has a bad id"; return false; }
            if ((size_t) vocab->o[i].second.n + 1 > ntok) ntok = (size_t) vocab->o[i].second.n + 1;
        }
    } else {
        for (size_t i = 0; i < vocab->a.size(); ++i)
            if (vocab->a[i].t != Json::Str) { err = path + ": vocab[" + std::to_string(i) + "] is not a string"; return false; }
        if (vocab->a.size() > ntok) ntok = vocab->a.size();
    }
    for (size_t i = 0; i < added.size(); ++i) if ((size_t) added[i].id >= ntok) ntok = (size_t) added[i].id + 1;

    out.tok_.assign(ntok, std::string());
    std::vector<char> filled(ntok, 0);
    if (vocab->t == Json::Obj) {
        for (size_t i = 0; i < vocab->o.size(); ++i) {
            const int id = (int) vocab->o[i].second.n;
            out.tok_[(size_t) id] = vocab->o[i].first;
            filled[(size_t) id] = 1;
        }
    } else {
        for (size_t i = 0; i < vocab->a.size(); ++i) {
            out.tok_[i] = vocab->a[i].s;
            filled[(size_t) i] = 1;
        }
    }
    for (size_t i = 0; i < added.size(); ++i) {   // the specials: their content fills whatever the BPE left empty
        if ((size_t) added[i].id >= ntok) { err = path + ": added token " + added[i].content + " has an id outside the vocab"; return false; }
        if (!filled[(size_t) added[i].id]) out.tok_[(size_t) added[i].id] = added[i].content;
        out.special_ids_.push_back(added[i].id);
    }
    for (size_t i = 0; i < ntok; ++i) {
        if (out.tok_[i].empty()) { err = path + ": vocab id " + std::to_string(i) + " is missing"; return false; }
        out.tok2id_.emplace(out.tok_[i], (int) i);
    }
    // Longest content first: `<|return|>` must win over any shorter special sharing its start.
    std::sort(out.special_ids_.begin(), out.special_ids_.end(), [&out](int a, int b) {
        const size_t la = out.tok_[(size_t) a].size(), lb = out.tok_[(size_t) b].size();
        return la != lb ? la > lb : a < b;
    });

    // The pre_tokenizer Split regex: a bare Split, or a Sequence containing one.
    const Json* pt = root.get("pre_tokenizer");
    if (!pt || pt->t != Json::Obj) { err = path + ": no pre_tokenizer"; return false; }
    const Json* split = nullptr;
    if (pt->str("type") == "Split") split = pt;
    else {
        const Json* subs = pt->get("pretokenizers");
        if (!subs || subs->t != Json::Arr) { err = path + ": pre_tokenizer is neither a Split nor a Sequence"; return false; }
        for (size_t i = 0; i < subs->a.size(); ++i)
            if (subs->a[i].str("type") == "Split") { split = &subs->a[i]; break; }
    }
    if (!split) { err = path + ": no Split in pre_tokenizer"; return false; }
    const Json* patj = split->get("pattern");
    std::string pat;
    if (patj && patj->t == Json::Obj) pat = patj->str("Regex", patj->str("String"));
    if (pat.empty()) { err = path + ": the Split has no pattern"; return false; }

    // Compile it once, UTF-8: \p{L}/\p{N} and the inline (?i:'s|'t|...) groups are PCRE2-native.  A regex that
    // does not compile here would fail on every encode with an opaque error later, so refuse at load.
    int ecode = 0; PCRE2_SIZE eoff = 0;
    pcre2_code* re = pcre2_compile((PCRE2_SPTR) pat.c_str(), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_ANCHORED, &ecode, &eoff, nullptr);
    if (!re) { err = path + ": pre_tokenizer regex does not compile (pcre2 error " + std::to_string(ecode) + ")"; return false; }
    pcre2_code_free(re);
    out.pre_pattern_ = pat;

    // The merge ranks: (left_id, right_id) -> position in the list.  A merge naming a token outside the vocab
    // would make BPE silently produce ids the model was never trained with, so it is an error here.
    for (size_t i = 0; i < merges->a.size(); ++i) {
        const Json& mg = merges->a[i];
        std::string l, r;
        if (mg.t == Json::Str) {
            const size_t sp = mg.s.find(' ');
            if (sp == std::string::npos || sp + 1 >= mg.s.size()) { err = path + ": merge " + std::to_string(i) + " is not a pair"; return false; }
            l = mg.s.substr(0, sp); r = mg.s.substr(sp + 1);
        } else if (mg.t == Json::Arr && mg.a.size() == 2 && mg.a[0].t == Json::Str && mg.a[1].t == Json::Str) {
            l = mg.a[0].s; r = mg.a[1].s;
        } else { err = path + ": merge " + std::to_string(i) + " has an unknown shape"; return false; }
        const int li = [&]() -> int { auto it = out.tok2id_.find(l); return it == out.tok2id_.end() ? -1 : it->second; }();
        const int ri = [&]() -> int { auto it = out.tok2id_.find(r); return it == out.tok2id_.end() ? -1 : it->second; }();
        if (li < 0 || ri < 0) { err = path + ": merge " + std::to_string(i) + " names a token outside the vocabulary"; return false; }
        out.rank_.emplace(((long long) li << 32) | (unsigned) ri, (int) i);
    }
    return true;
}

// ------------------------------------------------------------------ the Harmony chat template
std::string build_chat_prompt(const std::vector<ChatMsg>& msgs, const std::string& system, const std::string& reasoning) {
    char date[32] = "1970-01-01";
    const std::time_t now = std::time(nullptr);
    const std::tm* lt = std::localtime(&now);
    if (lt) std::strftime(date, sizeof date, "%Y-%m-%d", lt);

    std::string s;
    // The system message is ALWAYS rendered by the template: identity + cutoff + date + reasoning effort.
    s += "<|start|>system<|message|>";
    s += "You are ChatGPT, a large language model trained by OpenAI.\n";
    s += "Knowledge cutoff: 2024-06\n";
    s += std::string("Current date: ") + date + "\n\n";
    s += "Reasoning: " + (reasoning.empty() ? std::string("low") : reasoning) + "\n\n";
    s += "# Valid channels: analysis, commentary, final. Channel must be included for every message.";
    s += "<|end|>";

    // A system/developer message becomes the developer block ("# Instructions").  Skipped entirely when absent.
    if (!system.empty()) {
        s += "<|start|>developer<|message|# Instructions\n\n";
        s += system;
        s += "\n\n<|end|>";
    }

    // Every turn after that.  The template DROPS the analysis channel of every previous assistant turn during
    // inference ("CoT is dropped during all previous turns"), so only the final content goes back in.
    for (size_t i = 0; i < msgs.size(); ++i) {
        const ChatMsg& m = msgs[i];
        if (m.role == "user") {
            s += "<|start|>user<|message|>";
            s += m.content;
            s += "<|end|>";
        } else {   // assistant, a previous turn: the final channel only.
            s += "<|start|>assistant<|channel|>final<|message|>";
            s += m.content;
            s += "<|end|>";
        }
    }
    // The generation prompt: exactly this, no trailing newline (the template's add_generation_prompt branch).
    s += "<|start|>assistant";
    return s;
}

bool parse_assistant_reply(const std::string& raw, std::string& thinking, std::string& final_text) {
    thinking.clear();
    final_text.clear();
    const std::string a_tag = "<|channel|>analysis<|message|>";
    const std::string f_tag = "<|channel|>final<|message|>";
    const size_t fa = raw.find(a_tag);
    const size_t ff = raw.find(f_tag);
    if (fa == std::string::npos && ff == std::string::npos) { final_text = raw; return true; }   // no channels: plain content
    if (ff != std::string::npos) {
        const size_t b0 = ff + f_tag.size();
        const size_t e1 = raw.find("<|return|>", b0);
        const size_t e2 = raw.rfind("<|end|>");
        size_t end = raw.size();
        if (e1 != std::string::npos && (e2 == std::string::npos || e1 < e2)) end = e1;
        else if (e2 != std::string::npos) end = e2;
        final_text = raw.substr(b0, end > b0 ? end - b0 : 0);
    }
    if (fa != std::string::npos) {
        const size_t b0 = fa + a_tag.size();
        const size_t e1 = raw.find("<|end|>", b0);
        thinking = raw.substr(b0, e1 != std::string::npos ? e1 - b0 : std::string::npos);
    }
    return true;
}

}  // namespace gptoss

// m4/json.hpp - minimal JSON reader (config.json, safetensors headers, *.index.json). No dependencies.
#pragma once
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace m4 {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    const Json* get(const std::string& k) const {
        for (const auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    double num(const std::string& k, double d) const { const Json* j = get(k); return j && j->t == Num ? j->n : d; }
    bool boolean(const std::string& k, bool d) const { const Json* j = get(k); return j && j->t == Bool ? j->b : d; }
    std::string str(const std::string& k, const std::string& d = "") const { const Json* j = get(k); return j && j->t == Str ? j->s : d; }
};

namespace detail {
struct P {
    const std::string& s; size_t i = 0; std::string err;
    explicit P(const std::string& x) : s(x) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    bool fail(const char* m) { if (err.empty()) err = std::string(m) + " at byte " + std::to_string(i); return false; }
    bool str(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("string expected");
        ++i; out.clear();
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c == '\\' && i < s.size()) {
                char e = s[i++];
                switch (e) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                    case 'u': { unsigned v = (unsigned) std::strtoul(s.substr(i, 4).c_str(), nullptr, 16); i += 4;
                                if (v < 0x80) out += (char) v; else if (v < 0x800) { out += (char) (0xC0 | v >> 6); out += (char) (0x80 | (v & 63)); }
                                else { out += (char) (0xE0 | v >> 12); out += (char) (0x80 | ((v >> 6) & 63)); out += (char) (0x80 | (v & 63)); } break; }
                    default: out += e;
                }
            } else out += c;
        }
        if (i >= s.size()) return fail("unterminated string");
        ++i; return true;
    }
    bool val(Json& j) {
        ws();
        if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            j.t = Json::Obj; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                ws(); std::string k; if (!str(k)) return false; ws();
                if (i >= s.size() || s[i] != ':') return fail("':' expected");
                ++i; Json v; if (!val(v)) return false;
                j.o.emplace_back(std::move(k), std::move(v)); ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("',' or '}' expected");
            }
        }
        if (c == '[') {
            j.t = Json::Arr; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                Json v; if (!val(v)) return false; j.a.push_back(std::move(v)); ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("',' or ']' expected");
            }
        }
        if (c == '"') { j.t = Json::Str; return str(j.s); }
        if (s.compare(i, 4, "true") == 0) { j.t = Json::Bool; j.b = true; i += 4; return true; }
        if (s.compare(i, 5, "false") == 0) { j.t = Json::Bool; j.b = false; i += 5; return true; }
        if (s.compare(i, 4, "null") == 0) { j.t = Json::Null; i += 4; return true; }
        char* e = nullptr; double d = std::strtod(s.c_str() + i, &e);
        if (e == s.c_str() + i) return fail("value expected");
        j.t = Json::Num; j.n = d; i = (size_t) (e - s.c_str()); return true;
    }
};
}  // namespace detail

inline bool parse_json(const std::string& text, Json& out, std::string& err) {
    detail::P p(text);
    if (!p.val(out)) { err = p.err; return false; }
    return true;
}

}  // namespace m4

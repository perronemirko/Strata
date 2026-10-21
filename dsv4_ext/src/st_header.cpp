// dsv4 safetensors header reader - see dsv4/st_header.hpp.
// Reads ONLY the header of each shard (a few hundred KB at most) and mmap'es the file read-only,
// so the bytes an adapter needs are already mapped when it touches them. Nothing here decodes a
// weight: the dtype -> ggml conversion lives in the MTP adapter (src/mtp_adapter.cpp).
#include "dsv4/st_header.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dsv4 {
namespace {

// ------------------------------------------------------------------ the header is one JSON object
// of string -> {dtype, shape, data_offsets}. A minimal parser for exactly that grammar: the
// safetensors spec caps the header at 2^31 bytes, and its values are only objects, arrays of
// numbers, strings, booleans and null. Keeping it here (rather than a JSON library) keeps the
// extension dependency-free, like the GGUF reader.
struct Jsonv {
    enum { Obj, Arr, Str, Num, Bool, Null } k = Null;
    std::vector<std::pair<std::string, Jsonv>> o;   // Obj
    std::vector<Jsonv> a;                            // Arr
    std::string s;
    double n = 0;
};

struct Parse {
    const char* p; const char* e; std::string err;
    bool fail(const char* m) { if (err.empty()) err = m; return false; }
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; }
    bool lit(const char* w) { const size_t n = std::strlen(w); if ((size_t) (e - p) < n || std::strncmp(p, w, n) != 0) return false; p += n; return true; }
    bool str(std::string& out) {
        if (p >= e || *p != '"') return fail("string expected");
        ++p; out.clear();
        while (p < e && *p != '"') {
            char c = *p++;
            if (c != '\\') { out.push_back(c); continue; }
            if (p >= e) return fail("string ended");
            switch (*p++) {
                case 'n': out.push_back('\n'); break; case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break; case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break; case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break; case '/': out.push_back('/'); break;
                case 'u': {
                    if (e - p < 4) return fail("short \\u");
                    unsigned v = (unsigned) std::strtoul(std::string(p, p + 4).c_str(), nullptr, 16); p += 4;
                    if (v < 0x80) out.push_back((char) v);
                    else if (v < 0x800) { out.push_back((char) (0xC0 | v >> 6)); out.push_back((char) (0x80 | (v & 63))); }
                    else { out.push_back((char) (0xE0 | v >> 12)); out.push_back((char) (0x80 | ((v >> 6) & 63))); out.push_back((char) (0x80 | (v & 63))); }
                    break;
                }
                default: return fail("bad escape");
            }
        }
        if (p >= e) return fail("unterminated string");
        ++p; return true;
    }
    bool val(Jsonv& j) {
        ws();
        if (p >= e) return fail("value expected");
        if (*p == '{') {
            j.k = Jsonv::Obj; ++p; ws();
            if (p < e && *p == '}') { ++p; return true; }
            for (;;) {
                ws(); std::string key; if (!str(key)) return false; ws();
                if (p >= e || *p != ':') return fail("':' expected");
                ++p; Jsonv v; if (!val(v)) return false;
                j.o.emplace_back(std::move(key), std::move(v)); ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == '}') { ++p; return true; }
                return fail("',' or '}' expected");
            }
        }
        if (*p == '[') {
            j.k = Jsonv::Arr; ++p; ws();
            if (p < e && *p == ']') { ++p; return true; }
            for (;;) {
                Jsonv v; if (!val(v)) return false; j.a.push_back(std::move(v)); ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == ']') { ++p; return true; }
                return fail("',' or ']' expected");
            }
        }
        if (*p == '"') { j.k = Jsonv::Str; return str(j.s); }
        if (lit("true")) { j.k = Jsonv::Bool; j.n = 1; return true; }
        if (lit("false")) { j.k = Jsonv::Bool; j.n = 0; return true; }
        if (lit("null")) { j.k = Jsonv::Null; return true; }
        char* endp = nullptr;
        const double d = std::strtod(p, &endp);
        if (endp == p) return fail("value expected");
        p = endp; j.k = Jsonv::Num; j.n = d; return true;
    }
};

}  // namespace

int st_dtype_bytes(const std::string& d) {
    if (d == "F32" || d == "I32") return 4;
    if (d == "F64" || d == "I64") return 8;
    if (d == "BF16" || d == "F16") return 2;
    if (d == "F8_E4M3" || d == "F8_E5M2" || d == "F8_E8M0" || d == "I8" || d == "U8" || d == "B8") return 1;
    return 0;
}

bool st_read_header(const std::vector<std::string>& paths, StHeader& out, std::string& err) {
    out = StHeader();
    std::set<std::string> seen;
    for (size_t si = 0; si < paths.size(); ++si) {
        const std::string& p = paths[si];
        const int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open " + p; return false; }
        struct stat sb;
        if (::fstat(fd, &sb) != 0 || !S_ISREG(sb.st_mode)) { err = "not a regular file: " + p; ::close(fd); return false; }
        const uint64_t fsz = (uint64_t) sb.st_size;
        uint64_t hlen = 0;
        if (fsz < 8 || ::pread(fd, &hlen, 8, 0) != 8) { err = p + ": too short to be a safetensors file"; ::close(fd); return false; }
        // The spec caps the header at 2 GiB; anything larger is a corrupt or truncated file.
        if (hlen == 0 || hlen > (1ull << 31) || 8 + hlen > fsz) { err = p + ": bad header length (" + std::to_string(hlen) + ")"; ::close(fd); return false; }
        std::string hdr((size_t) hlen, '\0');
        if (::pread(fd, &hdr[0], (size_t) hlen, 8) != (ssize_t) hlen) { err = p + ": short header read"; ::close(fd); return false; }

        Jsonv root; Parse parser{hdr.c_str(), hdr.c_str() + hdr.size(), ""};
        if (!parser.val(root)) { err = p + ": header JSON: " + parser.err; ::close(fd); return false; }
        if (root.k != Jsonv::Obj) { err = p + ": header JSON is not an object"; ::close(fd); return false; }

        out.files.push_back(p);
        out.file_size.push_back(fsz);
        const uint64_t data0 = 8 + hlen;
        for (const auto& kv : root.o) {
            if (kv.first == "__metadata__") {
                // Free-form producer metadata (format, author, ...). Kept as a hint for the caller, never required.
                if (kv.second.k == Jsonv::Str) out.metadata = kv.second.s;
                continue;
            }
            const Jsonv* ent = &kv.second;
            if (ent->k != Jsonv::Obj) { err = p + ": tensor entry " + kv.first + " is not an object"; ::close(fd); return false; }
            StTensor t; t.name = kv.first; t.shard = (int) si;
            for (const auto& f : ent->o) {
                if (f.first == "dtype" && f.second.k == Jsonv::Str) t.dtype = f.second.s;
                else if (f.first == "shape" && f.second.k == Jsonv::Arr) { for (const Jsonv& d : f.second.a) if (d.k == Jsonv::Num) t.shape.push_back((uint64_t) d.n); }
                else if (f.first == "data_offsets" && f.second.k == Jsonv::Arr && f.second.a.size() == 2) {
                    if (f.second.a[0].k == Jsonv::Num) t.abs_offset = data0 + (uint64_t) f.second.a[0].n;
                    if (f.second.a[1].k == Jsonv::Num) t.nbytes = (uint64_t) f.second.a[1].n - (uint64_t) f.second.a[0].n;
                }
            }
            if (t.dtype.empty() || t.shape.empty() || !t.nbytes) { err = p + ": tensor " + t.name + " is missing dtype, shape or data_offsets"; ::close(fd); return false; }
            t.known_dtype = st_dtype_bytes(t.dtype) != 0;
            if (t.known_dtype) {
                const int bps = st_dtype_bytes(t.dtype);
                int64_t numel = 1; for (uint64_t d : t.shape) numel *= (int64_t) d;
                if (numel * bps != (int64_t) t.nbytes) {
                    err = p + ": tensor " + t.name + " (" + t.dtype + " x " + std::to_string(numel) +
                          " elements) does not match its byte range (" + std::to_string(t.nbytes) + ")";
                    ::close(fd); return false;
                }
            }
            if (t.abs_offset + t.nbytes > fsz) { err = p + ": tensor " + t.name + " lies outside the file"; ::close(fd); return false; }
            if (!seen.insert(t.name).second) { err = "tensor " + t.name + " appears in two files"; return false; }
            out.tensors.push_back(std::move(t));
        }
        ::close(fd);
    }
    std::sort(out.tensors.begin(), out.tensors.end(), [](const StTensor& a, const StTensor& b) { return a.name < b.name; });
    return true;
}

}  // namespace dsv4

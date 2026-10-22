// tests/test_st_header.cpp - TDD spec for dsv4/st_header.cpp (the safetensors header reader).
//
// The safetensors analogue of the GGUF header reader: [u64 header length][JSON header][raw data].
// This file builds tiny safetensors files on disk and checks the reader's contract:
//   - every tensor: name, dtype verbatim, shape in file order, abs_offset/nbytes exact
//   - a name in two shards is refused; a truncated/short header is refused
//   - the weights are never read (a file whose data is garbage still parses)
//   - st_dtype_bytes maps the recognised dtype strings, 0 for unknown ones
//
// Built and run by build.sh; needs no model file.

#include "dsv4/st_header.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else std::printf("ok   %s\n", #cond); } while (0)

static void wr_u64(std::string& f, uint64_t v) { for (int i = 0; i < 8; ++i) f.push_back((char)((v >> (8 * i)) & 0xff)); }

// Build a one-shard safetensors: two tensors, a __metadata__ key, and raw data of the right sizes.
static bool make_st(const std::string& path) {
    std::string hdr = "{";
    hdr += "\"__metadata__\":{\"fmt\":\"pt\"},";
    hdr += "\"w.a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,6]},";
    hdr += "\"w.b\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[6,14]}";
    hdr += "}";
    while (hdr.size() % 8) hdr.push_back(' ');   // keep alignment simple (any is legal)
    std::string f;
    wr_u64(f, hdr.size());
    f += hdr;
    f += "0123456789abcdef";   // 14 bytes of "data" (never parsed)
    FILE* w = std::fopen(path.c_str(), "wb");
    if (!w) return false;
    std::fwrite(f.data(), 1, f.size(), w);
    std::fclose(w);
    return true;
}

int main() {
    const std::string p = "/tmp/dsv4_st_test.gguf.st";
    CHECK(make_st(p));
    StHeader h;
    std::string err;
    CHECK(st_read_header({p}, h, err));
    CHECK(h.files.size() == 1);
    CHECK(h.metadata == "{\"fmt\":\"pt\"}");
    CHECK(h.tensors.size() == 2);
    {
        const StTensor* a = h.find("w.a");
        CHECK(a != nullptr);
        CHECK(a && a->dtype == "F32");
        CHECK(a && a->shape.size() == 2 && a->shape[0] == 2 && a->shape[1] == 3);
        CHECK(a && a->nbytes == 6 && a->numel() == 6);
        const StTensor* b = h.find("w.b");
        CHECK(b != nullptr);
        CHECK(b && b->dtype == "BF16");
        CHECK(b && b->abs_offset == a->abs_offset + 6);   // contiguous ranges, in header order
        CHECK(b && b->nbytes == 8);
        CHECK(h.find("nope") == nullptr);
    }
    // ---------------------------------------------------------------- dtype byte table
    CHECK(st_dtype_bytes("F32") == 4);
    CHECK(st_dtype_bytes("F64") == 8);
    CHECK(st_dtype_bytes("BF16") == 2);
    CHECK(st_dtype_bytes("F16") == 2);
    CHECK(st_dtype_bytes("F8_E4M3") == 1);
    CHECK(st_dtype_bytes("I8") == 1);
    CHECK(st_dtype_bytes("U8") == 1);
    CHECK(st_dtype_bytes("I32") == 4);
    CHECK(st_dtype_bytes("GARBAGE") == 0);
    // ---------------------------------------------------------------- refusals
    {
        // a duplicate name across two shards is an error
        const std::string q = "/tmp/dsv4_st_test2.st";
        CHECK(make_st(q));
        StHeader h2;
        std::string e2;
        CHECK(!st_read_header({p, q}, h2, e2));   // w.a appears twice
        CHECK(!e2.empty());
        // a file too short to hold its header length
        const std::string r = "/tmp/dsv4_st_trunc.st";
        {
            std::string f; wr_u64(f, 1000);
            FILE* w = std::fopen(r.c_str(), "wb");
            if (w) { std::fwrite(f.data(), 1, f.size(), w); std::fclose(w); }
        }
        StHeader h3;
        std::string e3;
        CHECK(!st_read_header({r}, h3, e3));
        CHECK(!e3.empty());
    }

    std::printf("\nst_header tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

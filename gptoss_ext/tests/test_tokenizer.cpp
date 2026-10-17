// tests/test_tokenizer.cpp - the o200k_harmony tokenizer checked against itself and its invariants.
//
// Usage: test_tokenizer DIR [mode]
//   mode enc  : read lines from stdin, print "id,id,id" per line (what tools/compare_tokenizer.py diffs)
//   mode dec  : read "id,id,id" lines from stdin, print the decoded text per line
//   mode rt    : read lines from stdin, assert decode(encode(s)) == s and print OK / FAIL <line>
// The real cross-check against transformers is tools/compare_tokenizer.py (it runs both and diffs).
#include "gptoss/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace gptoss;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s DIR [enc|dec|rt]\n", argv[0]); return 2; }
    Tokenizer tk;
    std::string err;
    if (!Tokenizer::load(argv[1], tk, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const std::string mode = argc > 2 ? argv[2] : "enc";
    if (mode == "arg") {   // encode the string given as argv[3], newlines and all: no line-based stdin involved
        const std::string text = argc > 3 ? argv[3] : "";
        std::vector<int> ids;
        if (tk.encode(text, ids) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%s", ids[i], i + 1 < ids.size() ? "," : "\n");
        return 0;
    }
    if (mode == "rtarg") {   // encode THEN decode argv[3] in one process: the exact round trip, no stdin involved
        const std::string text = argc > 3 ? argv[3] : "";
        std::vector<int> ids;
        if (tk.encode(text, ids) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        std::string out;
        if (!tk.decode(ids.data(), (int) ids.size(), out) || out != text) {
            std::fprintf(stderr, "FAIL round-trip: %zu in -> %zu out\n", text.size(), out.size());
            return 1;
        }
        std::printf("OK %zu tokens\n", ids.size());
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (mode == "enc") {
            std::vector<int> ids;
            if (tk.encode(line, ids) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
            for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%s", ids[i], i + 1 < ids.size() ? "," : "\n");
        } else if (mode == "dec") {
            std::vector<int> ids;
            size_t start = 0;
            for (;;) {
                const size_t c = line.find(',', start);
                ids.push_back(std::atoi(line.substr(start, c - start).c_str()));
                if (c == std::string::npos) break;
                start = c + 1;
            }
            std::string out;
            if (!tk.decode(ids.data(), (int) ids.size(), out)) { std::fprintf(stderr, "decode failed\n"); return 1; }
            std::printf("%s\n", out.c_str());
        } else if (mode == "rt") {
            std::vector<int> ids;
            if (tk.encode(line, ids) != 0) { std::fprintf(stderr, "FAIL encode: %s\n", line.substr(0, 64).c_str()); return 1; }
            std::string out;
            if (!tk.decode(ids.data(), (int) ids.size(), out) || out != line) { std::fprintf(stderr, "FAIL round-trip: %s -> %s\n", line.substr(0, 64).c_str(), out.substr(0, 64).c_str()); return 1; }
            std::printf("OK %zu tokens\n", ids.size());
        } else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2; }
    }
    return 0;
}

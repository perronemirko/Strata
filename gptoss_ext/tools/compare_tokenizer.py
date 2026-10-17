#!/usr/bin/env python3
"""compare_tokenizer.py - diff the C++ o200k_harmony tokenizer (build-out/test_tokenizer) against transformers.

Usage: compare_tokenizer.py MODEL_DIR [N]
  N = number of random-ish strings to generate on top of the fixed corpus (default 40).
Exits non-zero on the first disagreement, printing both id lists and the offending string.
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, "build-out", "test_tokenizer")


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/albus/.lmstudio/hub/models/lmstudio-community/gpt-oss-120b"
    n_rand = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    corpus = [
        "Ciao, come va?",
        "Hello world! This is a test.",
        "C++14 e CUDA 12.8: la GPU non si tocca.",
        "emoji 🚀🎉 and unicode èàòù ñ",
        "it's don't we're you've I'm let's they'd",          # the (?i:'s|'t|...) contraction classes
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "1234567890 0.5 -3 +42 1e10",                        # numbers: \p{N}{1,3} groups of three
        "line one\nline two\r\nwindows line\ttabbed",         # whitespace classes
        "punctuation !@#$%^&*()_+-=[]{};':\",.<>/?~`|\\",    # the ?[^\s\p{L}\p{N}]+ run
        "café naïve résumé über straße",                      # accented letters (multi-byte UTF-8)
        "日本語のテキストと英語が混ざっている",                  # CJK: long runs, one piece per character class
        "👨‍👩‍👧‍👦 family emoji with ZWJ and skin tone 👋🏽",      # 4-byte + combining sequences
        "a" * 300,                                            # a long run: the heap path of _bpe
        "word " * 200,                                        # repeated pieces (the cache case)
        "<|start|>user<|message|>hi<|end|>",                  # specials matched whole
        "text with <|return|> inside",                        # a special in the middle of plain text
        "",                                                   # empty string: zero tokens, no crash
        " ",                                                  # a lone space byte (Ġ)
        "\u0120t\u0120w",                                     # raw alphabet characters as input
    ]
    rng = random.Random(42)
    alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-éàòùñ🚀中")
    for _ in range(n_rand):
        corpus.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 400))))

    bad = 0
    for i, s in enumerate(corpus):
        want = tk.encode(s, add_special_tokens=False)
        # arg mode: the string goes through argv, so embedded newlines survive (stdin enc/dec are line-based).
        p = subprocess.run([EXE, model_dir, "arg", s], capture_output=True, text=True)
        if p.returncode != 0:
            print("FAIL encode rc=%d on %r\nstderr: %s" % (p.returncode, s[:80], p.stderr.strip()))
            bad += 1
            continue
        got = [int(x) for x in p.stdout.replace("\n", ",").split(",") if x.strip()]
        if got != want:
            print("MISMATCH on case %d: %r\n  cpp    %s\n  py     %s" % (i, s[:80], got[:24], want[:24]))
            bad += 1
        # round trip through the C++ decoder must be exact. rtarg does encode+decode in ONE process so an
        # embedded newline cannot be lost to a line-based stdin (an empty id list has nothing to decode).
        if want:
            p = subprocess.run([EXE, model_dir, "rtarg", s], capture_output=True, text=True)
            if p.returncode != 0:
                print("ROUND-TRIP FAIL on case %d: %r\n%s" % (i, s[:80], p.stderr.strip()))
                bad += 1

    print("%d cases, %d failures" % (len(corpus), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

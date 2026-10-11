#!/usr/bin/env python3
"""Builds the expert-usage profile from representative prompts. Routing does not depend on the device: the CPU binary (build-out/gptoss_run) is enough,
no GPU needed; the engine rewrites the profile after every request, so Ctrl-C loses nothing.

  python3 tools/calibrate_profile.py --model /path/gpt-oss-120b --profile /var/tmp/gptoss.prof --prompts prompts.txt --max-new 128

prompts.txt: one prompt per line, as comma-separated TOKEN IDS (e.g. 200006,17360,...). With --tokenizer-dir DIR (a HF snapshot, needs `transformers`)
plain-text lines are rendered with the chat template instead (that branch is not tested). Use prompts that look like what you will really ask: the
profile only helps if the calibration traffic resembles the real traffic."""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build-out", "gptoss_run"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("engine_args", nargs="*", help="extra engine flags after --")
    a = ap.parse_args()
    tok = None
    if a.tokenizer_dir:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    p = subprocess.Popen([a.exe, "--model", a.model, "--ctx", str(a.ctx), "--serve", "--expert-profile", a.profile] + a.engine_args,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
        if line.startswith("ERR"):
            sys.exit(line)
    n = 0
    for raw in open(a.prompts, encoding="utf-8"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.replace(",", "").isdigit():
            ids = raw
        elif tok:
            ids = ",".join(map(str, tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=True, add_generation_prompt=True)))
        else:
            sys.exit("a text line needs --tokenizer-dir: %r" % raw[:40])
        p.stdin.write("GEN %d temperature=0.7 %s\n" % (a.max_new, ids)); p.stdin.flush()
        for line in p.stdout:
            if line.startswith("DONE") or line.startswith("ERR"):
                break
        n += 1
        print("prompt %d done" % n, flush=True)
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait()
    print("profile: %s (%d prompts). Now: python3 tools/profile_report.py %s" % (a.profile, n, a.profile))


if __name__ == "__main__":
    main()

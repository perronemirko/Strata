#!/usr/bin/env python3
"""Fase 1: legge SOLO gli header dei file .safetensors (locali o su Hugging Face) e stampa nome, dtype, forma, byte.
Risolve le incognite 'forme dei tensori' di docs/mistral_small4_porting.md senza scaricare i pesi.

  python3 tools/dump_safetensors_shapes.py /path/to/snapshot/model-0000*.safetensors > shapes.txt
  python3 tools/dump_safetensors_shapes.py --hf mistralai/Mistral-Small-4-119B-2603 [--token HF_TOKEN] > shapes.txt

--hf scarica per ogni shard solo i primi 8 byte + l'header JSON (HTTP Range): decine di KB, non 242 GB.
(Il ramo --hf non e' stato eseguito dall'autore: il sandbox non raggiunge huggingface.co. Il ramo locale e' testato.)"""
import argparse
import json
import struct
import sys
import urllib.request

DT = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I32": 4, "I64": 8, "U8": 1, "I8": 1}


def local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def remote_header(url, token):
    def rng(a, b):
        r = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
        if token:
            r.add_header("Authorization", "Bearer " + token)
        return urllib.request.urlopen(r).read()
    n = struct.unpack("<Q", rng(0, 7))[0]
    return json.loads(rng(8, 8 + n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--hf")
    ap.add_argument("--token")
    ap.add_argument("--prefix", default="model-")
    a = ap.parse_args()
    heads = []
    if a.hf:
        idx = json.loads(urllib.request.urlopen("https://huggingface.co/%s/resolve/main/model.safetensors.index.json" % a.hf).read())
        for fn in sorted(set(idx["weight_map"].values())):
            if fn.startswith(a.prefix):
                heads.append((fn, remote_header("https://huggingface.co/%s/resolve/main/%s" % (a.hf, fn), a.token)))
    for p in a.files:
        heads.append((p, local_header(p)))
    tot, by_dt = 0, {}
    for fn, h in heads:
        print("# shard", fn)
        for k in sorted(h):
            if k == "__metadata__":
                continue
            t = h[k]
            nb = t["data_offsets"][1] - t["data_offsets"][0]
            n = 1
            for d in t["shape"]:
                n *= d
            if DT.get(t["dtype"]) and n * DT[t["dtype"]] != nb:
                print("!! %s: byte count %d != numel*dtype %d" % (k, nb, n * DT[t["dtype"]]), file=sys.stderr)
            print("%s\t%s\t%s\t%d" % (k, t["dtype"], "x".join(map(str, t["shape"])), nb))
            tot += nb
            by_dt[t["dtype"]] = by_dt.get(t["dtype"], 0) + nb
    print("# total %.3f GiB  %s" % (tot / 2 ** 30, "  ".join("%s=%.3f GiB" % (k, v / 2 ** 30) for k, v in sorted(by_dt.items()))))


if __name__ == "__main__":
    main()

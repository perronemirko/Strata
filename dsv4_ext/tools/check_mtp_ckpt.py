#!/usr/bin/env python3
"""check_mtp_ckpt.py - prints the tensor names/logical shapes of an MTP safetensors checkpoint.

Pure inspection: it never touches the GPU and never loads a model, so it is safe to run while
the dsv4 serve engine holds the whole VRAM. Use it to see WHICH names a checkpoint ships,
against the names src/mtp_adapter.cpp accepts.

    python3 tools/check_mtp_ckpt.py /media/.../DeepSeekV4/model.safetensors
"""
from __future__ import annotations

import json
import struct
import sys


def header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_mtp_ckpt.py FILE.safetensors")
        return 2
    h = header(sys.argv[1])
    meta = h.pop("__metadata__", None)
    print(f"tensors: {len(h)}  metadata: {meta}")
    for k in sorted(h):
        v = h[k]
        print(f"{k:52s} {v['dtype']:8s} {v['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

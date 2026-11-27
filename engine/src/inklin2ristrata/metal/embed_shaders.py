#!/usr/bin/env python3
# engine/src/inklin2ristrata/metal/embed_shaders.py - Embed Inklin2RiStrata-2-Mini MSL shaders into a C++ header.
# MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: embed_shaders.py <metal_src_dir> <out_header>\n")
        return 2
    base = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
    out = [
        "#pragma once",
        "namespace inklin2ristrata_metal {",
    ]
    pairs = [
        ("kInkMetalCodecsSrc", "inklin2ristrata_metal_codecs.metal"),
        ("kInkMetalOpsSrc", "inklin2ristrata_metal_ops.metal"),
        ("kInkMetalDsparkSrc", "inklin2ristrata_metal_dspark.metal"),
        ("kInkMetalVisionSrc", "inklin2ristrata_metal_vision.metal"),
        ("kInkMetalMppSrc", "inklin2ristrata_metal_mpp.metal"),
    ]
    for var, fn in pairs:
        txt = (base / fn).read_text(encoding="utf-8")
        out.append(f'static const char {var}[] = R"INKMSL({txt})INKMSL";')
    out.append("}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

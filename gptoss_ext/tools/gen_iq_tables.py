#!/usr/bin/env python3
# Estrae le tabelle dei codebook IQ da ggml-common.h (copia di riferimento della fonte, letta una volta sola)
# e genera include/gptoss/iq_tables.hpp. Nessuna trascrizione a mano: i numeri vengono copiati dal sorgente,
# e il numero di elementi viene verificato contro la dimensione dichiarata.
# Uso: python3 tools/gen_iq_tables.py <percorso/ggml-common.h> [output.hpp]
import hashlib, re, sys

TABLES = ["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid",
          "iq3xxs_grid", "iq3s_grid", "iq1s_grid", "kvalues_iq4nl"]

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "include/gptoss/iq_tables.hpp"
    src = open(src_path, encoding="utf-8").read()

    defines = dict(re.findall(r"^#define\s+(\w+)\s+(\d+)\s*$", src, re.M))

    def resolve(expr):
        expr = expr.strip()
        if expr.isdigit():
            return int(expr)
        if expr in defines:
            return int(defines[expr])
        raise SystemExit("dimensione non risolta: " + expr)

    out = [
        "// iq_tables.hpp - codebook IQ estratti da ggml-common.h (generato da tools/gen_iq_tables.py: NON modificare a mano).",
        "// sha256 del sorgente: " + hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace gptoss {",
        "namespace iq {",
        "",
    ]
    report = []
    for name in TABLES:
        # prima occorrenza della tabella (per iq1s_grid la variante C, non quella GPU)
        m = re.search(r"GGML_TABLE_BEGIN\((\w+),\s*" + name + r",\s*([^)]+)\)\s*\n(.*?)GGML_TABLE_END\(\)",
                      src, re.S)
        if not m:
            raise SystemExit("tabella non trovata: " + name)
        ctype, size_expr, body = m.group(1), m.group(2), m.group(3)
        size = resolve(size_expr)
        body = re.sub(r"//.*", "", body)
        values = [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
        if len(values) != size:
            raise SystemExit("%s: attesi %d elementi, trovati %d" % (name, size, len(values)))
        out.append("static const %s %s[%d] = {" % (ctype, name, size))
        for i in range(0, len(values), 8):
            out.append("    " + ", ".join(values[i:i + 8]) + ",")
        out.append("};")
        out.append("")
        report.append("%s %s[%d]" % (ctype, name, size))

    out += ["}  // namespace iq", "}  // namespace gptoss", ""]
    open(out_path, "w", encoding="utf-8").write("\n".join(out))
    print("ok: " + out_path)
    for r in report:
        print("  " + r)

if __name__ == "__main__":
    main()

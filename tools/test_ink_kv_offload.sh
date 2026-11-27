#!/bin/sh
# Run the static KV-offload test (no GPU, no downloads): the auto Host KV Offload split,
# the grow ledger, the DSpark ctx steps and the position mapping, with the numbers measured
# on the RTX 4090 on 2026-11-25 (models/inkling-2-mini/serve.log).
# Usage: sh tools/test_ink_kv_offload.sh
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/engine/tests/core/ink_kv_offload_test.cpp"
INC="$ROOT/engine/include"
OUT="${TMPDIR:-/tmp}/ink_kv_offload_test.$$"

[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

echo "compiling (C++14, host-only) ..."
g++ -std=c++14 -O2 -I "$INC" "$SRC" -o "$OUT"

echo "running ..."
"$OUT"
rm -f "$OUT"
echo "test_ink_kv_offload: ok"

#!/bin/sh
# Test of the gateway and the gate, WITHOUT a GPU: the emulated device and tests/tiny.
# It does not test the numbers of gpt-oss (test_gptoss_core does): it tests the routing and the certification.
# Idempotent: it removes its own reference and record file at the start.
set -e
cd "$(dirname "$0")/.."

BIN=build-out/gptoss_run_cpu_gate
REF=/var/tmp/gate_tiny_ref.txt
rm -f "$REF" arch/certifications.txt

if [ ! -f "$BIN" ]; then
  g++ -std=c++17 -O3 -Wall -Wextra -pthread -fopenmp -Iinclude \
    src/gptoss_run_main.cpp src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp \
    src/ops.cpp src/model.cpp src/sampler.cpp src/gguf_loader.cpp src/gpu.cpp src/prefill_dev_stub.cpp \
    -o "$BIN"
fi

fail() { echo "FAIL: $1"; exit 1; }

echo "1) no record: the gateway must refuse"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 64 --selfcheck --selfcheck-file "$REF" --gate-info > /var/tmp/gate1.txt 2>&1 || true
grep -q "refused" /var/tmp/gate1.txt || fail "1: the gateway admitted without a record"
grep -q "status=UNKNOWN" /var/tmp/gate1.txt || fail "1: status not UNKNOWN"

echo "2) selfcheck: writes the reference, the record is INCONCLUSIVE (no comparison happened)"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 64 --selfcheck --selfcheck-file "$REF" > /var/tmp/gate2.txt 2>&1 || true
grep -q "reference written" /var/tmp/gate2.txt || fail "2: reference not written"
grep -q "INCONCLUSIVE" arch/certifications.txt || fail "2: no INCONCLUSIVE record"

echo "3) selfcheck again: MATCH, and the record becomes MATCH"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 64 --selfcheck --selfcheck-file "$REF" > /var/tmp/gate3.txt 2>&1 || true
grep -q "MATCH with the reference" /var/tmp/gate3.txt || fail "3: no MATCH"
grep -q "MATCH$" arch/certifications.txt || fail "3: no MATCH record"

echo "4) same flags: the gateway admits the adapter"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 64 --selfcheck --selfcheck-file "$REF" --gate-info > /var/tmp/gate4.txt 2>&1 || true
grep -q "status=MATCH" /var/tmp/gate4.txt || fail "4: status not MATCH"
grep -q "open_from_spec -> adapter" /var/tmp/gate4.txt || fail "4: the gateway refused"

echo "5) one flag changed (--ctx 48): the record of --ctx 64 must NOT authorize it"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 48 --selfcheck --selfcheck-file "$REF" --gate-info > /var/tmp/gate5.txt 2>&1 || true
grep -q "refused" /var/tmp/gate5.txt || fail "5: a different configuration was admitted"

echo "6) the adapter path gives the same tokens as the direct path"
"$BIN" --model tests/tiny --gpu-emu --cuda --ctx 64 --gateway-info > /var/tmp/gate6.txt 2>&1
grep -q "^GATEWAY_TOKENS" /var/tmp/gate6.txt || fail "6: the gateway did not run"
gw=$(sed -n 's/^GATEWAY_TOKENS //p' /var/tmp/gate6.txt | head -1)
direct=$(sed -n 's/^selfcheck: .* tokens //p' /var/tmp/gate3.txt | head -1)
[ -n "$gw" ] || fail "6: no gateway tokens"
[ -n "$direct" ] || fail "6: no direct tokens"
[ "$gw" = "$direct" ] || fail "6: gateway $gw != direct $direct"

echo "7) a model without arch.json: refused, no fallback"
mkdir -p /var/tmp/noarch
cp tests/tiny/*.safetensors tests/tiny/config.json /var/tmp/noarch/ 2>/dev/null || true
"$BIN" --model /var/tmp/noarch --gpu-emu --cuda --ctx 64 --gate-info > /var/tmp/gate7.txt 2>&1 || true
grep -q "arch.json absent" /var/tmp/gate7.txt || fail "7: not refused on the missing spec"

echo "8) an architecture with no adapter (tests/gguf): refused, and the diagnostic says why"
python3 tools/inspect_model.py tests/gguf/test.gguf > /dev/null || true
"$BIN" --model tests/gguf --gpu-emu --cuda --ctx 64 --gate-info > /var/tmp/gate8.txt 2>&1 || true
grep -q "registered=0" /var/tmp/gate8.txt || fail "8: the registry says the adapter exists"
grep -q "not supported" /var/tmp/gate8.txt || fail "8: no diagnostic of what is not supported"
grep -q "semantics unknown" /var/tmp/gate8.txt || fail "8: no diagnostic of what is unknown"

echo "all ok: the gateway routes, the gate certifies, no silent fallback"

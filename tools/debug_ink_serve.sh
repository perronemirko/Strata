#!/bin/sh
# Debug run + check for the exit -11 at 43,008 of 43,649 prompt tokens (models/inkling-2-mini/serve.log, 2026-11-25).
#
# The engine now logs every request, step, checkpoint save and draft call with STRATA_INK_TRACE=1
# (inklin2ristrata_serve.h). A segfault leaves no error line, so the LAST [ink_trace] line before the
# crash names the call that faulted.
#
# Usage:
#   sh tools/debug_ink_serve.sh run      # 1. rebuild the CUDA engine (the card holder is never killed)
#                                        # 2. start the server with STRATA_INK_TRACE=1, log to debug-trace.log
#   sh tools/debug_ink_serve.sh check    # 3. print the last [ink_trace] line before the crash
#
# Bisect switches (export ONE at a time before run, to name the guilty path):
#   STRATA_INK_NO_GRAPH=1   eager launches: CUDA-Graph capture path out
#   STRATA_INK_SPEC=block   linear block-8 drafter instead of the beam tree
#   STRATA_INK_OOM_RETRY=0  no chunk halving retry: first failing allocation ends the engine
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# The engine's stderr (all [ink_trace] lines) goes to its own log, named in the crash message.
LOG=models/inkling-2-mini/serve.log
SERVER_LOG=models/inkling-2-mini/debug-trace.log

cmd="${1:-run}"

if [ "$cmd" = run ]; then
    # 1. rebuild the integrated CUDA engine so it carries the trace; build only, never kills the serving process
    sh tools/build_launch_inklin2ristrata.sh build --cuda
    # 2. serve with the trace on
    echo "serving with STRATA_INK_TRACE=1; send the long prompt, then run: sh tools/debug_ink_serve.sh check"
    STRATA_INK_TRACE=1 python3 -m terminal.serve.server \
        --engine strata \
        --config models/strata-inklin2ristrata.json \
        --host 127.0.0.1 --port 8080 \
        2>&1 | tee "$SERVER_LOG"
elif [ "$cmd" = check ]; then
    [ -f "$LOG" ] || { echo "no $LOG: run 'run' first"; exit 1; }
    # 1. the crash notice the server prints (server log)
    grep -n 'exit code -11' "$SERVER_LOG" | tail -n 3 || echo "no -11 in the server log"
    # 2. in the engine log a segfault leaves no line: a restart ("engine started") right after
    #    trace lines without a clean end = the crash; the last trace line before it names the guilty call
    awk '
        /\[ink_trace\]/ { last = NR; line = $0 }
        /engine started/ {
            if (last) { print "restart at line " NR ": last trace before it (line " last "): " line; last = 0; line = "" }
        }
        END { if (last) print "engine still alive: last trace line " last ": " line }
    ' "$LOG"
    # 3. the trace of the last (crashed or live) engine run, for reading the batch
    grep -n 'ink_trace' "$LOG" | tail -n 40 || true
else
    echo "usage: sh tools/debug_ink_serve.sh run | check"
    exit 1
fi

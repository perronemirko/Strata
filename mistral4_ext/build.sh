#!/bin/sh
# Builds the CPU engine and runs the tests. No CMake needed.  CXX=g++ sh build.sh
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
OUT=build-out
mkdir -p "$OUT"
OMP=
echo 'int main(){return 0;}' > "$OUT/omp_probe.cpp"
if $CXX -fopenmp "$OUT/omp_probe.cpp" -o "$OUT/omp_probe" 2>/dev/null; then OMP=-fopenmp; else echo "WARNING: no OpenMP: the engine will run on ONE core" >&2; fi
FLAGS="-std=c++17 -O3 -Wall -Wextra $OMP ${M4_MARCH:+-march=$M4_MARCH} -Iinclude"
SRC="src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp src/ops.cpp src/model.cpp src/sampler.cpp"
if [ ! -f tests/tiny/golden.txt ]; then python3 tools/tiny_model.py tests/tiny; fi
$CXX $FLAGS tests/test_m4_core.cpp $SRC -o "$OUT/test_m4_core"
"$OUT/test_m4_core" tests/tiny
if [ -f src/m4_run_main.cpp ]; then $CXX $FLAGS src/m4_run_main.cpp $SRC -o "$OUT/m4_run"; echo "built $OUT/m4_run"; fi

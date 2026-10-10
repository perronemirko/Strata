#!/bin/sh
# CPU build + tests (EMULATED device, no GPU needed):   sh build.sh
# CUDA build (real device, separate binary):             sh build.sh --cuda      [NVCC=nvcc CUDA_ARCH=sm_89 | native]
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}; OUT=build-out; mkdir -p "$OUT"; OMP=
echo 'int main(){return 0;}' > "$OUT/omp_probe.cpp"
if $CXX -fopenmp "$OUT/omp_probe.cpp" -o "$OUT/omp_probe" 2>/dev/null; then OMP=-fopenmp; else echo "WARNING: no OpenMP: the engine will run on ONE core" >&2; fi
COMMON="src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp src/ops.cpp src/model.cpp src/sampler.cpp"

if [ "$1" = "--cuda" ]; then
  NVCC=${NVCC:-nvcc}; ARCH=${CUDA_ARCH:-native}
  # never links src/gpu.cpp (the emulated device): src/gpu_cuda.cu defines the same symbols
  $NVCC -O3 -std=c++17 -arch="$ARCH" ${OMP:+-Xcompiler -fopenmp} -Iinclude src/gptoss_run_main.cpp $COMMON src/gpu_cuda.cu ${OMP:+-lgomp} -o "$OUT/gptoss_run_cuda"
  echo "built $OUT/gptoss_run_cuda"
  exit 0
fi

FLAGS="-std=c++17 -O3 -Wall -Wextra $OMP ${GPTOSS_MARCH:+-march=$GPTOSS_MARCH} -Iinclude"
SRC="$COMMON src/gpu.cpp"
if [ ! -f tests/tiny/golden.txt ]; then python3 tools/tiny_gptoss.py tests/tiny; fi
$CXX $FLAGS tests/test_gptoss_core.cpp $SRC -o "$OUT/test_gptoss_core"
"$OUT/test_gptoss_core" tests/tiny
$CXX $FLAGS src/gptoss_run_main.cpp $SRC -o "$OUT/gptoss_run"; echo "built $OUT/gptoss_run (CPU; its --cuda is refused: it only has the emulated device)"

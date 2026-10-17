#!/bin/sh
set -e
cd "$(dirname "$0")"

# C++14 ONLY (the target is CUDA 12.8 / RTX 4090 sm_89; no C++17/20 features anywhere in the tree).
# g++-20 when present (the machine this was developed on), otherwise the default g++.
CXX="g++"
if command -v g++-20 >/dev/null 2>&1; then CXX="g++-20"; fi
OUT=build-out
mkdir -p "$OUT"
OMP="-fopenmp"

# src/tokenizer.cpp: the o200k_harmony BPE tokenizer (chat mode), needs libpcre2-8 for the unicode pre-tokenizer.
COMMON="src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp src/ops.cpp src/model.cpp src/sampler.cpp src/gguf_loader.cpp src/tokenizer.cpp"
PCRE="-lpcre2-8"

if [ "$1" = "--cuda" ]; then
  # The target toolkit is CUDA 12.8 (the /usr/bin/nvcc wrapper needs the CUDAVER env var; point at the real one).
  NVCC=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}; ARCH=${CUDA_ARCH:-native}
  if [ "$ARCH" = "native" ]; then ARCH="sm_89"; fi   # RTX 4090 (Ada Lovelace)

  echo "Compilazione CUDA in corso (arch $ARCH, C++14)..."
  $NVCC -O3 -std=c++14 -arch="$ARCH" -Iinclude -c src/gguf_dequantize.cu -o "$OUT/gguf_dequantize.o"

  # gptoss_cuda: the real device. Chat mode runs on it by default (--no-cuda forces the CPU path).
  $NVCC -O3 -std=c++14 -arch="$ARCH" ${OMP:+-Xcompiler -fopenmp} -Xcompiler -pthread -Iinclude \
    src/gptoss_run_main.cpp $COMMON src/gpu_cuda.cu src/prefill_dev.cu "$OUT/gguf_dequantize.o" \
    ${OMP:+-lgomp} $PCRE -o "$OUT/gptoss_cuda"

  echo "built $OUT/gptoss_cuda (CUDA; chat mode defaults to the GPU)"
  exit 0
fi

FLAGS="-std=c++14 -O3 -Wall -Wextra -pthread $OMP ${GPTOSS_MARCH:+-march=$GPTOSS_MARCH} -Iinclude"
SRC="$COMMON src/gpu.cpp src/prefill_dev_stub.cpp"
if [ ! -f tests/tiny/golden.txt ]; then python3 tools/tiny_gptoss.py tests/tiny; fi
$CXX $FLAGS tests/test_gptoss_core.cpp $SRC -o "$OUT/test_gptoss_core" $PCRE
"$OUT/test_gptoss_core" tests/tiny
$CXX $FLAGS src/gptoss_run_main.cpp $SRC -o "$OUT/gptoss" $PCRE; echo "built $OUT/gptoss (CPU; its --cuda is refused: it only has the emulated device)"

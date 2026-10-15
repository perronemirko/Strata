#!/bin/sh
set -e
cd "$(dirname "$0")"

# Impostato g++-20 come compilatore predefinito e forzato l'uso di OpenMP
CXX="g++-20"
OUT=build-out
mkdir -p "$OUT"
OMP="-fopenmp"

# Incluso src/gguf_loader.cpp nei sorgenti comuni stabili
COMMON="src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp src/ops.cpp src/model.cpp src/sampler.cpp src/gguf_loader.cpp"

if [ "$1" = "--cuda" ]; then
  # Forzato l'utilizzo del compilatore CUDA 13.1 specifico
  NVCC=${NVCC:-nvcc}; ARCH=${CUDA_ARCH:-native}
  ARCH=${CUDA_ARCH:-native}
  
  # Se impostato native su RTX 4090, passiamo l'architettura corretta sm_89 a NVCC
  if [ "$ARCH" = "native" ]; then ARCH="sm_89"; fi
  
  echo "Compilazione dei moduli CUDA e GGUF in corso con CUDA 13.1..."
  
  # Compilazione asincrona del nuovo kernel di dequantizzazione GGUF Q4_K_L
  $NVCC -O3 -std=c++17 -arch="$ARCH" -Iinclude -c src/gguf_dequantize.cu -o "$OUT/gguf_dequantize.o"
  
  # Links finali unificati inclusi src/gpu_cuda.cu, il prefill device-resident (src/prefill_dev.cu) e il file oggetto del modulo GGUF
  $NVCC -O3 -std=c++17 -arch="$ARCH" ${OMP:+-Xcompiler -fopenmp} -Xcompiler -pthread -Iinclude \
    src/gptoss_run_main.cpp $COMMON src/gpu_cuda.cu src/prefill_dev.cu "$OUT/gguf_dequantize.o" \
    ${OMP:+-lgomp} -o "$OUT/gptoss_run_cuda"
    
  echo "built $OUT/gptoss_run_cuda"
  exit 0
fi

FLAGS="-std=c++20 -O3 -Wall -Wextra -pthread $OMP ${GPTOSS_MARCH:+-march=$GPTOSS_MARCH} -Iinclude"
SRC="$COMMON src/gpu.cpp src/prefill_dev_stub.cpp"
if [ ! -f tests/tiny/golden.txt ]; then python3 tools/tiny_gptoss.py tests/tiny; fi
$CXX $FLAGS tests/test_gptoss_core.cpp $SRC -o "$OUT/test_gptoss_core"
"$OUT/test_gptoss_core" tests/tiny
$CXX $FLAGS src/gptoss_run_main.cpp $SRC -o "$OUT/gptoss_run"; echo "built $OUT/gptoss_run (CPU; its --cuda is refused: it only has the emulated device)"

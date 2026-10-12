# #!/bin/sh
# CPU build + tests (EMULATED device, no GPU needed):   sh build.sh
# CUDA build (real device, separate binary):             sh build.sh --cuda      [NVCC=nvcc CUDA_ARCH=sm_89 | native]
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}; OUT=build-out; mkdir -p "$OUT"; OMP=
echo 'int main(){return 0;}' > "$OUT/omp_probe.cpp"
if $CXX -fopenmp "$OUT/omp_probe.cpp" -o "$OUT/omp_probe" 2>/dev/null; then OMP=-fopenmp; else echo "WARNING: no OpenMP: the engine will run on ONE core" >&2; fi

# Incluso src/gguf_loader.cpp nei sorgenti comuni stabili
COMMON="src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp src/ops.cpp src/model.cpp src/sampler.cpp src/gguf_loader.cpp"

if [ "$1" = "--cuda" ]; then
  NVCC=${NVCC:-nvcc}; ARCH=${CUDA_ARCH:-native}
  # Se impostato native su RTX 4090, passiamo l'architettura corretta sm_89 a NVCC
  if [ "$ARCH" = "native" ]; then ARCH="sm_89"; fi
  
  echo "Compilazione dei moduli CUDA e GGUF in corso..."
  
  # Compilazione asincrona del nuovo kernel di dequantizzazione GGUF Q4_K_L
  $NVCC -O3 -std=c++17 -arch="$ARCH" -Iinclude -c src/gguf_dequantize.cu -o "$OUT/gguf_dequantize.o"
  
  # Links finali unificati inclusi src/gpu_cuda.cu e il file oggetto del modulo GGUF
  $NVCC -O3 -std=c++17 -arch="$ARCH" ${OMP:+-Xcompiler -fopenmp} -Iinclude \
    src/gptoss_run_main.cpp $COMMON src/gpu_cuda.cu "$OUT/gguf_dequantize.o" \
    ${OMP:+-lgomp} -o "$OUT/gptoss_run_cuda"
    
  echo "built $OUT/gptoss_run_cuda"
  exit 0
fi

FLAGS="-std=c++17 -O3 -Wall -Wextra $OMP ${GPTOSS_MARCH:+-march=$GPTOSS_MARCH} -Iinclude"
SRC="$COMMON src/gpu.cpp"
if [ ! -f tests/tiny/golden.txt ]; then python3 tools/tiny_gptoss.py tests/tiny; fi
$CXX $FLAGS tests/test_gptoss_core.cpp $SRC -o "$OUT/test_gptoss_core"
"$OUT/test_gptoss_core" tests/tiny
$CXX $FLAGS src/gptoss_run_main.cpp $SRC -o "$OUT/gptoss_run"; echo "built $OUT/gptoss_run (CPU; its --cuda is refused: it only has the emulated device)"

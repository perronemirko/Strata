#!/usr/bin/env bash
# tools/build_launch_inklin2ristrata.sh - Build and launch the Inklin2RiStrata shape engine (strata-inklin2ristrata).
#
# Two steps, each usable alone:
#   build  [--cuda] [--arch sm_89|native]   CPU core through CMake; --cuda adds the CUDA plugin for this card
#   launch --model FILE [ENGINE FLAGS]      start it on a model file you already have
#
# The engine reads .strata v1.1, Unsloth .gguf (inkling2 arch) and HF safetensors. Get the files with
# tools/fetch_model_linklin2.sh, or let the Terminal do it all: strata launch --family inkling-2-mini --yes --no-tui
#
# MIT License - Copyright (c) 2026 Strata contributors
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${ENGINE_DIR:-$REPO_ROOT/engine}"
COMP_DIR="$ENGINE_DIR/src/inklin2ristrata"
BUILD_DIR="${BUILD_DIR:-$ENGINE_DIR/build-inklin2ristrata}"
# The CUDA binary that reads every model kind (.strata / .gguf / safetensors) is the integrated one: the root build
# with STRATA_BUILD_INKLIN2RISTRATA=ON + STRATA_ENABLE_CUDA=ON. The standalone cuda/build/strata-inklin2ristrata-cuda reads .strata only.
CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-$ENGINE_DIR/build-inklin2ristrata-cuda}"
CUDA_EXE="$CUDA_BUILD_DIR/src/inklin2ristrata/strata-inklin2ristrata"
STANDALONE_EXE="$COMP_DIR/cuda/build/strata-inklin2ristrata-cuda"

usage() {
    cat <<'EOF'
Usage: build_launch_inklin2ristrata.sh COMMAND [options]

Commands:
  build [--cuda] [--arch ARCH]     build strata-inklin2ristrata (CPU core; --cuda also builds the integrated CUDA
                                   binary, the one that reads .strata / .gguf / safetensors; ARCH = native (this
                                   card, default) | sm_89 | sm_120 | all for the standalone .strata-only binary)
  launch --model FILE [FLAGS]      run the engine: --selfcheck | --serve | --bench, plus --ctx N,
                                   --parallel N, --gpu N, --vram-pct P, --no-graph, --kv auto|fp16|fp8|k8v8|k8v4
                                   --cuda picks the integrated CUDA binary (required for .gguf / safetensors)
  all [--cuda] [--model FILE]      build and then launch (selfcheck when no --model, serve when given)

Examples:
  build_launch_inklin2ristrata.sh build --cuda
  build_launch_inklin2ristrata.sh launch --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --selfcheck
  build_launch_inklin2ristrata.sh launch --model models/Inkling-2-Mini-SQ-20.strata --serve --ctx 32768 --vram-pct 90
  build_launch_inklin2ristrata.sh all --cuda --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf
EOF
}

die() { printf 'build_launch_inklin2ristrata.sh: %s\n' "$*"; exit 1; }

note_gpu_busy() {
    # The engine process holds the card: say so, never kill it, never run a CUDA test while another one owns it.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local busy
        busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
        [ -n "$busy" ] && printf 'note: a CUDA process already holds the card (%s); this script builds, it does not kill that process\n' "$busy"
    fi
}

do_build() {
    local cuda="" arch="native"
    while [ $# -gt 0 ]; do
        case "$1" in
            --cuda) cuda="1"; shift ;;
            --arch) arch="$2"; shift 2 ;;
            *) die "unknown build option: $1" ;;
        esac
    done
    printf 'configure: %s -> %s\n' "$COMP_DIR" "$BUILD_DIR"
    cmake -S "$COMP_DIR" -B "$BUILD_DIR" -DSTRATA_BUILD_TESTS=OFF -DSTRATA_INK_BUILD_PREFIX_TESTS=OFF
    cmake --build "$BUILD_DIR" --target strata-inklin2ristrata -j "$(nproc)"
    printf 'engine (CPU core): %s\n' "$BUILD_DIR/strata-inklin2ristrata"
    if [ -n "$cuda" ]; then
        note_gpu_busy
        # The integrated binary is the one that reads every model kind: the root build with the component
        # registered and the CUDA lane on. CMAKE_CUDA_ARCHITECTURES picks the gencode set ("native" = this card).
        local archs="$arch"
        [ "$arch" = "native" ] && archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
        # CMake wants bare integers (89, 120); strip the sm_ prefix users pass for nvcc.
        archs="${archs#sm_}"
        # The nvcc wrapper at /usr/bin/nvcc needs CUDAVER; detect from the toolkit (prefer 12.x = project target).
        export CUDAVER="${CUDAVER:-$(ls /usr/local/cuda-12.*/bin/nvcc 2>/dev/null | sed 's|/usr/local/cuda-||;s|/bin/nvcc||' | sort -V | tail -1)}"
        local tk="/usr/local/cuda-${CUDAVER}"
        # Pin the toolkit explicitly: without CUDAToolkit_ROOT, CMake resolves cudart/cublas from the distro's
        # nvidia-cuda-toolkit (/usr/lib/x86_64-linux-gnu = CUDA 11) and links BOTH runtimes into one binary.
        # Two cudarts in one process corrupt the runtime state: crash inside cuGraphInstantiate at the first
        # decode step (#6905, SIGSEGV at address 0x6). Wipe a stale cache configured against another toolkit.
        if [ -f "$CUDA_BUILD_DIR/CMakeCache.txt" ] && ! grep -q "CUDAToolkit_ROOT:.*=${tk}" "$CUDA_BUILD_DIR/CMakeCache.txt"; then
            printf 'note: wiping %s (previous configure used a different CUDA toolkit)\n' "$CUDA_BUILD_DIR"
            rm -rf "$CUDA_BUILD_DIR/CMakeCache.txt" "$CUDA_BUILD_DIR/CMakeFiles"
        fi
        printf 'configure (integrated CUDA): %s -> %s (toolkit %s)\n' "$ENGINE_DIR" "$CUDA_BUILD_DIR" "$tk"
        cmake -S "$ENGINE_DIR" -B "$CUDA_BUILD_DIR" \
            -DCMAKE_CUDA_COMPILER="$tk/bin/nvcc" \
            -DCUDAToolkit_ROOT="$tk" \
            -DSTRATA_BUILD_INKLIN2RISTRATA=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
            -DCMAKE_CUDA_ARCHITECTURES="$archs"
        cmake --build "$CUDA_BUILD_DIR" --target strata-inklin2ristrata -j "$(nproc)"
        printf 'engine (CUDA, all model kinds): %s\n' "$CUDA_EXE"
        if [ "$arch" != "native" ]; then
            # The standalone .strata-only bench/parity binary, for the multi-arch portable set.
            ARCH_MODE="$arch" STRATA_CORE_LIB="$BUILD_DIR/libstrata_inklin2ristrata_core.a" \
            STRATA_QUANT_LIB="$BUILD_DIR/libstrata_quant.a" \
                sh "$COMP_DIR/cuda/build_cuda.sh"
            printf 'engine (CUDA standalone, .strata only): %s\n' "$STANDALONE_EXE"
        fi
    fi
}

do_launch() {
    local model="" exe=""
    local args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --cuda) exe="$CUDA_EXE"; shift ;;
            --*) args+=("$1"); shift ;;
            *) args+=("$1"); shift ;;
        esac
    done
    # --selfcheck alone runs on the built-in fixture and needs no model; every other mode does.
    local need_model="1" a
    for a in ${args[@]+"${args[@]}"}; do
        if [ "$a" = "--selfcheck" ]; then need_model=""; fi
    done
    if [ -n "$need_model" ]; then
        [ -n "$model" ] || die "give --model FILE (.strata / .gguf / safetensors directory), or use --selfcheck alone"
        [ -e "$model" ] || die "no such model: $model (fetch it first with tools/fetch_model_linklin2.sh)"
    fi
    if [ -z "$exe" ]; then
        exe="$BUILD_DIR/strata-inklin2ristrata"
    fi
    [ -x "$exe" ] || die "engine not built: run 'build' first (or pass --cuda for the CUDA binary)"
    if [ -n "$model" ]; then
        printf 'launch %s --native %s %s\n' "$exe" "$model" "${args[*]:-}"
        "$exe" --native "$model" ${args[@]+"${args[@]}"}
    else
        printf 'launch %s (fixture selfcheck, no model file) %s\n' "$exe" "${args[*]:-}"
        "$exe" ${args[@]+"${args[@]}"}
    fi
}

[ $# -eq 0 ] && { usage; exit 1; }
cmd="$1"; shift
case "$cmd" in
    build) do_build "$@" ;;
    launch) do_launch "$@" ;;
    all)
        do_build "$@" || true
        do_launch --model "${MODEL:-}" || die "for 'all' pass --model FILE or set MODEL=..."
        ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $cmd (build | launch | all)" ;;
esac

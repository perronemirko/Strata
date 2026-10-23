#!/bin/bash
# start_dsv4.sh - Avvia DeepSeek V4 Flash con Strata's API
#
# Questo script avvia el server DeepSeek V4 Flash integrado con Strata.
# Usa serve_dsv4.py per integrare dsv4_run con el API OpenAI/Anthropic di Strata.
#
# Nothing here is hardcoded to one machine: the model path, the profile and the GPU choice are
# resolved at start. Override with DSV4_MODEL / DSV4_PROFILE; a container without a card runs the
# CPU path and says so, rather than failing later inside the engine.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$SCRIPT_DIR"

MODEL_DIR_DEFAULT="/media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4"
MODEL="${DSV4_MODEL:-$MODEL_DIR_DEFAULT/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf}"
PROFILE="${DSV4_PROFILE:-$MODEL_DIR_DEFAULT/dsv4_experts.bin}"
PORT="${DSV4_PORT:-8095}"
CTX="${DSV4_CTX:-8192}"
THREADS="${DSV4_THREADS:-0}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL"
    echo "set DSV4_MODEL=/path/to/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf"
    exit 1
fi

# The GPU check decides the binary AND whether an expert tier on the card is worth asking for.
# A container without a card (nvidia-smi cannot initialize NVML) runs the CPU path only.
GPU=0
if nvidia-smi -L >/dev/null 2>&1; then GPU=1; fi

if [ ! -f "$EXT_ROOT/build-out/dsv4_run" ]; then
    echo "dsv4_run is not built yet. Build it first: cd $EXT_ROOT && sh build.sh"
    exit 1
fi
if [ "$GPU" = 1 ] && [ ! -f "$EXT_ROOT/build-out/dsv4_run_cuda" ]; then
    echo "a GPU is visible but dsv4_run_cuda is not built: run 'sh build.sh --cuda' for the CUDA path"
fi

export DSV4_API_KEY="${DSV4_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "starting DeepSeek V4 Flash (API on port $PORT)..."
if [ "$GPU" = 0 ]; then
    echo "  no CUDA device visible: using the CPU path (--cpu)"
    echo "  on the host card, build with 'sh build.sh --cuda' and this script picks --cuda by itself"
fi
echo "  model: $MODEL"
echo ""

ARGS="--model $MODEL --port $PORT --host 127.0.0.1 --ctx $CTX --threads $THREADS"
ARGS="$ARGS --prefill-chunk 256 --workdir $EXT_ROOT/serve-work --model-name deepseek-v4-flash"
if [ "$GPU" = 1 ]; then
    ARGS="$ARGS --cuda --expert-vram-pct auto --expert-cache 256 --expert-vram-reserve-mib 768"
else
    ARGS="$ARGS --cpu"
fi
[ -f "$PROFILE" ] && ARGS="$ARGS --expert-profile $PROFILE"
[ -n "$DSV4_API_KEY" ] && ARGS="$ARGS --api-key $DSV4_API_KEY"

cd "$EXT_ROOT"
# shellcheck disable=SC2086
python3 tools/serve_dsv4.py $ARGS

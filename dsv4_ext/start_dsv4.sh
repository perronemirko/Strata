#!/bin/bash
# start_dsv4.sh - Avvia DeepSeek V4 Flash con Strata's API
#
# Questo script avvia il server DeepSeek V4 Flash integrato con Strata.
# Usa serve_dsv4.py per integrare dsv4_run con l'API OpenAI/Anthropic di Strata.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$SCRIPT_DIR"
STRATA_ROOT="$(dirname "$EXT_ROOT")"

# Verifica che i binari esistano
if [ ! -f "$EXT_ROOT/build-out/dsv4_run" ]; then
    echo "⚠️  dsv4_run non trovato. Esecuzione in modalità CPU-emulated."
    echo "   Per prestazioni migliori, compila con: cd $EXT_ROOT && sh build.sh --cuda"
fi

# Configura le variabili d'ambiente
export DSV4_API_KEY="${DSV4_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Avvia il server
echo "🚀 Avvio DeepSeek V4 Flash con Strata's API..."
echo "   Port: 8095"
echo "   Host: 127.0.0.1"
echo "   Model: deepseek-v4-flash"
echo ""

cd "$EXT_ROOT"
python3 tools/serve_dsv4.py \
    --model "/path/to/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf" \
    --port 8095 \
    --host "127.0.0.1" \
    --ctx 8192 \
    --threads 0 \
    --expert-vram-pct auto \
    --expert-cache 256 \
    --expert-vram-reserve-mib 768 \
    --prefill-chunk 256 \
    --cpu \
    --workdir "$EXT_ROOT/serve-work" \
    --model-name "deepseek-v4-flash" \
    --api-key "$DSV4_API_KEY" \
    --open

#!/bin/bash
# start_dsv4_from_host.sh - Script da eseguire SUL TUO HOST (non nel container)
#
# Questo script avvia DeepSeek V4 Flash dal tuo sistema host,
# montando la directory del modello nel container.

set -e

MODEL_DIR="/home/albus/.lmstudio/hub/models/DeepSeekV4"
CONTAINER_NAME="strata-container"  # Nome del tuo container Docker

echo "🚀 Avvio DeepSeek V4 Flash dal sistema host..."
echo ""
echo "1. Verifica modello:"
ls -lh "$MODEL_DIR/"*.gguf 2>/dev/null || { echo "❌ Modello non trovato in $MODEL_DIR"; exit 1; }

echo ""
echo "2. Avvio container con modello montato..."
docker run -d \
    --name strata-dsv4 \
    --gpus all \
    -v "$MODEL_DIR:/mnt/models:ro" \
    -p 8095:8095 \
    -w /workspace/dsv4_ext \
    ubuntu:22.04 \
    bash -c "cd /workspace && sh dsv4_ext/build.sh --cuda && python3 tools/serve_dsv4.py --model /mnt/models/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf --cuda --port 8095"

echo ""
echo "✅ Container avviato!"
echo "📡 API: http://127.0.0.1:8095/v1/chat/completions"
echo "🌐 Web: http://127.0.0.1:8095/"
echo ""
echo "Per fermare: docker stop strata-dsv4 && docker rm strata-dsv4"

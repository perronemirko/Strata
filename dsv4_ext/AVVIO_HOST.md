# Guida Avvio DeepSeek V4 Flash - Istruzioni per Host

## 🚨 PROBLEMA RILEVATO
Sto girando in un container Docker isolato (`/workspace`). Non ho accesso alla tua directory locale:
```
/home/albus/.lmstudio/hub/models/DeepSeekV4/
```

Il modello DeepSeek V4 Flash è sul tuo host, non nel mio ambiente.

## ✅ SOLUZIONE: Monta il modello nel container

### Opzione 1: Avvio rapido (consigliata)
Esegui questo script SUL TUO HOST (non nel container):

```bash
# Copia start_dsv4_from_host.sh sul tuo host
chmod +x dsv4_ext/start_dsv4_from_host.sh
./dsv4_ext/start_dsv4_from_host.sh
```

### Opzione 2: Avvio manuale Docker

```bash
MODEL_DIR="/home/albus/.lmstudio/hub/models/DeepSeekV4"

docker run -it --rm \
    --gpus all \
    -v "$MODEL_DIR:/mnt/models:ro" \
    -p 8095:8095 \
    -w /workspace/dsv4_ext \
    ubuntu:22.04 bash << 'EOF'
# Installa dipendenze necessarie
apt-get update && apt-get install -y gcc-10 g++-10 build-essential python3-pip
pip3 install numpy

# Compila dsv4_ext
cd /workspace/dsv4_ext
sh build.sh --cuda

# Avvia il server
python3 tools/serve_dsv4.py \
    --model /mnt/models/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
    --cuda \
    --port 8095 \
    --ctx 8192
EOF
```

### Opzione 3: Usa il percorso diretto (se Docker non è necessario)
Se vuoi eseguire tutto direttamente sul tuo host senza container:

```bash
cd /path/to/Strata/dsv4_ext

# Compila
sh build.sh --cuda

# Avvia con modello locale
python3 tools/serve_dsv4.py \
    --model "/home/albus/.lmstudio/hub/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf" \
    --cuda \
    --port 8095 \
    --ctx 8192
```

## 📋 FILE CREATI PER TE

| File | Posizione | Descrizione |
|------|-----------|-------------|
| `dsv4_config_host.json` | `dsv4_ext/serve-work/` | Config per modello montato |
| `start_dsv4_from_host.sh` | `dsv4_ext/` | Script avvio rapido Docker |
| `INTEGRATION.md` | `dsv4_ext/` | Documentazione completa |

## 🔍 VERIFICA MODELLLO

Prima di avviare, verifica che il modello esista:

```bash
ls -lh /home/albus/.lmstudio/hub/models/DeepSeekV4/*.gguf
```

Devi vedere 3 shard:
- `DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf`
- `DeepSeek-V4-Flash-UD-IQ1_M-00002-of-00003.gguf`
- `DeepSeek-V4-Flash-UD-IQ1_M-00003-of-00003.gguf`

## 🚀 DOPO L'AVVIO

Una volta avviato, testa l'API:

```bash
curl http://127.0.0.1:8095/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Ciao, come stai?"}],
        "max_tokens": 64
    }'
```

## ⚠️ NOTE IMPORTANTI

1. **VRAM**: DeepSeek V4 Flash richiede ~24GB VRAM (RTX 3090/4090 consigliata)
2. **RAM**: Servono ~80GB RAM totali per il modello + cache
3. **Tempo avvio**: Il primo avvio può richiedere 5-10 minuti per caricare il modello
4. **Porta**: L'API è su `http://127.0.0.1:8095` (non 8080 come Strata default)

## 🆘 PROBLEMI COMUNI

**Errore "modello non trovato"**: Verifica il percorso esatto del GGUF
```bash
find /home/albus/.lmstudio/hub/models/DeepSeekV4/ -name "*.gguf"
```

**Errore VRAM insufficiente**: Riduci `--expert-cache` o usa `--cpu` (più lento)

**Container non parte**: Verifica che Docker sia attivo e i permessi GPU
```bash
docker info | grep -A 5 "GPU"
nvidia-smi
```

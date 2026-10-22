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

---

## 🩺 EngineDied: "the engine did not report READY within 900 s"

### Cosa e' happened (2026-10-22 18:04)

`tools/serve_dsv4.py --cuda` started `dsv4_run_cuda`; the engine log stopped right after
`[load 108.0s] experts copied to RAM`, and `serve/server.py` raised
`EngineDied: the engine did not report READY within 900 s`. The engine was not dead and the
model was not wrong: **it was swapping host RAM**. The kernel journal for that minute says it
plainly — `dsv4_run_cuda` at 61.5 GiB resident with 49 GiB pushed into swap, and an
`Out of memory` that killed other processes on the same machine.

The arithmetic of the flags that were used:

| Tier | Size |
|------|------|
| `--expert-ram all` (11008 experts) | 74.4 GiB private RAM |
| `--ram-cache-mib 32768` (LRU arena) | 32.0 GiB private RAM |
| **total wanted** | **106.4 GiB** |
| MemAvailable on this machine | 71.3 GiB |

The log stops exactly where the code allocates next: the arena, right after the copy. Nothing
to do with the GPU. The same command line loaded in 114.3 s at 15:01, when the machine had
more free RAM: the failure is **contention-dependent**, not a bad configuration.

### Two things to know before starting

1. **Never run two Strata engines on the same machine without saying so.** A sibling engine
   (here `strata`, 62.6 GiB RSS and 23.78 GiB of the 24 GiB card) is the usual cause of an
   engine that never reports READY. Do not kill it: tell the launcher it is there instead.
2. **`--expert-ram all` + a big `--ram-cache-mib` is a promise about RAM, not about speed.**
   On a 125 GiB machine with another engine resident, that promise cannot be kept.

### What the code does now (the fix)

* **The engine clamps the arena itself** ([`src/model.cpp`](src/model.cpp),
  `host_mem_available_bytes()` in [`src/mem_plan.cpp`](src/mem_plan.cpp)): it reads
  `MemAvailable` from `/proc/meminfo`, subtracts `--host-ram-reserve-mib` (what the other
  processes hold), and shrinks `--ram-cache-mib` to what is left, printing the numbers. A MISS
  expert outside the arena is read from the shard: **slower, still exact**. A start that used
  to hang for 900 s now loads.
* **The launcher says the numbers before starting** ([`tools/serve_dsv4.py`](tools/serve_dsv4.py),
  `preflight()`): it reads only the GGUF headers, `/proc/meminfo` and `nvidia-smi` — it starts
  no process and touches no GPU — and refuses (exit 3) only what the machine cannot do at all,
  i.e. a demand bigger than RAM **plus** swap.
* **`dsv4_plan` answers the same question as a what-if**, without loading:
  `--expert-ram all --ram-cache-mib 32768 --host-ram-reserve-mib 64000`.

### How to start it correctly on this machine

```bash
# The sibling engine holds ~64 GiB of RAM and the whole card: say so, and the arena is sized for what is left.
python3 tools/serve_dsv4.py --cuda --model .../DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
    --ctx 16384 --threads 12 \
    --expert-vram-pct 0 --expert-vram-reserve-mib 1024 \
    --host-ram-reserve-mib 64000 \
    --expert-ram all --ram-cache-mib 8192
```

`--expert-vram-pct 0` keeps the experts off a card another engine already owns; `--host-ram-reserve-mib`
is what makes the host-RAM plan honest. If the sibling engine is stopped, both can be dropped and
the original line works as it did at 15:01.

### Reading the engine log

| Line | Meaning |
|------|---------|
| `host RAM free: … MiB` / `host RAM wanted: … MiB` | the two numbers of this check |
| `WARNING: the expert tiers want … this load swaps …` | over-demand: it loads, it is slower |
| `note: --ram-cache-mib … is clamped to …` | the engine gave the arena up; correctness is kept |
| `preflight: refused: … RAM plus swap …` | not a slow start: an out-of-memory, nothing to wait for |

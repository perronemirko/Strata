# DeepSeek V4 Flash Integration - Strata

## Panoramica
DeepSeek V4 Flash è integrato in Strata come estensione additiva (`dsv4_ext`). Non modifica il codice sorgente di Strata, ma aggiunge un nuovo engine (`dsv4_run`) che può essere usato in modo indipendente o integrato con l'API di Strata.

## Struttura
```
dsv4_ext/
├── build-out/           # Binari compilati (dsv4_run, dsv4_run_cuda)
├── include/dsv4/        # Header C++
├── src/                 # Sorgenti C++
│   ├── gpu.cpp          # Backend CPU-emulated
│   ├── gpu_cuda.cu      # Backend CUDA reale
│   └── ...              # Altri sorgenti
├── tools/               # Utility Python
│   ├── serve_dsv4.py    # Script di integrazione con Strata API
│   ├── dsv4_tokenize.py # Tokenizer
│   └── ...
├── serve-work/          # Directory di lavoro
│   ├── tokenizer/       # Tokenizer estratto dal GGUF
│   └── dsv4_config.json # Configurazione JSON
├── build.sh             # Script di build
└── start_dsv4.sh        # Script di avvio rapido
```

## Build
```bash
cd dsv4_ext
sh build.sh              # Backend CPU-emulated (test inclusi)
sh build.sh --cuda       # Aggiunge backend CUDA reale
```

## Avvio Rapido
```bash
# Modalità CPU (emulata, per test)
./start_dsv4.sh

# Modalità CUDA (reale, richiede GPU NVIDIA)
python3 tools/serve_dsv4.py \
    --model /path/to/model.gguf \
    --cuda \
    --port 8095
```

## Configurazione JSON
Il file `serve-work/dsv4_config.json` definisce:
- `exe`: Percorso del binario (dsv4_run o dsv4_run_cuda)
- `args`: Argomenti della riga di comando
- `cwd`: Directory di lavoro
- `log`: File di log
- `tokenizer`: Directory del tokenizer
- `model_name`: Nome del modello per l'API

## API Endpoints
Una volta avviato, il server è disponibile su:
- **Web App**: http://127.0.0.1:8095/
- **OpenAI API**: http://127.0.0.1:8095/v1/chat/completions
- **Anthropic API**: http://127.0.0.1:8095/v1/messages

## Note Importanti
1. Il modello GGUF reale è necessario per l'inferenza (non incluso nel repo)
2. La build CUDA richiede nvcc e una GPU NVIDIA compatibile
3. Il backend CPU-emulated è più lento ma utile per test
4. Il file di configurazione deve essere aggiornato con il percorso corretto del modello

## Testing
```bash
# Test core
./build-out/test_dsv4_core

# Test operatori
./build-out/test_ops tests/golden.txt

# Selfcheck sul modello reale (richiede GGUF)
./build-out/dsv4_run --model /path/to/model.gguf --selfcheck
```

## Autore
Integrazione creata per Strata - DeepSeek V4 Flash Extension

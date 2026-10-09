# mistral4-ext

Motore CPU additivo per **Mistral Small 4 119B (2603)** (MLA + MoE 128x4, pesi FP8), costruito sullo schema di `dsv4_ext`.
Note, fatti verificati e incognite: [docs/mistral_small4_porting.md](docs/mistral_small4_porting.md). **Leggere prima la tabella "Non noto".**

    sh build.sh                       # compila (g++, OpenMP) e lancia test_m4_core sul checkpoint minuscolo
    python3 tools/tiny_model.py tests/tiny      # rigenera checkpoint + golden (solo numpy)

    ./build-out/m4_run --model SNAPSHOT --ctx 2048 --selfcheck
    ./build-out/m4_run --model SNAPSHOT --prompt-ids 1,5,9 --n-predict 32 --temperature 0.7
    python3 tools/serve_m4.py --model SNAPSHOT --port 8095          # API OpenAI (tokenizer HF, richiede transformers)

    # server residente: contesto che non si perde, prefill a batch, checkpoint KV a ogni richiesta
    ./build-out/m4_run --model SNAPSHOT --serve --ctx 32768 --prefill-chunk 512 --kv-unified --kv-checkpoint /var/tmp/m4.kv
    python3 tools/serve_m4.py --model SNAPSHOT --ctx 32768 -- --kv-checkpoint /var/tmp/m4.kv --prefill-chunk 512
    # piano VRAM (solo piano, nessun backend CUDA): --vram-pct 90 --vram-total-mib 24576

    # dentro Strata (cartella accanto a dsv4_ext/): API OpenAI/Anthropic + web app di Strata -- vedi docs, sezione "Integrazione in Strata"
    python3 tools/serve_m4_strata.py --model SNAPSHOT --port 8095 --ctx 32768 --kv-checkpoint /var/tmp/m4.kv

SNAPSHOT = cartella con `config.json` e `model-0000x-of-00003.safetensors` (i `consolidated-*` sono ignorati).

## Verificato / non verificato

| Area | Stato | Evidenza |
| --- | --- | --- |
| Tabella FP8 E4M3 (256 codici) | OK | `test_m4_core`: 0 differenze vs Python |
| Lettore safetensors multi-shard, config, BF16/F32/FP8, scale per-tensore / per-expert / gate+up | OK su sintetico | stesso test |
| Forward completo (MLA assorbita vs MLA naive numpy, router, MoE, shared, scala Llama-4, YaRN) | OK su sintetico | 4 varianti, \|diff\| max 1,9e-5 su logit di ordine 9, argmax identici; le varianti differiscono davvero (7,2 / 7,2 / 2,7) |
| `--serve` + `serve_m4.py` (`--tokenizer bytes`) | OK su sintetico | richiesta completa e SSE |
| `dump_safetensors_shapes.py` locale | OK | su shard sintetici |
| `serve_m4_strata.py`: conversione tokenizer su tokenizer.json sintetico | OK | `tests/test_convert_tokenizer.py` |
| `serve_m4_strata.py` contro un checkout di Strata, tokenizer Strata == transformers | **NON RUN** | Strata non e' nel patch |
| Modello vero (qualsiasi cosa) | **NON RUN** | 242 GB, nessuna rete verso HF nel sandbox |
| `--hf` del dump, tokenizer `hf` di `serve_m4.py` | NON RUN | idem |
| Incognite 1-4, 8-9 della tabella | **APERTE** | servono i logit di riferimento di transformers/vLLM |
| Prefill a batch (chunk 1/5/7/64) == token per token == golden | OK su sintetico | \|diff\| 5e-6 |
| Riuso del prefisso tra richieste, checkpoint KV per richiesta, ripristino al riavvio, rifiuto di checkpoint di altra configurazione | OK su sintetico | test + prova `--serve` (`40 of 44 cached`) |
| 32k token/s di prefill | **NON raggiunto, non raggiungibile su CPU/GPU consumer** | ~0,42 PFLOP/s necessari; vedi doc |
| `--vram-pct` | **solo piano**, non eseguito | nessun backend CUDA |
| GPU (kernel FP8), memoria a livelli, GEMM di prefill | NON FATTI | fasi 5, 6, 8 |

# gptoss-ext

Motore di inferenza per **gpt-oss** (GQA + sink + strati a finestra alternati, MoE 128x4 in MXFP4), con **modalità chat** integrata: tokenizer o200k_harmony in C++14, template Harmony, streaming del canale `final`. Note e incognite: [docs/gptoss_porting.md](docs/gptoss_porting.md).

```bash
sh build.sh            # CPU  -> build-out/gptoss        (+ test sul checkpoint minuscolo)
sh build.sh --cuda     # CUDA 12.8, sm_89 (RTX 4090) -> build-out/gptoss_cuda
# NVCC=/percorso/nvcc e CUDA_ARCH=sm_89|native per sovrascrivere i default

./gptoss --model ./models/12b --prompt "Ciao, come va?"     # one-shot: risponde ed esce
./gptoss --model ./models/12b                                # REPL interattivo (/exit /reset /system TEXT)
./gptoss --model ./models/12b --chat --system "Sei conciso" --reasoning medium --show-thinking
./gptoss --model ./models/12b --no-cuda                      # forza la CPU (il binario CUDA di default usa la GPU in chat)

# percorso a token (il vecchio modo, ancora disponibile):
./build-out/gptoss --model SNAPSHOT --prompt-ids 200006,17360,200008 --n-predict 32
./build-out/gptoss_cuda --model SNAPSHOT --cuda --vram-pct 90 --serve --ctx 32768 --kv-checkpoint /var/tmp/gptoss.kv

# dentro il checkout di Strata (cartella accanto a dsv4_ext/): UI + API di Strata sopra il motore
python3 tools/serve_gptoss_strata.py --model SNAPSHOT --port 8095 --ctx 32768 --kv-checkpoint /var/tmp/gptoss.kv
```

`SNAPSHOT` = cartella con `config.json`, i `model-0000x-of-*.safetensors` e (per la chat) `tokenizer.json`. La sottocartella `original/` non serve.

## Modalità chat

* **Tokenizer** ([`src/tokenizer.cpp`](src/tokenizer.cpp)): BPE byte-level o200k_harmony letto da `tokenizer.json`, pre-tokenizer PCRE2 (regex unicode di HF), merge lowest-rank-first con heap `(rank, pos)`. Verificato identico a `transformers` su 79 casi (`tools/compare_tokenizer.py`): encode + round-trip esatti.
* **Template Harmony** ([`build_chat_prompt`](include/gptoss/tokenizer.hpp:64)): system (identità + data + reasoning effort), blocco developer, turni user/assistant con canali `analysis`/`final`, prompt di generazione.
* **Streaming**: il canale `final` è stampato token per token man mano che si genera; i caratteri multi-byte incompleti (un emoji arriva in 4 token) restano nel buffer finché non sono completi (`utf8_printable_prefix`). `--show-thinking` streama anche l'analisi.
* **Riuso KV tra i turni**: ogni turno riprefill solo il delta (`Model::reuse_prefix`); `/reset` svuota cache e conversazione.
* **Stop**: `<|return|>` (200002, eos) e `<|call|>` (200012).

## Stato

| Area | Stato | Evidenza |
| --- | --- | --- |
| MXFP4, BF16, lettore multi-shard, config | OK su sintetico | test |
| Forward (GQA, sink, finestra, YaRN, router, swiglu, bias) vs numpy float64 | OK su sintetico | 4 varianti, \|diff\| max 4,3e-5, argmax identici |
| Prefill a batch (chunk 1/3/64, anello pieno e riavvolgimento) | OK su sintetico | \|diff\| 6e-6 |
| Riuso prefisso, checkpoint KV, rifiuto se cambia configurazione | OK su sintetico | test + prova `--serve` |
| Tokenizer o200k_harmony (encode/decode/template) | **OK vs transformers** | 79 casi, 0 fallimenti (`tools/compare_tokenizer.py`) |
| Chat mode (one-shot + REPL, streaming final, riuso KV tra turni) | OK su sintetico | `./gptoss --model tests/tiny` (pesi casuali: testo incomprensibile, pipeline verificata) |
| CUDA (`src/gpu_cuda.cu`, pool di expert LRU, matmat BF16/MXFP4, prefill a batch su device) | **compilato sm_89, non ancora eseguito** | `sh build.sh --cuda` pulito; il percorso è provato sul device emulato (test `device(emulated)`) |
| Modello vero 120b | **NON RUN** | 60,8 GiB, nessuna rete verso HF nel sandbox |
| Incognite 1-4, 6-8 della tabella "Non noto" | **APERTE** | servono i logit di riferimento |

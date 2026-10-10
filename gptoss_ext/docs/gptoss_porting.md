# gpt-oss-120b: motore di inferenza autonomo (`gptoss_ext`)

Motore residente, CPU, solo per gpt-oss-120b. Stessa logica dei motori del patch dev4 (processo che parla il protocollo a righe `READY/INFO/GEN/PP/T/DONE/ERR/STOP/QUIT`,
cache che sopravvive tra le richieste, checkpoint KV, piano VRAM) ma **nessuna integrazione con server o framework esterni**: il motore riceve id di token e restituisce id.
Il rendering Harmony e la tokenizzazione (o200k_harmony) stanno fuori dal motore.

## Fatti verificati e dove

| Voce | Valore | Fonte |
| --- | --- | --- |
| Struttura | 36 layer, hidden 2880, 64 teste q / 8 kv, head_dim 64, MoE 128 expert top-4, intermediate 2880, vocab 201088 | config.json, model card |
| Attenzione | layer **pari = finestra 128**, dispari = densa (`layer_types`, parte da `sliding_attention`); bias su q/k/v/o (`attention_bias: true`); sink appreso per testa nel denominatore dello softmax | config.json, model card |
| RoPE | YaRN, theta 150000, fattore 32, originale 4096, beta 32/1, **`truncate: false`** | config.json |
| Norm | RMSNorm pre-LN, eps 1e-5 | config.json, card |
| Router | proiezione lineare (con bias), top-4 sui logit, **softmax solo sui 4 scelti** | model card (peso), nomi da vLLM |
| SwiGLU | alpha 1.702, limite 7.0, `(up+1) * gate*sigmoid(alpha*gate)`, clamp su gate e up | config.json, doc llm-compressor/vLLM |
| Pesi | expert in MXFP4 (4,25 bit/param); `q/k/v/o`, router, embedding, `lm_head` in BF16 (`modules_to_not_convert`) | model card, config.json |
| Nomi | `model.layers.N.self_attn.{q,k,v,o}_proj.{weight,bias}`, `self_attn.sinks`, `mlp.router.{weight,bias}`, `mlp.experts.{gate_up,down}_proj_{blocks,scales,bias}`, `lm_head.weight` | mappa pesi di vLLM, doc NVIDIA |
| Token | EOS 200002, PAD 199999; formato Harmony con canali `analysis`/`commentary`/`final`; reasoning `low/medium/high` nel system prompt | config.json, card |
| Dimensioni | 116,83B totali, 5,13B attivi (MLP 114,71B, attenzione 0,96B, embed+unembed 1,16B); checkpoint 60,8 GiB in 15 file `model-0000x-of-00014` | card, tree HF |

Conti per token (aritmetica su queste cifre): 4 expert x 36 layer = 1,9 GB (MXFP4) + attenzione 1,9 GB (BF16) + `lm_head` 1,16 GB = **~5 GB letti per token**; un expert pesa ~13,2 MB.
A differenza di Mistral, qui le parti BF16 dense pesano quanto gli expert.

## Non noto (non assumere): ogni voce ha un interruttore o un controllo

Chiuderle richiede i logit di riferimento (transformers o l'implementazione di OpenAI) sul modello vero: i test sintetici provano il codice, non queste scelte.

| # | Incognita | Default | Interruttore |
| --- | --- | --- | --- |
| 1 | `gate_up_proj`: righe **interleaved** (gate = pari, up = dispari) o gate = prima meta' | interleaved | `--glu-halves` |
| 2 | RoPE: layout **half-split** (x[i], x[i+32]) o coppie adiacenti | half-split | `--rope-pairs` |
| 3 | Fattore YaRN `0.1*ln(32)+1` su cos/sin (quindi ~x1,35^2 sui logit di attenzione) | si' | `--no-concentration` |
| 4 | Sink: logit per testa **non** scalato per 1/sqrt(64), solo nel denominatore | cosi' | - |
| 5 | MXFP4: blocchi da 32 lungo l'ultima dimensione, nibble basso = elemento pari, scala `2^(s-127)`, forme `[E,righe,K/32,16]` e `[E,righe,K/32]` | cosi' | il caricatore controlla le forme e si ferma con un errore chiaro |
| 6 | Finestra: la query vede 128 chiavi, **token corrente incluso** | si' | - |
| 7 | Stop: oltre a 200002 (`<\|return\|>`) serve fermarsi su `<\|call\|>` per i tool; l'id (200012) **non e' verificato** | solo config.eos | `--eos-ids` |
| 8 | Attivazioni in float32 (il riferimento lavora in bf16): piccole differenze numeriche attese | - | - |

Primo passo sul modello vero: `python3 tools/dump_safetensors_shapes.py --hf openai/gpt-oss-120b --prefix model-` (chiude 5) e `gptoss_run --selfcheck`; poi confronto dei logit del primo token variando 1-3.

## Cache e contesto (le tre richieste)

- **Riuso del prefisso:** la cache tiene i token; a ogni `GEN` si ricalcola solo cio' che differisce (l'ultimo token del prompt si ricalcola sempre).
- **Strati a finestra:** tengono un anello di `128 + --kv-rollback` posizioni (default 8192 di margine), cosi' si puo' tornare indietro nel contesto. Se il riavvolgimento supera il margine il motore lo dice e ricalcola il prompt (provato).
  Con Harmony le tracce di ragionamento dei turni passati **vanno tolte dal prompt**: il prefisso comune finisce quindi al messaggio utente precedente, e la risposta precedente si ricalcola. Serve un margine grande quanto CoT + risposta.
- **Checkpoint:** `--kv-checkpoint FILE`, riscritto dopo ogni richiesta (tmp + rename), ripristinato all'avvio, rifiutato se cambiano modello/opzioni/anello (provato). Il file contiene token + K/V (la parte viva dell'anello nei layer a finestra).
- **Memoria KV (fp32):** layer densi 4 KiB/token/layer x 18 = 74 KiB/token (ctx 8192 = 0,6 GiB; 131072 = 9,7 GiB); layer a finestra ~34 MiB ciascuno con il margine di default.
- `--kv-unified` e' accettato per parita' con llama.cpp: il motore ha una sola sequenza e una sola cache.

## Prefill e VRAM: cosa e cosa non

- **Prefill a batch** (`--prefill-chunk`, default 512, ridotto da solo se l'anello lo richiede): layer per layer sul chunk, expert raggruppati (ogni expert si percorre una volta per chunk). Provato identico al percorso token per token.
- **32k token/s non e' raggiungibile con questo motore**: con ~10 GFLOP per token (5,13B parametri attivi) servono ~0,33 PFLOP/s sostenuti, hardware da H100 con kernel a matrice. Qui il calcolo nel chunk e' ancora matrice-vettore per token. `PP` riporta la velocita' misurata per chunk.
- **CUDA (`--cuda`, binario `gptoss_run_cuda`, `sh build.sh --cuda`):** stessa architettura di dev4 (API unica `gpu.hpp`, backend emulato `gpu.cpp` per i test e `gpu_cuda.cu` per la GPU, mai nello stesso binario).
  Sulla scheda: `lm_head` e `q/k/v/o` + router di ogni layer (BF16) finche' entrano nel budget (`--vram-pct` della memoria libera, o di `--vram-total-mib`, meno `--vram-reserve-mib`), poi **un pool di slot da ~13 MiB** per gli expert con evizione LRU; un expert mancante viene caricato dal file mappato e valutato sul device.
  Resta sul host: embedding, norme, RoPE, attenzione e KV cache, swiglu, bias, router top-k. Ogni chiamata copia x su e y giu' (semplice, non veloce). Prefill a batch: ogni proiezione e ogni expert e' **un solo passaggio sui pesi per tutto il chunk** (kernel a warp per riga, 8 token per volta in registri).
  Ogni errore CUDA spegne il device e la sessione prosegue su CPU. **Il codice CUDA non e' mai stato compilato ne' eseguito**; solo il percorso del Model e' coperto dal device emulato.

## Stato per fase della guida

| Fase | Stato |
| --- | --- |
| 0 serve un motore nuovo? | No in generale (vLLM, SGLang, transformers, llama.cpp lo supportano). Ha senso solo per la memoria a livelli |
| 1 fatti | tabella sopra; `dump_safetensors_shapes.py --hf` **non eseguito** (nessuna rete verso HF nel sandbox) |
| 2 riferimento | `tools/tiny_gptoss.py`: forward naive numpy float64 con gli stessi nomi di tensore, MXFP4 + BF16, 2 shard, 4 varianti. **Non e' il codice ufficiale** (non letto) |
| 3 base C++ | lettore safetensors multi-shard (mmap), JSON, config, BF16, MXFP4: provati |
| 4 CPU | forward completo, campionatore, cache. **Provato solo su modello sintetico** (differenza max sui logit 4e-5 su logit di ordine 9, argmax identici, 4 varianti). Non provato sul modello vero |
| 5 GPU | non fatta |
| 6 memoria a livelli | solo il piano VRAM |
| 7 protocollo | `gptoss_run --serve` (stesso protocollo dei motori dev4), provato con richieste ravvicinate e riavvio |
| 8 prestazioni | prefill a batch; nessuna misura sul modello vero |
| 9 operativita' | checkpoint per richiesta; profilo expert non fatto |

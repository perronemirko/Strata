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
  An allocation that does not fit is now recoverable (that one call runs on the host, the chunk is halved, the device stays on); any other CUDA error still turns the device off and the session continues on the CPU. **The CUDA code compiles clean (`nvcc` sm_89, `g++ -Wall -Wextra`); it has never been executed on a real GPU** — only the Model path is covered by the emulated device.
  The BF16 bias rows of an expert live inside its slot (`gu_bias`, `dn_bias` device copies, `(2I + H) * 2` bytes ~ 14 KiB on a 13.2 MiB slot), uploaded with the expert; the kernels of the expert add them, so the decode path needs no host add step.
  Decode of one token (`experts_decode`): the top-k experts run **at the same time**, one per device context (`kExpertCtxBatch = 4`): `expert_run_async` enqueues the whole expert without waiting, `expert_collect` waits per context and accumulates. Each expert's result goes into its own `part[k]` slice, so an expert without a slot (all slots pinned or in flight, upload or enqueue failed) runs on the host **while the device experts are still computing**, and a failed collect re-runs just that expert on the host; the sum into `acc` stays in k order. A slot in flight carries `busy = true` and cannot be chosen as the LRU victim. The scratch of the 4 contexts is reserved in `gpu_setup` before the pool, so no allocation happens during decode once VRAM is full.
  Proof of the async path (emulated device, never on CUDA): `test_gptoss_core` with `slots = 5`, `slots = 0` and `slots = -1` gives the same logits as the CPU path (max |diff| 3.3e-05 on decode, 6.9e-06 on the last token of a full prefill), identical to the run without the batched path.

### The dense scratch and the VRAM budget (measured failure, 2026-10-12)

The first real run on this PC (i7-14700K, 28 threads, RTX 4090 24564 MiB) never used the GPU. The engine log says so:

```
gptoss: CUDA device 'NVIDIA GeForce RTX 4090' (real): dense weights 2.88 GiB on device, 1470 expert slots of 12.62 MiB (32% of the 4608 experts)
gptoss_run: GEN max_new=130638 ... prompt=426
gptoss: CUDA call failed (cudaMalloc y: out of memory): continuing on the CPU for the rest of the session
gptoss_run: 426 prompt -> 200 tokens, cancel (103823 ms prefill, 44416 ms decode)   = 4.1 tok/s prefill, 4.5 tok/s decode
```

Two facts, both arithmetic on the numbers above:

- `reserve_scratch` asked for the scratch of **one** token (`hidden`, `vocab` floats). `prefill_gpu` calls `mm(..., T = chunk_max)` on q/k/v/o/router, so `run_matmat` grows it to `T * cols` and `T * rows` floats: with `--prefill-chunk 512` that is 5.9 MiB in and 8.4 MiB out (q has 4096 rows). The first prefill call asked for 8.4 MiB and got `out of memory`.
- The plan left no room: 1470 slots x 12.62 MiB = 18.1 GiB + 2.88 GiB dense = 21.0 GiB of the 24.5 GiB, and `--vram-pct 100 --vram-reserve-mib 0` spent the rest. `gpu_fail` then set `gpu_on = false` for the whole session, so the 4.5 tok/s measured are the CPU path.

Fixed in this session (compiles clean with `nvcc` sm_89 and with `g++ -Wall -Wextra`; **not yet run on the GPU**, the Qwen engine holds the card):

- `gpu_setup` computes the real `max_rows` of the batched projections and reserves the dense scratch for `chunk_max` tokens **before** the slots take the VRAM, subtracting it from the budget; if the scratch does not fit, it halves the chunk (and says so) instead of failing.
- `gpu_fail` now separates the two errors: `cudaErrorMemoryAllocation` (a size that did not fit, not a poisoned context) runs that one call on the host, halves `chunk_max`, and keeps the device; any other CUDA error stays sticky as before. The per-request line ends with `oom fallbacks N`, so a run that keeps hitting the host path is visible.
- One GPU, one engine: the run before this one died at `cudaSetDevice: out of memory` because the Qwen engine already held the card. The two cannot share it.

To measure the GPU path once the card is free: `python3 tools/serve_gptoss_strata.py --model <gpt-oss-120b> --cuda --exe build-out/gptoss_run_cuda --ctx 131072 --prefill-chunk 512 --vram-pct 90 --vram-reserve-mib 2048 --expert-profile /var/tmp/gptoss-expert.bin --kv-checkpoint /var/tmp/gptoss.kv` and read the `PP`/`T` lines plus `experts: ... hit rate ...%, oom fallbacks ...`.

## End of a turn: which id stops

The engine stops on `--eos-ids` (`serve_loop.cpp:196`), prints the token as one `T` line and breaks (`serve_loop.cpp:350`). The server reads that `T` line and must know it is the end, or it reports the request as cut by max tokens.

`Service.stop_ids` (`serve/server.py:2862`) used to be the ids of `<|im_end|>` and `<|endoftext|>` in whatever vocabulary was loaded. In this vocabulary `<|im_end|>` does not exist: its spelling encodes to six ordinary text ids (`27, 91, 321, 13707, 91, 29`), which an answer can contain. The real end of a turn is 200002, and the pack says so in `tokenizer.json` (`tokenizer.ggml.eos_token_id`). So: a literal counts only when it is one control token in this vocabulary, and the vocabulary's own eos id is always a stop.

What it looked like before, on the real run (the engine's log against the server window):

```
gptoss_run: finish=stop generated=190 max_new=130814
[strata] done: 190 tokens in 24 s (7.9 tok/s) (length, cancel=False)
```

The model said "stop", the client heard "length": the answer was cut and its reasoning channel leaked as visible text. Checked now with a fake engine that writes the model's own Harmony text and ends on 200002: `finish == "stop"`, the analysis channel comes out as reasoning, the final channel as content, nothing lost at the end. The Qwen path is unchanged: with a vocabulary that has both markers the set is still `{248044, 248046}`.

## Stato per fase della guida

| Fase | Stato |
| --- | --- |
| 0 serve un motore nuovo? | No in generale (vLLM, SGLang, transformers, llama.cpp lo supportano). Ha senso solo per la memoria a livelli |
| 1 fatti | tabella sopra; `dump_safetensors_shapes.py --hf` **non eseguito** (nessuna rete verso HF nel sandbox) |
| 2 riferimento | `tools/tiny_gptoss.py`: forward naive numpy float64 con gli stessi nomi di tensore, MXFP4 + BF16, 2 shard, 4 varianti. **Non e' il codice ufficiale** (non letto) |
| 3 base C++ | lettore safetensors multi-shard (mmap), JSON, config, BF16, MXFP4: provati |
| 4 CPU | forward completo, campionatore, cache. **Provato solo su modello sintetico** (differenza max sui logit 4e-5 su logit di ordine 9, argmax identici, 4 varianti). Non provato sul modello vero |
| 5 GPU | compila (nvcc sm_89); never executed on a real GPU; the first real run fell back to the CPU (see the section above) |
| 6 memoria a livelli | solo il piano VRAM |
| 7 protocollo | `gptoss_run --serve` (stesso protocollo dei motori dev4), provato con richieste ravvicinate e riavvio |
| 8 prestazioni | prefill a batch; nessuna misura sul modello vero |
| 9 operativita' | checkpoint per richiesta; profilo expert non fatto |

## Profilo degli expert: quali stanno in VRAM

- Il motore conta quante volte ogni (layer, expert) viene scelto dal router (anche sul binario CPU: il routing non dipende dal device) e riscrive il profilo **dopo ogni richiesta** (`--expert-profile FILE`, tmp + rename).
- All'avvio con `--cuda`, il `--pin-pct` % (default 80) degli slot viene riempito con gli expert piu' usati del profilo (classifica globale) e **non viene mai evitto**; il resto degli slot e' LRU per tutti gli altri.
- Senza profilo (primo avvio) tutti gli slot sono LRU. Il profilo si accumula tra le sessioni; il pinning si aggiorna al riavvio.
- Dopo ogni richiesta il motore scrive su stderr: `experts: N slots (M pinned), hits ... (of which pinned ...), lookups ..., hit rate ...%; P of the R routed experts read over PCIe, H on the host, oom fallbacks ...`. **Quella riga dice se il bus e' il limite.** Gli counters e' per request e e' del decode: `Model::begin_request_stats` / `request_stats` (model.cpp:838), si' `Model::forward` si conta (`counting`), prefill e' ne' conta (si' dilute la hit rate).
- The `DONE` line now carries the same field layout as Strata's engine (`serve/server.py:915` `_parse_done`), so the server window prints the same engineering lines for gpt-oss as for Qwen: `[strata] thinking: N of max M tokens, X tok/s` and `[strata] done: N tokens in T s (X tok/s) (stop, cancel=False), expert cache H% hit (+P% of the routed experts over PCIe)`. Fields 10-11 are the request's `hits lookups`, field 14 the prompt tokens actually read, field 15 the experts uploaded over PCIe; drafts (fields 7-9) are 0/0 because there is no speculative decoding here, and the RAM/file tiers (fields 12-14) are 0 because this engine has only VRAM + host.
- Provato su device emulato (tests/tiny, `--gpu-emu --cuda`, ctx 64, 1 request 4 prompt / 8 generated): `DONE 8 4 5.1 10.8 length 0 0 0 47 47 0 0 0.0 4 9` = 7 forward decode x 4 layer x top-2 = 56 routed = 47 hit VRAM + 9 over PCIe (16.1% of the routed). Mai ne'执行 on a real CUDA card (the Qwen engine holds it).
- `tools/calibrate_profile.py` riempie il profilo con prompt rappresentativi; `tools/profile_report.py` stampa, per N slot, la quota di attivazioni coperta dai N expert piu' usati (hit rate di un pinning statico ideale) contro la linea uniforme. Se la curva e' piatta, il pinning non aiuta.
- Limiti: il pinning dipende dalla somiglianza fra prompt di calibrazione e uso reale; la classifica e' globale (un layer poco usato puo' restare con pochi expert fissi); **codice mai compilato ne' eseguito**.

## Three knobs that only move WHICH experts hold the slots

The line `gptoss: expert slots (1407): 9169 hits, 1859 misses` says 16.8% of the routed activations of that run missed. Two things decide that number, and both are blunt: the pinned set is chosen once, at load, from a profile that sums every past session; and the LRU region is `100 - --pin-pct` % of the slots (20% by default, ~280 slots spread over the model's 36 layers x 128 experts = 4608 experts, of which a 24.5 GiB card can hold ~1470). These three flags act on that, and on nothing else: no slot is added, no arithmetic changes, the GGUF is untouched. All are opt-in; with the defaults (`0`, off, `0`) the code path is the one of the previous release and the profile file is byte-identical.

`--expert-promote` keeps the pinned set at the size `--pin-pct` asked for: below that cap a promoted expert only pins, at the cap it trades with the coldest pinned one. With `--pin-pct 0` the cap is 0 and the flag does nothing — a pure-LRU run has no pinned set to correct, so raise `--pin-pct` above 0 if the goal is to fight misses.

| Flag | What it does | Where | Cost |
| --- | --- | --- | --- |
| `--expert-promote N` | an expert served N times from an LRU slot during the decode pins itself, and at the `--pin-pct` cap it trades with the coldest pinned expert; the trade happens only if the demoted one really is colder | `Model::Impl::promote_locked`, `src/model.cpp` | no PCIe, no upload: both experts are already in VRAM, only the flag trades |
| `--expert-lru-layered` | on a miss, evict first the coldest LRU slot **of the requesting layer**; fall back to the global LRU scan when that layer has none | `Model::Impl::victim_slot`, `src/model.cpp` | one extra scan over the pool per miss |
| `--profile-decay N` | after a profile save, halve the whole usage table once its sum passes N; the ranking is kept, the weight of old sessions is bounded | `Model::decay_profile`, called from `save_prof` (`src/serve_loop.cpp`) and from the one-shot exit path (`src/gptoss_run_main.cpp`) | one pass over `layers*experts` u64s, only at a save |

Why they should help, on the shape of that run:

- The pinned set follows the profile of every session ever run. A long run of a different task dilutes it, and the next start pins the wrong experts; `--profile-decay` keeps the ranking fresh without re-spawning the process.
- A global LRU lets one chatty layer evict the LRU slots of every other layer. With ~280 LRU slots over 36 layers, one layer's traffic can evict another layer's hot expert; `--expert-lru-layered` keeps the eviction where it belongs.
- `--expert-promote` fixes both at runtime, without a restart: whatever the decode really keeps using and was found by LRU becomes pinned, and the pinned expert the profile overestimated goes back to the LRU pool.

**Not measured.** No run with these flags was started here (the GPU holds the next task). The `experts:` line printed after each request ends now with `promotions N` when `--expert-promote` is on (cumulative, like the other counters in that line; with the flag off the line is byte-identical to the last release, so the server window's `_parse_done` fields do not move). A run that never promotes is visible and one that promotes on every token is too. To compare: same binary, same prompts, one arm per flag, and report the `experts:` line of each arm.

Suggested order to try, cheapest first: `--profile-decay 200000`, then `--expert-lru-layered`, then `--expert-promote 40`. `--pin-pct` stays the blunt lever, and `tools/profile_report.py` says where it stops helping (a flat curve = pinning gains nothing).

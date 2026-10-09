# Mistral Small 4 119B (2603): note di porting

Regola dell'albero: **additivo**, come `dsv4_ext`. Nessun file del progetto ospite viene toccato. Namespace `m4`, cartella a parte.
Fonti lette (tutte pubbliche, 8 ott 2026): `config.json`, `params.json`, `model.safetensors.index.json` (prima parte, layer 0-35 compresi
quelli visibili), model card HF, ricetta vLLM, pagina doc `transformers/model_doc/mistral4`, pagine NeMo AutoModel.
**Non ho potuto leggere `modeling_mistral4.py`** (il sandbox raggiunge solo pagine gia' comparse nelle ricerche): tutto cio' che dipende
da quel file e' nella tabella "Non noto", con un interruttore a runtime.

## Fase 0 - serve un motore nuovo?
**Probabilmente no.** L'architettura e' gia' supportata da vLLM, SGLang, transformers (main), llama.cpp / LM Studio (GGUF Unsloth:
`unsloth/Mistral-Small-4-119B-2603-GGUF`). Un motore su misura ha senso solo per la gestione a livelli della memoria degli expert
(stesso scopo di `dsv4_ext`). Qui si applicano le fasi 6 e 9: 120 GB di pesi FP8 non stanno in una GPU consumer.

## Fase 1 - fatti verificati

| Voce | Valore | Fonte |
| --- | --- | --- |
| Architettura | `Mistral3ForConditionalGeneration` (wrapper multimodale) + testo `model_type: mistral4` | config.json |
| Layer / hidden / vocab | 36 / 4096 / 131072 | config.json |
| Attenzione | **MLA** (come DeepSeek-V3): 32 teste, `q_lora_rank` 1024, `kv_lora_rank` 256, `qk_nope` 64 + `qk_rope` 64 (= `qk_head_dim` 128), `v_head_dim` 128 | config.json, params.json |
| Tensori attenzione | `q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj` | index.json |
| MoE | 128 expert, top-4, 1 condiviso, `moe_intermediate_size` 2048, `first_k_dense_replace` 0 (**tutti** i 36 layer sono MoE), `n_group`=`topk_group`=1, `norm_topk_prob` true, `routed_scaling_factor` 1.0 | config.json |
| Expert | tensori **fusi** per layer: `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` (+ `_scale_inv`, `_activation_scale`); `mlp.gate.weight` senza bias; `mlp.shared_experts.{gate,up,down}_proj` | index.json |
| Attivazione | `silu`, senza clamp (a differenza di DeepSeek-V4) | config.json (`hidden_act`) |
| RoPE | YaRN, theta 10000, fattore 128, orig 8192, beta_fast 32, beta_slow 1, `mscale` = `mscale_all_dim` = 1, `rope_interleave` true | config.json |
| Scala Llama-4 | `llama_4_scaling_beta` 0.1, orig 8192 -> **vale esattamente 1 per pos < 8192** | config.json, params.json |
| Pesi | FP8 E4M3 (`quant_method: fp8`, `weight_block_size: null` = per-tensore, attivazioni statiche); BF16 per vision tower, projector, `lm_head` | config.json |
| Embedding / head | non legati (`tie_word_embeddings` false) | config.json |
| Contesto | 256K raccomandato (config dice 1.048.576 posizioni) | model card |
| Token | BOS 1, EOS 2, PAD 11; template con `reasoning_effort` in {`none`, `high`} (tag `[THINK]`) | config.json, model card |
| MTP / hash routing / hyper-connections | **assenti** (nessun tensore, nessun campo) | index.json, config.json |
| Dimensione | `total_size` 120.922.770.560 B in 3 shard `model-0000x-of-00003` (49,1 + 49,1 + 22,7 GB); il repo contiene anche `consolidated-*` (nomi nativi Mistral, **non usati**) | tree HF |

**Controllo di coerenza dei conti** (non e' una prova, ma le forme assunte devono tornare): expert 36 x 128 x (2x2048x4096 + 4096x2048) = 115.964.116.992 B;
attenzione + condiviso + gate 36 x 54.263.808 = 1.953.497.088 B; embedding + lm_head BF16 = 2.147.483.648 B; somma 120.065.097.728 B; la differenza
con `total_size` (857.672.832 B) e' il tower visivo Pixtral (24 layer, hidden 1024 => ~805 MB in BF16) piu' il projector. Tornano anche i "6,5B attivi":
4 expert x 36 layer = 3,62 GB + 1,95 GB + 1,07 GB (lm_head) = 6,65 GB letti per token.

## Non noto (non assumere) e come si risolve

Ogni voce ha un interruttore in `m4_run` / `RunOpts`; si chiude con un confronto di logit contro transformers/vLLM (Fase 4).

| # | Incognita | Default | Interruttore | Perche' non basta il test sintetico |
| --- | --- | --- | --- | --- |
| 1 | **Funzione del router**: softmax o sigmoid? (la pagina HF parla di "post-softmax" ma e' testo generico; il checkpoint non ha `e_score_correction_bias`) | softmax | `--router sigmoid` | il golden numpy e' scritto da me: prova il codice C++, non la scelta |
| 2 | **mscale^2 nello scale dello softmax** (DeepSeek-V3 HF lo applica; `params.json` dice `apply_scale: false`) | no | `--mscale-softmax` | idem; cambia **tutte** le posizioni, quindi e' decidibile anche con un prompt corto |
| 3 | **Scala Llama-4 su tutto q o solo su q_pe** (vLLM: `q *= scaling`; NeMo: "to q_pe after RoPE") | tutto q | `--l4-qpe` | e' esattamente 1 sotto 8192 token: serve un prompt **> 8192** per distinguerle |
| 4 | **Convenzione di `*_scale_inv`**: w = fp8 * s oppure fp8 / s | moltiplica | `--fp8-scale-div` | `--selfcheck` lo rivela: con la convenzione sbagliata i logit esplodono o sono tutti ~0 |
| 5 | Forma di `gate_up_proj`: `[E, 2I, H]` con gate nella prima meta' (convenzione transformers v5) | si' | (errore chiaro al caricamento) | `dump_safetensors_shapes.py` lo mostra in un secondo |
| 6 | Forma di `*_scale_inv` degli expert: `[E]` o `[2E]` (gate/up separati) sono gestiti; blocchi 2D **no** | - | errore chiaro | idem |
| 7 | Nome del norm finale (`language_model.model.norm.weight` assunto: il mio estratto dell'indice si ferma al layer 5 della terza parte) | si' | errore "missing tensor" | idem |
| 8 | Quantizzazione delle **attivazioni** (`activation_scale`, FP8 statico): vLLM le quantizza, qui restano float32 | float32 | - | differenze numeriche piccole attese, non nulle |
| 9 | RoPE interleaved: qui rotazione di coppie adiacenti; HF de-interleava e ruota le meta'. Equivalenti per il prodotto q_pe.k_pe **se** i pesi `q_b`/`kv_a` sono nell'ordine interleaved | si' | - | si verifica solo sul modello vero |
| 10 | Tokenizer: `tokenizer.json` (HF) vs `tekken.json`; qui solo il primo, via `transformers` | HF | `--tokenizer` | ramo non eseguito |

## Differenze rispetto a `dsv4_ext` (dev4.patch)

| dev4 (DeepSeek-V4-Flash) | Mistral Small 4 |
| --- | --- |
| GGUF 3 shard, quantizzazioni IQ1_M/IQ2/Q4_K..., codebook da llama.cpp | safetensors, FP8 E4M3 + BF16: **nessun codebook**, niente `extract_iq_tables.py` ne' `dq_traits.hpp` |
| CSA/HCA, compressori, indexer, finestra | MLA a cache latente (320 float/token/layer) |
| Hyper-connections (Sinkhorn) | residui normali |
| 256 expert top-6, hash routing sui primi 3 layer, clamp swiglu, sqrtsoftplus | 128 expert top-4, router a punteggio + renorm, silu puro |
| MTP layer | assente |
| `dump_gguf_shapes.py` | `dump_safetensors_shapes.py` (anche con Range HTTP) |
| `mem_plan`, `gpu*`, profilo expert | **non ancora portati** (vedi sotto) |

Riuso diretto dal patch: `sampler.{hpp,cpp}` (solo `sed dsv4->m4`), `serve_loop` (stesso protocollo `READY/GEN/PP/T/DONE`; tolti profilo e token-testo).

## Stato per fase (guida in 10 fasi)

| Fase | Stato |
| --- | --- |
| 0 decisione | scritta qui |
| 1 fatti | fatta dalle fonti; **`dump_safetensors_shapes.py --hf` non eseguito** (nessuna rete verso HF): eseguirlo e confrontare con le voci 5-7 |
| 2 riferimento | `tools/tiny_model.py`: forward naive numpy float64, checkpoint minuscolo con gli stessi nomi, FP8/BF16, 2 shard, 1 layer con scale `[2E]`. Non e' il codice ufficiale (non letto): e' una trascrizione dell'architettura DeepSeek-V3 + scala Llama-4 |
| 3 base C++ | lettore safetensors multi-shard (mmap), JSON, config, decodifica FP8/BF16/F16/F32, tutti con test |
| 4 CPU | forward completo (MLA assorbita, MoE, shared), campionatore, KV cache latente. **Provato su modello sintetico** (4 varianti, differenza max sui logit 1,9e-5). **Non provato sul modello vero**: manca `compare_logits` contro transformers/vLLM |
| 5 GPU | non fatta. Il formato e' piu' semplice di IQ: un kernel FP8 (LUT da 256 valori o `cvt` hardware su sm_89) + scale per-tensore. Riusare `gpu.hpp` con due backend (emulato / CUDA) come in dev4 |
| 6 memoria a livelli | **solo il piano** (`--vram-pct`, vedi sotto); nessun backend CUDA, nessun expert caricato in VRAM. Numeri: expert = 24 MiB ciascuno (gate_up 16 MiB + down 8 MiB), 3 GiB/layer, 108 GiB totali; 4 x 36 = 144 letture di expert per token |
| 7 server | `m4_run --serve` + `tools/serve_m4.py` (OpenAI `/v1/chat/completions`, SSE). Protocollo provato su modello sintetico; tokenizer HF reale non provato |
| 8 prestazioni | `matvec` OpenMP; **prefill a batch fatto** (vedi sotto); nessuna misura sul modello vero |
| 9 operativita' | non fatta (profilo expert, salvataggio dopo ogni richiesta) |

## Prima esecuzione sul modello vero (ordine consigliato)

    python3 tools/dump_safetensors_shapes.py --hf mistralai/Mistral-Small-4-119B-2603 > shapes.txt   # chiude le voci 5-7
    sh build.sh
    ./build-out/m4_run --model /path/snapshot --ctx 2048 --selfcheck                                 # logit finiti? (voce 4)
    # poi: stesso prompt su transformers/vLLM e su m4_run, confronto dei logit del primo token;
    # variare --router / --mscale-softmax finche' i logit coincidono (voci 1-2); --l4-qpe solo con prompt > 8192 (voce 3)

Limite di velocita' (aritmetica, non misura): ~6,65 GB letti per token; a 40 GB/s di RAM <= ~6 tok/s, da NVMe a 3 GB/s <= ~0,5 tok/s.

## Errori da non ripetere (dalla guida, ancora validi)
Patch generato al contrario; segnaposto nei comandi di test; statistiche che contano gli admit come hit; avviare la calibrazione senza `--cuda`;
strutture non inizializzate passate ai kernel (`= {}`); **dato per noto qualcosa che e' nella tabella "Non noto"**.

## Prefill, contesto persistente, VRAM (aggiunti dopo la prima consegna)

**Prefill a batch** (`--prefill-chunk N`, default 512). Ogni chunk percorre i layer uno alla volta: prima la cache MLA dell'intero chunk, poi attenzione
per token, poi gli expert **raggruppati** (i pesi di un expert si percorrono una volta per chunk, non una per token: e' questo che conta quando i pesi
arrivano dal disco via mmap). Provato: i logit coincidono con il percorso token-per-token e con il golden per chunk 1/5/7/64, a 1 e a piu' thread.

**Sui 32k token/s: non e' raggiungibile con questo codice e va detto chiaro.** E' un obiettivo di hardware, non un parametro. Aritmetica: ~6,5 G parametri
attivi = ~13 GFLOP/token, quindi 32.000 tok/s richiedono ~0,42 PFLOP/s sostenuti. Una CPU o una GPU consumer sono ordini di grandezza sotto; serve una
classe H100/B200 con kernel FP8 a matrice (GEMM) veri. Qui il prefill e' ancora matvec per token dentro un chunk: elimina il traffico dei pesi
ripetuto, non porta il throughput a quel livello. Non ho misure sul modello vero, quindi non do nessun numero di tok/s. Il PP del protocollo riporta il
tasso reale misurato su ogni chunk: e' li' che lo leggerai.

**Il contesto non si perde piu' tra le domande** (`--kv-unified` accettato: il motore ha una sola sequenza e una sola cache condivisa da tutte le richieste).
- Riuso del prefisso: la cache tiene i token; a ogni `GEN` si calcola il prefisso comune con quanto e' in cache e si prefill-a solo il resto
  (l'ultimo token del prompt si ricalcola sempre, per averne i logit). Provato: domanda 2 = domanda 1 + 4 token -> `40 of 44 already cached, 4 to prefill`.
  Vale anche per le risposte generate se il template le rimanda uguali; dove il template cambia i token (EOS, tag), si riparte dal primo token diverso.
- Checkpoint (`--kv-checkpoint FILE`): scritto **dopo ogni richiesta** (tmp + rename, non solo a QUIT) e ripristinato all'avvio. Porta una impronta di
  config + router + convenzione scale + dimensione pesi: un checkpoint di un'altra configurazione viene rifiutato (provato). Costo: 46 KiB/token in float32
  (a 32k token = 1,5 GiB riscritti a ogni richiesta; se pesa, la prossima mossa e' scrivere solo il delta).
- Un bug trovato provandolo: il controllo di `STOP` scartava le righe in coda (anche un `GEN`); ora vengono tenute.

**Percentuale di VRAM** (`--vram-pct P --vram-total-mib N [--vram-reserve-mib N]`): **solo piano, niente esecuzione.** Non c'e' backend CUDA in questo albero
(fase 5 non fatta) e senza driver non posso interrogare la scheda, quindi la dimensione va data a mano. Il piano calcola: pesi densi (attenzione, condiviso,
gate, embedding, lm_head, ~4,1 GiB) in VRAM, il resto del budget diventa slot di expert da 24 MiB, ripartiti per layer. Nessun expert viene caricato.
Per eseguirlo servono il backend CUDA (kernel FP8 + `gpu.hpp` a due backend come in dev4) e il livello VRAM/RAM/disco della fase 6.

## Integrazione in Strata

Stesso meccanismo di `dsv4_ext` (letto in `dev4.patch`): **additivo, Strata non si modifica.** Tre pezzi.

1. **Posizione.** La cartella va dentro il checkout di Strata, accanto a `dsv4_ext/` (`<Strata>/mistral4_ext/`): lo script risale da li' al root per importare `serve.server`.
   Il patch dev4 aggiungeva `add_subdirectory(dsv4_ext)` al `CMakeLists.txt` di root; **qui non serve**: `sh build.sh` e' autonomo (g++ + OpenMP, nessuna dipendenza da Strata).
2. **Protocollo.** `m4_run --serve` parla la stessa riga di comandi di `strata --serve` (`READY/INFO`, `GEN`, `PP`, `T`, `DONE`, `ERR`, `STOP`, `QUIT`): il serve loop e' quello di dev4.
   `tools/serve_m4_strata.py` importa `serve.server`, sostituisce `StrataEngine` con una sottoclasse che lancia `build-out/m4_run` e chiama il suo `main()`:
   API OpenAI `/v1/chat/completions`, Anthropic `/v1/messages` e web app (Chat/Monitor/About) restano quelle di Strata.

       python3 tools/serve_m4_strata.py --model /path/Mistral-Small-4-119B-2603 --port 8095 --ctx 32768 \
           --kv-checkpoint /var/tmp/m4.kv --prefill-chunk 512

3. **Tokenizer: e' il punto debole.** dev4 lo estraeva dal GGUF con `tools/strata_tokenizer.py`; qui non c'e' GGUF. `convert_tokenizer()` legge il `tokenizer.json` HF
   (BPE byte-level) e scrive i cinque file che il server di Strata si aspetta (`vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json` di config con `pre_pattern`,
   `chat_template.jinja`), con i formati ricavati dal tokenizer estratto presente nel patch. Rifiuta con un errore esplicito un vocabolario non byte-level.

**Cosa NON e' verificato** (il checkout di Strata non e' nel patch e non ce l'ho): la sottoclasse e gli argomenti di `S.main()` sono copiati da `serve_dsv4.py`, mai eseguiti;
la conversione e' provata solo su un `tokenizer.json` sintetico; non so se Strata accetta `"pre": "mistral4"` (dev4 usava `joyai-llm`, un nome che il suo tokenizer conosce)
ne' se il suo tokenizer riproduce il pre-tokenizzatore del modello: se la tokenizzazione differisce da transformers, **i logit saranno sbagliati senza nessun errore**.
Controllo obbligatorio prima di fidarsi: stessi 5-10 prompt tokenizzati da Strata e da `transformers.AutoTokenizer`, id identici. Il chat template di Mistral usa `reasoning_effort`
("none"/"high"): il server di Strata potrebbe non inoltrarlo (dev4 gestiva `thinking`); se manca, la modalita' ragionamento non si attiva da API.
Se la conversione non basta, il ripiego e' `tools/serve_m4.py` (server autonomo, tokenizer HF via `transformers`, nessuna dipendenza da Strata).

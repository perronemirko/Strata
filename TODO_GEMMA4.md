Stato verificato adesso
Repo pulito. Lavoro già committato: bb3a3e1 "Gemma improvements", 4 file, +390/−14 (model.py, ops.py, backends.py, test_gemma4.py). git status vuoto → nessun commit pendente.

Test: 74 passed in 1.74s.

GPU: 23953 / 24564 MiB, util 0%. Quel processo sono io. Non la libero, non misuro su CUDA. RAM: 51 GiB available.

---

## Aggiornato 2026-10-03, dopo la misura su 4 profili (bench/results/2026-10-03-expert-residency/)

Test ora: 90 passed in 1.82s. Numeri sotto: torch CPU float32, layer 0 reale, Q4_0, un processo per config.

**Voce 1 chiusa, ma la premessa era sbagliata.** Stack a blocchi e scratch riutilizzabile esistono e batchano (`BlockedExperts`, `ScratchExperts`, `batched_views = True`), ma nel regime di una card da 24 GB sono il SBAGLIO. A parità di memoria (48 esperti/layer): stream 2109 ms, scratch 2823 ms, block 3252 ms → block è 1.55x più lento di stream. Motivo: `_fill` copia ogni esperto in un buffer stacked (`store_row`, 3.7 s su 128 esperti contro 1.9 s di letture) e con keep piccolo la copia si ripete ogni forward (`built=32`), mentre il batching che la paga non arriva mai, perché i run del router coprono più blocchi di quanti ne tenga.

**Voce 2, il numero decisivo che mancava: la spaccatura è resident vs non-resident, non modalità vs modalità.** 4 profili con 128/128 esperti resident: 95–167 ms a 128 tok. 7 profili sotto: 2.1–3.3 s. Gap ~20x, e l'etichetta sulla riga non decide da che parte cade. Lettura+dequant di un esperto = 14.9 ms (22.7 MiB) → un forward che tocca tutti e 128 paga ~1.9 s di letture comunque si raggruppino i matmul. Il batching vale 1.1–1.2x SOPRA la residency: `block 16 keep=8` 132 ms a 512 tok contro `stack` 147 ms (1.11x). Vero, piccolo, non chiude il gap.

**Nuova misura, quella che decide davvero (cross-forward.log).** 4 turni = prefill 512 tok + 64 decode, routing skew zipf 0.9. `block 8 keep=4` legge ancora 3522 volte negli ultimi 3 turni: 3.2 s/prefill, 5.4 s/decode, contro 171 ms e 383 ms con il layer tutto resident. La residency NON è warm-up: è per-forward, perché un prefill tocca tutti gli esperti del layer e un LRU da 48 non li tiene.

**Voci 8 e 9 fatte.** `TorchBackend` docstring: tabella completa con cosa è stato misurato dove. `bench/results/2026-10-03-expert-residency/README.md`: tabelle, comando di riproduzione, cosa ha deciso.

**Default cambiato: `--expert-mode auto`.** `auto_expert_profile` sceglie per regime, non per etichetta: se tutti gli esperti entrano → `stack` (perché `block` con block×blocks ≥ num_experts costa gli STESSI byte e ha misurato più lento: 95 ms vs 110–167 ms); altrimenti → `stream` dimensionato sui byte liberi. Su 24 GB float16: `stream keep=47`, 19.9 GiB totali. Residency completa = 42.5 GiB di esperti in float16 → nessuna 24 GB ci arriva. Il server ora stampa quale regime hai comprato invece di far finta che il default sia veloce.

**Due bug trovati e corretti, non dal bench ma dai test:** `_resident_experts` divideva per la dimensione di un esperto senza `num_hidden_layers` (avrebbe detto "12 di 8"); `auto_expert_profile` dimensionava `keep` come se il budget coprisse un layer invece di 30 (42 GiB su una card da 24).

Rimane aperto: voci 3 (prefill 2k–8k mai corso), 4 (sweep `batched_min_rows`), 5 (CUDA quando la card è libera), 6 (`stack_first` codice morto), 7 (`concat_last` non misurato), 10 (breakdown gate_up/down/scatter).

Cosa rimane — in ordine di valore
1. Il buco vero: il batch non si attiva mai dove serve.
Il percorso batchato vive solo su batched_views = True (model.py:333). Il profilo che sta su una card da 24 GB usa StreamedExperts, che dichiara batched_views = False (backends.py:168), ed è il default del torch backend (expert_stream: int = 24, backends.py:373). Quindi su GPU il codice nuovo non gira mai. È la voce grossa.

Due strade:

stack a blocchi: tenere su device 16–32 esperti impilati, batchare dentro i run contigui, far ruotare i blocchi.
scratch riutilizzabile: dequant del gruppo in un buffer fisso [G, hidden, 2*inter], come fa Gemm::native in C++. Costo copia gruppo da 8: gate_up 15.9 MiB + down 7.9 MiB per esperto → ~190 MiB contro ~100 ms di matmul. Copiabile, ma da misurare, non da dichiarare.
2. Numeri da correggere — i miei erano disonesti.
Soglia batched_min_rows = 8 (model.py:373). Righe per esperto e gruppi batchati reali:

tok	righe/esperto	gruppi batchati
1	0–1	0
8	0–1	0
32	2	0
128	8	1
512	32	1
Quindi 0.97x, 1.19x, 1.12x a 1/8/32 tok erano loop contro loop = rumore. Unico guadagno reale misurato: 1.12x a 128 tok, 1.08x a 512 tok, torch CPU, layer 0 reale, Q4_0 dequant in F32. Parità max|diff| ~1e-5.

Il bench calcola groups (/tmp/bench_gemma4_moe.py:61) ma non lo stampa → ogni riga va etichettata con quanti gruppi batchano davvero.

3. Prefill reale non misurato. Bench si ferma a 512 tok. Il prefill vero è 2k–32k, ed è lì che il batch paga. 2048 e 8192 mai corsi.

4. Sweep soglia. batched_min_rows = 8 mai tarato. Sweep {2,4,8,16,32} a 128/512/2048 tok.

5. GPU rimandata, non fatta. Su CUDA il bmm dovrebbe vincere molto di più (launch overhead domina), ma è un'aspettativa, non una misura. Da correre con device="cuda" quando la card è libera.

6. stack_first = codice morto. Zero call site fuori da ops.py:146, :295, :452. model.py non lo usa più. Togliere abstract + entrambe le implementazioni, o documentare perché resta. Il test a test_gemma4.py:1256 confronta abstract - have == set(), quindi va toccato insieme.

7. concat_last nel path gate/up separati (model.py:341): copia il gruppo, non lo stack. Corretto, coperto da test, ma costo mai misurato. Il file testato ha gate_up_exps fuso → quel ramo non è mai passato nel bench.

8. Docs: zero righe scritte. grep su docs/ e AGENTS.md per batched_views|batched matmul|matmul3 → nessun risultato. AGENTS.md chiede parole piane e numeri misurati con cosa li ha misurati. Da scrivere.

9. Vincolo di memoria da scrivere nero su bianco. Stack F32 esperti 26B: 2.84 GiB/layer × 30 = 85.1 GiB (verificato ora dal config del GGUF: 30 layer, hidden 2816, moe_inter 704, 128 esperti, top_k 8). Non entra nei 24 GB di GPU, e non entra nei 51 GiB available di RAM. Il profilo stacked F32 è eseguibile solo layer-per-layer, come fa il bench.

10. Bug scratch, non repo. /tmp/probe_routed_parts.py:65: rb.reshape(NE, rows, INTER) fallisce, rb è (NE*rows, H) e H=2816 ≠ INTER=704. Fix: rb.reshape(NE, rows, H). Il breakdown gate_up / down / scatter non l'ho mai ottenuto — senza quello non so se il 1.12x viene dal bmm o dallo scatter index_add_ (ops.py:413).


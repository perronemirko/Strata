Stato verificato adesso
Repo pulito. Lavoro già committato: bb3a3e1 "Gemma improvements", 4 file, +390/−14 (model.py, ops.py, backends.py, test_gemma4.py). git status vuoto → nessun commit pendente.

Test: 74 passed in 1.74s.

GPU: 23953 / 24564 MiB, util 0%. Quel processo sono io. Non la libero, non misuro su CUDA. RAM: 51 GiB available.

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


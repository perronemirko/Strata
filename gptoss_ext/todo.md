Un expert non presente in VRAM lo calcola la CPU in place dalla RAM, in parallelo con la GPU che lavora su quelli in cache.
Tutti gli expert stanno in RAM, bloccati (pinned), con kernel AVX-512/AVX2 per la CPU.
Sulla GPU stanno anche l’attenzione, la KV cache (oltre i 64K solo la parte più letta, il resto in streaming dalla RAM) e l’output head.
La cache degli expert si adatta alla conversazione mentre chatti.
Gli expert del layer successivo viaggiano sulla GPU mentre gira l’attenzione del layer corrente, e il prefill usa pezzi fino a 8192 token.
Se possibile Decodifica speculativa: il livello MTP del modello propone fino a 3 token, e prompt lookup fino a 5, verificati in un solo passaggio.

# CLI to run DeepSeek-V4-Flash in the container

Model: `/media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf` (3 shards, auto-loaded from the first).
Profile that already exists and is valid for this model: `/media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin` (DSV4PROF v1, 43 layers x 256 experts, 11008 entries, 87.6% seen).

## 1. Engine, --serve (the live config + the two missing levers)

    /workspace/dsv4_ext/build-out/dsv4_run_cuda --serve \
      --model /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
      --ctx 16384 --threads 12 \
      --expert-vram-pct auto --expert-vram-reserve-mib 1024 \
      --expert-ram all --ram-cache-mib 32768 \
      --expert-profile /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin \
      --expert-profile-save /var/tmp/dsv4_flash_expert.bin \
      --expert-adapt-swaps 8 \
      --prefill-chunk 4096 -k e4m3 -v e4m3 \
      --kv-checkpoint /var/tmp/dsv4_flash.kv

## 2. Strata API on top (serve_dsv4.py has no -k/-v/--kv-checkpoint: use --engine-arg)

    python3 /workspace/dsv4_ext/tools/serve_dsv4.py --cuda \
      --model /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
      --port 8095 --host 127.0.0.1 --ctx 16384 --threads 12 \
      --expert-vram-pct auto --expert-vram-reserve-mib 1024 \
      --expert-ram all --ram-cache-mib 32768 --expert-adapt-swaps 8 \
      --expert-profile /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin \
      --engine-arg -k --engine-arg e4m3 --engine-arg -v --engine-arg e4m3 \
      --engine-arg --kv-checkpoint --engine-arg /var/tmp/dsv4_flash.kv \
      --engine-arg --prefill-chunk --engine-arg 4096

## 3. Profile refresh (one-shot; ONLY while the resident engine is stopped)

The engine holds 20.17/23.60 GiB of VRAM (2.52 GiB free): a second CUDA process would OOM.
Tokenize with the model's own BPE + chat template (no GPU, no weights):

    M=/media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf
    IDS=$(python3 /workspace/dsv4_ext/tools/dsv4_tokenize.py --model "$M" --chat --ids \
          --messages '[{"role":"user","content":"ciao, come stai oggi?"}]')

    /workspace/dsv4_ext/build-out/dsv4_run_cuda --model "$M" --prompt-ids "$IDS" --n-predict 256 \
      --ctx 4096 --expert-ram all --ram-cache-mib 32768 --prefill-chunk 4096 -k e4m3 -v e4m3 \
      --dump-routing /var/tmp/run1.trace

    python3 /workspace/dsv4_ext/tools/make_expert_profile.py /var/tmp/run1.trace -o /var/tmp/dsv4_flash_expert.bin

Ready-made ids (from dsv4_tokenize.py, chat template applied):

    ciao, come stai oggi?                       128803,6539,81,14,2822,370,2238,65099,33,128804,128822
    QuickSort in Python (it)                    128803,7508,84,18463,612,33121,1298,33531,10824,26618,22156,295,15255,40450,7944,84255,16,128804,128822
    MoE vs dense (en)                           128803,89422,624,25805,532,114735,7752,1469,260,21857,10672,294,58124,343,28608,39,11,305,260,27091,2645,16,128804,128822

## 4. Flags that do NOT exist in this engine (exit 2 at dsv4_run_main.cpp:130)

- `--expert-promote` is a gptoss_ext flag; the dsv4 spelling is `--expert-adapt-swaps N`.
- `-k` and `-v` must match (dsv4_run_main.cpp:136: one fused KV row per token).
- A gpt-oss profile is refused: "not a DSV4 expert profile (version 1)".

## 5. Measured

| config | experts in RAM | decode |
| --- | --- | --- |
| `--expert-ram all --ram-cache-mib 32768` | 11008 / 74.42 GiB | 3.8-6.0 tok/s |
| no `--expert-ram`, 8 GiB arena | 0 | 0.51 tok/s |

`--expert-ram all` costs 109.7 s of the 133.8 s load, once. Keep it.

## 6. Measured on 2026-10-21 13:29 (one-shot, container, threads 28)

Run: `--prompt-ids 128803,6539,...` (11 tokens) `--n-predict 256 --ctx 4096 --expert-ram all --ram-cache-mib 32768 --prefill-chunk 4096 -k e4m3 -v e4m3 --dump-routing /var/tmp/run1.trace`

    load    112.0 s  (94.9 s = 74.42 GiB experts copied to RAM)
    prefill  11 tokens in 4.94 s  = 2.23 tok/s   (1 chunk, chunk cap 128)
    decode   24 tokens in 5.85 s  = 4.10 tok/s
    experts  hits 8064 / misses 708 = 91.93% hit rate, admits 2011, swaps 0
    H2D      13.581 GiB  (cold admission: this is what a profile moves off the hot path)
    RAM arena 0 hits, 0 loads, 0 evictions  (unused: --expert-ram all keeps everything resident)
    CPU miss time 1.99 s, GPU hit wait 1.27 s
    text: "Ciao! Sto benissimo, grazie per avermelo asked. Come posso aiutarti oggi?"  -> correct

## 7. Profile merged and verified

    python3 /workspace/dsv4_ext/tools/make_expert_profile.py /var/tmp/run1.trace \
      --merge /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin \
      --bytes-per-expert 6.9 -o /var/tmp/dsv4_flash_expert.bin

Result: DSV4PROF v1, 43 layers x 256 experts, model_hash 0xc15740f33aa31097 (matches this model), 9644/11008 experts seen (87.6%), estimated RAM 64.98 GiB. Coverage curve: 50% of the activations need 1123 experts, 80% need 3397, 90% need 4933, 99% need 8133.

Use it as `--expert-profile /var/tmp/dsv4_flash_expert.bin --expert-ram all`: the profile decides only WHERE the experts live (it preloads the hottest per layer into VRAM, so `preloaded: 0` becomes ~2070 and the 13.58 GiB of H2D move to load time); the router still decides which run.

## 8. --prefill-chunk above 128 is a no-op

model.cpp:692 caps the chunk at the sliding window: `if (win > 0 && want > win) want = win;` with win = 128. So `--prefill-chunk 4096` runs as chunks of 128 (the log says so: `prefill: chunks of 128 tokens`). 128 is the ceiling for this model; ask for it, do not expect more.

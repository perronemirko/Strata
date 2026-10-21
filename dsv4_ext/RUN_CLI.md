# DSv4-Flash — the CLI to run (container)

Paths are the container's (`/workspace`). The model lives on the host mount `/media/albus/windows/...`.
The resident engine holds 20.17/23.60 GiB of VRAM (2.52 GiB free): never start a second CUDA process
while it is up, and never kill it.

## 1. Serve — the engine

```bash
/workspace/dsv4_ext/build-out/dsv4_run_cuda --serve \
  --model /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
  --ctx 16384 --threads 12 \
  --expert-vram-pct auto --expert-vram-reserve-mib 1024 \
  --expert-ram all --ram-cache-mib 32768 \
  --expert-adapt-swaps 8 \
  --expert-profile /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin \
  --expert-profile-save /var/tmp/dsv4_flash_expert.bin \
  --prefill-chunk 4096 -k e4m3 -v e4m3 \
  --kv-checkpoint /var/tmp/dsv4_flash.kv
```

- `--expert-profile` is optional; without it the load report says `expert profile: none` and `preloaded: 0`,
  so every VRAM slot is uploaded cold during decode.
- `--expert-ram all` costs ~110 s of load and is the measured 9x win over leaving experts on disk
  (4.7-6.0 tok/s vs 0.51 tok/s on this card).
- `--expert-promote` does NOT exist in `dsv4_ext` (it is a `gptoss_ext` flag): an unknown argument makes
  `dsv4_run` print the usage and exit 2. The dsv4 equivalent is `--expert-adapt-swaps N`.
- `-k` and `-v` must match: DSv4-Flash caches one fused row per token.

## 2. Strata API on top of the engine

`tools/serve_dsv4.py` has no `-k`/`-v`/`--kv-checkpoint` options, so they go through `--engine-arg`:

```bash
python3 /workspace/dsv4_ext/tools/serve_dsv4.py \
  --model /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf \
  --cuda --port 8095 --host 127.0.0.1 --ctx 16384 --threads 12 \
  --expert-vram-pct auto --expert-vram-reserve-mib 1024 \
  --expert-ram all --ram-cache-mib 32768 --expert-adapt-swaps 8 \
  --expert-profile /media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/dsv4_experts.bin \
  --engine-arg "-k" --engine-arg "e4m3" --engine-arg "-v" --engine-arg "e4m3" \
  --engine-arg "--kv-checkpoint" --engine-arg "/var/tmp/dsv4_flash.kv" \
  --engine-arg "--prefill-chunk" --engine-arg "4096"
```

## 3. The ids for `--prompt-ids`

They come from `tools/dsv4_tokenize.py`, which reads the GGUF's own BPE vocabulary and applies the
      --messages '[{"role":"user","content":"ciao, come stai oggi?"}]')
```

Real ids already produced (chat template applied, 11 tokens):

```
128803,6539,81,14,2822,370,2238,65099,33,128804,128822
```


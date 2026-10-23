# DSv4-Flash — the CLI to run (container)

Paths are the container's (`/workspace`). The model lives on the host mount `/media/albus/windows/...`
(`dsv4_ext/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf` auto-loads all three shards).
The resident engine holds 20.17/23.60 GiB of VRAM (2.52 GiB free): never start a second CUDA process
while it is up, and never kill it.

**Check the GPU before choosing the binary.** `nvidia-smi -L` answering `Failed to initialize NVML`
means this container has no card: only `dsv4_run` (CPU-emulated, `--cpu`) runs here, and `dsv4_run_cuda`
must be started on the host. The same check decides whether `--expert-vram-pct` is worth anything.

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

## 4. The expert-tier knob (new in this build)

```
--pcie-frac F   share of a layer's MISS experts sent to the device by DMA from the page-locked
                LRU arena. -1 = auto (0.55, the native-pack value Strata uses), 0 = compute every
                miss in place on the CPU tier with the OpenMP threads, 1 = stage every miss.
```

`0` is the CPU tier the OpenMP threads were idle for before; `1` is the old all-or-nothing staging.
The run report now says which path was actually taken, so the effect is measured, not assumed:

```
pinned DMA: 12.345 GiB direct from the arena, 0.000 GiB still staged
readback:   4128 doorbell, 0 synchronous DtoH | graphs: 43 captured, 1720 replays
```

Zeros in the first pair mean the LRU arena was not page-locked — the load report says why (see
[`docs/TODO_IMPLEMENTED.md`](docs/TODO_IMPLEMENTED.md)).

## 5. RAM before starting

`--expert-ram all` is 74.4 GiB of private RAM on this model. With `MemAvailable` below that plus a
big `--ram-cache-mib`, the load swaps and the engine never reports `READY` — say so with
`--host-ram-reserve-mib`, or keep `--expert-ram profile`. The arithmetic and the fix are in
[`AVVIO_HOST.md`](AVVIO_HOST.md).


# How to run gptoss_ext

Every command here is a real path in this repository. Every result marked *verified* was produced on this machine; the rest are the commands to run, without a claimed result. Nothing in this file states a number that was not measured.

Target hardware assumed: one NVIDIA RTX 4090 (sm_89, 24.5 GiB), Linux, 128 GB RAM. The CPU path needs no GPU.

---

## 0. Before anything: the toolchain

`build.sh` asks for two things this machine does not provide by default. Both fail loudly, and both have a one-line fix.

| What `build.sh` assumes | What is here | Fix |
| --- | --- | --- |
| `CXX="g++-20"` hardcoded at line 6 of `build.sh` | `g++` is 12.3; `/usr/bin/g++-20` does not exist (available: `-10`, `-11`, `-12`, `-13`, `-14`, `-16`) | edit that line, or build the CPU binary directly (see below) — `CXX=g++-12 sh build.sh` does **not** work, the assignment in the script wins |
| `nvcc` in `PATH` | `/usr/bin/nvcc` is a **wrapper** that reads `$CUDAVER`; with `CUDAVER` unset it fails with `/usr/local/cuda-/bin/nvcc: No such file or directory` | `NVCC=/usr/local/cuda/bin/nvcc` (that is CUDA 13.1), **or** `export CUDAVER=12.8` so the wrapper resolves `/usr/local/cuda-12.8/bin/nvcc` |

Check it in one command:

```sh
g++ --version | head -1
ls /usr/local/cuda*/bin/nvcc
/usr/local/cuda/bin/nvcc --version | tail -1
```

The CUDA build passes `-arch=sm_89` when `CUDA_ARCH=native` (the default in `build.sh`), which is what an RTX 4090 needs.

---

## 1. Build

### CPU (no GPU, runs the tiny test)

```sh
cd /home/albus/workspaces/RiStarta/Strata/gptoss_ext
sed -i 's/^CXX="g++-20"/CXX="g++-12"/' build.sh && sh build.sh
```

Verified: without that edit the CPU build fails with `build.sh: line 39: g++-20: command not found`.

The equivalent direct command (what `build.sh` runs, verified on this machine with `g++ -std=c++17 -O3 -Wall -Wextra -pthread -fopenmp -Iinclude`):

```sh
g++ -std=c++17 -O3 -Wall -Wextra -pthread -fopenmp -Iinclude \
  src/gptoss_run_main.cpp \
  src/safetensors.cpp src/serve_loop.cpp src/config.cpp src/weights.cpp \
  src/ops.cpp src/model.cpp src/sampler.cpp src/gguf_loader.cpp \
  src/gpu.cpp src/prefill_dev_stub.cpp -o build-out/gptoss_run
```

`build.sh` does, in order: regenerate `tests/tiny` if `golden.txt` is missing, compile `build-out/test_gptoss_core` and **run it on `tests/tiny`**, then compile `build-out/gptoss_run`. The CPU binary carries the **emulated device**: its `--cuda` is refused on purpose.

Verified on this machine (the same sources, `g++ -std=c++17 -O3 -Wall -Wextra -pthread -fopenmp -Iinclude`): `test_gptoss_core` on `tests/tiny` ends with `all ok`.

### CUDA (separate binary)

```sh
cd /home/albus/workspaces/RiStarta/Strata/gptoss_ext
NVCC=/usr/local/cuda/bin/nvcc bash build.sh --cuda
```

Produces `build-out/gptoss_run_cuda`. Verified: `built build-out/gptoss_run_cuda`.

To use the CUDA 12.8 toolkit instead of 13.1:

```sh
export CUDAVER=12.8
bash build.sh --cuda          # the wrapper now finds /usr/local/cuda-12.8/bin/nvcc
```

`CUDA_ARCH` overrides the target: `CUDA_ARCH=sm_89 bash build.sh --cuda`.

---

## 2. The model directory

`SNAPSHOT` = a directory holding `config.json` and the `model-0000x-of-000NN.safetensors` shards (the `original/` subfolder of an HF checkout is not needed). The engine reads token ids; the tokenizer and the Harmony format stay outside.

The path used by the regression script on this machine:

```
/media/albus/windows/Users/mio/models/llm_models/models/lmstudio-community/gpt-oss-120b
```

It holds 15 shards, 60.77 GiB mapped. The engine prints that count at load: `15 shard(s), 687 tensors, 60.77 GiB mapped (not read yet)`.

---

## 3. Extract the structure first

`tools/inspect_model.py` reads only the headers of safetensors/GGUF — no weights are loaded (0.2 s on the 60 GiB model, verified).

```sh
python3 tools/inspect_model.py SNAPSHOT
```

Writes two files next to the model:

- `layer_map.json` — the raw report: every tensor template, its dtype, its shape, how many.
- `arch.json` — the contract `arch-spec/1`: `arch`, `layers`, `experts`, `components` + `component_evidence` (the tensor names each rule matched), `params` (each value with `{value, from, how}`: `declared` = the file states it, `deduced` = arithmetic from declared values), `unknown_params`, `layer_types` (which layer is sliding — `layer_kinds` cannot express that), `unbound_tensors` (stems no rule recognizes), `routing`, `missing`, `engine_ready`.

The gateway refuses a model without `arch.json`; it does not fall back to a guess.

---

## 4. Selfcheck and the regression test

The selfcheck is deterministic: 64 fixed prompt ids, then 16 greedy tokens. With `--selfcheck-file` the sequence is compared to the reference in that file (written on the first run). Same flags, same reference.

```sh
cd /home/albus/workspaces/RiStarta/Strata/gptoss_ext
sh run_regression_test.sh
```

`run_regression_test.sh` is that one command with `$@` appended, so extra flags pass through. Expanded, it is:

```sh
./build-out/gptoss_run_cuda \
  --model /media/albus/windows/Users/mio/models/llm_models/models/lmstudio-community/gpt-oss-120b \
  --cuda --ctx 128000 --prefill-chunk 0 --vram-pct 97 --vram-reserve-mib 768 \
  --expert-profile /var/tmp/gptoss-expert.bin --pin-ram-gib 58 \
  -k e4m3 -v e4m3 --expert-promote 3 \
  --selfcheck --selfcheck-file /var/tmp/selfcheck_ref.txt
```

Expected output lines: the model card (`36 layers (18 sliding/18 dense, window 128), hidden 2880, 64 heads / 8 kv x 64 | MoE 128 experts top-4 ff 2880 limit 7 | yarn x32 orig 4096 theta 150000 ... | vocab 201088 eos 200002`), the shard count, the KV plan, then `selfcheck: 201088 logits, non-finite 0, tokens ...` and `selfcheck: MATCH with the reference ...`.

**Status on this machine:** the reference `/var/tmp/selfcheck_ref.txt` exists (76 bytes, from a previous run). The run of the **new** binary stopped at `gpt_oss: CUDA: cudaSetDevice: out of memory` because another process held 23790 of the 24564 MiB of the card. The `MATCH` reported earlier belongs to the **previous** binary and is not evidence about this one. Re-run when the card is free.

The selfcheck also feeds the gate: its result is recorded for the exact configuration (see §6).

---

## 5. Generate

One-shot, ids in and ids out:

```sh
./build-out/gptoss_run_cuda --model SNAPSHOT --cuda --ctx 8192 \
  --prompt-ids 200006,17360,200008 --n-predict 32
```

Resident server (the line protocol READY/INFO/GEN/PP/T/DONE/ERR/STOP/QUIT):

```sh
./build-out/gptoss_run_cuda --model SNAPSHOT --cuda --serve --ctx 32768 \
  --kv-checkpoint /var/tmp/gptoss.kv --kv-rollback 8192 --eos-ids 200002,200012
```

Strata's OpenAI/Anthropic API and web app on top of it — needs the folder to sit inside a Strata checkout (`<Strata>/gptoss_ext/`):

```sh
python3 tools/serve_gptoss_strata.py --model SNAPSHOT --port 8095 --ctx 32768 \
  --kv-checkpoint /var/tmp/gptoss.kv --prefill-chunk 512
```

Status of that script: never run against a Strata checkout (stated in its own header).

Ids come from a tokenizer. `tools/tokeniser_prompts.py` (HF `AutoTokenizer` over `tools/prompts.txt`) produces them; it is a script with a hard-coded `MODEL_DIR` to edit.

---

## 6. The gateway, the adapters and the gate

One entry point, four roles:

| Role | File | Job |
| --- | --- | --- |
| Extractor | `tools/inspect_model.py` | describes the structure (`arch.json`, `arch-spec/1`) |
| Adapter | `include/gptoss/adapter_gpt_oss.hpp` | implements one model: load, prefill, decode. Thin: it calls the `Model` already verified by the selfcheck |
| Gateway | `include/gptoss/gateway.hpp` | reads `arch.json`, validates the contract, finds the adapter in the registry, admits it only if the gate approved it |
| Gate | `include/gptoss/gate.hpp` | certifies: only `MATCH` admits |

The architecture comes from `arch.json`, never from a flag or the directory name. There is no fallback: an architecture without an approved adapter stops before the inference.

`--gate-info` answers what the gateway would do with those flags, **without loading the model** — no GPU needed:

```sh
./build-out/gptoss_run_cuda --model SNAPSHOT --cuda --ctx 128000 --prefill-chunk 0 \
  --vram-pct 97 --vram-reserve-mib 768 --expert-profile /var/tmp/gptoss-expert.bin \
  --pin-ram-gib 58 -k e4m3 -v e4m3 --expert-promote 3 \
  --selfcheck --selfcheck-file /var/tmp/selfcheck_ref.txt --gate-info
```

It prints `arch=`, `schema=`, the `cfg=` fingerprint (the flags that change the numeric path: `cuda`, `ctx`, `prefill_chunk`, `gpu_attn`, `prefill_dev_pct`, `key_format`, `value_format`, `device_graphs`, `async_dense`, `glu_halves`, `rope_pairs`, `no_concentration`, `vram_pct`, `vram_reserve_mib`, `pin_ram_gib`, `expert_promote`), `spec=` hash of `arch.json`, `ref=` hash of the reference, `status=`, `registered=` and the decision. Exit 0 only on `MATCH`.

`--gateway-info` opens the model **through** the adapter and runs the same sequence as the selfcheck, printing `GATEWAY_TOKENS ...`.

Certification lives in `arch/certifications.txt`, append-only, one line per verification:

```
arch | config-fingerprint | code-version | hash(arch.json) | hash(reference) | status
```

A record authorizes a run only if all five match. `NO_MATCH` and `INCONCLUSIVE` block the admission. Writing the reference for the first time is `INCONCLUSIVE`: no comparison happened. `GATE_CODE_VERSION` (a macro in `gate.hpp`, currently `gpt_oss/adapter/phase1`) must be raised at every change to an adapter or numeric path — a record of an old version never authorizes a new binary.

The gate certifies correctness within the tests run, not for every possible input.

---

## 7. Tests

| Test | Command | Needs GPU | Covers |
| --- | --- | --- | --- |
| Core numeric | `./build-out/test_gptoss_core tests/tiny` (built by `build.sh`) | no | forward, prefill, KV reuse, checkpoints, device(emulated) paths vs the numpy reference |
| Gateway + gate | `sh tests/gate_test.sh` | no | the 8 checks below |
| Dequantization | `tests/iq/test_dequant_*.cpp` — **sources only**, not wired into `build.sh`; compile one with the same `g++` line used for the CPU binary | no | MXFP4, IQ, k-quants vs the ggml reference (`tests/iq/ref_*.inc`) |
| GGUF reader | `tests/gguf/test_gguf_*.cpp`, `tests/gguf/deepseek4_contract.cpp` — **sources only**, not wired into `build.sh` | no | header parsing, shard rules, the expected tensor shapes of deepseek4 |

`tests/gate_test.sh` builds its own CPU binary (`build-out/gptoss_run_cpu_gate`) if missing, uses `--gpu-emu` (the device is host memory) and `tests/tiny`, and removes its own reference and record file at the start, so it is idempotent. Verified on this machine: `all ok: the gateway routes, the gate certifies, no silent fallback`. Its 8 checks:

1. no record → the gateway refuses;
2. first selfcheck writes the reference → record `INCONCLUSIVE`;
3. second selfcheck → `MATCH with the reference`, record `MATCH`;
4. same flags → `status=MATCH`, `gateway.open_from_spec -> adapter`;
5. one flag changed (`--ctx 48`) → refused: the record of `--ctx 64` does not authorize it;
6. the adapter path prints the same tokens as the direct path;
7. a directory without `arch.json` → `spec not valid: arch.json absent`;
8. an architecture with no adapter (`tests/gguf`, arch `test`) → `registered=0`, and the diagnostic separates `not supported` from `semantics unknown`.

The tiny checkpoint is regenerable: `python3 tools/tiny_gptoss.py tests/tiny` (numpy float64 reference + the same tensor names as the real model).

---

## 8. Flags that matter

| Flag | Meaning |
| --- | --- |
| `--ctx N` | positions of the dense layers' KV. 128000 costs ~9.36 GiB of KV at fp32; `-k e4m3 -v e4m3` cuts the card windows 4x (the host cache and the KV checkpoint stay fp32) |
| `--prefill-chunk 0` | auto: 8192 on CUDA, 512 on CPU. `N` forces it |
| `--vram-pct P` | % of the card used; `--vram-reserve-mib N` keeps a margin. The regression line uses 97 / 768 |
| `--pin-ram-gib N` | GiB of expert weights page-locked in RAM (58 on a 128 GB machine) |
| `--expert-profile FILE` | expert-usage profile: loaded at start, rewritten after every request; decides which experts get pinned in VRAM |
| `--expert-promote N` | an LRU expert reaching N decode hits becomes pinned, trading with the coldest pinned one |
| `--glu-halves`, `--rope-pairs`, `--no-concentration` | the facts the model card does not settle (`docs/gptoss_porting.md`, "Non noto"); defaults = best reading |
| `--gpu-emu` | tests only: the "device" is host memory |

`--kv-unified` is accepted and ignored (one sequence, one shared cache already). Unknown `GPTOSS_*` environment variables are reported as ignored, not silently dropped.

---

## 9. What is verified, what is not

| Area | Status |
| --- | --- |
| CPU build + `test_gptoss_core` on `tests/tiny` | verified `all ok` on this machine |
| CUDA build | verified `built build-out/gptoss_run_cuda` |
| `tools/inspect_model.py` on the real gpt-oss-120b | verified: 0.2 s, 23 params declared, `unbound_tensors` empty, `layer_types` 18 sliding / 18 full, `unknown_params = [swiglu_alpha, bos]` (not in `config.json`, not guessed) |
| `tests/gate_test.sh` | verified `all ok` |
| Selfcheck of gpt-oss-120b on the **new** binary | **not verified** — the card was occupied; the earlier `MATCH` belongs to the previous binary |
| `tools/serve_gptoss_strata.py` | never run against a Strata checkout |
| `--gateway-info` on the real model | not run (needs the card) |
| Throughput | ~17 tok/s at 11–18k context was reported by a user, not measured here; the ~14–15 tok/s estimate at 123k context has never been run |

Multi-architecture is **not** claimed: one adapter exists. A second model enters only with its own adapter and its own `MATCH`.

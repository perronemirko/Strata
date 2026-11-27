# inklin2ristrata

`strata-inklin2ristrata` — successor of `strata-inkling` (Inklin2RiStrata-2-Mini, `.strata v1.1`). Same shape engine: 48-visit looped MoE, 1..64 continuous batching, vision, DSpark drafting, 4-tier GPU+RAM+SSD experts, CUDA/HIP/Metal/SYCL backends.

Opt-in. The default path stays byte-identical to the last release: the root [`engine/CMakeLists.txt`](engine/CMakeLists.txt:1006) builds `src/inkling2` as before, and `src/inklin2ristrata` only with `-DSTRATA_BUILD_INKLIN2RISTRATA=ON`.

## What is new over linklin2

Modes taken from [`gptoss_ext`](gptoss_ext/README.md) (the gpt-oss-120b engine), which the card's memory system makes worth having:

| Control | What it does | Where |
| --- | --- | --- |
| `--vram-pct P` | use at most P% of the card; the rest is held back as a reservation, so the expert placement and the auto prefill chunk both see the smaller card | [`inklin2ristrata_main.cpp`](engine/src/inklin2ristrata/inklin2ristrata_main.cpp:915) → [`BackendOpts::vram_pct`](engine/src/inklin2ristrata/inklin2ristrata_backend.h:28), applied in [`cuda/inklin2ristrata_cuda_backend.cu`](engine/src/inklin2ristrata/cuda/inklin2ristrata_cuda_backend.cu:228) and [`cuda/inklin2ristrata_cuda_main.cu`](engine/src/inklin2ristrata/cuda/inklin2ristrata_cuda_main.cu:113) |
| `--no-graph` | same kernels, launched one by one, no CUDA Graph capture — the control arm that separates a graph fault from a stream/event fault | [`BackendOpts::no_device_graphs`](engine/src/inklin2ristrata/inklin2ristrata_backend.h:29), sets `runner_->use_cuda_graph = false` in [`cuda/inklin2ristrata_cuda_backend.cu`](engine/src/inklin2ristrata/cuda/inklin2ristrata_cuda_backend.cu:315) |

`--vram-pct` is the gptoss_ext flag of the same name (`--vram-pct` + `--vram-total-mib` in [`gptoss_ext/README.md`](gptoss_ext/README.md:10)); `--no-graph` matches gptoss_ext's `--no-device-graphs`. `inklin2ristrata_cuda_main.cu` already had `--no-graph`; the integrated `strata-inklin2ristrata` binary now has both.

## Model files

The engine reads three kinds of model file; all are the Inkling-2-Mini weights, hosted by Thinking Machines and quantized by Unsloth:

| Kind | What it is | Where it comes from |
| --- | --- | --- |
| `.strata v1.1` (`.strata`) | Strata's own SQ pack (lossless L0..23 expert codes) — the fastest path, no transcode | Strata's release feed, [`terminal/strata_control/inkling_family.py`](terminal/strata_control/inkling_family.py:1) |
| `.gguf` (Unsloth Dynamic, `UD-*`) | `inkling2` arch GGUFs, Q8_K_XL down to IQ1_S | `huggingface.co/unsloth/Inkling-2-Mini-GGUF` |
| HF safetensors | the original `thinkingmachines/Inkling-2-Mini` shards | `huggingface.co/thinkingmachines/Inkling-2-Mini` |

The GGUFs carry `general.architecture = "inkling2"` / `"inkling2_loop"` — the Linklin2 names. This engine accepts both its own names and those, so the existing files load unchanged ([`inklin2ristrata_gguf_loader.h`](engine/src/inklin2ristrata/inklin2ristrata_gguf_loader.h:344)).

The Terminal downloads and verifies them for you (sizes and SHA-256 are pinned in `inkling_family.py`):

```sh
strata launch --family inkling-2-mini --model q4_k_xl --yes --no-tui   # picks the GGUF for the card and downloads it
strata launch --family inkling-2-mini --model-path /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf-00001-of-00002.gguf --yes
```

For a plain download without serving, [`tools/fetch_model_linklin2.sh`](tools/fetch_model_linklin2.sh) does the same fetch with `curl` + resume + size check.

## Build

One script does both steps: [`tools/build_launch_inklin2ristrata.sh`](tools/build_launch_inklin2ristrata.sh).

```sh
sh tools/build_launch_inklin2ristrata.sh build                 # CPU core (CMake)
sh tools/build_launch_inklin2ristrata.sh build --cuda --arch sm_89   # + the CUDA plugin for this card
sh tools/build_launch_inklin2ristrata.sh launch --selfcheck    # the built-in fixture, no model file
sh tools/build_launch_inklin2ristrata.sh launch --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --serve --ctx 32768 --vram-pct 90
```

The script never kills the process that holds the card; it says so and builds. Model files come from [`tools/fetch_model_linklin2.sh`](tools/fetch_model_linklin2.sh) (same weights, the Linklin2 names) or from the Terminal.

The plain CMake path, for reference:

```sh
cmake -S engine/src/inklin2ristrata -B build && cmake --build build --target strata-inklin2ristrata -j
export CUDAVER=12.8
STRATA_CORE_LIB=<path>/libstrata_inklin2ristrata_core.a STRATA_QUANT_LIB=<path>/libstrata_quant.a \
  sh engine/src/inklin2ristrata/cuda/build_cuda.sh
```

The standalone CUDA link needs the `strata_quant` dispatch and the host AVX2/AVX-512 dot kernels; [`cuda/build_cuda.sh`](engine/src/inklin2ristrata/cuda/build_cuda.sh:73) takes them through `STRATA_CORE_LIB` / `STRATA_QUANT_LIB` (from the CMake build). Without them the link behaves as the old script did.

## Two CUDA binaries, one difference that matters

| Binary | Reads | Built by |
| --- | --- | --- |
| `strata-inklin2ristrata` from the root build (`-DSTRATA_BUILD_INKLIN2RISTRATA=ON -DSTRATA_ENABLE_CUDA=ON`) | `.strata`, `.gguf`, HF safetensors — every kind, through `InkCudaBackend::init` → `load_from_model_file` | the integrated path, [`inklin2ristrata_cuda_backend.cu`](engine/src/inklin2ristrata/cuda/inklin2ristrata_cuda_backend.cu:237) |
| `cuda/build/strata-inklin2ristrata-cuda` (standalone) | `.strata` only — `InkCudaModel::load` checks the `STRATA\0\x01` magic | [`cuda/build_cuda.sh`](engine/src/inklin2ristrata/cuda/build_cuda.sh) |

A `.gguf` given to the standalone binary fails with `Invalid .strata magic`. [`tools/build_launch_inklin2ristrata.sh`](tools/build_launch_inklin2ristrata.sh) sends `--cuda` to the integrated binary for that reason; `--arch` other than `native` also builds the standalone one for the portable multi-arch set.

## Validation (this session, 2026-11-25, RTX 4090 + i7-14700K)

Passed:

- `cmake -S engine/src/inklin2ristrata` configures and builds `strata-inklin2ristrata` (CPU core) clean.
- Integrated: `cmake -S engine -DSTRATA_BUILD_INKLIN2RISTRATA=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89` configures with no error, and `cmake --build … --target strata-inklin2ristrata` links it (core + `strata_inklin2ristrata_ep_cuda`).
- The default configure (`cmake -S engine`, no flag) is unchanged: `src/inklin2ristrata` is not registered.
- Test targets of the shared `tests/` sources stay owned by the inkling2 component; this component declares only its own `inklin2ristrata_prefix_mock` + `inklin2ristrata_shared_prefix` (off by default, `STRATA_INK_BUILD_PREFIX_TESTS=ON`), so no target name clashes in a combined build.
- `inklin2ristrata_cuda_main.cu` and `inklin2ristrata_cuda_backend.cu` compile clean with nvcc at `sm_89`, no error.
- `sh cuda/build_cuda.sh` produces `build/strata-inklin2ristrata-cuda` (8.6 MB) and `build/inklin2ristrata_kernel_test`.
- `strata-inklin2ristrata --version` → `0.1.67`; `--help` shows the two new controls.
- [`test_shared_prefix.py`](engine/src/inklin2ristrata/test_shared_prefix.py) against `inklin2ristrata_prefix_mock`: 16 tests, `OK` (GPU-free protocol regression).
- `tools/build_launch_inklin2ristrata.sh`: `bash -n` clean; `build` links the core; `build --cuda --arch sm_89` builds the CUDA plugin and prints the busy-card note without killing the holder; `launch --selfcheck` → `[selfcheck] PASS backend=cpu (CPU (avx2, 28 threads))`; a missing model file is refused with a clear message.
- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17,629,760,736 bytes, fetched to `/media/albus/windows/strata-models/`): SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3` = the pinned ALT_GGUF_FILES digest in [`inkling_family.py`](terminal/strata_control/inkling_family.py:145). `--info --cpu` reads it: `arch=inklin2ristrata_loop format=gguf build=GGUF-NVFP4 layers=24 visits=48 hidden=2304 vocab=200058 experts=128 eos=200006 drafter=dspark dspark_f32_widen=1`. The `mtp.safetensors` drafter (635,801,416 bytes) fetched beside it.
- The integrated CUDA binary on that GGUF: the file is accepted (magic + arch gate pass, the model parses) and reaches the device allocation; it stops at `inklin2ristrata_runner.cuh:1013` with `CUDA error 2: out of memory` because the card had 597 MiB free (the Qwen engine holds 23.5 GiB). Not a model fault — the same command with the card free is what settles it.

Failed: none.

Fixed on the way: the rename had made the GGUF gate expect `inklin2ristrata` / `inklin2ristrata_loop` only, and the file on disk says `inkling2_loop` — the gate now accepts both names ([`inklin2ristrata_gguf_loader.h`](engine/src/inklin2ristrata/inklin2ristrata_gguf_loader.h:344)).

Not run:

- Any CUDA execution. The card is held by the Qwen engine (`nvidia-smi`: `strata-inkling` at 22.6 GiB / 100% util); the process was not killed. `--vram-pct` and `--no-graph` are compiled, not measured.
- HIP, Metal, SYCL backends (no such hardware here).

Limits:

- `--vram-pct` bounds the reservation only; it does not re-plan an already-placed expert set. On the 4090 (24 GiB) it is the way to leave room for a second engine on the same card.
- The two controls are new and opt-in; nothing on the `strata-inkling` default path changed.

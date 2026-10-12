# Test requests and hardware wanted

Strata is built and measured on a few machines. Results from other hardware are welcome and are used: they tell us
what a setup really does and show where the engine picks something wrong. Write a report as described in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md): the build, the card and PCIe width, the CPU and RAM speed, the
model file, the exact command, three runs per size, and what you did not test. Post it in an issue, or on the pull
request it is about.

This list is edited by the maintainers. An item stays until a report answers it.

| Wanted | Why | Example of an earlier answer |
|---|---|---|
| Two NVIDIA cards of different sizes (a large and a small one), in both card orders, with `--layer-split auto` | There is no unequal pair here to check the split planner; the card order and the small card's free memory changed the split | #880, #1238 |
| Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880 |
| Pascal (sm_61: Tesla P40, P100, GTX 10 series) | Older cards run on the fp32 path and can behave differently from newer ones | #875, #876 |
| A Linux host with 32 GB of RAM or less | The expert file tier and the resident mode depend on how much of the model fits in RAM | #1194 |
| A long soak (an hour or more of back-to-back requests) on a split or resident setup | Shows crashes, drift and heat that a short run does not | #848 |
| The gpt-oss engine (`gptoss_ext/`) on a free NVIDIA card, three runs of the same prompt: with the CUDA graph, with `--no-device-graphs`, with `--gpu-slots 0` | The async decode of the top-k experts has never run on a card: on the emulated device it is correct (max |diff| 3.3e-05 against the CPU path), on the RTX 4090 at 2026-10-12 21:50 the same code answered with gibberish while its own counters said the cache was fine (100% hit rate, 0 host fallbacks, 29.5 tok/s). The three runs tell whether the graph or the stream/event ordering is at fault | none yet |

To add a request, open a pull request that adds a row, or tell the maintainers in an issue.

## The gpt-oss async decode (exact commands)

The card must be free: the Qwen engine and the gpt-oss engine cannot share it (the attempt at 2026-10-12 15:51 died at
`cudaSetDevice: out of memory` with the Qwen engine holding 23810 of 24564 MiB). Build first: `sh gptoss_ext/build.sh --cuda`
(nvcc sm_89, clean). Then, on the same PC and model file, three runs of one prompt:

```
python3 gptoss_ext/tools/serve_gptoss_strata.py --model /media/albus/windows/Users/mio/models/llm_models/models/lmstudio-community/gpt-oss-120b \
  --cuda --exe gptoss_ext/build-out/gptoss_run_cuda --ctx 8192 --threads 10 --prefill-chunk 4 --kv-rollback 4 --pin-pct 80 \
  --expert-profile gptoss_ext/profile.prof --expert-profile-save gptoss_ext/profile.prof --expert-profile-save-every 1 \
  --expert-lru-layered --expert-promote 25 --profile-decay 200000 --glu-halves --rope-pairs --no-concentration
```

- run 1: as above (the graph path).
- run 2: add `--engine-arg --no-device-graphs` (the same kernels one at a time, no graph captured).
- run 3: add `--engine-arg --gpu-slots 0` (no expert on the card: the control, must answer correctly).

Report the answer text of each run, the `gptoss_run: experts:` line and the `T`/`DONE` lines from `gptoss_ext/serve-work/gptoss_engine.log`.
What is not tested here: any CUDA path at all — the card is always held by the Qwen engine.

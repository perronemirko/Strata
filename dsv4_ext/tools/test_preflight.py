#!/usr/bin/env python3
"""test_preflight.py - check tools/serve_dsv4.py's preflight against the real command lines.

It imports the launcher and calls its preflight() with the exact argparse namespace main() would
build, so the check runs the same code the start runs - without starting an engine, without
extracting a tokenizer, and without touching the GPU. Two cases, both from serve-work/dsv4_engine.log:

  the 18:04 start that ended in EngineDied after 900 s (--expert-ram all + --ram-cache-mib 32768)
  the 15:01 start that loaded in 114.3 s (same model, the machine had more free RAM then)

The point of the test is not the arithmetic but the decision: over-booking RAM must be a NOTE
(preflight True: the engine clamps and loads), while wanting more than RAM plus swap must be a
REFUSAL (preflight False: main() exits 3 instead of waiting 900 s for an EngineDied).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("serve_dsv4", HERE / "serve_dsv4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

MODEL = ("/media/albus/windows/Users/mio/models/llm_models/models/DeepSeekV4/"
         "DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf")


def mkargs(**kw) -> argparse.Namespace:
    """The namespace main() would hold for these flags (only the ones preflight reads matter)."""
    a = argparse.Namespace(model=MODEL, cuda=True, ctx=16384, threads=12,
                          expert_vram_pct="auto", expert_cache=None, expert_vram_reserve_mib=1024,
                          vram_total_mib=None, vram_free_mib=None, expert_profile="/var/tmp/dsv4_flash_expert.bin",
                          expert_profile_save="/var/tmp/dsv4_flash_expert.bin", expert_ram="all",
                          ram_cache_mib=32768, prefill_chunk=4096, expert_adapt_swaps=8,
                          host_ram_reserve_mib=None, no_preflight=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def main() -> int:
    print("=== case 1: the exact 18:04 command line that ended in EngineDied ===")
    ok1 = mod.preflight(mkargs())
    print(f"preflight -> {ok1} (expected True: over-booking RAM is a note, the engine clamps and loads)")

    print("\n=== case 2: the same model, asked for more than RAM plus swap ===")
    ok2 = mod.preflight(mkargs(ram_cache_mib=900000))
    print(f"preflight -> {ok2} (expected False: main() exits 3 instead of waiting 900 s)")

    print("\n=== case 3: --expert-ram profile, the shape that loaded in 114.3 s ===")
    ok3 = mod.preflight(mkargs(expert_ram="profile", ram_cache_mib=8192))
    print(f"preflight -> {ok3} (expected True)")

    print("\n=== case 4: a non-Linux machine (no /proc/meminfo) must not block the start ===")
    real = mod.read_host_memory
    mod.read_host_memory = lambda: (None, None)
    ok4 = mod.preflight(mkargs())
    mod.read_host_memory = real
    print(f"preflight -> {ok4} (expected True: the check is skipped, not guessed)")

    want = [True, False, True, True]
    got = [ok1, ok2, ok3, ok4]
    if got != want:
        print(f"MISMATCH: got {got}, wanted {want}")
        return 1
    print("\nALL PREFLIGHT DECISIONS CORRECT")
    return 0


if __name__ == "__main__":
    sys.exit(main())

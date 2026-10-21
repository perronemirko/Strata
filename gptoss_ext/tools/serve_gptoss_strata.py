#!/usr/bin/env python3
"""serve_gptoss_strata.py - put Strata's OpenAI/Anthropic API and web app on top of gptoss_run (gpt-oss-120b).

Same mechanism as dsv4_ext/tools/serve_dsv4.py (read from dev4.patch): NOTHING under the Strata checkout is edited. The script imports
serve.server, swaps its StrataEngine for a subclass that spawns build-out/gptoss_run --serve (same line protocol: READY/INFO/GEN/PP/T/DONE/
ERR/STOP/QUIT) and calls its main().

Place this folder INSIDE the Strata checkout, next to dsv4_ext/ :   <Strata>/gptoss_ext/

    python3 tools/serve_gptoss_strata.py --model /path/to/gpt-oss-120b --port 8095 --ctx 32768 \\
        --kv-checkpoint /var/tmp/m4.kv --prefill-chunk 512

STATO: NON eseguito contro un checkout di Strata (non e' nel patch e non l'ho a disposizione). Le parti ipotizzate dal solo patch dev4 sono:
StrataEngine(exe, args, cwd, log, env, lazy), cfg["tokenizer"] = cartella con vocab.json/merges.txt/token_type.json/tokenizer.json/
chat_template.jinja, argv "--engine strata --config ...". La conversione tokenizer.json -> quella cartella e' provata solo su un
tokenizer.json sintetico (tests/test_convert_tokenizer.py)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent
STRATA_ROOT = EXT_ROOT.parent          # gptoss_ext sits inside the Strata checkout
for p in (str(STRATA_ROOT), str(STRATA_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


# Flags whose value is a path: the engine runs with cwd = EXT_ROOT (see cfg["cwd"] below), so a
# relative path handed over through --engine-arg would be resolved against the wrong directory
# ("gptoss_ext/profile.prof" -> "gptoss_ext/gptoss_ext/profile.prof": fopen fails, the profile
# never saves and the engine spams "profile save FAILED" on every request). Absolutize them here.
PATH_FLAGS = ("--expert-profile", "--expert-profile-save", "--kv-checkpoint")


def absolutize_paths(tokens, cwd):
    """Rewrite the value of every PATH_FLAGS token as an absolute path under cwd."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        out.append(t)
        if t in PATH_FLAGS and i + 1 < len(tokens):
            p = Path(tokens[i + 1])
            out.append(str(p if p.is_absolute() else (Path(cwd) / p).resolve()))
            i += 2
            continue
        i += 1
    return out


def _find_regex(node):
    """The Split regex of the HF pre_tokenizer (a Split, or a Sequence containing one)."""
    if not isinstance(node, dict):
        return None
    if node.get("type") == "Split":
        pat = node.get("pattern", {})
        return pat.get("Regex") or pat.get("String")
    for sub in node.get("pretokenizers", []) or []:
        r = _find_regex(sub)
        if r:
            return r
    return None


def convert_tokenizer(model_dir: Path, out: Path, bos: int, eos: int, pad: int) -> dict:
    """HF tokenizer.json (byte-level BPE) -> the 5 files Strata's server reads. Fails loudly on anything it does not recognise."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    m = tj.get("model", {})
    if m.get("type") != "BPE":
        raise SystemExit("tokenizer.json: model.type is %r, only BPE is handled" % m.get("type"))
    vocab = dict(m["vocab"])
    added = tj.get("added_tokens", [])
    for t in added:
        vocab.setdefault(t["content"], t["id"])
    # GPT-2 byte-level check: its alphabet writes a space as U+0120 'Ġ'. A tiktoken/tekken-style vocab (raw bytes) would need another converter.
    if sum(1 for k in vocab if k.startswith("\u0120")) < 50:
        raise SystemExit("tokenizer.json: vocabulary is not GPT-2 byte-level (almost no 'Ġ' tokens): a different converter is needed")
    merges = []
    for mg in m["merges"]:
        merges.append(mg if isinstance(mg, str) else " ".join(mg))
    n = max(vocab.values()) + 1
    ttype = [1] * n                                   # 1 = normal (llama.cpp numbering), 3 = control
    for t in added:
        if t.get("special"):
            ttype[t["id"]] = 3
    pat = _find_regex(tj.get("pre_tokenizer"))
    if not pat:
        raise SystemExit("tokenizer.json: no Split regex in pre_tokenizer: cannot build pre_pattern")
    out.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(vocab.items(), key=lambda kv: kv[1]))
    (out / "vocab.json").write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")
    # No trailing newline: the reader does read_text().split("\n"), so a final "\n" would add an empty merge
    # ("merge N is not a pair: ''"). Same format as the proven writer in tools/strata_tokenizer.py.
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(ttype), encoding="utf-8")
    cfg = {"model": "gpt2", "pre": "o200k_harmony", "vocab_size": n, "n_merges": len(merges),
           "special_ids": {"tokenizer.ggml.bos_token_id": bos, "tokenizer.ggml.eos_token_id": eos, "tokenizer.ggml.padding_token_id": pad},
           "add_bos_token": False,                      # Harmony has no BOS: the template writes everything
           "pre_pattern": pat}
    (out / "tokenizer.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    ct = model_dir / "chat_template.jinja"
    if ct.exists():
        shutil.copyfile(ct, out / "chat_template.jinja")
    else:
        print("WARNING: no chat_template.jinja in the snapshot", file=sys.stderr)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="snapshot directory (config.json + model-*.safetensors + tokenizer.json)")
    ap.add_argument("--exe", default=None, help="the gptoss_run binary (default build-out/gptoss_run)")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--prefill-chunk", type=int, default=None)
    ap.add_argument("--kv-checkpoint", default=None, help="KV checkpoint file, rewritten after every request, restored at start")
    ap.add_argument("--vram-pct", type=float, default=None, help="VRAM plan only (no CUDA backend yet)")
    ap.add_argument("--vram-total-mib", type=int, default=None)
    ap.add_argument("--vram-reserve-mib", type=int, default=None, help="MiB kept out of the VRAM budget (the engine's default is 1024)")
    ap.add_argument("--cuda", action="store_true", help="run on the GPU (needs --exe build-out/gptoss_run_cuda from `sh build.sh --cuda`)")
    ap.add_argument("--cuda-device", type=int, default=None)
    ap.add_argument("--gpu-slots", type=int, default=None, help="cap on expert slots in VRAM")
    ap.add_argument("--expert-profile", default=None, help="expert-usage profile: pins the hottest experts in VRAM, rewritten after every request")
    ap.add_argument("--pin-ram-gib", type=float, default=None, help="GiB of expert weights to page-lock in RAM (faster prefill and misses; keep below free RAM; default off)")
    ap.add_argument("--pin-pct", type=float, default=None, help="%% of the slots reserved for pinned experts (default 80)")
    ap.add_argument("--kv-rollback", type=int, default=None)
    ap.add_argument("--prefill-dev-pct", type=float, default=None, help="percent of the VRAM left for the device-resident prefill (its KV windows + chunk scratch); 0 = the old prefill")
    ap.add_argument("--eos-ids", default="200002,200012", help="stop ids: <|return|>=200002 (config) and <|call|>=200012 (NOT verified)")
    ap.add_argument("--glu-halves", action="store_true")
    ap.add_argument("--rope-pairs", action="store_true")
    ap.add_argument("--no-concentration", action="store_true")
    ap.add_argument("--workdir", default=str(EXT_ROOT / "serve-work"))
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--model-name", default="gpt-oss-120b")
    ap.add_argument("--api-key", default=os.environ.get("GPTOSS_API_KEY", ""))
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--engine-arg", action="append", help="extra gptoss_run flags, verbatim (repeatable)")
    ap.add_argument("--engine-env", action="append", metavar="KEY=VAL")
    a = ap.parse_args()

    exe = Path(a.exe or EXT_ROOT / "build-out" / "gptoss_run").resolve()
    if not exe.exists():
        raise SystemExit("%s not found: run sh build.sh first" % exe)
    model = Path(a.model).resolve()
    if model.is_file() or model.suffix.lower() == ".gguf":
        raise SystemExit("--model must be the Hugging Face snapshot DIRECTORY (config.json + model-*.safetensors), not a file. "
                         "A .gguf (LM Studio) is a different format that this engine does not read.")
    if not (model / "config.json").exists() or not list(model.glob("model-*.safetensors")):
        raise SystemExit("%s: no config.json / model-*.safetensors (download openai/gpt-oss-120b, see README)" % model)
    if a.cuda and Path(a.exe or "").name == "gptoss_run":
        raise SystemExit("--cuda needs the CUDA binary: --exe build-out/gptoss_run_cuda")
    conf = json.loads((model / "config.json").read_text(encoding="utf-8"))
    t = conf.get("text_config", conf)
    workdir = Path(a.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    tdir = Path(a.tokenizer_dir) / "tokenizer" if a.tokenizer_dir else workdir / "tokenizer"
    if not (tdir / "vocab.json").exists():
        cfg_t = convert_tokenizer(model, tdir, t.get("bos_token_id", -1), t.get("eos_token_id", 200002), t.get("pad_token_id", 199999))
        print("tokenizer/: vocab %d, merges %d" % (cfg_t["vocab_size"], cfg_t["n_merges"]), flush=True)

    args = ["--model", str(model), "--ctx", str(a.ctx), "--kv-unified"]
    for flag, val in (("--threads", a.threads), ("--prefill-chunk", a.prefill_chunk), ("--vram-pct", a.vram_pct),
                      ("--vram-total-mib", a.vram_total_mib), ("--vram-reserve-mib", a.vram_reserve_mib), ("--kv-rollback", a.kv_rollback), ("--cuda-device", a.cuda_device), ("--gpu-slots", a.gpu_slots), ("--pin-pct", a.pin_pct), ("--pin-ram-gib", a.pin_ram_gib),
                      ("--prefill-dev-pct", a.prefill_dev_pct), ("--eos-ids", a.eos_ids)):
        if val is not None:
            args += [flag, str(val)]
    if a.expert_profile:
        args += ["--expert-profile", str(Path(a.expert_profile).resolve())]
    if a.kv_checkpoint:
        args += ["--kv-checkpoint", str(Path(a.kv_checkpoint).resolve())]
    for flag, on in (("--cuda", a.cuda), ("--glu-halves", a.glu_halves), ("--rope-pairs", a.rope_pairs), ("--no-concentration", a.no_concentration)):
        if on:
            args.append(flag)
    for extra in a.engine_arg or []:
        args += extra.split()
    # cwd is EXT_ROOT: any relative path flag inside --engine-arg must be resolved there, not where
    # the shell happens to be (the user's typo "gptoss_ext/profile.prof" becomes EXT_ROOT/profile.prof).
    args = absolutize_paths(args, str(EXT_ROOT))
    log = str(workdir / "gptoss_engine.log")
    cfg = {"exe": str(exe), "args": args, "cwd": str(EXT_ROOT), "log": log, "tokenizer": str(tdir), "model_name": a.model_name}
    env = dict(kv.partition("=")[::2] for kv in (a.engine_env or []) if "=" in kv)
    if env:
        cfg["env"] = env
    cfg_path = workdir / "gptoss_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    import serve.server as S   # Strata's own server; imported, never edited

    class GptossEngine(S.StrataEngine):
        """Same protocol, different process: the pump thread, STOP handling, watchdog, restart and sessions are inherited."""

        def __init__(self, exe_, args_, cwd=None, log=None, env=None, lazy=False):
            super().__init__(exe_, args_, cwd=cwd, log=log, env=env, lazy=lazy)
            self.model_path = next((args_[i + 1] for i, x in enumerate(args_) if x == "--model"), str(exe_))
            self.info.setdefault("engine", "gptoss_ext")

    S.StrataEngine = GptossEngine
    argv = ["serve_gptoss", "--engine", "strata", "--config", str(cfg_path), "--tokenizer", str(tdir), "--port", str(a.port), "--host", a.host]
    if a.api_key:
        argv += ["--api-key", a.api_key]
    if a.open:
        argv.append("--open")
    if a.lazy:
        argv.append("--lazy")
    print("gptoss_run: %s\nengine log: %s" % (" ".join(args), log), file=sys.stderr, flush=True)
    sys.argv = argv
    return S.main()


if __name__ == "__main__":
    sys.exit(main())

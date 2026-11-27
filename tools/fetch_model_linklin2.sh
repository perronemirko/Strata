#!/usr/bin/env bash
# tools/fetch_model_linklin2.sh - Download, build and launch for the Linklin2 (Inkling-2-Mini) shape engine.
#
# Three steps in one script, each usable alone:
#   fetch   download the model files (HF safetensors, Unsloth GGUF, or Strata .strata) with resume + size check
#   build   build the engine binary (CPU core, or the CUDA plugin for this card)
#   launch  start the engine on a file you already have or just fetched
#
# The Terminal does the same and more (card auto-selection, presets, the API): prefer
#   strata launch --family inkling-2-mini --model q4_k_xl --yes --no-tui
# This script is the plain-shell path for a developer in a source checkout.
#
# MIT License - Copyright (c) 2026 Strata contributors
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${ENGINE_DIR:-$REPO_ROOT/engine}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/models}"

HF_SAFETENSORS_REPO="thinkingmachines/Inkling-2-Mini"
HF_GGUF_REPO="unsloth/Inkling-2-Mini-GGUF"
CURL="curl -fL --connect-timeout 15 --max-time 1800 --retry 3 --retry-delay 5"

usage() {
    cat <<'EOF'
Usage: fetch_model_linklin2.sh COMMAND [options]

Commands:
  fetch --gguf QUANT [--out DIR]        download one Unsloth Dynamic GGUF (single file)
  fetch --safetensors [--out DIR]      download the HF safetensors checkpoint (+ config, tokenizer, index)
  fetch --strata BUILD [--out DIR]     download Strata's own .strata v1.1 SQ pack (20 | 15 | 10)
  build [--cuda]                       build the engine (CPU core; --cuda adds the sm_89 native CUDA plugin)
  launch --model FILE [--serve|--selfcheck] [--ctx N] [--vram-pct P] [--no-graph]
                                       start the engine on that file

Examples:
  fetch_model_linklin2.sh fetch --gguf q4_k_xl
  fetch_model_linklin2.sh fetch --safetensors
  fetch_model_linklin2.sh build --cuda
  fetch_model_linklin2.sh launch --model "$OUT_DIR/Inkling-2-Mini-UD-Q4_K_XL.gguf" --selfcheck
EOF
}

die() { printf 'fetch_model_linklin2.sh: %s\n' "$*"; exit 1; }

# ---- fetch ----

hf_files_json() { # $1 = repo
    local api="https://huggingface.co/api/models/$1?blobs=true"
    $CURL "$api" 2>/dev/null || die "cannot reach the Hugging Face API for $1"
}

# The GGUF names are the pinned ones in terminal/strata_control/inkling_family.py (UNSLOTH_GGUF_FILES).
gguf_name_for() {
    case "$1" in
        q8_k_xl) echo "Inkling-2-Mini-UD-Q8_K_XL.gguf" ;;
        q4_k_xl) echo "Inkling-2-Mini-UD-Q4_K_XL.gguf" ;;
        q4_k_m)  echo "Inkling-2-Mini-UD-Q4_K_M.gguf" ;;
        iq4_xs)  echo "Inkling-2-Mini-UD-IQ4_XS.gguf" ;;
        q3_k_xl) echo "Inkling-2-Mini-UD-Q3_K_XL.gguf" ;;
        iq3_xxs) echo "Inkling-2-Mini-UD-IQ3_XXS.gguf" ;;
        q2_k_xl) echo "Inkling-2-Mini-UD-Q2_K_XL.gguf" ;;
        iq2_xxs) echo "Inkling-2-Mini-UD-IQ2_XXS.gguf" ;;
        iq1_m)   echo "Inkling-2-Mini-UD-IQ1_M.gguf" ;;
        iq1_s)   echo "Inkling-2-Mini-UD-IQ1_S.gguf" ;;
        *) die "unknown GGUF quant: $1 (use q8_k_xl, q4_k_xl, q4_k_m, iq4_xs, q3_k_xl, iq3_xxs, q2_k_xl, iq2_xxs, iq1_m, iq1_s)" ;;
    esac
}

get_size() { # $1 = repo, $2 = filename -> bytes or empty
    hf_files_json "$1" | python3 -c '
import sys, json
want = sys.argv[1]
d = json.load(sys.stdin)
for s in d.get("siblings", []):
    if s.get("rfilename") == want:
        print(s.get("size", "")); break
' "$2" 2>/dev/null || true
}

fetch_one() { # $1 = repo, $2 = filename, $3 = out dir
    local repo="$1" name="$2" dir="$3"
    mkdir -p "$dir"
    local url="https://huggingface.co/$repo/resolve/main/$name"
    local to="$dir/$name"
    local want
    want="$(get_size "$repo" "$name")"
    if [ -f "$to" ] && [ -n "$want" ] && [ "$(stat -c%s "$to")" = "$want" ]; then
        printf 'have %s (%s bytes), skip\n' "$to" "$want"
        return 0
    fi
    printf 'get %s -> %s\n' "$url" "$to"
    $CURL -C - -o "$to" "$url" || die "download of $name failed"
    if [ -n "$want" ]; then
        local got
        got="$(stat -c%s "$to")"
        [ "$got" = "$want" ] || die "$name is $got bytes, expected $want"
        printf 'size ok: %s bytes\n' "$got"
    else
        printf 'size not checked (the API gave none)\n'
    fi
}

do_fetch() {
    local kind="" quant="" build="" dir="$OUT_DIR"
    while [ $# -gt 0 ]; do
        case "$1" in
            --gguf) kind="gguf"; quant="$2"; shift 2 ;;
            --safetensors) kind="safetensors"; shift ;;
            --strata) kind="strata"; build="$2"; shift 2 ;;
            --out) dir="$2"; shift 2 ;;
            *) die "unknown fetch option: $1" ;;
        esac
    done
    case "$kind" in
        gguf)
            local name
            name="$(gguf_name_for "$quant")"
            fetch_one "$HF_GGUF_REPO" "$name" "$dir"
            # The drafter of a GGUF build (mtp.safetensors) comes from the original repository.
            fetch_one "$HF_SAFETENSORS_REPO" "mtp.safetensors" "$dir"
            ;;
        safetensors)
            # The index first: it names the shards, so only the files that exist are fetched.
            fetch_one "$HF_SAFETENSORS_REPO" "model.safetensors.index.json" "$dir"
            local shards
            shards="$(python3 - "$dir/model.safetensors.index.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
seen = []
for f in d.get("weight_map", {}).values():
    if f not in seen:
        seen.append(f)
print("\n".join(seen))
PY
)"
            [ -n "$shards" ] || die "no shards in the index"
            local s
            for s in $shards; do fetch_one "$HF_SAFETENSORS_REPO" "$s" "$dir"; done
            for s in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
                fetch_one "$HF_SAFETENSORS_REPO" "$s" "$dir" || true
            done
            ;;
        strata)
            # Strata's own SQ pack: served by Strata's feed, not by Hugging Face.
            die "the .strata SQ pack comes from Strata's release feed; use 'strata launch --family inkling-2-mini --model sq-$build' or set STRATA_INKLING_SOURCE to a mirror"
            ;;
        *) die "give one of --gguf QUANT, --safetensors, --strata BUILD" ;;
    esac
}

# ---- build ----

do_build() {
    local cuda=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --cuda) cuda="1"; shift ;;
            *) die "unknown build option: $1" ;;
        esac
    done
    local bdir="$ENGINE_DIR/build-fetch"
    cmake -S "$ENGINE_DIR/src/inklin2ristrata" -B "$bdir" \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_INK_BUILD_PREFIX_TESTS=OFF
    cmake --build "$bdir" --target strata-inklin2ristrata -j "$(nproc)"
    printf 'engine (CPU core): %s\n' "$bdir/strata-inklin2ristrata"
    if [ -n "$cuda" ]; then
        # The standalone CUDA link needs the strata_quant dispatch and the host dot kernels from the core build.
        # ARCH_MODE=native reads this card (nvidia-smi compute_cap); CUDAVER selects the nvcc wrapper.
        STRATA_CORE_LIB="$bdir/libstrata_inklin2ristrata_core.a" \
        STRATA_QUANT_LIB="$bdir/libstrata_quant.a" \
            sh "$ENGINE_DIR/src/inklin2ristrata/cuda/build_cuda.sh"
        printf 'engine (CUDA): %s\n' "$ENGINE_DIR/src/inklin2ristrata/cuda/build/strata-inklin2ristrata-cuda"
    fi
}

# ---- launch ----

do_launch() {
    local model="" mode="serve" extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --serve) mode="serve"; shift ;;
            --selfcheck) mode="selfcheck"; shift ;;
            # everything else goes to the engine as it was typed (--ctx, --vram-pct, --no-graph, --gpu, --parallel, --kv, --draft)
            --*) extra+=("$1"); shift ;;
            *) extra+=("$1"); shift ;;
        esac
    done
    [ -f "$model" ] || die "no such model file: $model (run 'fetch' first, or pass --model PATH)"
    local exe="$ENGINE_DIR/build-fetch/strata-inklin2ristrata"
    [ -x "$exe" ] || die "engine not built: run 'build' first"
    printf 'launch %s\n' "$exe"
    if [ "$mode" = "selfcheck" ]; then
        "$exe" --selfcheck --native "$model" ${extra+"${extra[@]}"}
    else
        "$exe" --serve --native "$model" ${extra+"${extra[@]}"}
    fi
}

# ---- main ----

[ $# -eq 0 ] && { usage; exit 1; }
cmd="$1"; shift
case "$cmd" in
    fetch) do_fetch "$@" ;;
    build) do_build "$@" ;;
    launch) do_launch "$@" ;;
    all)
        do_fetch "$@" || true
        do_build
        ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $cmd (fetch | build | launch | all)" ;;
esac

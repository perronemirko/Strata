#!/usr/bin/env python3
"""Ispettore di modelli: legge solo gli header, nessun peso viene caricato. Supporta safetensors e GGUF.

Uso:  python3 tools/inspect_model.py DIR_O_FILE [--out layer_map.json]

Mostra: impronta dei componenti, tipi di layer (layer con lo stesso insieme di tensori = stesso tipo),
gruppi di tensori (indice del layer -> {i}, indice dell'esperto -> {e}) con TUTTE le varianti di dtype e forma,
e scrive layer_map.json. Le regole di riconoscimento sono ipotesi da verificare, non certezze.
"""
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

LAYER_RE = re.compile(r"(?:(?<=layers\.)|(?<=blk\.))(\d+)(?=\.)")   # "model.layers.17.x" e "blk.17.x" -> 17
EXPERT_RE = re.compile(r"(?<=experts\.)(\d+)(?=\.)")           # "mlp.experts.42.x" -> 42 (safetensors per-esperto)
PREFIX_RE = re.compile(r"^(?:model\.)?(?:layers|blk)\.\{i\}\.")

# Tipi GGML, verificati da ggml/include/ggml.h (enum ggml_type, llama.cpp): gli id 4, 5, 31-33, 36-38 sono ritirati e non compaiono.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}


def read_safetensors(path):
    """u64 little-endian con la lunghezza, poi un JSON con nome, dtype, forma e offset di ogni tensore."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in head.items()}, {}


def read_gguf(path):
    """Header GGUF: magic, versione, numero di tensori e di metadati, poi i metadati e le descrizioni dei tensori (nessun dato)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(path + ": non e' un file GGUF")
        version, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def s():
            (ln,) = struct.unpack("<Q", f.read(8))
            return f.read(ln).decode("utf-8", "replace")

        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(t):
            if t == 8:
                return s()
            if t == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    return [s() for _ in range(cnt)]
                fmt = "<" + scalar[et]
                f.seek(struct.calcsize(fmt) * cnt, 1)
                return {"array_len": cnt}
            fmt = "<" + scalar[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        meta = {}
        for _ in range(n_kv):
            key = s()
            (t,) = struct.unpack("<I", f.read(4))
            v = value(t)
            if key.startswith("general.") or key.endswith(("block_count", "embedding_length", "expert_count", "expert_used_count",
                                                          "context_length", "vocab_size", "attention.head_count", "attention.head_count_kv",
                                                          "attention.sliding_window", "expert_feed_forward_length", "attention.type",
                                                          "rope.theta", "rope.type", "rope.factor", "rope.scale", "rope.attention_factor",
                                                          "rope.original_ctx", "rope.beta_fast", "rope.beta_slow", "rope.ext_factor",
                                                          "rms_norm_eps", "rope.length_scale", "feed_forward_length")) or isinstance(v, str):
                meta[key] = v
        tensors = {}
        for _ in range(n_tensors):
            name = s()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            (gt,) = struct.unpack("<I", f.read(4))
            f.read(8)   # offset dei dati: non serve
            # GGUF scrive le dimensioni al contrario rispetto a safetensors: le giriamo per confrontarle
            tensors[name] = {"dtype": GGML_TYPES.get(gt, "ggml_%d" % gt), "shape": list(reversed(dims))}
    return tensors, meta


def collect(target):
    if os.path.isfile(target):
        files = [target]
    else:
        files = sorted(os.path.join(target, f) for f in os.listdir(target) if f.endswith((".safetensors", ".gguf")))
    if not files:
        sys.exit("nessun .safetensors o .gguf in " + target)
    tensors, meta, kinds = {}, {}, set()
    for p in files:
        reader = read_gguf if p.endswith(".gguf") else read_safetensors
        kinds.add("gguf" if p.endswith(".gguf") else "safetensors")
        t, m = reader(p)
        tensors.update(t)
        meta.update(m)
    return tensors, meta, kinds


def template_of(name):
    return EXPERT_RE.sub("{e}", LAYER_RE.sub("{i}", name))


AUX_RE = re.compile(r"^(v|mm|a)\.")   # visione (v.), proiettore (mm.), audio (a.): non sono layer di testo


def is_aux(name):
    return bool(AUX_RE.match(name))


def layer_of(name):
    if is_aux(name):
        return None
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def text_arch(meta, fallback):
    """Un file mmproj/multimodale dichiara arch 'clip' ma contiene il testo: l'architettura vera e' il prefisso di '*.block_count'."""
    if meta.get("general.architecture") in ("clip", None) or meta.get("general.type") == "mmproj":
        cands = [k.split(".")[0] for k in meta if k.endswith(".block_count") and not k.startswith("clip.")]
        if cands:
            return cands[0]
    return fallback


def stem_of(name):
    """Nome del tensore dentro al layer, senza prefisso: 'attn_q.weight', 'mlp.experts.{e}.gate_proj.weight', ..."""
    return PREFIX_RE.sub("", template_of(name))


def analyze(tensors):
    groups = defaultdict(Counter)     # template -> Counter((dtype, shape))
    stems = defaultdict(set)          # layer -> insieme dei nomi (stem) dei suoi tensori
    experts = 0
    for name, info in tensors.items():
        groups[template_of(name)][(info["dtype"], tuple(info["shape"]))] += 1
        L = layer_of(name)
        if L is not None:
            stems[L].add(stem_of(name))
        for m in EXPERT_RE.finditer(name):
            experts = max(experts, int(m.group(1)) + 1)
    kinds = defaultdict(list)         # insieme di tensori -> layer che lo hanno
    for L in sorted(stems):
        kinds[frozenset(stems[L])].append(L)
    ordered = sorted(kinds.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
    return groups, ordered, experts


def fingerprint(tensors, config, meta):
    """Riconosce i componenti da nomi e dtype. Ogni voce e' una regola da verificare con il selfcheck, non una certezza."""
    names = list(tensors)
    has = lambda *parts: any(all(p in n for p in parts) for n in names)
    rx = lambda pat: any(re.search(pat, n) for n in names)
    dtypes = {t["dtype"] for t in tensors.values()}
    fp = {
        "attention_sinks": has("sinks"),
        "qk_norm": has("q_norm") and has("k_norm"),
        "mla_latent_kv": has("kv_a_proj_with_mqa") or has("q_a_proj") or has("attn_kv_a_mqa") or has("attn_q_a"),
        "router_bias": rx(r"exp_probs_b|e_score_correction_bias") or has("router", "bias"),
        "shared_experts": has("shared_expert") or has("shexp"),
        "mxfp4_experts": any(t["dtype"] == "MXFP4" and ("exps" in n or "experts" in n) for n, t in tensors.items())
                         or (has("_blocks") and has("_scales")),
        "fp8_block_scales": has("weight_scale_inv"),
        "attention_bias": rx(r"(attn_q|q_proj)\.bias"),
        "linear_recurrent_ssm": rx(r"(^|[._])ssm_") or has("linear_attn") or has("mamba"),
        "sparse_indexer": rx(r"(^|[._])indexer"),
        "kv_compressor": rx(r"(^|[._])compressor"),
        "hash_routing_tid2eid": rx("tid2eid"),
        "hyper_connections": rx(r"(^|[._])hc_"),
        "per_layer_embeddings": rx(r"(^|[._])ple_"),
        "mtp_head": rx(r"(^|[._])nextn\."),
        "quant_dtypes": sorted(d for d in dtypes if d != "F32"),
    }
    if config:
        fp["model_type"] = config.get("model_type")
        fp["architectures"] = config.get("architectures")
    if "general.architecture" in meta:
        fp["gguf_architecture"] = meta["general.architecture"]
    return fp


# Evidence of the fingerprint: which tensor names matched each rule. The adapter reads the names,
# not just the boolean: a rule that matches the wrong tensor must be visible as such.
def evidence(tensors):
    names = list(tensors)
    def has(*parts):
        return [n for n in names if all(p in n for p in parts)]
    def rx(pat):
        return [n for n in names if re.search(pat, n)]
    return {
        "attention_sinks": has("sinks"),
        "qk_norm": has("q_norm") + has("k_norm"),
        "mla_latent_kv": has("kv_a_proj_with_mqa") + has("q_a_proj") + has("attn_kv_a_mqa") + has("attn_q_a"),
        "router_bias": rx(r"exp_probs_b|e_score_correction_bias") + has("router", "bias"),
        "shared_experts": has("shared_expert") + has("shexp"),
        "mxfp4_experts": [n for n, t in tensors.items() if t["dtype"] == "MXFP4" and ("exps" in n or "experts" in n)]
                         + (has("_blocks") if has("_scales") else []),
        "fp8_block_scales": has("weight_scale_inv"),
        "attention_bias": rx(r"(attn_q|q_proj)\.bias"),
        "linear_recurrent_ssm": rx(r"(^|[._])ssm_") + has("linear_attn") + has("mamba"),
        "sparse_indexer": rx(r"(^|[._])indexer"),
        "kv_compressor": rx(r"(^|[._])compressor"),
        "hash_routing_tid2eid": rx("tid2eid"),
        "hyper_connections": rx(r"(^|[._])hc_"),
        "per_layer_embeddings": rx(r"(^|[._])ple_"),
        "mtp_head": rx(r"(^|[._])nextn\."),
    }


# Structural parameters an adapter needs. Each value carries its source; a parameter not declared
# by the model's own files is "unknown", never a guess. "deduced" = arithmetic from declared values.
PARAM_KEYS = {  # canonical name -> (config.json key, GGUF key suffix)
    "layers": ("num_hidden_layers", "block_count"),
    "hidden": ("hidden_size", "embedding_length"),
    "vocab": ("vocab_size", "vocab_size"),
    "heads": ("num_attention_heads", "attention.head_count"),
    "kv_heads": ("num_key_value_heads", "attention.head_count_kv"),
    "head_dim": ("head_dim", "attention.head_dim_k"),
    "ffn_intermediate": ("intermediate_size", "feed_forward_length"),
    "n_experts": ("num_local_experts", "expert_count"),
    "topk": ("experts_per_token", "expert_used_count"),
    "window": ("sliding_window", "attention.sliding_window"),
    "pos_max": ("max_position_embeddings", "context_length"),
    "rope_theta": ("rope_theta", "rope.theta"),
    "rope_type": ("rope_scaling.rope_type", "rope.type"),
    "yarn_factor": ("rope_scaling.factor", "rope.factor"),
    "yarn_beta_fast": ("rope_scaling.beta_fast", "rope.beta_fast"),
    "yarn_beta_slow": ("rope_scaling.beta_slow", "rope.beta_slow"),
    "yarn_orig": ("rope_scaling.original_max_position_embeddings", "rope.original_ctx"),
    "rms_eps": ("rms_norm_eps", "rms_norm_eps"),
    "swiglu_limit": ("swiglu_limit", "swiglu_limit"),
    "swiglu_alpha": ("swiglu_alpha", "swiglu_alpha"),
    "attention_bias": ("attention_bias", "attention.bias"),
    "tie_word_embeddings": ("tie_word_embeddings", "tensor_length"),
    "bos": ("bos_token_id", "bos_token_id"),
    "eos": ("eos_token_id", "eos_token_id"),
    "pad": ("pad_token_id", "pad_token_id"),
}


def params_of(config, meta, tensors):
    """Let the structural parameters from the files that declare them. Returns (params, unknown).
    Never invents: a parameter not declared stays in unknown."""
    out, unknown = {}, []
    for canon, (cfg_key, gg_key) in PARAM_KEYS.items():
        val, src = None, None
        if config:
            cur = config
            for part in cfg_key.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                val, src = cur, "config.json:" + cfg_key
        if val is None:
            for k, v in meta.items():
                if isinstance(k, str) and k.endswith("." + gg_key):
                    val, src = v, "gguf:" + k
                    break
        if val is None:
            unknown.append(canon)
        else:
            out[canon] = {"value": val, "from": src, "how": "declared"}
    # head_dim = hidden / kv_heads: only when both are declared and the division is exact.
    if "head_dim" not in out and {"hidden", "kv_heads"} <= set(out):
        h, kh = out["hidden"]["value"], out["kv_heads"]["value"]
        if isinstance(h, int) and isinstance(kh, int) and kh > 0 and h % kh == 0:
            out["head_dim"] = {"value": h // kh, "from": "hidden/kv_heads", "how": "deduced"}
    # experts from the per-experto names, when the layout shows them.
    if "n_experts" not in out:
        e = max((int(m.group(1)) for n in tensors for m in EXPERT_RE.finditer(n)), default=0)
        if e:
            out["n_experts"] = {"value": e, "from": "tensor names experts.{e}", "how": "deduced"}
    return out, unknown


def layer_types_of(config, meta, n_layers):
    """Type of each layer (sliding/full/...), only if the file declares it. Returns (list, source).
    layer_kinds groups layers by their TENSORS; it cannot express a model whose layers share the same
    tensors but differ in behaviour (gpt-oss: 36 identical stems, alternating sliding/full)."""
    if config and isinstance(config.get("layer_types"), list) and len(config["layer_types"]) == n_layers:
        norm = {"sliding_attention": "sliding", "full_attention": "full"}
        return [norm.get(t, t) for t in config["layer_types"]], "config.json:layer_types"
    for key, v in meta.items():
        if key.endswith(".attention.type") and isinstance(v, list) and len(v) == n_layers:
            return list(v), "gguf:" + key
    return None, None


# Standard structural roles. A stem that matches none of these is not bound to any known component:
# the spec says so, it does not invent a meaning.
KNOWN_STEMS_RE = re.compile(
    r"^(?:"
    r"model\.(?:embed_tokens|norm)\.weight|lm_head\.weight"
    r"|(?:input|post_attention|output)_layernorm\.weight|norm(?:_b)?\.weight"
    r"|self_attn\.(?:q|k|v|o)_proj\.(?:weight|bias)|self_attn\.sinks"
    r"|attn_(?:q|q_b|q_a|k|k_a|kv|kv_a|v|o|o_a|o_b)(?:\.(?:weight|bias|norm))?|attn_sinks|attn_max_bias"
    r"|mlp\.router\.(?:weight|bias)|ffn_gate_inp\.weight|exp_probs_b\.bias"
    r"|mlp\.experts\.(?:gate_up|down)_proj(?:_blocks|_scales|_bias)?|mlp\.experts\.{e}\..*"
    r"|ffn_(?:gate|up|down)_(?:exps|shexp|inp)\.weight|ffn_gate_proj\.weight|ffn_(?:gate|up|down)_proj\.weight"
    r"|gate_proj\.weight|up_proj\.weight|down_proj\.weight"
    r")$")


def unbound_stems(layer_kinds, ev_hits):
    """Stem templates of the layers that no component rule and no standard role recognizes."""
    bound = set()
    for names in ev_hits.values():
        for n in names:
            bound.add(stem_of(n))
    out = set()
    for stems, _ls in layer_kinds:
        for s in stems:
            if s not in bound and not KNOWN_STEMS_RE.match(s):
                out.add(s)
    return sorted(out)


def ranges(xs):
    xs = sorted(xs)
    out, start, prev = [], None, None
    for x in xs:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    if start is not None:
        out.append((start, prev))
    return ", ".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)


def short(v, n=110):
    s = str(v).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[:n] + " ..."


def short_list(xs, n=12):
    return ", ".join(xs[:n]) + (" ... (+%d)" % (len(xs) - n) if len(xs) > n else "")


SUPPORTED_FALLBACK = {"gpt_oss": ["attention_sinks", "router_bias", "mxfp4_experts", "attention_bias"]}


def supported_for(arch):
    """Componenti che il motore esegue per questa architettura, letti da arch/supported.json (dentro gptoss_ext). Senza file: tabella di riserva."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arch", "supported.json")
    try:
        with open(p, encoding="utf-8") as f:
            archs = json.load(f).get("architectures", {})
        return set(archs.get(arch, {}).get("components", []))
    except (OSError, ValueError, AttributeError):
        return set(SUPPORTED_FALLBACK.get(arch, []))


def missing_for(arch, components, layer_kinds):
    """Cio' che il motore non sa eseguire per questo modello. Lista vuota = eseguibile."""
    sup = supported_for(arch)
    miss = []
    if arch not in SUPPORTED_FALLBACK and not sup:
        miss.append("architettura '%s' non registrata nel motore" % arch)
    miss += [k for k, v in components.items() if v is True and k not in sup]
    if len(layer_kinds) > 1 and "multiple_layer_kinds" not in sup:
        miss.append("layer di %d tipi diversi" % len(layer_kinds))
    return miss


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    base = target if os.path.isdir(target) else (os.path.dirname(target) or ".")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(base, "layer_map.json")

    tensors, meta, fmts = collect(target)
    config = None
    cfg_path = os.path.join(base, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)

    groups, layer_kinds, experts = analyze(tensors)
    expert_src = "dai nomi"
    if not experts:
        # Gli esperti fusi (un solo tensore per tutti) non mostrano il numero nei nomi: lo prendiamo dal config o dai metadati GGUF.
        cand = [config.get(k) for k in ("num_local_experts", "num_experts", "n_routed_experts")] if config else []
        cand += [v for k, v in meta.items() if k.endswith("expert_count") and isinstance(v, int)]
        cand = [c for c in cand if isinstance(c, int) and c > 0]
        if cand:
            experts, expert_src = cand[0], "dal config/metadati"
    layers = max((L for _, ls in layer_kinds for L in ls), default=-1) + 1
    fp = fingerprint(tensors, config, meta)
    ev = evidence(tensors)
    dtypes = {t["dtype"] for t in tensors.values()}
    unknown = sorted(d for d in dtypes if d.startswith("ggml_"))
    params, unknown_params = params_of(config, meta, tensors)
    ltypes, ltypes_src = layer_types_of(config, meta, layers)
    unbound = unbound_stems(layer_kinds, ev)

    report = {
        "source": target,
        "formats": sorted(fmts),
        "tensors": len(tensors),
        "layers": layers,
        "experts_in_names": experts,
        "gguf_metadata": meta,
        "fingerprint": fp,
        "layer_kinds": [{"count": len(ls), "layers": ls, "stems": sorted(s)} for s, ls in layer_kinds],
        "groups": {t: [{"dtype": d, "shape": list(s), "count": c} for (d, s), c in cnt.most_common()]
                   for t, cnt in sorted(groups.items())},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Specifica per gli adapter: formato arch-spec/1 (contratto da cui si scrive ogni modello)
    arch_name = text_arch(meta, fp.get("model_type") or fp.get("gguf_architecture") or "unknown")
    n_aux = sum(1 for n in tensors if is_aux(n))
    fp_components = {k: v for k, v in fp.items() if isinstance(v, bool)}
    missing = missing_for(arch_name, fp_components, layer_kinds)
    spec = {
        "schema": "arch-spec/1",
        "source": target,
        "arch": arch_name,
        "layers": layers,
        "experts": experts,
        "experts_source": expert_src,
        "components": {k: v for k, v in fp.items() if isinstance(v, bool)},
        "component_evidence": {k: v for k, v in ev.items() if v},
        "dtypes": sorted(dtypes),
        "layer_kinds": [{"id": i + 1, "layers": ls, "stems": sorted(s)} for i, (s, ls) in enumerate(layer_kinds)],
        "tensors": report["groups"],
        "params": params,
        "unknown_params": unknown_params,
        "layer_types": {"value": ltypes, "from": ltypes_src} if ltypes else {"value": None, "from": None},
        "unbound_tensors": unbound,
        "routing": {"algorithm": "unknown", "topk": params.get("topk", {}).get("value"),
                    "note": "the algorithm (softmax/renorm/bias order) is not declared by the model files; read it from the reference code"},
        "missing": missing,
        "engine_ready": not missing,
    }
    spec_path = os.path.join(base, "arch.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print("scritto: " + spec_path)
    print("motore: " + ("eseguibile, nessun componente mancante" if not missing
                        else "NON eseguibile. Mancano: " + "; ".join(missing)))

    print("formato: %s   tensori: %d   layer: %d   esperti: %d (%s)"
          % ("+".join(sorted(fmts)), len(tensors), layers, experts, expert_src))
    print("parametri structurali (value <- sorgli):")
    for k in sorted(params):
        print("  %-18s %s <- %s (%s)" % (k, short(params[k]["value"]), params[k]["from"], params[k]["how"]))
    if unknown_params:
        print("PARAMETRI UNKNOWN (not declared by the files, not guessed): %s" % ", ".join(unknown_params))
    print("types of layers: %s" % (ltypes_src if ltypes else "NOT DECLARED (layer_kinds groups by tensors, not by behaviour)"))
    if unbound:
        print("TENSORI NOT BOUND to a component or a standard role: %s" % ", ".join(unbound))
    if unknown:
        print("ATTENZIONE tipi non in tabella: %s -> controllare ggml.h e aggiungerli a GGML_TYPES" % ", ".join(unknown))
    if n_aux:
        print("tensori ausiliari (visione/audio/proiettore, esclusi dai layer): %d" % n_aux)
    print("impronta:")
    for k, v in fp.items():
        print("  %-22s %s" % (k, v))
    print("tipi di layer (stessi tensori = stesso tipo):")
    if layer_kinds:
        base_set = layer_kinds[0][0]
        for i, (s, ls) in enumerate(layer_kinds):
            print("  tipo %d: %d layer  [%s]  componenti: %d" % (i + 1, len(ls), ranges(ls), len(s)))
            if i:
                plus, minus = sorted(s - base_set), sorted(base_set - s)
                if plus:
                    print("     in piu' rispetto al tipo 1: " + short_list(plus))
                if minus:
                    print("     in meno rispetto al tipo 1: " + short_list(minus))
    if meta:
        print("metadati GGUF (selezionati):")
        for k, v in meta.items():
            print("  %-40s %s" % (k, short(v)))
    print("gruppi (template -> dtype, forma, quanti):")
    for t, cnt in sorted(groups.items()):
        if len(cnt) == 1:
            (d, s), c = next(iter(cnt.items()))
            print("  %-70s %-10s %-26s x%d" % (t, d, list(s), c))
        else:
            print("  %s   <- %d varianti" % (t, len(cnt)))
            for (d, s), c in cnt.most_common():
                print("      %-10s %-26s x%d" % (d, list(s), c))
    print("scritto: " + out)


if __name__ == "__main__":
    main()
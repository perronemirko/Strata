#!/usr/bin/env python3
"""convert_tokenizer on a SYNTHETIC HF tokenizer.json (GPT-2 byte-level alphabet, 2 specials, a Split regex). Not the real tokenizer."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from serve_m4_strata import convert_tokenizer

def gpt2_bytes():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

with tempfile.TemporaryDirectory() as d:
    d = Path(d); u = gpt2_bytes()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256): vocab[u[b]] = len(vocab)
    vocab["\u0120t"] = len(vocab); 
    for i in range(60): vocab["\u0120w%d" % i] = len(vocab)
    tj = {"model": {"type": "BPE", "vocab": vocab, "merges": [["\u0120", "t"], "a b"]},
          "added_tokens": [{"id": 1, "content": "<s>", "special": True}, {"id": 2, "content": "</s>", "special": True}],
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{L}+|\\p{N}"}}, {"type": "ByteLevel"}]}}
    (d / "tokenizer.json").write_text(json.dumps(tj)); (d / "chat_template.jinja").write_text("{{ x }}")
    cfg = convert_tokenizer(d, d / "out", 1, 2, 11)
    assert json.loads((d / "out/vocab.json").read_text())["</s>"] == 2
    assert (d / "out/merges.txt").read_text().splitlines() == ["\u0120 t", "a b"]
    tt = json.loads((d / "out/token_type.json").read_text()); assert tt[1] == 3 and tt[2] == 3 and tt[10] == 1 and len(tt) == len(vocab)
    assert cfg["pre_pattern"] == "\\p{L}+|\\p{N}" and (d / "out/chat_template.jinja").exists()
    tj["model"]["vocab"] = {k: v for k, v in vocab.items() if not k.startswith("\u0120")}
    (d / "tokenizer.json").write_text(json.dumps(tj))
    try: convert_tokenizer(d, d / "out2", 1, 2, 11); raise AssertionError("must refuse a non byte-level vocab")
    except SystemExit as e: assert "byte-level" in str(e)
print("ok   convert_tokenizer (synthetic tokenizer.json): files, token types, regex, refusal of a non byte-level vocab")

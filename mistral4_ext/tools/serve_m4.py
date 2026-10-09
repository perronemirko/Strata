#!/usr/bin/env python3
"""Fase 7: server HTTP compatibile OpenAI (/v1/chat/completions, /v1/models) sopra `m4_run --serve`.
Solo libreria standard + (per --tokenizer hf) `transformers`. Nessun file del progetto ospite viene toccato.

  python3 tools/serve_m4.py --model /path/to/snapshot --port 8095                 # tokenizer HF reale
  python3 tools/serve_m4.py --model tests/tiny --tokenizer bytes --port 8095      # solo per provare il protocollo

Il tokenizer del modello e' in tokenizer.json / chat_template.jinja nella cartella dello snapshot (il repo ha anche tekken.json:
non usato qui). `reasoning_effort` ("none" | "high") e' passato al chat template.
STATO: il protocollo e' provato sul modello minuscolo con --tokenizer bytes; il ramo --tokenizer hf NON e' stato eseguito
(niente transformers ne' rete nel sandbox)."""
import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Engine:
    def __init__(self, exe, model, ctx, extra):
        self.p = subprocess.Popen([exe, "--model", model, "--ctx", str(ctx), "--serve"] + extra, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.info, self.ctx = {}, ctx
        for line in self.p.stdout:
            if line.startswith("INFO "):
                k, _, v = line[5:].strip().partition("=")
                self.info[k] = v
            elif line.startswith("READY"):
                self.ctx = int(line.split()[1])
                break
            elif line.startswith("ERR"):
                raise RuntimeError(line)
        else:
            raise RuntimeError("m4_run exited before READY")

    def generate(self, ids, max_new, sampling):
        """Generatore di (kind, valore): ('tok', id) ... poi ('done', dict)."""
        with self.lock:
            kv = " ".join("%s=%s" % (k, v) for k, v in sampling.items() if v is not None)
            self.p.stdin.write("GEN %d %s %s\n" % (max_new, kv, ",".join(map(str, ids))))
            self.p.stdin.flush()
            for line in self.p.stdout:
                f = line.split()
                if not f:
                    continue
                if f[0] == "T":
                    yield "tok", int(f[1])
                elif f[0] == "DONE":
                    yield "done", {"generated": int(f[1]), "prompt": int(f[2]), "finish": f[5]}
                    return
                elif f[0] == "ERR":
                    raise RuntimeError(line.strip())


class Tok:
    def __init__(self, kind, model, vocab_hint=96):
        self.kind = kind
        if kind == "hf":
            from transformers import AutoTokenizer
            self.t = AutoTokenizer.from_pretrained(model)
        self.vocab = vocab_hint

    def encode_chat(self, messages, effort):
        if self.kind == "hf":
            kw = {"reasoning_effort": effort} if effort else {}
            return self.t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw)
        text = "\n".join(m["content"] if isinstance(m["content"], str) else "" for m in messages)
        return [3 + b % (self.vocab - 3) for b in text.encode()] or [3]

    def decode(self, ids):
        if self.kind == "hf":
            return self.t.decode(ids, skip_special_tokens=False)
        return "".join(chr(32 + i % 95) for i in ids)


def make_handler(eng, tok, name):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                ids = tok.encode_chat(req["messages"], req.get("reasoning_effort"))
                max_new = int(req.get("max_tokens") or 256)
                samp = {"temperature": req.get("temperature", 0.0), "top_p": req.get("top_p"), "seed": req.get("seed")}
                samp = {k: v for k, v in samp.items() if v is not None}
                stream, out, done = bool(req.get("stream")), [], {}
                cid = "chatcmpl-%d" % int(time.time() * 1000)
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                for kind, v in eng.generate(ids, max_new, samp):
                    if kind == "tok":
                        out.append(v)
                        if stream:
                            ch = {"id": cid, "object": "chat.completion.chunk", "model": name,
                                  "choices": [{"index": 0, "delta": {"content": tok.decode([v])}, "finish_reason": None}]}
                            self.wfile.write(("data: %s\n\n" % json.dumps(ch)).encode())
                            self.wfile.flush()
                    else:
                        done = v
                fr = "stop" if done.get("finish") == "stop" else "length"
                if stream:
                    self.wfile.write(("data: %s\n\ndata: [DONE]\n\n" % json.dumps(
                        {"id": cid, "object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]})).encode())
                else:
                    self._json(200, {"id": cid, "object": "chat.completion", "model": name,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(out)}, "finish_reason": fr}],
                                     "usage": {"prompt_tokens": done.get("prompt", len(ids)), "completion_tokens": done.get("generated", len(out)),
                                               "total_tokens": done.get("prompt", len(ids)) + done.get("generated", len(out))}})
            except Exception as e:  # noqa: BLE001 - the client gets the reason
                try:
                    self._json(500, {"error": str(e)})
                except Exception:  # noqa: BLE001
                    pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(__file__), "..", "build-out", "m4_run"))
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    ap.add_argument("--name", default="mistral-small-4-119b")
    ap.add_argument("engine_args", nargs="*", help="extra m4_run args after --, e.g. -- --threads 8 --router sigmoid")
    a = ap.parse_args()
    eng = Engine(a.exe, a.model, a.ctx, a.engine_args)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng, Tok(a.tokenizer, a.model), a.name)).serve_forever()


if __name__ == "__main__":
    main()

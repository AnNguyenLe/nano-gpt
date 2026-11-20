"""serve.py -- a small local chat-style web UI for the NumPy nanoGPT.

    python serve.py                       # loads model.ckpt if present, else untrained random weights
    python serve.py --checkpoint my.ckpt --port 8000

Then open http://127.0.0.1:8000 . Standard library only (http.server); no new dependencies.
The server binds to 127.0.0.1 by default and never touches the network otherwise.

IMPORTANT, and the UI says so: this is a tiny character-level language model, not an assistant.
It CONTINUES the text you give it. It has a context window of cfg.T characters (64 by default) and
a vocabulary of only the characters in its training corpus. Prompt + new characters can never
exceed cfg.T (there is deliberately no sliding window; see README / C8), so the server keeps the
LAST (T - max_new) characters of your prompt and tells the client what it actually used.

Endpoints:
    GET  /               the chat page (web/index.html)
    GET  /api/info       model + vocabulary + limits
    POST /api/generate   JSON body -> streamed NDJSON, one {"token": "x"} line per character,
                         then a final {"done": true, ...} line (or {"error": "..."}).
"""
from __future__ import annotations

import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from nanogpt import checkpoint, sampling
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, ModelConfig, make_rng
from nanogpt.kv_cache import KVCache
from nanogpt.model import GPT
from nanogpt.tokenizer import load_corpus_and_tokenizer

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "web" / "index.html"
MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS_IN = 10_000  # reject absurd inputs before doing any work


class RequestError(ValueError):
    """A client error: becomes HTTP 400 with the message in the body."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without any HTTP)
# ---------------------------------------------------------------------------
def sanitize_prompt(text: str, stoi: dict):
    """Drop characters the tokenizer cannot encode. Returns (clean_text, dropped_chars)."""
    clean, dropped = [], []
    for ch in text:
        (clean if ch in stoi else dropped).append(ch)
    return "".join(clean), sorted(set(dropped))


def default_max_new(cfg: ModelConfig) -> int:
    """40 characters, but never more than the context allows (T - 1) so a short-context model works."""
    return min(40, cfg.T - 1)


def parse_request(body: dict, cfg: ModelConfig, stoi: dict) -> dict:
    """Validate a /api/generate body and return the normalised parameters."""
    if not isinstance(body, dict):
        raise RequestError("body must be a JSON object")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise RequestError("prompt must be a string")
    if len(prompt) > MAX_PROMPT_CHARS_IN:
        raise RequestError(f"prompt too long (limit {MAX_PROMPT_CHARS_IN} characters before truncation)")

    def num(name, default, lo, hi, integer=False):
        v = body.get(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RequestError(f"{name} must be a number")
        if integer and int(v) != v:
            raise RequestError(f"{name} must be an integer")
        if not (lo <= v <= hi):
            raise RequestError(f"{name} must be between {lo} and {hi}")
        return int(v) if integer else float(v)

    max_new = num("max_new", default_max_new(cfg), 1, cfg.T - 1, integer=True)
    temperature = num("temperature", 0.8, 0.05, 5.0)
    greedy = body.get("greedy", False)
    if not isinstance(greedy, bool):
        raise RequestError("greedy must be true or false")
    top_k = body.get("top_k", None)
    if top_k is not None:
        top_k = num("top_k", None, 1, cfg.V, integer=True)
    seed = body.get("seed", None)
    if seed is not None:
        seed = num("seed", None, 0, 2**31 - 1, integer=True)

    clean, dropped = sanitize_prompt(prompt, stoi)
    if not clean:
        raise RequestError(
            "nothing left to continue after removing characters this model does not know"
            + (f" (removed: {''.join(dropped)!r})" if dropped else "")
        )
    keep = cfg.T - max_new  # characters of prompt that fit alongside max_new new ones
    used = clean[-keep:]
    return {
        "prompt": used,
        "truncated": len(clean) - len(used),
        "dropped": dropped,
        "max_new": max_new,
        "temperature": temperature,
        "top_k": top_k,
        "greedy": greedy,
        "seed": seed,
    }


def stream_completion(model, params, tok, prompt, max_new, rng, *, temperature=0.8, top_k=None, greedy=False):
    """Yield the generated characters one at a time. Same cached decoding path as generate.py;
    for the same rng it produces exactly the same ids (tested)."""
    cfg = model.cfg
    ids = tok.encode(prompt)
    if ids.size + max_new > cfg.T:
        raise RequestError("prompt + max_new exceeds the context length")
    kv = KVCache(cfg, 1)
    logits = model.prefill(ids[None, :].astype(INT_DTYPE), params, kv)[:, -1, :]
    for i in range(max_new):
        if greedy:
            nxt = sampling.greedy(logits)
        else:
            nxt = sampling.sample(logits, rng, temperature=temperature, top_k=top_k)
        yield tok.decode(nxt)
        if i < max_new - 1:
            logits = model.forward_step(nxt[:, None].astype(INT_DTYPE), params, kv)[:, -1, :]


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
class ChatState:
    """Everything the handler needs; built once, shared read-only (plus one compute lock)."""

    def __init__(self, model, params, tok, *, source: str, trained: bool):
        self.model = model
        self.params = params
        self.tok = tok
        self.source = source
        self.trained = trained
        self.lock = threading.Lock()  # one generation at a time: NumPy is CPU-bound anyway

    def info(self) -> dict:
        c = self.model.cfg
        return {
            "source": self.source,
            "trained": self.trained,
            "context_length": c.T,
            "vocab_size": c.V,
            "vocab": "".join(self.tok.itos),
            "config": {"L": c.L, "d": c.d, "H": c.H, "T": c.T, "V": c.V},
            "default_max_new": default_max_new(c),
        }


def make_handler(state: ChatState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nanogpt-numpy"

        def log_message(self, fmt, *args):  # quiet by default
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    self._json(500, {"error": "web/index.html is missing"})
            elif path == "/api/info":
                self._json(200, state.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/generate":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._json(400, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                req = parse_request(body, state.model.cfg, state.tok.stoi)
            except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as e:
                return self._json(400, {"error": str(e)})

            seed = req["seed"] if req["seed"] is not None else secrets.randbelow(2**31)
            # NDJSON stream; the connection closes at the end (HTTP/1.0 semantics).
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with state.lock:
                    n = 0
                    for ch in stream_completion(
                        state.model, state.params, state.tok, req["prompt"], req["max_new"],
                        make_rng(seed), temperature=req["temperature"], top_k=req["top_k"],
                        greedy=req["greedy"],
                    ):
                        self.wfile.write((json.dumps({"token": ch}) + "\n").encode("utf-8"))
                        self.wfile.flush()
                        n += 1
                done = {"done": True, "tokens": n, "seed": seed, "used_prompt": req["prompt"],
                        "truncated": req["truncated"], "dropped": req["dropped"]}
                self.wfile.write((json.dumps(done) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away mid-stream
            except Exception as e:  # report instead of a silent half-stream
                try:
                    self.wfile.write((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode("utf-8"))
                except OSError:
                    pass

    return Handler


def build_state(checkpoint_path: str | None, *, random_init_seed: int = DEFAULT_SEED) -> ChatState:
    _, tok = load_corpus_and_tokenizer()
    if checkpoint_path and Path(checkpoint_path).exists():
        cfg, params, _ = checkpoint.load(checkpoint_path)
        if cfg.V != tok.vocab_size:
            raise SystemExit(f"checkpoint vocab {cfg.V} does not match the corpus tokenizer {tok.vocab_size}")
        return ChatState(GPT(cfg), params, tok, source=str(checkpoint_path), trained=True)
    cfg = ModelConfig(V=tok.vocab_size)
    model = GPT(cfg)
    return ChatState(model, model.init_params(make_rng(random_init_seed)), tok,
                     source="random initialisation (untrained)", trained=False)


def make_server(state: ChatState, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(state))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local chat-style web UI for the NumPy nanoGPT.")
    ap.add_argument("--checkpoint", default="model.ckpt", help="checkpoint to load (default: model.ckpt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    state = build_state(args.checkpoint)
    server = make_server(state, args.host, args.port)
    c = state.model.cfg
    print(f"model: {state.source} | L={c.L} d={c.d} H={c.H} T={c.T} V={c.V}")
    if not state.trained:
        print(f"NOTE: {args.checkpoint!r} not found, serving UNTRAINED weights (output will be noise).")
        print("      Train one first:  python -u train.py --save model.ckpt")
    print(f"open http://{args.host}:{server.server_address[1]}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

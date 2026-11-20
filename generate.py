"""generate.py -- text generation with the KV-cache and the sampling module.

SPRINT_5_PLAN.md TL-39 and section 3.5. Same shape as ``train.py``: a pure
``generate`` function plus an argparse ``main``. Lives at the repo root; the root
``conftest.py`` puts the repo root on ``sys.path`` for the tests.

Generation uses the cached path only (``GPT.prefill`` / ``GPT.forward_step``) and is
deterministic for a fixed seed. ``len(prompt) + tokens`` must not exceed the context
length ``cfg.T``: there is deliberately no sliding window (it would shift absolute
positions and invalidate the C8 equivalence). The CLI prints text for a human to
read; nothing in the test suite treats the content of generated text as evidence.
"""
from __future__ import annotations

import argparse

import numpy as np

from nanogpt import checkpoint, sampling
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, ModelConfig, make_rng
from nanogpt.kv_cache import KVCache
from nanogpt.model import GPT
from nanogpt.tokenizer import load_corpus_and_tokenizer

DEFAULT_TOKENS = 200


def generate(
    model: GPT,
    params: dict,
    kv: KVCache,
    prompt_ids,
    n_tokens: int,
    rng,
    *,
    temperature: float = 1.0,
    top_k=None,
    greedy_decode: bool = False,
) -> np.ndarray:
    """Return ``n_tokens`` new ids (shape ``(n_tokens,)``, INT_DTYPE) continuing
    ``prompt_ids``. ``kv`` must be an empty batch-1 cache for ``model.cfg``."""
    prompt = np.asarray(prompt_ids)
    if prompt.ndim != 1 or prompt.size < 1:
        raise ValueError(f"generate: prompt_ids must be a non-empty 1-D array, got shape {prompt.shape}")
    if not np.issubdtype(prompt.dtype, np.integer):
        raise TypeError(f"generate: prompt_ids must be integer, got {prompt.dtype!r}")
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, (int, np.integer)) or n_tokens < 1:
        raise ValueError(f"generate: n_tokens must be a positive int, got {n_tokens!r}")
    T = model.cfg.T
    if prompt.size + n_tokens > T:
        raise ValueError(
            f"generate: len(prompt)={prompt.size} + n_tokens={n_tokens} exceeds context length "
            f"T={T} (no sliding window)"
        )
    if kv.B != 1 or kv.length != 0:
        raise ValueError("generate: kv must be an empty batch-1 KVCache")
    if prompt.min() < 0 or prompt.max() >= model.cfg.V:
        raise ValueError(f"generate: prompt ids must lie in [0, {model.cfg.V})")

    logits = model.prefill(prompt[None, :].astype(INT_DTYPE), params, kv)[:, -1, :]  # (1, V)
    out = np.empty(n_tokens, dtype=INT_DTYPE)
    for i in range(n_tokens):
        if greedy_decode:
            nxt = sampling.greedy(logits)
        else:
            nxt = sampling.sample(logits, rng, temperature=temperature, top_k=top_k)
        out[i] = nxt[0]
        if i < n_tokens - 1:
            logits = model.forward_step(nxt[:, None].astype(INT_DTYPE), params, kv)[:, -1, :]
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate text with the NumPy nanoGPT (KV-cache).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", metavar="PATH", help="load model weights from a checkpoint")
    src.add_argument("--random-init", action="store_true", help="use freshly initialised (untrained) weights")
    ap.add_argument("--prompt", default="", help="prompt text (default: the first corpus character)")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true", help="argmax decoding (ignores temperature/top-k)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for name, default in (("L", 2), ("d", 64), ("T", 64), ("H", 4)):
        ap.add_argument(f"--{name}", type=int, default=default, help="--random-init only")
    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    info, tok = load_corpus_and_tokenizer()
    V = tok.vocab_size

    if args.checkpoint:
        try:
            cfg, params, _ = checkpoint.load(args.checkpoint)
        except (OSError, ValueError) as e:  # missing file, not an .npz, or a rejected checkpoint
            ap.error(f"cannot load checkpoint {args.checkpoint!r}: {e}")
        model = GPT(cfg)
    else:
        try:
            cfg = ModelConfig(V=V, T=args.T, d=args.d, L=args.L, H=args.H)
        except ValueError as e:
            ap.error(str(e))
        model = GPT(cfg)
        params = model.init_params(make_rng(args.seed))
    if cfg.V != V:
        ap.error(f"model vocab size {cfg.V} does not match the corpus tokenizer's {V}")

    prompt_text = args.prompt if args.prompt else info.text[0]
    try:
        prompt_ids = tok.encode(prompt_text)
    except ValueError as e:
        ap.error(f"prompt contains a character outside the vocabulary: {e}")
    if prompt_ids.size + args.tokens > cfg.T:
        ap.error(
            f"len(prompt)={prompt_ids.size} + --tokens={args.tokens} exceeds context length "
            f"T={cfg.T} (there is no sliding window)"
        )
    try:
        ids = generate(
            model, params, KVCache(cfg, 1), prompt_ids, args.tokens, make_rng(args.seed + 2),
            temperature=args.temperature, top_k=args.top_k, greedy_decode=args.greedy,
        )
    except ValueError as e:
        ap.error(str(e))
    print(prompt_text + tok.decode(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

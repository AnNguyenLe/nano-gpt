"""nanogpt.model -- full GPT assembly with a tied head (Sprint 3, Engineer B).

See SPRINT_3_PLAN.md section 3.4 (TL-16 .. TL-21) for the derivation this
module implements exactly.

Composition (one rng, in this order): wte = ops.Embedding("wte", V, d),
wpe = ops.Embedding("wpe", T, d), L transformer Blocks "h.0".."h.{L-1}",
ln_f = LayerNorm("ln_f", d). param_keys(): wte.W, wpe.W, h.0.* (16), ...,
h.{L-1}.* (16), ln_f.gamma, ln_f.beta -- 4 + 16L tensors total.

TL-17 (tied head, frozen): params holds exactly ONE array, "wte.W" (V, d),
C-contiguous, owning its data. logits = y @ params["wte.W"].T -- the `.T`
is a transient matmul operand, never stored in params, cache (as a
parameter) or handed to gradcheck. There is no "lm_head.W" key and no head
bias in params. `named_parameters(params)` returns a per-role view that
additionally maps "lm_head.W" to the SAME object as params["wte.W"]
(identity, not a copy or a transpose).

TL-16 (three layers of entry point, each with its own backward -- the
backward half: embed_backward, backward_from_embedding, backward with the
two C7 instrumentation flags, backward_loss -- lands in milestone B3, after
this forward-only structure is frozen and committed):
    embed(idx, params) -> (h0, cache)
    forward_from_embedding(h0, params) -> (logits, cache)
    forward(idx, params) -> (logits, cache)        # = embed then forward_from_embedding
    forward_loss(idx, targets, params) -> (loss, cache)   # via nanogpt/loss.py

Forward:
    h0     = wte.W[idx] + wpe.W[arange(T)]           # ops.Embedding for both
    h_l    = Block_l(h_{l-1}),  l = 0..L-1
    y      = LN_f(h_L)
    logits = y @ params["wte.W"].T                   # tied head, NO bias

Validation: `idx` must be a 2-D integer array with 1 <= T <= cfg.T (else
ValueError); out-of-range ids raise via `ops.Embedding`. No KV-cache
argument, no sampling, no training code (out of scope, section 13).

Module contract (unchanged): forward(x, params) -> (out, cache);
backward(d_out, cache) -> (d_x, d_params).

Milestone B2 (SPRINT_3_PLAN.md section 6.1): constructor, param_keys,
init_params, named_parameters, embed, forward_from_embedding, forward,
forward_loss -- the param STRUCTURE IS FROZEN as of this commit (subject
line ends "(structure final)"); any later change to keys/shapes needs TL
approval and invalidates C1 (A's live param_count check).

Owner: Engineer B. Files: nanogpt/model.py, tests/test_model.py.
"""
from __future__ import annotations

import numpy as np

from nanogpt import loss as loss_mod
from nanogpt import ops
from nanogpt.attention_mask import causal_mask
from nanogpt.block import Block
from nanogpt.config import INT_DTYPE, ModelConfig
from nanogpt.layernorm import LayerNorm


def _resolve_mask_fns(mask_fn, L: int) -> list:
    """mask_fn is one callable (used for all L blocks) or a list/tuple of
    exactly L callables (one per block); anything else -> ValueError."""
    if isinstance(mask_fn, (list, tuple)):
        if len(mask_fn) != L:
            raise ValueError(
                f"GPT: mask_fn sequence must have exactly L={L} entries, got {len(mask_fn)}"
            )
        if not all(callable(f) for f in mask_fn):
            raise ValueError("GPT: every entry of a mask_fn sequence must be callable")
        return list(mask_fn)
    if callable(mask_fn):
        return [mask_fn] * L
    raise ValueError(
        f"GPT: mask_fn must be a callable or a list/tuple of exactly L={L} callables, "
        f"got {mask_fn!r}"
    )


class GPT:
    """Full transformer: token + positional embeddings, L blocks, final LN,
    weight-tied language-model head.

    Module contract: forward(x, params) -> (out, cache);
    backward(d_out, cache) -> (d_x, d_params).
    """

    def __init__(self, cfg: ModelConfig, mask_fn=causal_mask):
        self.cfg = cfg
        mask_fns = _resolve_mask_fns(mask_fn, cfg.L)

        self.wte = ops.Embedding("wte", cfg.V, cfg.d)
        self.wpe = ops.Embedding("wpe", cfg.T, cfg.d)
        self.blocks = [Block(f"h.{l}", cfg.d, cfg.H, mask_fns[l]) for l in range(cfg.L)]
        self.ln_f = LayerNorm("ln_f", cfg.d)

    # -- structure ----------------------------------------------------------
    def param_keys(self) -> list:
        keys = [f"{self.wte.prefix}.W", f"{self.wpe.prefix}.W"]
        for b in self.blocks:
            keys.extend(b.param_keys())
        keys.extend(self.ln_f.param_keys())
        return keys

    def init_params(self, rng) -> dict:
        params = {}
        params.update(self.wte.init_params(rng))
        params.update(self.wpe.init_params(rng))
        for b in self.blocks:
            params.update(b.init_params(rng))
        params.update(self.ln_f.init_params(rng))
        return params

    def named_parameters(self, params: dict) -> dict:
        """Every role, INCLUDING "lm_head.W" mapped to the SAME object as
        params["wte.W"] (TL-17: identity, not a copy or a transpose)."""
        named = dict(params)
        named["lm_head.W"] = params[f"{self.wte.prefix}.W"]
        return named

    # -- entry point 1: embed -------------------------------------------------
    def embed(self, idx, params: dict):
        idx = np.asarray(idx)
        if idx.ndim != 2:
            raise ValueError(f"GPT.embed: idx must be 2-D (B, T), got ndim={idx.ndim}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"GPT.embed: idx must have an integer dtype, got {idx.dtype!r}")
        B, T = idx.shape
        if not (1 <= T <= self.cfg.T):
            raise ValueError(
                f"GPT.embed: T={T} must satisfy 1 <= T <= cfg.T={self.cfg.T}"
            )

        tok_emb, c_wte = self.wte.forward(idx, params)  # (B, T, d)
        pos_idx = np.arange(T, dtype=INT_DTYPE)
        pos_emb, c_wpe = self.wpe.forward(pos_idx, params)  # (T, d)
        h0 = tok_emb + pos_emb  # broadcasts (B,T,d) + (T,d) -> (B,T,d)

        cache = {"c_wte": c_wte, "c_wpe": c_wpe, "B": B, "T": T}
        return h0, cache

    # -- entry point 2: forward_from_embedding -------------------------------
    def forward_from_embedding(self, h0, params: dict):
        h = h0
        block_caches = []
        for b in self.blocks:
            h, c_b = b.forward(h, params)
            block_caches.append(c_b)

        y, c_lnf = self.ln_f.forward(h, params)
        wte_W = params[f"{self.wte.prefix}.W"]
        logits = y @ wte_W.T  # transient operand; wte_W itself (not .T) is cached

        cache = {"block_caches": block_caches, "c_lnf": c_lnf, "y": y, "wte_W": wte_W}
        return logits, cache

    # -- entry point 3: forward (ids in, logits out) -------------------------
    def forward(self, idx, params: dict):
        h0, c_embed = self.embed(idx, params)
        logits, c_fe = self.forward_from_embedding(h0, params)
        cache = {"c_embed": c_embed, "c_fe": c_fe}
        return logits, cache

    # -- loss entry point -----------------------------------------------------
    def forward_loss(self, idx, targets, params: dict):
        logits, c_fwd = self.forward(idx, params)
        loss, c_loss = loss_mod.cross_entropy_forward(logits, targets)
        cache = {"c_fwd": c_fwd, "c_loss": c_loss}
        return loss, cache

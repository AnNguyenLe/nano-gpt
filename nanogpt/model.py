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

TL-16 (three layers of entry point, each with its own backward):
    embed(idx, params) -> (h0, cache)
    embed_backward(d_h0, cache) -> {"wte.W": input-side, "wpe.W": ...}
    forward_from_embedding(h0, params) -> (logits, cache)
    backward_from_embedding(d_logits, cache) -> (d_h0, d_params)
        # every key except "wpe.W"; "wte.W" holds the OUTPUT-side
        # contribution only (g_out below)
    forward(idx, params) -> (logits, cache)        # = embed then forward_from_embedding
    backward(d_logits, cache, *, include_input_side=True, include_output_side=True)
        -> (None, d_params)                        # all 4 + 16L keys, always present
    forward_loss(idx, targets, params) -> (loss, cache)   # via nanogpt/loss.py
    backward_loss(d_loss, cache) -> (None, d_params)

Forward:
    h0     = wte.W[idx] + wpe.W[arange(T)]           # ops.Embedding for both
    h_l    = Block_l(h_{l-1}),  l = 0..L-1
    y      = LN_f(h_L)
    logits = y @ params["wte.W"].T                   # tied head, NO bias

Backward (TL-20 / C7):
    g_out  = d_logits.reshape(-1,V).T @ y.reshape(-1,d)   # (V,d) output-side
    dy     = d_logits @ params["wte.W"]
    dh, g  = ln_f.backward(dy); blocks backward in reverse order
    g_pos  = wpe.backward(dh.sum(axis=0))             # np.add.at inside Embedding
    g_in   = wte.backward(dh)                         # input-side (np.add.at)
    grads  : ops.accumulate_grads(grads, {"wte.W": g_out}) then
             ops.accumulate_grads(grads, g_in)         # same array, += (C7)

Every merge in this module goes through `ops.accumulate_grads`, called as a
module attribute (`ops.accumulate_grads(...)`, never a bare import of the
function), so C7 can monkeypatch it. There is no statement of the form
`<dict>["wte.W"] = ...` anywhere in this file; the "wte.W" key is always
built by passing a literal `{"wte.W": ...}` dict into `ops.accumulate_grads`.
When a flag excludes a side, that side is simply not accumulated; "wte.W"
is always present in the result (zeros if both sides are excluded).

Validation: `idx` must be a 2-D integer array with 1 <= T <= cfg.T (else
ValueError); out-of-range ids raise via `ops.Embedding`. No KV-cache
argument, no sampling, no training code (out of scope, section 13).

Module contract (unchanged): forward(x, params) -> (out, cache);
backward(d_out, cache) -> (d_x, d_params).

Milestone B2 (SPRINT_3_PLAN.md section 6.1, committed first): constructor,
param_keys, init_params, named_parameters, embed, forward_from_embedding,
forward, forward_loss -- the param structure was frozen as of that commit
("structure final"); any later key/shape change needs TL approval and
invalidates C1 (A's live param_count check). Milestone B3 (this commit):
embed_backward, backward_from_embedding, backward (with the two C7 flags),
backward_loss, and the dev gradcheck in tests/test_model.py.

Owner: Engineer B. Files: nanogpt/model.py, tests/test_model.py.
"""
from __future__ import annotations

import numpy as np

from nanogpt import loss as loss_mod
from nanogpt import ops
from nanogpt.attention_mask import causal_mask
from nanogpt.block import Block
from nanogpt.config import INT_DTYPE, ModelConfig, check_float64
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

    def embed_backward(self, d_h0, cache: dict) -> dict:
        check_float64(d_h0, names=["d_h0"])
        _, g_wte = self.wte.backward(d_h0, cache["c_wte"])
        d_pos = d_h0.sum(axis=0)  # (T, d), broadcast's adjoint
        _, g_wpe = self.wpe.backward(d_pos, cache["c_wpe"])

        grads: dict = {}
        ops.accumulate_grads(grads, g_wte)
        ops.accumulate_grads(grads, g_wpe)
        return grads

    # -- entry point 2: forward_from_embedding -------------------------------
    def forward_from_embedding(self, h0, params: dict):
        h = h0
        block_caches = []
        for b in self.blocks:
            h, c_b = b.forward(h, params)
            block_caches.append(c_b)

        y, c_lnf = self.ln_f.forward(h, params)
        wte_W = params[f"{self.wte.prefix}.W"]
        logits = ops.matmul(y, wte_W.T)  # transient operand; wte_W itself (not .T) is cached

        cache = {"block_caches": block_caches, "c_lnf": c_lnf, "y": y, "wte_W": wte_W}
        return logits, cache

    def _backward_through_lnf_and_blocks(self, d_logits, c_fe: dict):
        """Shared by forward_from_embedding's and forward's backward: the
        output-side contribution g_out, the gradient dh flowing back into
        h0, and the merged grads for ln_f + every block (never "wte.W" or
        "wpe.W", those are layered on by the two callers)."""
        y = c_fe["y"]
        wte_W = c_fe["wte_W"]
        V, d = wte_W.shape
        check_float64(d_logits, names=["d_logits"])

        g_out = ops.matmul(d_logits.reshape(-1, V).T, y.reshape(-1, d))  # (V, d)
        dy = ops.matmul(d_logits, wte_W)  # (B, T, d)

        grads: dict = {}
        dh, g_lnf = self.ln_f.backward(dy, c_fe["c_lnf"])
        ops.accumulate_grads(grads, g_lnf)
        for b, c_b in zip(reversed(self.blocks), reversed(c_fe["block_caches"])):
            dh, g_b = b.backward(dh, c_b)
            ops.accumulate_grads(grads, g_b)
        return dh, g_out, grads

    def backward_from_embedding(self, d_logits, cache: dict):
        dh, g_out, grads = self._backward_through_lnf_and_blocks(d_logits, cache)
        ops.accumulate_grads(grads, {f"{self.wte.prefix}.W": g_out})
        return dh, grads

    # -- entry point 3: forward / backward (ids in, logits out) --------------
    def forward(self, idx, params: dict):
        h0, c_embed = self.embed(idx, params)
        logits, c_fe = self.forward_from_embedding(h0, params)
        cache = {"c_embed": c_embed, "c_fe": c_fe}
        return logits, cache

    def backward(
        self,
        d_logits,
        cache: dict,
        *,
        include_input_side: bool = True,
        include_output_side: bool = True,
    ):
        c_embed = cache["c_embed"]
        c_fe = cache["c_fe"]
        wte_key = f"{self.wte.prefix}.W"

        dh, g_out, grads = self._backward_through_lnf_and_blocks(d_logits, c_fe)

        d_pos = dh.sum(axis=0)
        _, g_wpe = self.wpe.backward(d_pos, c_embed["c_wpe"])
        ops.accumulate_grads(grads, g_wpe)
        _, g_wte_in = self.wte.backward(dh, c_embed["c_wte"])

        if include_output_side:
            ops.accumulate_grads(grads, {wte_key: g_out})
        if include_input_side:
            ops.accumulate_grads(grads, g_wte_in)
        if wte_key not in grads:
            ops.accumulate_grads(grads, {wte_key: np.zeros_like(c_fe["wte_W"])})

        return None, grads

    # -- loss entry point -----------------------------------------------------
    def forward_loss(self, idx, targets, params: dict):
        logits, c_fwd = self.forward(idx, params)
        loss, c_loss = loss_mod.cross_entropy_forward(logits, targets)
        cache = {"c_fwd": c_fwd, "c_loss": c_loss}
        return loss, cache

    def backward_loss(self, d_loss, cache: dict):
        d_logits, _ = loss_mod.cross_entropy_backward(d_loss, cache["c_loss"])
        _, grads = self.backward(d_logits, cache["c_fwd"])
        return None, grads

    # -- Sprint 5 [A2] (TL-32/36): inference-only cached decoding -------------
    # ADDITIVE: nothing above changed. No backward, never gradchecked.
    def forward_step(self, idx_step, params: dict, kv):
        """Decode one position. idx_step: (B,1) ids; returns logits (B,1,V).

        The new token is embedded at absolute position ``pos = kv.length``,
        read BEFORE the layers append (TL-36); each block appends its K/V at
        that same index; ``kv.advance()`` runs once after the last block.
        """
        idx_step = np.asarray(idx_step)
        if idx_step.ndim != 2 or idx_step.shape[1] != 1:
            raise ValueError(f"GPT.forward_step: idx_step must be (B, 1), got {idx_step.shape}")
        if not np.issubdtype(idx_step.dtype, np.integer):
            raise TypeError(f"GPT.forward_step: idx_step must be integer, got {idx_step.dtype!r}")
        pos = kv.length
        if pos >= self.cfg.T:
            raise ValueError(
                f"GPT.forward_step: position {pos} >= context length T={self.cfg.T} "
                "(no sliding window)"
            )

        tok_emb, _ = self.wte.forward(idx_step, params)  # (B,1,d)
        pos_emb, _ = self.wpe.forward(np.array([pos], dtype=INT_DTYPE), params)  # (1,d)
        h = tok_emb + pos_emb
        for l, b in enumerate(self.blocks):
            h = b.forward_step(h, params, kv, l)
        kv.advance()

        y, _ = self.ln_f.forward(h, params)
        wte_W = params[f"{self.wte.prefix}.W"]
        return ops.matmul(y, wte_W.T)

    def prefill(self, idx, params: dict, kv):
        """Run positions 0..T0-1 through ``forward_step`` one at a time and
        return the stacked logits (B, T0, V). One code path on purpose (TL-36);
        a batched prefill is a performance optimisation and out of scope."""
        idx = np.asarray(idx)
        if idx.ndim != 2:
            raise ValueError(f"GPT.prefill: idx must be 2-D (B, T0), got ndim={idx.ndim}")
        outs = [self.forward_step(idx[:, t : t + 1], params, kv) for t in range(idx.shape[1])]
        return np.concatenate(outs, axis=1)
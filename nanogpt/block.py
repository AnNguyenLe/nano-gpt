"""nanogpt.block -- pre-LN transformer block (Sprint 3, Engineer B).

See SPRINT_3_PLAN.md section 3.3 for the derivation this module implements
exactly.

Composition (one rng, in this order): LayerNorm(f"{prefix}.ln1"),
CausalSelfAttention(f"{prefix}.attn"), LayerNorm(f"{prefix}.ln2"),
MLP(f"{prefix}.mlp"). 16 parameter keys total: ln1 (gamma, beta), attn (8,
TL-10 order: q.W,q.b,k.W,k.b,v.W,v.b,o.W,o.b), ln2 (gamma, beta), mlp (4:
fc.W,fc.b,proj.W,proj.b).

Pre-LN residual block:
    x1  = x  + attn(LN1(x))
    out = x1 + mlp(LN2(x1))

Hand-derived backward (restated so nobody re-derives it differently):
    dz2, g_mlp  = mlp.backward(d_out)        ;  du2, g_ln2  = ln2.backward(dz2)
    dx1         = d_out + du2                # residual: identity path + branch path, SUM
    dz1, g_attn = attn.backward(dx1)         ;  du1, g_ln1  = ln1.backward(dz1)
    dx          = dx1 + du1                  # residual again, SUM
    d_params    = merge of g_ln1, g_attn, g_ln2, g_mlp via ops.accumulate_grads

`block.attn.mask_fn` is exactly the `mask_fn` passed to the constructor
(identity, not a copy): this is what lets the model thread a per-block mask
list straight through without any extra plumbing, and what C5 (Sprint 3)
substitutes through to prove the mask is really wired in.

This module contains NO LayerNorm/softmax/GELU arithmetic of its own (no
np.mean, np.sqrt, np.exp, np.tanh): it only composes the four modules
above and sums residual branches. All of that math lives in layernorm.py,
attention.py (which reuses ops.softmax_backward) and mlp.py.

Module contract (unchanged): forward(x, params) -> (out, cache);
backward(d_out, cache) -> (d_x, d_params).

Owner: Engineer B. Files: nanogpt/block.py, tests/test_block.py.
"""
from __future__ import annotations

from nanogpt import ops
from nanogpt.layernorm import LayerNorm
from nanogpt.attention import CausalSelfAttention
from nanogpt.attention_mask import causal_mask
from nanogpt.mlp import MLP


class Block:
    """One pre-LN transformer block.

    Module contract: forward(x, params) -> (out, cache);
    backward(d_out, cache) -> (d_x, d_params).
    """

    def __init__(self, prefix: str, d: int, H: int, mask_fn=causal_mask):
        self.prefix = prefix
        self.d = d
        self.H = H
        self.mask_fn = mask_fn

        self.ln1 = LayerNorm(f"{prefix}.ln1", d)
        self.attn = CausalSelfAttention(f"{prefix}.attn", d, H, mask_fn=mask_fn)
        self.ln2 = LayerNorm(f"{prefix}.ln2", d)
        self.mlp = MLP(f"{prefix}.mlp", d)

    @classmethod
    def from_config(cls, cfg, prefix: str = "h.0", mask_fn=causal_mask):
        return cls(prefix, cfg.d, cfg.H, mask_fn=mask_fn)

    def param_keys(self) -> list:
        """ln1 (gamma, beta), attn (8, TL-10 order), ln2 (gamma, beta), mlp (4) = 16 keys."""
        keys = []
        keys.extend(self.ln1.param_keys())
        keys.extend(self.attn.param_keys())
        keys.extend(self.ln2.param_keys())
        keys.extend(self.mlp.param_keys())
        return keys

    def init_params(self, rng) -> dict:
        """ln1, attn, ln2, mlp, in that order, drawn from the SAME rng."""
        params = {}
        params.update(self.ln1.init_params(rng))
        params.update(self.attn.init_params(rng))
        params.update(self.ln2.init_params(rng))
        params.update(self.mlp.init_params(rng))
        return params

    def forward(self, x, params):
        u1, c_ln1 = self.ln1.forward(x, params)
        a, c_attn = self.attn.forward(u1, params)
        x1 = x + a

        u2, c_ln2 = self.ln2.forward(x1, params)
        m, c_mlp = self.mlp.forward(u2, params)
        out = x1 + m

        cache = {
            "c_ln1": c_ln1,
            "c_attn": c_attn,
            "c_ln2": c_ln2,
            "c_mlp": c_mlp,
        }
        return out, cache

    def backward(self, d_out, cache):
        dz2, g_mlp = self.mlp.backward(d_out, cache["c_mlp"])
        du2, g_ln2 = self.ln2.backward(dz2, cache["c_ln2"])
        dx1 = d_out + du2  # residual: identity path (d_out) + branch path (du2), SUM

        dz1, g_attn = self.attn.backward(dx1, cache["c_attn"])
        du1, g_ln1 = self.ln1.backward(dz1, cache["c_ln1"])
        dx = dx1 + du1  # residual again, SUM

        d_params = {}
        ops.accumulate_grads(d_params, g_ln1)
        ops.accumulate_grads(d_params, g_attn)
        ops.accumulate_grads(d_params, g_ln2)
        ops.accumulate_grads(d_params, g_mlp)
        return dx, d_params

    def forward_step(self, x_step, params, kv=None, layer: int = 0):
        """Sprint 5 [A2] (TL-32): inference-only one-token cached forward.
        Same pre-LN residual structure as ``forward``; no cache dict, no backward."""
        u1, _ = self.ln1.forward(x_step, params)
        a = self.attn.forward_step(u1, params, kv, layer)
        x1 = x_step + a
        u2, _ = self.ln2.forward(x1, params)
        m, _ = self.mlp.forward(u2, params)
        return x1 + m
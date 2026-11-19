"""nanogpt.attention -- vectorized causal multi-head self-attention.

See SPRINT_2_PLAN.md section 3.4 (TL-9, TL-10, TL-11, TL-12, TL-15) for the
derivation this module implements exactly.

Head-column convention (TL-11, frozen): head h in {0..H-1} owns feature
columns [h*hd, (h+1)*hd) of q, k, v, and writes the same columns of the
pre-projection concat Y. hd = d // H.

Forward:
    q, cq = self.q.forward(x, params) ; k, ck = ... ; v, cv = ...     # (B,T,d)
    qh, kh, vh = _split_heads(q,H), _split_heads(k,H), _split_heads(v,H)  # (B,H,T,hd)
    S  = (qh @ kh.swapaxes(-1,-2)) * self.scale                       # (B,H,T,T)
    allowed = self.mask_fn(T)                                         # (T,T) bool
    Z  = np.where(allowed, S, -np.inf)
    P, _ = ops.softmax_forward(Z)                                     # masked P == 0.0 exactly
    Y  = _merge_heads(P @ vh)                                         # (B,T,d)
    out, co = self.o.forward(Y, params)

No KV-cache argument, no einsum, no perf work (R21).

Milestone B1 (SPRINT_2_PLAN.md section 6.1): constructor, _split_heads /
_merge_heads, init_params, forward. The hand-derived backward lands in B3,
after B2's oracle-match hard gate is committed (S2-07).

Owner: Engineer B. Files: nanogpt/attention.py, tests/test_attention.py.
"""
from __future__ import annotations

import math

import numpy as np

from nanogpt.config import LINEAR_HAS_BIAS, N_HEAD, ModelConfig, check_float64
from nanogpt import ops
from nanogpt.attention_mask import causal_mask


def _split_heads(a: np.ndarray, H: int) -> np.ndarray:
    """(B,T,d) -> (B,H,T,hd), TL-11: head h owns columns [h*hd,(h+1)*hd)."""
    B, T, d = a.shape
    hd = d // H
    return a.reshape(B, T, H, hd).transpose(0, 2, 1, 3)


def _merge_heads(y: np.ndarray) -> np.ndarray:
    """(B,H,T,hd) -> (B,T,d). Exact inverse of `_split_heads`."""
    B, H, T, hd = y.shape
    return y.transpose(0, 2, 1, 3).reshape(B, T, H * hd)


class CausalSelfAttention:
    """Vectorized causal multi-head self-attention.

    Module contract: forward(x, params) -> (out, cache);
    backward(d_out, cache) -> (d_x, d_params).
    """

    def __init__(self, prefix: str, d: int, H: int = N_HEAD, mask_fn=causal_mask):
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"CausalSelfAttention: d must be a positive int, got {d!r}")
        if isinstance(H, bool) or not isinstance(H, (int, np.integer)) or H <= 0:
            raise ValueError(f"CausalSelfAttention: H must be a positive int, got {H!r}")
        d = int(d)
        H = int(H)
        if d % H != 0:
            raise ValueError(f"CausalSelfAttention: d={d} must be divisible by H={H}")

        self.prefix = prefix
        self.d = d
        self.H = H
        self.head_dim = d // H
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_fn = mask_fn

        self.q = ops.Linear(f"{prefix}.q", d, d, bias=LINEAR_HAS_BIAS)
        self.k = ops.Linear(f"{prefix}.k", d, d, bias=LINEAR_HAS_BIAS)
        self.v = ops.Linear(f"{prefix}.v", d, d, bias=LINEAR_HAS_BIAS)
        self.o = ops.Linear(f"{prefix}.o", d, d, bias=LINEAR_HAS_BIAS)

    @classmethod
    def from_config(cls, cfg: ModelConfig, prefix: str = "attn", mask_fn=causal_mask):
        return cls(prefix, cfg.d, cfg.H, mask_fn=mask_fn)

    def param_keys(self) -> list:
        """[q.W, q.b, k.W, k.b, v.W, v.b, o.W, o.b], each prefixed (TL-10)."""
        keys = []
        for lin in (self.q, self.k, self.v, self.o):
            keys.append(f"{lin.prefix}.W")
            keys.append(f"{lin.prefix}.b")
        return keys

    def init_params(self, rng) -> dict:
        """q, k, v, o drawn from the SAME rng, in that order."""
        params = {}
        for lin in (self.q, self.k, self.v, self.o):
            params.update(lin.init_params(rng))
        return params

    def forward(self, x: np.ndarray, params: dict):
        x = np.asarray(x)
        check_float64(x, names=["x"])
        if x.ndim != 3:
            raise ValueError(f"CausalSelfAttention.forward: x.ndim must be 3, got {x.ndim}")
        if x.shape[-1] != self.d:
            raise ValueError(
                f"CausalSelfAttention.forward: expected last axis {self.d}, got {x.shape[-1]}"
            )
        B, T, d = x.shape

        q, cq = self.q.forward(x, params)
        k, ck = self.k.forward(x, params)
        v, cv = self.v.forward(x, params)

        qh = _split_heads(q, self.H)
        kh = _split_heads(k, self.H)
        vh = _split_heads(v, self.H)

        S = (qh @ kh.swapaxes(-1, -2)) * self.scale
        allowed = self.mask_fn(T)
        Z = np.where(allowed, S, -np.inf)
        P, _ = ops.softmax_forward(Z)

        Y = _merge_heads(P @ vh)
        out, co = self.o.forward(Y, params)

        cache = {
            "cq": cq,
            "ck": ck,
            "cv": cv,
            "co": co,
            "qh": qh,
            "kh": kh,
            "vh": vh,
            "P": P,
            "allowed": allowed,
        }
        return out, cache

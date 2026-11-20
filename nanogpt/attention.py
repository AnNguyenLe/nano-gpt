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

Backward (hand-derived; restated so no one re-derives it differently):
    dY, g_o   = self.o.backward(d_out, co)
    dYh       = _split_heads(dY, H)
    dP        = dYh @ vh.swapaxes(-1,-2)
    dvh       = P.swapaxes(-1,-2) @ dYh
    dZ, _     = ops.softmax_backward(dP, {"p": P})   # P * (dP - sum_j dP*P), TL-4 reused
    dS        = np.where(allowed, dZ, 0.0)           # TL-15: backward never touches -inf
    dqh       = self.scale * (dS @ kh)               # S = scale * Q K^T  =>  dQ = scale * dS K
    dkh       = self.scale * (dS.swapaxes(-1,-2) @ qh)   #                     dK = scale * dS^T Q
    dx_q, g_q = self.q.backward(_merge_heads(dqh), cq)
    dx_k, g_k = self.k.backward(_merge_heads(dkh), ck)
    dx_v, g_v = self.v.backward(_merge_heads(dvh), cv)
    d_x       = dx_q + dx_k + dx_v                   # x fans out to 3 consumers: SUM, never overwrite
    d_params  = {**g_q, **g_k, **g_v, **g_o}          # exactly the 8 keys

TL-9: attn.k.b has an identically zero true gradient at every T. b_k shifts
every allowed score in a row by a j-independent constant, and softmax is
shift-invariant, so out (and hence every downstream loss) does not depend on
b_k. The analytic backward agrees: db_k = sum_j dk_j, and every row of the
softmax backward sums to 0, so db_k is 0 up to roundoff. This is NOT a bug;
see tests/test_attention.py for the TL-9 (a)-(d) gating protocol.

No KV-cache argument, no einsum, no perf work (R21).

Milestone B3 (SPRINT_2_PLAN.md section 6.1): hand-derived backward + tiny-
config module gradcheck + the TL-9 protocol, committed strictly after B2's
oracle-match hard gate (S2-07).

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

        S = ops.matmul(qh, kh.swapaxes(-1, -2), kind="attention") * self.scale
        allowed = self.mask_fn(T)
        Z = np.where(allowed, S, -np.inf)
        P, _ = ops.softmax_forward(Z)

        Y = _merge_heads(ops.matmul(P, vh, kind="attention"))
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

    def backward(self, d_out: np.ndarray, cache: dict):
        d_out = np.asarray(d_out)
        check_float64(d_out, names=["d_out"])

        cq, ck, cv, co = cache["cq"], cache["ck"], cache["cv"], cache["co"]
        qh, kh, vh = cache["qh"], cache["kh"], cache["vh"]
        P, allowed = cache["P"], cache["allowed"]

        dY, g_o = self.o.backward(d_out, co)
        dYh = _split_heads(dY, self.H)

        dP = ops.matmul(dYh, vh.swapaxes(-1, -2), kind="attention")
        dvh = ops.matmul(P.swapaxes(-1, -2), dYh, kind="attention")

        dZ, _ = ops.softmax_backward(dP, {"p": P})
        dS = np.where(allowed, dZ, 0.0)

        dqh = self.scale * ops.matmul(dS, kh, kind="attention")
        dkh = self.scale * ops.matmul(dS.swapaxes(-1, -2), qh, kind="attention")

        dx_q, g_q = self.q.backward(_merge_heads(dqh), cq)
        dx_k, g_k = self.k.backward(_merge_heads(dkh), ck)
        dx_v, g_v = self.v.backward(_merge_heads(dvh), cv)

        d_x = dx_q + dx_k + dx_v
        d_params = {**g_q, **g_k, **g_v, **g_o}
        return d_x, d_params

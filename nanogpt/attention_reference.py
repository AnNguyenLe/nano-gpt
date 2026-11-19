"""nanogpt.attention_reference -- naive loop-over-heads oracle (Sprint 2, Engineer A).

This module exists ONLY to be an independent correctness oracle for
Engineer B's vectorized `nanogpt.attention.CausalSelfAttention` (R12). It is
never imported by production code outside tests, and it deliberately avoids
every array-reshaping primitive that the vectorized module's head-split /
head-merge bookkeeping relies on (`reshape`, `transpose`, `swapaxes`,
`moveaxis`, `einsum`, `.T`), plus `inf`. This way an interleaved- or
wrong-axis head split in the vectorized code cannot also be present here by
construction: there is no reshape/transpose machinery here to share the bug
with (R12, TL-11, F7).

Forward only: `reference_attention_forward` returns `out` with no cache and
no backward. It is not meant to be differentiated; Sprint 2's gradcheck
lives entirely on the vectorized module (`nanogpt/attention.py`,
Engineer B). See SPRINT_2_PLAN.md section 3.3.
"""

from __future__ import annotations

import math

import numpy as np

from nanogpt import ops
from nanogpt.attention_mask import causal_mask
from nanogpt.config import check_float64


def reference_attention_forward(x, params, H, prefix="attn", mask_fn=causal_mask):
    """Naive, loop-over-heads reference (oracle) for causal multi-head
    self-attention.

    x: (B, T, d) float64.
    params: dict with the 8 TL-10 keys under `prefix`:
        f'{prefix}.q.W' (d, d), f'{prefix}.q.b' (d,),
        f'{prefix}.k.W' (d, d), f'{prefix}.k.b' (d,),
        f'{prefix}.v.W' (d, d), f'{prefix}.v.b' (d,),
        f'{prefix}.o.W' (d, d), f'{prefix}.o.b' (d,).
    H: number of heads; d % H == 0 is required.
    mask_fn: a function T -> (T, T) bool ndarray, allowed[i, j] True iff
        query i may attend to key j. Default `causal_mask`. Substituting
        the one-extra-token adversary mask from `attention_mask.py` here
        (via this argument -- never by monkeypatching) is the ONLY
        sanctioned way to reintroduce the C5 adversary against this oracle
        (TL-12).

    Returns `out`, shape (B, T, d). No cache, no backward: forward only.

    Never mutates `x` or `params`. No `reshape`, `transpose`, `swapaxes`,
    `moveaxis`, `einsum`, `.T`, or `inf` appears anywhere in this function
    (enforced by an AST-scan test in `tests/test_attention_reference.py`):
    softmax is taken over the allowed-keys subset directly, so no -inf
    padding is ever needed here, unlike the vectorized module's masking.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3-D (B, T, d), got ndim={x.ndim}")
    d = x.shape[-1]
    if d % H != 0:
        raise ValueError(f"d={d} must be divisible by H={H}")

    q_W = params[f"{prefix}.q.W"]
    q_b = params[f"{prefix}.q.b"]
    k_W = params[f"{prefix}.k.W"]
    k_b = params[f"{prefix}.k.b"]
    v_W = params[f"{prefix}.v.W"]
    v_b = params[f"{prefix}.v.b"]
    o_W = params[f"{prefix}.o.W"]
    o_b = params[f"{prefix}.o.b"]

    check_float64(
        x, q_W, q_b, k_W, k_b, v_W, v_b, o_W, o_b,
        names=[
            "x",
            f"{prefix}.q.W", f"{prefix}.q.b",
            f"{prefix}.k.W", f"{prefix}.k.b",
            f"{prefix}.v.W", f"{prefix}.v.b",
            f"{prefix}.o.W", f"{prefix}.o.b",
        ],
    )

    B, T, _ = x.shape
    hd = d // H
    scale = 1.0 / math.sqrt(hd)

    # Direct matmul, no head split yet -- (B, T, d).
    q = x @ q_W + q_b
    k = x @ k_W + k_b
    v = x @ v_W + v_b

    allowed = mask_fn(T)

    y = np.zeros((B, T, d), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            cols = slice(h * hd, (h + 1) * hd)  # TL-11, frozen head-column convention
            for i in range(T):
                js = [j for j in range(T) if allowed[i, j]]
                s = np.array(
                    [scale * np.dot(q[b, i, cols], k[b, j, cols]) for j in js],
                    dtype=np.float64,
                )
                p, _ = ops.softmax_forward(s)  # 1-D softmax over the allowed keys only
                for jj, j in enumerate(js):
                    y[b, i, cols] += p[jj] * v[b, j, cols]

    out = y @ o_W + o_b
    return out

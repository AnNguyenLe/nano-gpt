"""Stable cross-entropy loss and its exact gradient.

cross_entropy_forward(logits, targets) -> (loss: float, cache)
cross_entropy_backward(d_loss, cache)   -> (d_logits, {})

logits: (..., V) float64. targets: integers, matching leading shape,
validated 0 <= t < V. N = targets.size (PO ruling R4; B*T for a (B, T) batch).

    lse       = logsumexp(z, axis=-1)
    loss      = mean(lse - take_along_axis(z, t, axis=-1))
    p         = exp(z - lse[..., None])
    d_logits  = d_loss * (p - onehot(t)) / N

The one-hot subtraction is done in place on a *copy* of p (subtract 1 at the
target index), never by materializing a dense one-hot via np.eye.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import DTYPE, INT_DTYPE, check_float64
from nanogpt.ops import logsumexp as _logsumexp


def cross_entropy_forward(logits: np.ndarray, targets: np.ndarray):
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    check_float64(logits, names=["logits"])
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError(f"targets must be an integer dtype, got {targets.dtype}")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape {targets.shape} must match logits leading shape {logits.shape[:-1]}"
        )

    V = logits.shape[-1]
    if targets.size:
        tmin = int(targets.min())
        tmax = int(targets.max())
        if tmin < 0 or tmax >= V:
            raise ValueError(f"targets must satisfy 0 <= t < V={V}; got min={tmin}, max={tmax}")

    N = targets.size
    lse_kd = _logsumexp(logits, axis=-1, keepdims=True)
    t_idx = targets[..., None].astype(INT_DTYPE)
    z_t = np.take_along_axis(logits, t_idx, axis=-1)[..., 0]
    lse = lse_kd[..., 0]

    loss = float(np.sum(lse - z_t) / N)

    cache = {"logits": logits, "lse_kd": lse_kd, "targets": targets, "N": N, "V": V}
    return loss, cache


def cross_entropy_backward(d_loss, cache: dict):
    logits = cache["logits"]
    lse_kd = cache["lse_kd"]
    targets = cache["targets"]
    N = cache["N"]

    p = np.exp(logits - lse_kd)  # softmax probabilities, computed stably from lse
    d_logits = p.copy()

    flat_d_logits = d_logits.reshape(-1, d_logits.shape[-1])
    flat_targets = np.asarray(targets, dtype=INT_DTYPE).reshape(-1)
    flat_d_logits[np.arange(flat_targets.size), flat_targets] -= 1.0  # subtract onehot, no np.eye

    d_logits = (d_logits * (float(d_loss) / N)).astype(DTYPE, copy=False)
    return d_logits, {}

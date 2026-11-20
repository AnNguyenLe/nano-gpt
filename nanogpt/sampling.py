"""nanogpt.sampling -- temperature scaling and top-k sampling over logits.

SPRINT_5_PLAN.md TL-37 and section 3.3. Works on a logits array of shape
``(..., V)``; never touches the model. Every function is pure (never mutates
its input).

Milestone [B1]: ``apply_temperature`` and ``greedy``.
Milestone [B2] (appended later): ``apply_top_k``, ``probabilities``, ``sample``.

``temperature=0`` RAISES instead of being special-cased to argmax: ``logits/0``
overflows, and silently branching to argmax would hide the division. Greedy
decoding is the separate, explicit ``greedy``.
"""
from __future__ import annotations

import math

import numpy as np

from nanogpt.config import INT_DTYPE, check_float64


def _check_logits(logits) -> np.ndarray:
    logits = np.asarray(logits)
    check_float64(logits, names=["logits"])
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError(f"logits must have a non-empty last axis, got shape {logits.shape}")
    # -inf is legitimate (a masked entry); NaN and +inf never are, and would otherwise silently
    # turn into token 0 downstream.
    if np.any(np.isnan(logits)) or np.any(np.isposinf(logits)):
        raise ValueError("logits contain NaN or +infinity")
    return logits


def _check_has_a_finite_entry(logits: np.ndarray) -> None:
    """Every row needs at least one finite logit, or the softmax has nothing to normalise."""
    if not np.all(np.any(np.isfinite(logits), axis=-1)):
        raise ValueError("a row of logits has no finite entry (all -infinity); nothing can be sampled")


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Return ``logits / temperature`` for ``temperature > 0`` (a new array)."""
    logits = _check_logits(logits)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float, np.integer, np.floating))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    with np.errstate(over="ignore"):  # an overflow is detected and reported below, not warned about
        out = logits / float(temperature)
    if np.any(np.isfinite(logits) & ~np.isfinite(out)):
        raise ValueError(f"temperature {temperature!r} is so small that logits / temperature overflows")
    return out


def greedy(logits) -> np.ndarray:
    """argmax over the last axis (ties -> lowest index), as INT_DTYPE ids."""
    logits = _check_logits(logits)
    _check_has_a_finite_entry(logits)
    return np.argmax(logits, axis=-1).astype(INT_DTYPE)


def apply_top_k(logits, k: int) -> np.ndarray:
    """Keep the ``k`` largest entries of each row; set the rest to ``-inf``.

    Requires ``1 <= k <= V``; exact no-op (values equal) at ``k == V``. Ties are
    broken by index (lowest index kept first) via a STABLE sort, so the result is
    deterministic and never depends on ``np.argpartition`` ordering. Returns a
    new array.
    """
    logits = _check_logits(logits)
    V = logits.shape[-1]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not (1 <= k <= V):
        raise ValueError(f"top_k must be an int with 1 <= k <= V={V}, got {k!r}")
    k = int(k)
    order = np.argsort(-logits, axis=-1, kind="stable")  # descending, ties -> lowest index first
    keep = np.zeros(logits.shape, dtype=bool)
    np.put_along_axis(keep, order[..., :k], True, axis=-1)
    return np.where(keep, logits, -np.inf)


def probabilities(logits, *, temperature: float = 1.0, top_k=None) -> np.ndarray:
    """softmax(top_k(logits / temperature)); masked entries are EXACTLY 0 and each
    row sums to 1 (the renormalisation is the softmax itself)."""
    from nanogpt import ops  # local import: keeps module import order trivial

    z = apply_temperature(logits, temperature)
    if top_k is not None:
        z = apply_top_k(z, top_k)
    _check_has_a_finite_entry(z)
    p, _ = ops.softmax_forward(z)
    return p


def sample(logits, rng, *, temperature: float = 1.0, top_k=None) -> np.ndarray:
    """Draw one id per row of ``logits`` (shape ``(..., V)`` -> ``(...,)``, INT_DTYPE).

    p = softmax(top_k(logits / temperature)); inverse-CDF draw with one
    ``rng.random()`` per row. The CDF is normalised by its own last value, so its
    final entry is exactly 1.0 and ``u < 1`` can never land past the last allowed id
    (and never on a masked, zero-probability id).
    """
    p = probabilities(logits, temperature=temperature, top_k=top_k)
    cdf = np.cumsum(p, axis=-1)
    cdf = cdf / cdf[..., -1:]
    u = rng.random(size=p.shape[:-1])
    ids = np.sum(cdf <= u[..., None], axis=-1)
    return ids.astype(INT_DTYPE)

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
    return logits


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
    return logits / float(temperature)


def greedy(logits) -> np.ndarray:
    """argmax over the last axis (ties -> lowest index), as INT_DTYPE ids."""
    logits = _check_logits(logits)
    return np.argmax(logits, axis=-1).astype(INT_DTYPE)

"""nanogpt.attention_mask -- causal mask functions (Sprint 2, Engineer A).

Module contract (SPRINT_2_PLAN.md section 3.2 / TL-12):
    mask_fn(T) -> np.ndarray of dtype bool, shape (T, T), where
    allowed[i, j] is True iff query position i may attend to key position j.

Two masks live here:

- `causal_mask` (the only default used anywhere in production code):
  `allowed[i, j] = (j <= i)`, i.e. query i may see every key at or before
  its own position, and nothing strictly above the diagonal. The diagonal
  is always True, so no row is ever all-False (TL-5 precondition from
  Sprint 1: softmax must always see at least one finite/allowed entry).

- `shifted_causal_mask` (the C5 ADVERSARY, TESTS ONLY):
  `allowed[i, j] = (j <= i + 1)`. Exactly one extra future token (at
  j = i + 1) is visible beyond what the causal mask allows. This exists
  solely so that Sprint 2's C5 future-leak test has a concrete, precise
  bug to catch (substituted via the `mask_fn` argument of
  `reference_attention_forward` / `CausalSelfAttention`, never by
  monkeypatching). It must NEVER be used as a default anywhere in
  `nanogpt/attention.py`, `nanogpt/attention_reference.py`, or any other
  production code path. A regression test
  (`tests/test_attention_reference.py` /
  `tests/test_causal_mask.py`) greps for the name `shifted_causal_mask`
  and asserts it occurs only in this file and in test files.

A DIFFERENT bug, the "diagonal-masked" variant `allowed[i, j] = (j < i)`,
is explicitly NOT implemented here and must never be used as the C5
adversary (TL-12). That variant masks out the diagonal itself, so row 0
has zero allowed keys: every score in that row is -inf, and the ensuing
softmax (max of an all--inf row, `exp(-inf - -inf) = exp(nan) = nan`)
produces NaN. That is a different failure mode (a degenerate, ill-posed
mask) from the one-extra-token leak this sprint's adversary is designed to
probe, and conflating the two would make the C5 test-the-test meaningless.
"""

from __future__ import annotations

import numpy as np


def _validate_T(T) -> None:
    # bool is a subclass of int in Python, so isinstance(True, int) is True.
    # TL-12 / spec: T must be a *positive int*, and True/False are rejected
    # even though int(True) == 1.
    if isinstance(T, bool) or not isinstance(T, (int, np.integer)):
        raise ValueError(f"T must be a positive int, got {T!r} (type {type(T).__name__})")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T!r}")


def causal_mask(T: int) -> np.ndarray:
    """The correct causal mask: allowed[i, j] = (j <= i).

    Query i attends to every key j at or before its own position i, and
    nothing strictly after it (nothing above the main diagonal). The
    diagonal (j == i) is always True, so every row has at least one
    allowed key.

    Returns a fresh (T, T) bool ndarray on every call; no cached global is
    ever returned, so mutating the result of one call can never affect a
    later call.
    """
    _validate_T(T)
    rows = np.arange(T)[:, None]
    cols = np.arange(T)[None, :]
    return cols <= rows


def shifted_causal_mask(T: int) -> np.ndarray:
    """The C5 ADVERSARY (tests only): allowed[i, j] = (j <= i + 1).

    Exactly one extra future token, at j = i + 1, is visible beyond the
    causal mask. Never use this as a default anywhere outside tests: see
    the module docstring.

    Returns a fresh (T, T) bool ndarray on every call.
    """
    _validate_T(T)
    rows = np.arange(T)[:, None]
    cols = np.arange(T)[None, :]
    return cols <= (rows + 1)

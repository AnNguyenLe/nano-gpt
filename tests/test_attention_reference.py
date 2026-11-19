"""tests/test_attention_reference.py -- Engineer A's tests (Sprint 2, Phase 2).

Covers:
- nanogpt/attention_mask.py: causal_mask, shifted_causal_mask (unit tests
  against an explicit double-loop reference, edge cases, freshness).
- nanogpt/attention_reference.py: the naive loop-over-heads oracle's own
  closed-form anchors (it is the ground truth, so its correctness rests on
  these, never on agreement with Engineer B's vectorized module), plus its
  AST source guard.

Milestone A1 (this commit): attention_mask.py + its mask tests below.
Milestones A2/A3 (later commits): the oracle, its closed-form anchors, and
its AST source guard, added incrementally to this same file.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanogpt.attention_mask import causal_mask, shifted_causal_mask


# ===========================================================================
# attention_mask.py: causal_mask / shifted_causal_mask
# ===========================================================================
@pytest.mark.parametrize("T", range(1, 7))
def test_causal_mask_matches_explicit_loop(T):
    expected = np.array([[j <= i for j in range(T)] for i in range(T)], dtype=bool)
    actual = causal_mask(T)
    assert actual.dtype == np.bool_
    assert actual.shape == (T, T)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("T", range(1, 7))
def test_causal_mask_diagonal_and_row_counts(T):
    m = causal_mask(T)
    assert np.all(np.diag(m))
    for i in range(T):
        assert int(m[i].sum()) == i + 1


@pytest.mark.parametrize("T", range(1, 7))
def test_shifted_causal_mask_matches_explicit_loop(T):
    expected = np.array([[j <= i + 1 for j in range(T)] for i in range(T)], dtype=bool)
    actual = shifted_causal_mask(T)
    assert actual.dtype == np.bool_
    assert actual.shape == (T, T)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("T", range(1, 7))
def test_shifted_causal_mask_row_counts(T):
    m = shifted_causal_mask(T)
    for i in range(T):
        assert int(m[i].sum()) == min(i + 2, T)


def test_shifted_minus_causal_is_exactly_the_superdiagonal():
    for T in range(1, 7):
        c = causal_mask(T)
        s = shifted_causal_mask(T)
        diff = s & ~c
        expected = np.zeros((T, T), dtype=bool)
        for i in range(T - 1):
            expected[i, i + 1] = True
        assert np.array_equal(diff, expected)


@pytest.mark.parametrize("T", [0, -1, 2.0, True])
def test_mask_functions_reject_invalid_T(T):
    with pytest.raises(ValueError):
        causal_mask(T)
    with pytest.raises(ValueError):
        shifted_causal_mask(T)


def test_mask_functions_return_a_fresh_array_each_call():
    T = 5
    m1 = causal_mask(T)
    m1[:, :] = False
    m2 = causal_mask(T)
    assert m2.any()
    assert np.array_equal(m2, np.array([[j <= i for j in range(T)] for i in range(T)]))

    s1 = shifted_causal_mask(T)
    s1[:, :] = False
    s2 = shifted_causal_mask(T)
    assert s2.any()
    assert np.array_equal(s2, np.array([[j <= i + 1 for j in range(T)] for i in range(T)]))

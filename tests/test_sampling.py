"""tests/test_sampling.py -- Engineer B, Sprint 5 (SPRINT_5_PLAN.md sections 3.3 / 6.2).

[B1] (this section): temperature + greedy, with closed-form anchors.
[B2] sections (top-k, sample, test-the-tests) are appended later, never rewritten.
Nothing here asserts anything about generated TEXT; only closed forms and invariants.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt import ops, sampling
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, ROUNDOFF_ATOL, make_rng

_SEED = DEFAULT_SEED + 701


def _softmax(z):
    return ops.softmax_forward(z)[0]


def _logits(shape=(3, 7), off=0):
    return make_rng(_SEED + off).standard_normal(shape) * 2.0


# --------------------------------------------------------------------------
# apply_temperature
# --------------------------------------------------------------------------
def test_temperature_one_is_exact_identity():
    l = _logits()
    assert np.array_equal(sampling.apply_temperature(l, 1.0), l)


def test_temperature_divides_logits():
    l = _logits()
    assert np.array_equal(sampling.apply_temperature(l, 4.0), l / 4.0)
    assert np.array_equal(sampling.apply_temperature(l, 0.25), l * 4.0)


def test_temperature_returns_new_array_and_never_mutates_input():
    l = _logits()
    snap = l.copy()
    out = sampling.apply_temperature(l, 2.0)
    assert out is not l and np.array_equal(l, snap)
    same = sampling.apply_temperature(l, 1.0)
    assert same is not l  # even the identity case returns a fresh array


def test_temperature_probability_ratio_closed_form():
    # For two logits a > b: p_a / p_b = exp((a - b) / t).
    a, b = 1.7, -0.6
    for t in (0.5, 1.0, 3.0, 10.0):
        p = _softmax(sampling.apply_temperature(np.array([a, b, 0.2]), t))
        assert abs(np.log(p[0] / p[1]) - (a - b) / t) <= 1e-12


def test_high_temperature_converges_monotonically_to_uniform():
    l = _logits((1, 9))[0]
    prev = 2.0
    for t in (1.0, 10.0, 100.0, 1000.0):
        p = _softmax(sampling.apply_temperature(l, t))
        assert p.max() < prev  # strictly decreasing max prob
        prev = p.max()
    assert abs(prev - 1.0 / 9) < 1e-3


def test_low_temperature_converges_to_argmax():
    l = np.array([0.3, 2.0, 1.9, -1.0])
    p = _softmax(sampling.apply_temperature(l, 0.01))
    assert p[1] > 0.999 and np.argmax(p) == 1


def test_temperature_preserves_argmax_and_ordering():
    l = _logits((4, 11), off=1)
    for t in (0.3, 1.0, 5.0):
        out = sampling.apply_temperature(l, t)
        assert np.array_equal(np.argsort(out, axis=-1), np.argsort(l, axis=-1))


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, -1e-9, float("nan"), float("inf"), True, None, "1"])
def test_temperature_rejects_non_positive_or_non_finite(bad):
    with pytest.raises(ValueError):
        sampling.apply_temperature(_logits(), bad)


def test_temperature_zero_is_not_silently_argmax():
    with pytest.raises(ValueError):
        sampling.apply_temperature(_logits(), 0.0)


def test_temperature_rejects_float32_logits():
    with pytest.raises((TypeError, ValueError)):
        sampling.apply_temperature(_logits().astype(np.float32), 1.0)


def test_temperature_handles_minus_inf_entries_untouched():
    l = np.array([1.0, -np.inf, 2.0])
    out = sampling.apply_temperature(l, 2.0)
    assert out[1] == -np.inf and out[0] == 0.5 and out[2] == 1.0


def test_temperature_works_on_any_leading_shape():
    for shape in ((5,), (2, 5), (2, 3, 5)):
        l = _logits(shape)
        assert sampling.apply_temperature(l, 2.0).shape == shape


# --------------------------------------------------------------------------
# greedy
# --------------------------------------------------------------------------
def test_greedy_picks_argmax_with_int_dtype_and_batch_shape():
    l = np.array([[0.1, 3.0, 2.0], [5.0, -1.0, 4.9]])
    g = sampling.greedy(l)
    assert g.dtype == INT_DTYPE and g.shape == (2,)
    assert np.array_equal(g, [1, 0])


def test_greedy_ties_go_to_lowest_index():
    assert sampling.greedy(np.array([2.0, 2.0, 1.0]))[()] == 0
    assert sampling.greedy(np.array([[1.0, 5.0, 5.0]]))[0] == 1


def test_greedy_is_invariant_to_temperature():
    l = _logits((6, 9), off=2)
    for t in (0.1, 1.0, 7.0):
        assert np.array_equal(sampling.greedy(sampling.apply_temperature(l, t)), sampling.greedy(l))


def test_greedy_does_not_mutate_input_and_rejects_bad_input():
    l = _logits()
    snap = l.copy()
    sampling.greedy(l)
    assert np.array_equal(l, snap)
    with pytest.raises((TypeError, ValueError)):
        sampling.greedy(l.astype(np.float32))
    with pytest.raises(ValueError):
        sampling.greedy(np.zeros((3, 0)))

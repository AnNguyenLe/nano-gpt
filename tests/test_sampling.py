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


# ===========================================================================
# [B2] top-k, probabilities, sample, test-the-tests
# ===========================================================================
def test_top_k_one_leaves_exactly_the_argmax_finite():
    l = _logits((4, 9), off=10)
    out = sampling.apply_top_k(l, 1)
    assert np.all(np.isfinite(out).sum(axis=-1) == 1)
    assert np.array_equal(np.argmax(out, axis=-1), np.argmax(l, axis=-1))
    assert np.array_equal(sampling.greedy(out), sampling.greedy(l))


def test_top_k_equal_to_V_is_a_noop():
    l = _logits((3, 6), off=11)
    out = sampling.apply_top_k(l, 6)
    assert np.array_equal(out, l) and out is not l


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_top_k_leaves_exactly_k_finite_per_row_and_they_are_the_largest(k):
    l = _logits((5, 8), off=12)
    out = sampling.apply_top_k(l, k)
    assert np.all(np.isfinite(out).sum(axis=-1) == k)
    for row_in, row_out in zip(l, out):
        kept = np.sort(row_in[np.isfinite(row_out)])
        assert np.array_equal(kept, np.sort(row_in)[-k:])
        assert np.array_equal(row_out[np.isfinite(row_out)], row_in[np.isfinite(row_out)])  # values untouched


def test_top_k_tie_break_is_by_lowest_index_and_deterministic():
    l = np.array([1.0, 3.0, 3.0, 3.0, 0.0])
    out = sampling.apply_top_k(l, 2)
    assert np.isfinite(out).tolist() == [False, True, True, False, False]
    assert np.array_equal(out, sampling.apply_top_k(l.copy(), 2))


def test_top_k_never_mutates_input():
    l = _logits((2, 7), off=13)
    snap = l.copy()
    sampling.apply_top_k(l, 3)
    assert np.array_equal(l, snap)


@pytest.mark.parametrize("bad", [0, -1, 8, 100, 2.0, True, None, "3"])
def test_top_k_rejects_invalid_k(bad):
    with pytest.raises(ValueError):
        sampling.apply_top_k(_logits((2, 7)), bad)


def test_probabilities_sum_to_one_with_exact_zeros_on_masked_entries():
    l = _logits((4, 10), off=14)
    for k in (1, 3, 10):
        p = sampling.probabilities(l, temperature=0.7, top_k=k)
        assert np.allclose(p.sum(axis=-1), 1.0, atol=ROUNDOFF_ATOL)
        masked = ~np.isfinite(sampling.apply_top_k(l, k))
        assert np.all(p[masked] == 0.0)
        assert np.all((p > 0).sum(axis=-1) == k)


def test_probabilities_match_plain_softmax_without_options():
    l = _logits((3, 6), off=15)
    assert np.array_equal(sampling.probabilities(l), _softmax(l))


def test_sample_returns_int_ids_in_range_with_batch_shape():
    l = _logits((2, 3, 9), off=16)
    ids = sampling.sample(l, make_rng(_SEED + 20))
    assert ids.dtype == INT_DTYPE and ids.shape == (2, 3)
    assert ids.min() >= 0 and ids.max() < 9


def test_sample_is_reproducible_for_the_same_seed_and_differs_otherwise():
    l = _logits((64, 12), off=17)
    a = sampling.sample(l, make_rng(_SEED + 21))
    b = sampling.sample(l, make_rng(_SEED + 21))
    c = sampling.sample(l, make_rng(_SEED + 22))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("t", [0.3, 1.0, 5.0])
def test_sample_with_top_k_one_equals_greedy_at_any_temperature(t):
    l = _logits((50, 11), off=18)
    ids = sampling.sample(l, make_rng(_SEED + 23), temperature=t, top_k=1)
    assert np.array_equal(ids, sampling.greedy(l))


def test_sample_never_returns_a_masked_id():
    l = _logits((200, 12), off=19)
    k = 3
    keep = np.isfinite(sampling.apply_top_k(l, k))
    for s in range(5):
        ids = sampling.sample(l, make_rng(_SEED + 30 + s), top_k=k)
        assert np.all(keep[np.arange(200), ids])


def test_sample_respects_a_near_deterministic_distribution():
    l = np.array([[0.0, 50.0, 0.0]])
    ids = [int(sampling.sample(l, make_rng(_SEED + 40 + s))[0]) for s in range(30)]
    assert set(ids) == {1}


def test_sample_u_zero_and_near_one_edges_stay_in_the_allowed_set():
    class _Rng:
        def __init__(self, u):
            self.u = u

        def random(self, size):
            return np.full(size, self.u)

    l = np.array([[1.0, 0.5, 2.0, -1.0, 0.0]])
    keep = np.isfinite(sampling.apply_top_k(l, 2))[0]  # ids {2, 0}
    for u in (0.0, np.nextafter(1.0, 0.0)):
        i = int(sampling.sample(l, _Rng(u), top_k=2)[0])
        assert keep[i], (u, i)
    assert int(sampling.sample(l, _Rng(0.0), top_k=2)[0]) == 0  # lowest allowed id at u = 0
    assert int(sampling.sample(l, _Rng(np.nextafter(1.0, 0.0)), top_k=2)[0]) == 2  # highest allowed id


def test_sample_statistical_sanity_matches_softmax_within_4_standard_errors():
    # STATISTICAL SANITY ONLY (seeded, so deterministic) -- not a correctness anchor.
    l = np.array([[1.0, 0.0, -1.0]])
    p = _softmax(l)[0]
    n = 20000
    draws = sampling.sample(np.repeat(l, n, axis=0), make_rng(_SEED + 50))
    freq = np.bincount(draws, minlength=3) / n
    se = np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(freq - p) < 4 * se), (freq, p)


def test_sample_statistical_sanity_with_top_k_and_temperature():
    l = np.array([[2.0, 1.0, 0.0, -3.0]])
    p = sampling.probabilities(l, temperature=2.0, top_k=2)[0]
    n = 20000
    draws = sampling.sample(np.repeat(l, n, axis=0), make_rng(_SEED + 51), temperature=2.0, top_k=2)
    freq = np.bincount(draws, minlength=4) / n
    assert freq[2] == 0 and freq[3] == 0
    se = np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(freq[:2] - p[:2]) < 4 * se[:2])


def test_sample_validation_propagates():
    l = _logits((2, 5))
    with pytest.raises(ValueError):
        sampling.sample(l, make_rng(_SEED), temperature=0.0)
    with pytest.raises(ValueError):
        sampling.sample(l, make_rng(_SEED), top_k=6)


# --------------------------------------------------------------------------
# Test-the-tests: four buggy variants, each must FAIL the anchor battery that
# the real implementation passes.
# --------------------------------------------------------------------------
def _real_impl():
    return dict(temperature=sampling.apply_temperature, top_k=sampling.apply_top_k, sample=sampling.sample)


def _bad_keeps_smallest(l, k):  # (1) keeps the k SMALLEST
    order = np.argsort(l, axis=-1, kind="stable")
    keep = np.zeros(l.shape, dtype=bool)
    np.put_along_axis(keep, order[..., :k], True, axis=-1)
    return np.where(keep, l, -np.inf)


def _bad_off_by_one_k(l, k):  # (2) keeps k-1 (strict '>' threshold bug)
    return sampling.apply_top_k(l, max(k - 1, 1)) if k > 1 else sampling.apply_top_k(l, 1)


def _bad_unrenormalised_sample(l, rng, *, temperature=1.0, top_k=None):  # (3)
    # masks AFTER softmax and never renormalises, so the CDF tops out below 1.
    z = sampling.apply_temperature(l, temperature)
    p = _softmax(z)
    if top_k is not None:
        keep = np.isfinite(sampling.apply_top_k(z, top_k))
        p = np.where(keep, p, 0.0)
    cdf = np.cumsum(p, axis=-1)  # NOT divided by cdf[-1]
    u = rng.random(size=p.shape[:-1])
    return np.minimum(np.sum(cdf <= u[..., None], axis=-1), p.shape[-1] - 1).astype(INT_DTYPE)


def _bad_temperature_multiplies(l, t):  # (4) logits * t instead of logits / t
    return np.asarray(l) * float(t)


def _anchor_battery(temperature, top_k, sample):
    """Return the list of anchors an implementation FAILS (empty list = passes all)."""
    failed = []
    l = np.array([1.7, -0.6, 0.2, 0.9, -1.4])

    # temperature ratio anchor: log(p_a/p_b) = (a-b)/t, and t=4 shrinks the logits
    p = _softmax(temperature(l[:2], 4.0))
    if abs(np.log(p[0] / p[1]) - (l[0] - l[1]) / 4.0) > 1e-9:
        failed.append("temperature_ratio")
    # top-k keeps the LARGEST k, exactly k of them
    out = top_k(l, 2)
    if not np.array_equal(np.isfinite(out), [True, False, False, True, False]):
        failed.append("top_k_largest")
    if np.isfinite(top_k(l, 3)).sum() != 3:
        failed.append("top_k_count")
    # sampling must match the renormalised distribution (statistical, 4 SE)
    n = 8000
    draws = sample(np.repeat(l[None, :], n, axis=0), make_rng(_SEED + 60), top_k=2)
    pk = sampling.probabilities(l[None, :], top_k=2)[0]
    freq = np.bincount(draws, minlength=5) / n
    se = np.sqrt(np.maximum(pk * (1 - pk), 1e-12) / n)
    if np.any(np.abs(freq - pk) > 4 * se + 1e-12):
        failed.append("sample_distribution")
    return failed


def test_anchor_battery_passes_for_the_real_implementation():
    r = _real_impl()
    assert _anchor_battery(r["temperature"], r["top_k"], r["sample"]) == []


def test_bug_top_k_keeps_the_smallest_is_caught():
    r = _real_impl()
    failed = _anchor_battery(r["temperature"], _bad_keeps_smallest, r["sample"])
    assert "top_k_largest" in failed


def test_bug_top_k_off_by_one_is_caught():
    r = _real_impl()
    failed = _anchor_battery(r["temperature"], _bad_off_by_one_k, r["sample"])
    assert "top_k_count" in failed


def test_bug_sampling_from_unrenormalised_probabilities_is_caught():
    r = _real_impl()
    failed = _anchor_battery(r["temperature"], r["top_k"], _bad_unrenormalised_sample)
    assert "sample_distribution" in failed


def test_bug_temperature_applied_as_multiply_is_caught():
    r = _real_impl()
    failed = _anchor_battery(_bad_temperature_multiplies, r["top_k"], r["sample"])
    assert "temperature_ratio" in failed

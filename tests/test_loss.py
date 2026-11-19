"""Gradcheck + closed-form tests for nanogpt.loss. Owner: Engineer C.

See SPRINT_1_PLAN.md section 8.3.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt.loss import cross_entropy_forward, cross_entropy_backward
from nanogpt.gradcheck import gradcheck
from nanogpt.config import GRADCHECK_TOL, ROUNDOFF_ATOL, DEFAULT_SEED, INT_DTYPE


def _rng():
    return np.random.default_rng(DEFAULT_SEED)


def test_gradcheck_logits_with_repeated_targets():
    B, T, V = 2, 3, 7
    rng = _rng()
    logits = rng.standard_normal((B, T, V))
    targets = np.array([[0, 0, 3], [6, 0, 3]], dtype=INT_DTYPE)  # repeats forced

    def fwd(x, params):
        loss, cache = cross_entropy_forward(x, targets)
        return np.array(loss), cache

    def bwd(d_out, cache):
        d_logits, d_params = cross_entropy_backward(float(np.asarray(d_out)), cache)
        return d_logits, d_params

    report = gradcheck(fwd, bwd, {}, logits)
    print(report)
    assert report.passed, str(report)
    assert report.tensors["x"].rel_err < GRADCHECK_TOL


def test_all_zero_logits_give_ln_v():
    V = 7
    logits = np.zeros((2, 3, V))
    targets = np.zeros((2, 3), dtype=INT_DTYPE)
    loss, _ = cross_entropy_forward(logits, targets)
    assert abs(loss - math.log(V)) < ROUNDOFF_ATOL * max(1.0, math.log(V))


def test_all_zero_logits_give_ln_v_integration_anchor():
    """Integration anchor (precursor to C2): same identity using V from the
    resolved corpus's tokenizer. Skips gracefully if tokenizer.py (Engineer A,
    milestone A3) hasn't landed yet -- this is a Phase 1 sign-off item per the
    A3 -> C integration-anchor dependency in SPRINT_1_PLAN.md section 10.
    """
    try:
        from nanogpt.tokenizer import load_corpus_and_tokenizer
    except ImportError:
        pytest.skip("nanogpt.tokenizer not available yet (Engineer A, milestone A3)")

    _, tok = load_corpus_and_tokenizer()
    V = tok.vocab_size
    logits = np.zeros((2, 3, V))
    targets = np.zeros((2, 3), dtype=INT_DTYPE)
    loss, _ = cross_entropy_forward(logits, targets)
    assert abs(loss - math.log(V)) < ROUNDOFF_ATOL * max(1.0, math.log(V))


def test_confident_correct_logit_gives_tiny_loss():
    V = 5
    logits = np.zeros((1, V))
    logits[0, 2] = 50.0
    targets = np.array([2], dtype=INT_DTYPE)
    loss, _ = cross_entropy_forward(logits, targets)
    assert loss < 1e-20


def test_backward_rows_sum_to_zero_and_batch_shape_invariance():
    V = 7
    rng = _rng()
    logits_2d = rng.standard_normal((2, 3, V))
    targets_2d = np.array([[0, 2, 2], [5, 0, 6]], dtype=INT_DTYPE)

    loss, cache = cross_entropy_forward(logits_2d, targets_2d)
    d_logits, _ = cross_entropy_backward(1.0, cache)
    N = targets_2d.size
    row_sums = np.sum(d_logits * N / 1.0, axis=-1)
    assert np.max(np.abs(row_sums)) < ROUNDOFF_ATOL

    # R4: a (2,3) batch gives the same per-position gradient as the flattened (6,) batch.
    logits_flat = logits_2d.reshape(-1, V)
    targets_flat = targets_2d.reshape(-1)
    loss_flat, cache_flat = cross_entropy_forward(logits_flat, targets_flat)
    d_logits_flat, _ = cross_entropy_backward(1.0, cache_flat)

    assert abs(loss - loss_flat) < ROUNDOFF_ATOL * max(1.0, abs(loss))
    assert np.max(np.abs(d_logits.reshape(-1, V) - d_logits_flat)) < ROUNDOFF_ATOL


def test_lse_smoke_large_logits_finite():
    rng = _rng()
    logits = 1e3 * rng.standard_normal((4, 7))
    targets = rng.integers(0, 7, size=(4,)).astype(INT_DTYPE)
    loss, cache = cross_entropy_forward(logits, targets)
    assert np.isfinite(loss)
    d_logits, _ = cross_entropy_backward(1.0, cache)
    assert np.all(np.isfinite(d_logits))


def test_out_of_range_targets_raise():
    logits = np.zeros((2, 5))
    targets_high = np.array([0, 5], dtype=INT_DTYPE)  # V=5, 5 is out of range
    with pytest.raises(ValueError):
        cross_entropy_forward(logits, targets_high)

    targets_neg = np.array([0, -1], dtype=INT_DTYPE)
    with pytest.raises(ValueError):
        cross_entropy_forward(logits, targets_neg)


def test_non_float64_logits_raise():
    logits32 = np.zeros((2, 5), dtype=np.float32)
    targets = np.array([0, 1], dtype=INT_DTYPE)
    with pytest.raises(TypeError):
        cross_entropy_forward(logits32, targets)

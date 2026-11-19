"""tests/test_gradcheck_harness.py — Engineer A, milestone A2.

PHASE 0 EXIT CRITERION: the generic central-difference harness
(nanogpt/gradcheck.py) must be proven correct on a toy y = x @ W BEFORE any
real layer (linear/embedding/GELU/softmax/layernorm/loss) is checked with
it. This file is that proof: it shows the harness passes on a correct
backward, and — just as importantly — that it FAILS on each of several
deliberately broken backwards (test-the-test), so a future false "PASS"
from a genuinely broken layer backward can be trusted to be a real bug.
"""

import numpy as np
import pytest

from nanogpt.config import DEFAULT_SEED, make_rng
from nanogpt.gradcheck import gradcheck, numerical_jacobian


# ---------------------------------------------------------------------------
# A minimal, deliberately-not-nanogpt.ops "linear" used only to prove the
# harness. Real linear/embedding/etc. correctness is B's/C's job; this file
# only proves the gradcheck MACHINERY is sound.
# ---------------------------------------------------------------------------
def _linear_forward(x, params):
    W = params["W"]
    out = x @ W
    cache = {"x": x, "W": W}
    return out, cache


def _linear_backward_correct(d_out, cache):
    x, W = cache["x"], cache["W"]
    x2 = x.reshape(-1, x.shape[-1])
    g2 = d_out.reshape(-1, d_out.shape[-1])
    dx = d_out @ W.T
    dW = x2.T @ g2
    return dx, {"W": dW}


def _linear_with_unused_bias_forward(x, params):
    # params has an extra 'b' that the forward pass never reads.
    W = params["W"]
    out = x @ W
    cache = {"x": x, "W": W, "b_shape": params["b"].shape}
    return out, cache


def _linear_with_unused_bias_backward(d_out, cache):
    x, W = cache["x"], cache["W"]
    x2 = x.reshape(-1, x.shape[-1])
    g2 = d_out.reshape(-1, d_out.shape[-1])
    dx = d_out @ W.T
    dW = x2.T @ g2
    db = np.zeros(cache["b_shape"], dtype=np.float64)  # true gradient IS exactly zero
    return dx, {"W": dW, "b": db}


def _make_xW(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 4))
    W = rng.standard_normal((4, 5))
    return x.astype(np.float64), {"W": W.astype(np.float64)}


# ---------------------------------------------------------------------------
# 1. Correct y = x @ W passes with every rel_err < 1e-8
# ---------------------------------------------------------------------------
def test_correct_linear_passes_tight_tolerance():
    x, params = _make_xW(seed=1)
    report = gradcheck(
        _linear_forward, _linear_backward_correct, params, x, tol=1e-6, seed=DEFAULT_SEED
    )
    assert report.passed
    for t in report.tensors.values():
        assert t.rel_err < 1e-8, str(report)


# ---------------------------------------------------------------------------
# 2. The harness FAILS: (i) dW*1.01
# ---------------------------------------------------------------------------
def test_harness_fails_on_scaled_dW():
    x, params = _make_xW(seed=2)

    def backward_scaled(d_out, cache):
        dx, d_params = _linear_backward_correct(d_out, cache)
        d_params = dict(d_params)
        d_params["W"] = d_params["W"] * 1.01
        return dx, d_params

    report = gradcheck(_linear_forward, backward_scaled, params, x, tol=1e-6)
    assert not report.passed
    assert report.tensors["W"].rel_err > 1e-4


# ---------------------------------------------------------------------------
# 2. (ii) transposed dW on square W
# ---------------------------------------------------------------------------
def test_harness_fails_on_transposed_dW_square_W():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4)).astype(np.float64)
    W = rng.standard_normal((4, 4)).astype(np.float64)
    params = {"W": W}

    def backward_transposed(d_out, cache):
        dx, d_params = _linear_backward_correct(d_out, cache)
        d_params = dict(d_params)
        d_params["W"] = d_params["W"].T  # wrong operand order, same shape (square)
        return dx, d_params

    report = gradcheck(_linear_forward, backward_transposed, params, x, tol=1e-6)
    assert not report.passed
    assert report.tensors["W"].rel_err > 1e-3


# ---------------------------------------------------------------------------
# 2. (iii) wrong dx with a correct dW — proves the input IS checked
# ---------------------------------------------------------------------------
def test_harness_fails_on_wrong_dx_with_correct_dW():
    x, params = _make_xW(seed=4)

    def backward_wrong_dx(d_out, cache):
        dx_correct, d_params = _linear_backward_correct(d_out, cache)
        wrong_dx = dx_correct * 0.5  # right shape, wrong values
        return wrong_dx, d_params

    report = gradcheck(_linear_forward, backward_wrong_dx, params, x, tol=1e-6)
    assert not report.passed
    assert report.tensors["x"].rel_err > 1e-3
    # dW itself is still correct and tight:
    assert report.tensors["W"].rel_err < 1e-8


# ---------------------------------------------------------------------------
# 3. Missing key / wrong-shape key in d_params raises ValueError
# ---------------------------------------------------------------------------
def test_missing_key_in_d_params_raises_value_error():
    x, params = _make_xW(seed=5)

    def backward_missing_key(d_out, cache):
        return cache["x"] @ 0, {}  # drops 'W' entirely

    with pytest.raises(ValueError):
        gradcheck(_linear_forward, backward_missing_key, params, x)


def test_wrong_shape_key_in_d_params_raises_value_error():
    x, params = _make_xW(seed=6)

    def backward_wrong_shape(d_out, cache):
        dx, d_params = _linear_backward_correct(d_out, cache)
        d_params = dict(d_params)
        d_params["W"] = d_params["W"][:, :-1]  # wrong shape
        return dx, d_params

    with pytest.raises(ValueError):
        gradcheck(_linear_forward, backward_wrong_shape, params, x)


# ---------------------------------------------------------------------------
# 4. Unused bias with exactly-zero true gradient reports rel_err == 0.0
#    (the REL_ERR_FLOOR works: stand-in for the Phase-2 T=1 case)
# ---------------------------------------------------------------------------
def test_unused_bias_zero_gradient_reports_exact_zero_rel_err():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 4)).astype(np.float64)
    W = rng.standard_normal((4, 5)).astype(np.float64)
    b = rng.standard_normal((5,)).astype(np.float64)  # never read by forward
    params = {"W": W, "b": b}

    report = gradcheck(
        _linear_with_unused_bias_forward, _linear_with_unused_bias_backward, params, x
    )
    assert report.passed
    assert report.tensors["b"].rel_err == 0.0


# ---------------------------------------------------------------------------
# 5. Integer x with d_x=None is accepted (input check skipped)
# ---------------------------------------------------------------------------
def test_integer_x_with_none_dx_is_accepted():
    rng = make_rng(8)
    num, d = 5, 3
    W = rng.standard_normal((num, d)).astype(np.float64)
    idx = np.array([[0, 2, 2], [1, 0, 2]], dtype=np.int64)  # duplicates on purpose
    params = {"W": W}

    def emb_forward(x, params):
        out = params["W"][x]
        cache = {"idx": x, "W_shape": params["W"].shape}
        return out, cache

    def emb_backward(d_out, cache):
        dW = np.zeros(cache["W_shape"], dtype=np.float64)
        np.add.at(dW, cache["idx"], d_out)
        return None, {"W": dW}

    report = gradcheck(emb_forward, emb_backward, params, idx)
    assert report.passed
    assert report.tensors["W"].rel_err < 1e-8
    assert "x" not in report.tensors  # input check was skipped (integer x)


def test_integer_x_wrt_input_true_does_not_require_dx():
    # wrt_input defaults to True, but since x is integer it must still be
    # accepted with d_x=None — integer dtype always skips the input check.
    rng = make_rng(9)
    num, d = 4, 2
    W = rng.standard_normal((num, d)).astype(np.float64)
    idx = np.array([0, 1, 3, 2, 0], dtype=np.int64)
    params = {"W": W}

    def emb_forward(x, params):
        return params["W"][x], {"idx": x, "W_shape": params["W"].shape}

    def emb_backward(d_out, cache):
        dW = np.zeros(cache["W_shape"], dtype=np.float64)
        np.add.at(dW, cache["idx"], d_out)
        return None, {"W": dW}

    report = gradcheck(emb_forward, emb_backward, params, idx, wrt_input=True)
    assert report.passed


# ---------------------------------------------------------------------------
# 6. params are bit-identical after the run
# ---------------------------------------------------------------------------
def test_params_bit_identical_after_run():
    x, params = _make_xW(seed=10)
    W_before = np.array(params["W"], copy=True)
    gradcheck(_linear_forward, _linear_backward_correct, params, x)
    np.testing.assert_array_equal(params["W"], W_before)


def test_params_bit_identical_even_on_failing_check():
    x, params = _make_xW(seed=11)

    def backward_scaled(d_out, cache):
        dx, d_params = _linear_backward_correct(d_out, cache)
        d_params = dict(d_params)
        d_params["W"] = d_params["W"] * 1.01
        return dx, d_params

    W_before = np.array(params["W"], copy=True)
    gradcheck(_linear_forward, backward_scaled, params, x)
    np.testing.assert_array_equal(params["W"], W_before)


# ---------------------------------------------------------------------------
# 7. numerical_jacobian of x -> A x equals A, rel_err < 1e-9
# ---------------------------------------------------------------------------
def test_numerical_jacobian_of_linear_map_equals_matrix():
    rng = np.random.default_rng(12)
    m, n = 5, 4
    A = rng.standard_normal((m, n)).astype(np.float64)
    x0 = rng.standard_normal((n,)).astype(np.float64)

    def f(x):
        return A @ x

    J = numerical_jacobian(f, x0)
    assert J.shape == (m, n)
    from nanogpt.config import rel_err as _rel_err

    assert _rel_err(J, A) < 1e-9


def test_numerical_jacobian_handles_nontrivial_shapes():
    # f: R^(2,3) -> R^(4,) via a fixed linear reshape+matmul, to confirm
    # numerical_jacobian flattens arbitrary input/output shapes correctly.
    rng = np.random.default_rng(13)
    A = rng.standard_normal((4, 6)).astype(np.float64)
    x0 = rng.standard_normal((2, 3)).astype(np.float64)

    def f(x):
        return A @ x.reshape(-1)

    J = numerical_jacobian(f, x0)
    assert J.shape == (4, 6)
    from nanogpt.config import rel_err as _rel_err

    assert _rel_err(J, A) < 1e-9


# ---------------------------------------------------------------------------
# TL integration: a non-contiguous param (e.g. a transposed view, likely in
# Phase 2/3 weight tying) must be rejected, not silently FD-checked on a copy.
# ---------------------------------------------------------------------------
def test_non_contiguous_param_raises_value_error():
    x, params = _make_xW(seed=14)
    params = {"W": np.asfortranarray(params["W"])}  # (4,5) F-ordered, not C
    assert not params["W"].flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        gradcheck(_linear_forward, _linear_backward_correct, params, x)

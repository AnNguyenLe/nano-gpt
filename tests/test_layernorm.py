"""Gradcheck + closed-form tests for nanogpt.layernorm.LayerNorm.

Owner: Engineer C. See SPRINT_1_PLAN.md section 8.3.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt.layernorm import LayerNorm, c3_test_instance
from nanogpt.gradcheck import gradcheck
from nanogpt.config import GRADCHECK_TOL, ROUNDOFF_ATOL, DEFAULT_SEED


# Fixture seed deliberately DIFFERENT from gradcheck's internal DEFAULT_SEED:
# gradcheck() scalarizes its upstream gradient R via make_rng(DEFAULT_SEED),
# so building x from that same seed would make R coincide with x's own
# random draw whenever shapes match. For the eps=0 instance that lands R
# exactly in the degenerate null directions (span{1, yhat}), making the true
# gradient ~0 and leaving only FD roundoff noise relative to REL_ERR_FLOOR --
# a spurious failure, not an implementation bug.
_FIXTURE_SEED = DEFAULT_SEED + 90210


def _rng():
    return np.random.default_rng(_FIXTURE_SEED)


def test_gradcheck_production_affine():
    """Production eps=LN_EPS, affine, x (2,3,6); gamma/beta non-init (TL-6)."""
    d = 6
    rng = _rng()
    ln = LayerNorm("ln_prod", d)  # eps=LN_EPS, affine=True
    x = rng.standard_normal((2, 3, d))
    gamma = 1.0 + 0.5 * rng.standard_normal(d)
    beta = rng.standard_normal(d)
    params = {ln.gamma_key: gamma, ln.beta_key: beta}

    report = gradcheck(ln.forward, ln.backward, params, x)
    print(report)
    assert report.passed, str(report)
    for name in ("x", ln.gamma_key, ln.beta_key):
        assert report.tensors[name].rel_err < GRADCHECK_TOL, (name, report.tensors[name])


def test_gradcheck_eps0_no_affine():
    """eps=0 no-affine instance on non-constant x (2,3,6); check x."""
    d = 6
    rng = _rng()
    ln = c3_test_instance(d)
    x = 3.0 + rng.standard_normal((2, 3, d))  # non-constant rows guaranteed w.h.p.
    report = gradcheck(ln.forward, ln.backward, {}, x)
    print(report)
    assert report.passed, str(report)
    assert report.tensors["x"].rel_err < GRADCHECK_TOL


def test_ddof0_population_variance():
    """x=[1,2,3,4], eps=0: yhat == (x-2.5)/sqrt(1.25). Catches ddof=1."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ln = c3_test_instance(4)
    out, cache = ln.forward(x, {})
    expected = (x - 2.5) / np.sqrt(1.25)
    assert np.max(np.abs(out - expected)) < ROUNDOFF_ATOL
    assert np.max(np.abs(cache["yhat"] - expected)) < ROUNDOFF_ATOL
    # If ddof=1 had been used, population variance 1.25 would become 1.25*4/3,
    # giving a visibly different (smaller-magnitude) yhat; make that explicit.
    ddof1_yhat = (x - 2.5) / np.sqrt(1.25 * 4 / 3)
    assert np.max(np.abs(out - ddof1_yhat)) > 1e-3


def test_eps0_constant_row_raises():
    ln = c3_test_instance(4)
    x = np.array([5.0, 5.0, 5.0, 5.0])
    with pytest.raises(ValueError):
        ln.forward(x, {})


def test_production_constant_row_no_nan():
    d = 4
    ln = LayerNorm("ln_prod4", d)
    rng = _rng()
    params = ln.init_params(rng)
    x = np.full((2, d), 7.0)
    out, cache = ln.forward(x, params)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(cache["yhat"])) < ROUNDOFF_ATOL  # yhat == 0 for a constant row


def test_dtype_strictness_float32_raises():
    ln = LayerNorm("ln_dtype", 4)
    rng = _rng()
    params = ln.init_params(rng)
    x = rng.standard_normal((2, 4)).astype(np.float32)
    with pytest.raises(TypeError):
        ln.forward(x, params)

    # TL-7 also covers backward: a float32 upstream gradient must raise,
    # not be silently cast (added at TL integration).
    _, cache = ln.forward(x.astype(np.float64), params)
    with pytest.raises(TypeError):
        ln.backward(np.ones((2, 4), dtype=np.float32), cache)


# ---------------------------------------------------------------------------
# TEST-THE-TEST: deliberately broken backwards must FAIL gradcheck.
# ---------------------------------------------------------------------------

def _backward_missing_yhat_term(d_out, cache):
    """BUG: drops the yhat * mean(g*yhat) correction term."""
    yhat = cache["yhat"]
    rstd = cache["rstd"]
    gamma = cache["gamma"]
    d = cache["d"]
    affine = cache["affine"]
    d_out = np.asarray(d_out, dtype=np.float64)

    if affine:
        dbeta = d_out.reshape(-1, d).sum(axis=0)
        dgamma = (d_out * yhat).reshape(-1, d).sum(axis=0)
        g = d_out * gamma
        d_params = {cache["gamma_key"]: dgamma, cache["beta_key"]: dbeta}
    else:
        g = d_out
        d_params = {}

    mean_g = np.mean(g, axis=-1, keepdims=True)
    dx = rstd * (g - mean_g)  # missing: - yhat * mean(g * yhat)
    return dx, d_params


def _backward_missing_gamma(d_out, cache):
    """BUG: forgets g = d_out * gamma (uses g = d_out directly)."""
    yhat = cache["yhat"]
    rstd = cache["rstd"]
    d = cache["d"]
    d_out = np.asarray(d_out, dtype=np.float64)

    dbeta = d_out.reshape(-1, d).sum(axis=0)
    dgamma = (d_out * yhat).reshape(-1, d).sum(axis=0)
    g = d_out  # BUG: should be d_out * gamma
    d_params = {cache["gamma_key"]: dgamma, cache["beta_key"]: dbeta}

    mean_g = np.mean(g, axis=-1, keepdims=True)
    mean_g_yhat = np.mean(g * yhat, axis=-1, keepdims=True)
    dx = rstd * (g - mean_g - yhat * mean_g_yhat)
    return dx, d_params


def test_test_the_test_missing_yhat_term_fails():
    d = 6
    rng = _rng()
    ln = LayerNorm("ln_tt1", d)
    x = rng.standard_normal((2, 3, d))
    gamma = 1.0 + 0.5 * rng.standard_normal(d)
    beta = rng.standard_normal(d)
    params = {ln.gamma_key: gamma, ln.beta_key: beta}

    report = gradcheck(ln.forward, _backward_missing_yhat_term, params, x)
    print(report)
    assert not report.passed
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL


def test_test_the_test_missing_gamma_fails_only_because_gamma_ne_1():
    d = 6
    rng = _rng()
    ln = LayerNorm("ln_tt2", d)
    x = rng.standard_normal((2, 3, d))
    gamma_non1 = 1.0 + 0.5 * rng.standard_normal(d)
    beta = rng.standard_normal(d)
    params_non1 = {ln.gamma_key: gamma_non1, ln.beta_key: beta}

    report_non1 = gradcheck(ln.forward, _backward_missing_gamma, params_non1, x)
    print("gamma != 1:", report_non1)
    assert not report_non1.passed
    assert report_non1.tensors["x"].rel_err >= GRADCHECK_TOL

    # At gamma == 1 exactly the bug is invisible (TL-6): g = d_out*1 == d_out.
    params_at_1 = {ln.gamma_key: np.ones(d), ln.beta_key: beta}
    report_at_1 = gradcheck(ln.forward, _backward_missing_gamma, params_at_1, x)
    print("gamma == 1:", report_at_1)
    assert report_at_1.tensors["x"].rel_err < GRADCHECK_TOL

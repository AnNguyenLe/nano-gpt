"""tests/test_mlp.py -- Engineer A, Sprint 3 milestone A1 (S3-01).

Covers `nanogpt.mlp.MLP`: construction, init, forward shapes, the harness
gradcheck (tiny + a d=64 spot check), a closed-form identity-padded anchor,
and test-the-tests (gelu_backward skipped; W_proj ignored in the backward).
See SPRINT_3_PLAN.md section 5.3.

Fixture seed (per-file, never DEFAULT_SEED itself): `DEFAULT_SEED + 301`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt import mlp as mlp_module
from nanogpt.mlp import MLP
from nanogpt.config import (
    DEFAULT_SEED,
    GELU_A,
    GELU_C,
    GRADCHECK_TOL,
    MLP_HIDDEN_MULT,
    ROUNDOFF_ATOL,
    make_rng,
)
from nanogpt.gradcheck import gradcheck

_FIXTURE_SEED = DEFAULT_SEED + 301

_ALL_TENSOR_NAMES = ("mlp.fc.W", "mlp.fc.b", "mlp.proj.W", "mlp.proj.b", "x")


# ---------------------------------------------------------------------------
# Fixtures: non-init params, W = N(0,1)/sqrt(d_in), b = 0.1*N(0,1), drawn
# sequentially fc then proj from one Generator; x drawn next from the same rng.
# ---------------------------------------------------------------------------
def _non_init_params(rng, d, prefix="mlp"):
    hidden = MLP_HIDDEN_MULT * d
    W_fc = np.ascontiguousarray((rng.standard_normal((d, hidden)) / math.sqrt(d)).astype(np.float64))
    b_fc = np.ascontiguousarray((0.1 * rng.standard_normal(hidden)).astype(np.float64))
    W_proj = np.ascontiguousarray(
        (rng.standard_normal((hidden, d)) / math.sqrt(hidden)).astype(np.float64)
    )
    b_proj = np.ascontiguousarray((0.1 * rng.standard_normal(d)).astype(np.float64))
    return {
        f"{prefix}.fc.W": W_fc, f"{prefix}.fc.b": b_fc,
        f"{prefix}.proj.W": W_proj, f"{prefix}.proj.b": b_proj,
    }


def _gradcheck_fixture(d, B, T, seed_offset):
    rng = make_rng(_FIXTURE_SEED + seed_offset)
    m = MLP("mlp", d)
    params = _non_init_params(rng, d)
    x = rng.standard_normal((B, T, d)).astype(np.float64)
    return m, params, x


def _tiny_gradcheck_fixture():
    """d=8, B=2, T=4, non-init fixture."""
    return _gradcheck_fixture(8, 2, 4, 100)


# ---------------------------------------------------------------------------
# Construction (S3-01)
# ---------------------------------------------------------------------------
def test_constructor_rejects_non_positive_or_non_int_d():
    with pytest.raises(ValueError):
        MLP("mlp", 0)
    with pytest.raises(ValueError):
        MLP("mlp", -3)
    with pytest.raises(ValueError):
        MLP("mlp", 8.0)
    with pytest.raises(ValueError):
        MLP("mlp", True)


def test_hidden_width_and_param_keys():
    m = MLP("mlp", 8)
    assert m.hidden == 32
    assert m.param_keys() == ["mlp.fc.W", "mlp.fc.b", "mlp.proj.W", "mlp.proj.b"]


def test_init_params_shapes_dtype_contiguous_and_zero_bias():
    rng = make_rng(_FIXTURE_SEED)
    m = MLP("mlp", 8)
    params = m.init_params(rng)
    assert set(params) == set(m.param_keys())
    assert params["mlp.fc.W"].shape == (8, 32)
    assert params["mlp.fc.b"].shape == (32,)
    assert params["mlp.proj.W"].shape == (32, 8)
    assert params["mlp.proj.b"].shape == (8,)
    for key, arr in params.items():
        assert arr.dtype == np.float64, key
        assert arr.flags["C_CONTIGUOUS"], key
    assert np.all(params["mlp.fc.b"] == 0.0)
    assert np.all(params["mlp.proj.b"] == 0.0)


def test_init_params_draws_fc_then_proj_from_same_rng():
    d = 6
    rng_a = make_rng(_FIXTURE_SEED + 1)
    m = MLP("mlp", d)
    params = m.init_params(rng_a)

    # Re-derive by hand from a fresh Generator with the same seed, drawing
    # fc's init_params then proj's, exactly as MLP.init_params must do.
    rng_b = make_rng(_FIXTURE_SEED + 1)
    expected = {}
    expected.update(m.fc.init_params(rng_b))
    expected.update(m.proj.init_params(rng_b))

    for key in params:
        assert np.array_equal(params[key], expected[key]), key


# ---------------------------------------------------------------------------
# Forward shapes / dtype
# ---------------------------------------------------------------------------
def test_forward_shape_dtype_finite_and_hidden_width():
    m, params, x = _tiny_gradcheck_fixture()
    out, cache = m.forward(x, params)
    assert out.shape == x.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    # the pre-activation h cached inside gelu's cache has width 4d
    h = cache["c_gelu"]["x"]
    assert h.shape[-1] == MLP_HIDDEN_MULT * m.d
    assert h.shape[:-1] == x.shape[:-1]


def test_backward_d_params_has_exactly_four_keys():
    m, params, x = _tiny_gradcheck_fixture()
    out, cache = m.forward(x, params)
    rng = make_rng(_FIXTURE_SEED + 2)
    d_out = rng.standard_normal(out.shape).astype(np.float64)
    d_x, d_params = m.backward(d_out, cache)
    assert set(d_params.keys()) == set(m.param_keys())
    assert d_x.shape == x.shape


def test_dtype_strictness_float32_x_raises():
    m, params, x = _tiny_gradcheck_fixture()
    with pytest.raises(TypeError):
        m.forward(x.astype(np.float32), params)


def test_dtype_strictness_float32_param_raises():
    m, params, x = _tiny_gradcheck_fixture()
    bad_params = dict(params)
    bad_params["mlp.fc.W"] = bad_params["mlp.fc.W"].astype(np.float32)
    with pytest.raises(TypeError):
        m.forward(x, bad_params)


def test_dtype_strictness_float32_d_out_raises():
    m, params, x = _tiny_gradcheck_fixture()
    out, cache = m.forward(x, params)
    with pytest.raises(TypeError):
        m.backward(out.astype(np.float32), cache)


def test_value_error_wrong_last_dim():
    m = MLP("mlp", 8)
    rng = make_rng(_FIXTURE_SEED + 3)
    params = m.init_params(rng)
    x_wrong = rng.standard_normal((2, 4, 7)).astype(np.float64)
    with pytest.raises(ValueError):
        m.forward(x_wrong, params)


def test_no_mutation_of_params_x_and_d_out():
    m, params, x = _tiny_gradcheck_fixture()
    params_before = {k: v.copy() for k, v in params.items()}
    x_before = x.copy()

    out, cache = m.forward(x, params)
    for k in params:
        assert np.array_equal(params[k], params_before[k])
    assert np.array_equal(x, x_before)

    rng = make_rng(_FIXTURE_SEED + 4)
    d_out = rng.standard_normal(out.shape).astype(np.float64)
    d_out_before = d_out.copy()
    m.backward(d_out, cache)
    assert np.array_equal(d_out, d_out_before)
    for k in params:
        assert np.array_equal(params[k], params_before[k])
    assert np.array_equal(x, x_before)


# ---------------------------------------------------------------------------
# Gradcheck (harness). Tiny (<10s) + a d=64,B=1,T=2 spot check.
# ---------------------------------------------------------------------------
def test_module_gradcheck_tiny_full_table():
    m, params, x = _tiny_gradcheck_fixture()
    report = gradcheck(m.forward, m.backward, params, x)
    print(report)
    assert set(report.tensors) == set(_ALL_TENSOR_NAMES)
    for name, t in report.tensors.items():
        assert t.rel_err < GRADCHECK_TOL, (name, t)


def test_module_gradcheck_d64_B1_T2_spot_check():
    m, params, x = _gradcheck_fixture(64, 1, 2, 200)
    report = gradcheck(m.forward, m.backward, params, x)
    print(report)
    for name, t in report.tensors.items():
        assert t.rel_err < GRADCHECK_TOL, (name, t)


# ---------------------------------------------------------------------------
# Closed-form anchor: identity-padded W_fc/W_proj, zero biases ->
# out == GELU_tanh(x), computed independently from the GELU formula here
# (not via ops.gelu_forward), to roundoff.
# ---------------------------------------------------------------------------
def test_closed_form_identity_padded_anchor():
    d = 6
    m = MLP("mlp", d)
    hidden = MLP_HIDDEN_MULT * d

    W_fc = np.zeros((d, hidden), dtype=np.float64)
    W_fc[:, :d] = np.eye(d, dtype=np.float64)
    b_fc = np.zeros(hidden, dtype=np.float64)

    W_proj = np.zeros((hidden, d), dtype=np.float64)
    W_proj[:d, :] = np.eye(d, dtype=np.float64)
    b_proj = np.zeros(d, dtype=np.float64)

    params = {
        "mlp.fc.W": W_fc, "mlp.fc.b": b_fc,
        "mlp.proj.W": W_proj, "mlp.proj.b": b_proj,
    }

    rng = make_rng(_FIXTURE_SEED + 50)
    x = rng.standard_normal((2, 3, d)).astype(np.float64)
    out, _ = m.forward(x, params)

    # Independent closed-form GELU-tanh, from the formula constants directly.
    u = GELU_C * (x + GELU_A * x ** 3)
    expected = 0.5 * x * (1.0 + np.tanh(u))

    max_abs = float(np.max(np.abs(out - expected)))
    print("closed-form identity-padded anchor max_abs =", max_abs)
    assert max_abs <= ROUNDOFF_ATOL


# ---------------------------------------------------------------------------
# Test-the-tests (monkeypatch only; nothing broken is ever committed).
# ---------------------------------------------------------------------------
def test_tt_skipping_gelu_backward_fails_x_and_fc(monkeypatch):
    """(i) A backward that skips gelu_backward (passes da straight through
    as dh) fails the x and fc.* gradchecks."""
    m, params, x = _tiny_gradcheck_fixture()

    def _identity_gelu_backward(dy, cache):
        return dy, {}

    monkeypatch.setattr(mlp_module.ops, "gelu_backward", _identity_gelu_backward)
    report = gradcheck(m.forward, m.backward, params, x)
    print("gelu-backward-skipped gradcheck:", report)
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL
    assert report.tensors["mlp.fc.W"].rel_err >= GRADCHECK_TOL
    assert report.tensors["mlp.fc.b"].rel_err >= GRADCHECK_TOL


def test_tt_ignoring_proj_W_in_backward_fails_fc_and_x():
    """(ii) A backward that ignores W_proj when propagating into `a`/`h`
    (tiling d_out across the hidden width instead of multiplying by
    W_proj.T) fails the fc.* and x gradchecks, even though proj's own
    d_params (computed via the real proj.backward) are untouched."""
    m, params, x = _tiny_gradcheck_fixture()

    def _broken_backward(d_out, cache):
        c_fc, c_gelu, c_proj = cache["c_fc"], cache["c_gelu"], cache["c_proj"]
        a = c_proj["x"]
        hidden = a.shape[-1]
        d = d_out.shape[-1]
        reps = hidden // d
        # BUG: W_proj is ignored entirely when computing the upstream
        # gradient into the hidden activation; d_out is merely tiled.
        da = np.tile(d_out, (1,) * (d_out.ndim - 1) + (reps,))
        _, g_proj = m.proj.backward(d_out, c_proj)  # proj's own d_params: correct
        dh, _ = mlp_module.ops.gelu_backward(da, c_gelu)
        dx, g_fc = m.fc.backward(dh, c_fc)
        return dx, {**g_fc, **g_proj}

    report = gradcheck(m.forward, _broken_backward, params, x)
    print("W_proj-ignored-in-backward gradcheck:", report)
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL
    assert report.tensors["mlp.fc.W"].rel_err >= GRADCHECK_TOL

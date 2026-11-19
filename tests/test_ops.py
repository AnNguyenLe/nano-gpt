"""Tests for nanogpt/ops.py (Engineer B).

The gating checks use ``nanogpt.gradcheck.gradcheck``. The small *scoped*
directional central-difference spot checks (``*_directional_fd_check``) are
kept as an independent second opinion; they are not a substitute for the
harness-based tests. (Sprint 1 integration: the section 10 fallback
skip-guards were removed by the Tech Lead once gradcheck.py landed.)
"""

import numpy as np
import pytest

from nanogpt import ops
from nanogpt.config import GRADCHECK_TOL, ROUNDOFF_ATOL, DEFAULT_SEED, rel_err, make_rng
from nanogpt.gradcheck import gradcheck


def _directional_derivative_fd(f, x0, v, h=1e-5):
    """Central-difference directional derivative of scalar f along v at x0.

    A small, explicitly scoped check (a single number per call), distinct
    from the generic per-element harness in nanogpt.gradcheck -- used here
    only to give real, currently-executable gradient verification while that
    harness is not yet available. Never mutates x0.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (f(x0 + h * v) - f(x0 - h * v)) / (2.0 * h)


def _reference_embedding_dW(W_shape, idx, d_out):
    """Independent reference accumulation for Embedding.backward (a python
    loop, NOT a finite-difference estimate): re-derives dW analytically by
    summing contributions per row index. Used to verify np.add.at usage and
    to demonstrate the fancy-indexing duplicate-index bug (TEST-THE-TEST).
    """
    dW = np.zeros(W_shape, dtype=np.float64)
    flat_idx = np.asarray(idx).reshape(-1)
    flat_d = np.asarray(d_out).reshape(-1, W_shape[-1])
    for i, n in enumerate(flat_idx):
        dW[int(n)] += flat_d[i]
    return dW


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------

def test_linear_keys_shapes_dtype_and_no_mutation():
    rng = make_rng(1)
    lin = ops.Linear("lin", d_in=4, d_out=5, bias=True)
    params = lin.init_params(rng)
    assert set(params) == {"lin.W", "lin.b"}
    assert params["lin.W"].shape == (4, 5)
    assert params["lin.b"].shape == (5,)
    assert params["lin.W"].dtype == np.float64
    assert params["lin.b"].dtype == np.float64

    x = rng.normal(size=(2, 3, 4))
    params_before = {k: v.copy() for k, v in params.items()}
    x_before = x.copy()

    out, cache = lin.forward(x, params)
    assert out.shape == (2, 3, 5)
    for k in params:
        assert np.array_equal(params[k], params_before[k]), f"forward mutated {k}"
    assert np.array_equal(x, x_before), "forward mutated x"

    d_out = rng.normal(size=out.shape)
    d_out_before = d_out.copy()
    dx, d_params = lin.backward(d_out, cache)
    assert np.array_equal(d_out, d_out_before), "backward mutated d_out"
    assert dx.shape == x.shape
    assert set(d_params) == {"lin.W", "lin.b"}
    assert d_params["lin.W"].shape == (4, 5)
    assert d_params["lin.b"].shape == (5,)
    for v in d_params.values():
        assert v.dtype == np.float64


def test_linear_bias_false_has_no_b_key_anywhere():
    rng = make_rng(2)
    lin = ops.Linear("lin2", d_in=4, d_out=5, bias=False)
    params = lin.init_params(rng)
    assert set(params) == {"lin2.W"}
    x = rng.normal(size=(2, 3, 4))
    out, cache = lin.forward(x, params)
    d_out = rng.normal(size=out.shape)
    dx, d_params = lin.backward(d_out, cache)
    assert set(d_params) == {"lin2.W"}
    assert dx.shape == x.shape


@pytest.mark.parametrize("bias", [True, False])
def test_linear_directional_fd_check(bias):
    # TL-6: non-init params (gradient w.r.t. an unused .b would be invisible
    # with b at its init value, but b has no "unused" risk here; the real
    # risk TL-6 guards against is gamma=1 in LayerNorm -- for Linear we still
    # follow the same non-init convention for consistency).
    rng = make_rng(3)
    lin = ops.Linear("lin3", d_in=4, d_out=5, bias=bias)
    params = lin.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    x = rng.normal(size=(2, 3, 4))

    out, cache = lin.forward(x, params)
    R = make_rng(DEFAULT_SEED).standard_normal(out.shape)
    dx, d_params = lin.backward(R, cache)

    def loss_x(xx):
        o, _ = lin.forward(xx, params)
        return float(np.sum(o * R))

    v = rng.normal(size=x.shape)
    fd = _directional_derivative_fd(loss_x, x, v)
    analytic = float(np.sum(dx * v))
    assert rel_err(analytic, fd) < 1e-7

    for key in params:
        def loss_param(flat, key=key):
            p2 = dict(params)
            p2[key] = flat.reshape(params[key].shape)
            o, _ = lin.forward(x, p2)
            return float(np.sum(o * R))

        vp = rng.normal(size=params[key].shape)
        fd_p = _directional_derivative_fd(loss_param, params[key], vp)
        analytic_p = float(np.sum(d_params[key] * vp))
        assert rel_err(analytic_p, fd_p) < 1e-7, f"Linear directional FD mismatch on {key}"


def test_linear_gradcheck_with_bias():
    rng = make_rng(4)
    lin = ops.Linear("lin4", d_in=4, d_out=5, bias=True)
    params = lin.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    x = rng.normal(size=(2, 3, 4))
    report = gradcheck(lin.forward, lin.backward, params, x, tol=GRADCHECK_TOL, seed=DEFAULT_SEED)
    assert report.passed, str(report)


def test_linear_gradcheck_no_bias():
    rng = make_rng(5)
    lin = ops.Linear("lin5", d_in=4, d_out=5, bias=False)
    params = lin.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    x = rng.normal(size=(2, 3, 4))
    report = gradcheck(lin.forward, lin.backward, params, x, tol=GRADCHECK_TOL, seed=DEFAULT_SEED)
    assert report.passed, str(report)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def test_embedding_backward_matches_reference_with_duplicates():
    rng = make_rng(6)
    emb = ops.Embedding("emb", num_embeddings=5, d=3)
    params = emb.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    idx = np.array([[0, 2, 2], [1, 0, 2]], dtype=np.int64)  # duplicates required

    out, cache = emb.forward(idx, params)
    assert out.shape == (2, 3, 3)
    d_out = rng.normal(size=out.shape)
    dx, d_params = emb.backward(d_out, cache)
    assert dx is None
    assert set(d_params) == {"emb.W"}

    ref = _reference_embedding_dW(params["emb.W"].shape, idx, d_out)
    assert rel_err(d_params["emb.W"], ref) < ROUNDOFF_ATOL


def test_embedding_test_the_test_naive_fancy_index_fails_on_duplicates_passes_on_unique():
    rng = make_rng(7)
    emb = ops.Embedding("emb2", num_embeddings=5, d=3)
    params = emb.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    W_shape = params["emb2.W"].shape

    idx_dup = np.array([[0, 2, 2], [1, 0, 2]], dtype=np.int64)
    out_dup, cache_dup = emb.forward(idx_dup, params)
    d_out_dup = rng.normal(size=out_dup.shape)
    _, d_params_dup = emb.backward(d_out_dup, cache_dup)
    ref_dup = _reference_embedding_dW(W_shape, idx_dup, d_out_dup)

    naive_dup = np.zeros(W_shape, dtype=np.float64)
    naive_dup[idx_dup] += d_out_dup  # the bug: last write wins, no accumulation

    # correct (np.add.at) implementation matches the reference exactly
    assert rel_err(d_params_dup["emb2.W"], ref_dup) < ROUNDOFF_ATOL
    # the naive fancy-index variant FAILS on duplicate indices
    assert rel_err(naive_dup, ref_dup) > 1e-3

    idx_unique = np.array([0, 1, 2], dtype=np.int64)
    out_u, cache_u = emb.forward(idx_unique, params)
    d_out_u = rng.normal(size=out_u.shape)
    _, d_params_u = emb.backward(d_out_u, cache_u)
    ref_u = _reference_embedding_dW(W_shape, idx_unique, d_out_u)

    naive_u = np.zeros(W_shape, dtype=np.float64)
    naive_u[idx_unique] += d_out_u

    assert rel_err(d_params_u["emb2.W"], ref_u) < ROUNDOFF_ATOL
    # the naive variant PASSES when indices are all unique
    assert rel_err(naive_u, ref_u) < ROUNDOFF_ATOL


def test_embedding_invalid_idx_raises():
    rng = make_rng(8)
    emb = ops.Embedding("emb3", num_embeddings=5, d=3)
    params = emb.init_params(rng)

    with pytest.raises(TypeError):
        emb.forward(np.array([0.0, 1.0], dtype=np.float64), params)

    with pytest.raises(ValueError):
        emb.forward(np.array([0, 5], dtype=np.int64), params)  # 5 is out of range

    with pytest.raises(ValueError):
        emb.forward(np.array([-1, 2], dtype=np.int64), params)


def test_embedding_positional_style_idx_arange():
    rng = make_rng(9)
    T = 4
    emb = ops.Embedding("wpe", num_embeddings=10, d=3)
    params = emb.init_params(rng)
    idx = np.arange(T, dtype=np.int64)
    out, cache = emb.forward(idx, params)
    assert out.shape == (T, 3)
    assert np.array_equal(out, params["wpe.W"][:T])


def test_embedding_gradcheck_duplicates():
    rng = make_rng(10)
    emb = ops.Embedding("emb4", num_embeddings=5, d=3)
    params = emb.init_params(rng)
    params = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in params.items()}
    idx = np.array([[0, 2, 2], [1, 0, 2]], dtype=np.int64)
    # idx is integer, so the harness skips the input ("x") check automatically
    # and checks only params ("emb4.W").
    report = gradcheck(emb.forward, emb.backward, params, idx, tol=GRADCHECK_TOL, seed=DEFAULT_SEED)
    assert report.passed, str(report)


def test_embedding_test_the_test_naive_backward_fails_gradcheck_on_duplicates_only():
    """Section 7.3 wording: the naive ``dW[idx] += d_out`` backward must FAIL
    the gradcheck harness on the duplicate idx and PASS it on an all-unique
    idx (added at TL integration; the reference-loop test above is kept)."""
    rng = make_rng(15)
    emb = ops.Embedding("emb5", num_embeddings=5, d=3)
    params = {"emb5.W": rng.normal(0.0, 1.0, size=(5, 3))}

    def naive_backward(d_out, cache):
        dW = np.zeros(cache["W_shape"], dtype=np.float64)
        dW[cache["idx"]] += d_out  # the bug: duplicate indices do not accumulate
        return None, {"emb5.W": dW}

    idx_dup = np.array([[0, 2, 2], [1, 0, 2]], dtype=np.int64)
    report_dup = gradcheck(emb.forward, naive_backward, params, idx_dup)
    assert not report_dup.passed, str(report_dup)

    idx_unique = np.array([0, 2, 4], dtype=np.int64)
    report_unique = gradcheck(emb.forward, naive_backward, params, idx_unique)
    assert report_unique.passed, str(report_unique)


# --------------------------------------------------------------------------
# GELU
# --------------------------------------------------------------------------

def test_gelu_closed_form_values():
    xs = np.array([0.0, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0], dtype=np.float64)
    y, cache = ops.gelu_forward(xs)
    assert y[0] == 0.0
    assert abs(y[5] - 10.0) < 1e-12
    # sanity: y(-10) should have collapsed to (numerically) 0 as well
    assert abs(y[6]) < 1e-9
    # odd-ish small-x region sanity: sign matches x
    assert y[1] > 0.0 and y[2] < 0.0


def test_gelu_dtype_float32_raises():
    x32 = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        ops.gelu_forward(x32)


def test_gelu_directional_fd_check():
    rng = make_rng(11)
    x = rng.normal(size=(2, 3, 4))
    # include the required probe points alongside random values
    x = np.concatenate([x.reshape(-1), np.array([0.0, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0])]).reshape(-1, 1)

    y, cache = ops.gelu_forward(x)
    R = rng.normal(size=x.shape)
    dx, d_params = ops.gelu_backward(R, cache)
    assert d_params == {}

    def loss(xx):
        yy, _ = ops.gelu_forward(xx)
        return float(np.sum(yy * R))

    v = rng.normal(size=x.shape)
    fd = _directional_derivative_fd(loss, x, v)
    analytic = float(np.sum(dx * v))
    assert rel_err(analytic, fd) < 1e-6


def test_gelu_gradcheck():
    """DoD item 4: GELU passes the generic harness (every element), on random
    (2,3,4) plus the section 7.3 probe points 0, +-1e-3, +-1, +-10.
    Added at TL integration (previously only a directional spot check)."""
    rng = make_rng(16)
    probes = np.array([0.0, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0])
    x = np.concatenate([rng.normal(size=24), probes, rng.normal(size=5)]).reshape(6, 6)
    report = gradcheck(
        lambda xx, params: ops.gelu_forward(xx),
        ops.gelu_backward,
        {},
        x,
        tol=GRADCHECK_TOL,
    )
    assert report.passed, str(report)


# --------------------------------------------------------------------------
# Softmax / logsumexp
# --------------------------------------------------------------------------

def test_softmax_float32_raises():
    z32 = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        ops.softmax_forward(z32)


def test_softmax_directional_fd_check():
    rng = make_rng(12)
    z = rng.normal(size=(2, 3, 6))
    p, cache = ops.softmax_forward(z)
    assert p.shape == z.shape
    assert np.allclose(p.sum(axis=-1), 1.0, atol=ROUNDOFF_ATOL * 10)

    dp = rng.normal(size=z.shape)
    dz, d_params = ops.softmax_backward(dp, cache)
    assert d_params == {}

    def loss(zz):
        pp, _ = ops.softmax_forward(zz)
        return float(np.sum(pp * dp))

    v = rng.normal(size=z.shape)
    fd = _directional_derivative_fd(loss, z, v)
    analytic = float(np.sum(dz * v))
    assert rel_err(analytic, fd) < 1e-6


def test_softmax_gradcheck():
    rng = make_rng(13)
    z = rng.normal(size=(2, 3, 6))

    def fwd(x, params):
        return ops.softmax_forward(x)

    def bwd(d_out, cache):
        return ops.softmax_backward(d_out, cache)

    report = gradcheck(fwd, bwd, {}, z, tol=GRADCHECK_TOL, seed=DEFAULT_SEED)
    assert report.passed, str(report)


def test_lse_smoke_and_naive_overflow_test_the_test():
    rng = make_rng(14)
    z = 1e3 * rng.normal(size=(4, 7))

    p, cache = ops.softmax_forward(z)
    assert np.all(np.isfinite(p))
    row_sums = p.sum(axis=-1)
    assert np.max(np.abs(row_sums - 1.0)) < ROUNDOFF_ATOL * 1e3  # tiny float64 roundoff

    lse = ops.logsumexp(z)
    assert np.all(np.isfinite(lse))

    with np.errstate(over="ignore", invalid="ignore"):
        e_naive = np.exp(z)
        p_naive = e_naive / np.sum(e_naive, axis=-1, keepdims=True)
    assert not np.all(np.isfinite(p_naive))  # test-the-test: naive softmax overflows


def test_masked_softmax_exact_zero_no_nan_and_finite_backward():
    z = np.array([[1.0, 2.0, -np.inf, -np.inf],
                  [0.0, -np.inf, 3.0, -np.inf]], dtype=np.float64)
    p, cache = ops.softmax_forward(z)
    assert not np.any(np.isnan(p))
    assert p[0, 2] == 0.0 and p[0, 3] == 0.0
    assert p[1, 1] == 0.0 and p[1, 3] == 0.0
    assert np.allclose(p.sum(axis=-1), 1.0, atol=ROUNDOFF_ATOL * 10)

    dp = np.array([[0.5, -0.3, 2.0, -7.0],
                   [1.0, 5.0, -0.2, 9.0]], dtype=np.float64)  # arbitrary finite upstream grad
    dz, d_params = ops.softmax_backward(dp, cache)
    assert d_params == {}
    assert np.all(np.isfinite(dz))
    assert dz[0, 2] == 0.0 and dz[0, 3] == 0.0
    assert dz[1, 1] == 0.0 and dz[1, 3] == 0.0


# --------------------------------------------------------------------------
# accumulate_grads
# --------------------------------------------------------------------------

def test_accumulate_grads_copy_sum_and_identity():
    grads = {}
    d1 = {"a": np.array([1.0, 2.0])}
    ops.accumulate_grads(grads, d1)
    assert grads["a"] is not d1["a"], "first insert must be a copy"
    d1["a"][:] = 999.0  # mutate the source after inserting
    assert np.array_equal(grads["a"], np.array([1.0, 2.0])), "accumulator aliased the source"

    acc_ref = grads["a"]
    d2 = {"a": np.array([10.0, 20.0])}
    ops.accumulate_grads(grads, d2)
    assert grads["a"] is acc_ref, "second accumulation must be += in place, not a rebind"
    assert np.array_equal(grads["a"], np.array([11.0, 22.0]))

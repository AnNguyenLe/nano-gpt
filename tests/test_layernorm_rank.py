"""C3: LayerNorm Jacobian rank tests. Owner: Engineer C.

See SPRINT_1_PLAN.md section 2.2 (TL-2, TL-3) and section 8.3.

Closed form (any eps >= 0), single row x of shape (d,):
    J = (1/s) * (I - (1/d) 1 1^T - (1/d) yhat yhat^T)        s = sqrt(var+eps)
This is symmetric with eigenvalues:
  * 0                              in direction 1            (always, any eps)
  * (1/s) * eps / (var + eps)       in direction yhat          (0 iff eps=0)
  * 1/s = sigma_max                 on the (d-2)-dim complement
so sorted descending singular values are [sigma_max]*(d-2) + [eps/(var+eps)*sigma_max, 0].
s[-2]/s[0] == eps/(var+eps) is the TL-3 boundary relation this file checks directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt.layernorm import LayerNorm, c3_test_instance
from nanogpt.gradcheck import numerical_jacobian
from nanogpt.config import GRADCHECK_TOL, ROUNDOFF_ATOL, C3_SVD_REL_TOL, DEFAULT_SEED, LN_EPS


def _build_jacobian_via_backward(ln: LayerNorm, x: np.ndarray, params: dict) -> np.ndarray:
    """J_impl[i, :] = backward(e_i)[0], extracted from the implementation (not the formula)."""
    d = x.shape[-1]
    _, cache = ln.forward(x, params)
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        dx, _ = ln.backward(e, cache)
        J[i, :] = dx
    return J


def _closed_form_jacobian(x: np.ndarray, eps: float) -> np.ndarray:
    d = x.shape[-1]
    mu = np.mean(x)
    var = np.mean((x - mu) ** 2)
    s = np.sqrt(var + eps)
    yhat = (x - mu) / s
    I = np.eye(d)
    ones = np.ones((d, d)) / d
    outer_yhat = np.outer(yhat, yhat) / d
    return (I - ones - outer_yhat) / s


def _svd_rank(J: np.ndarray, rel_tol: float = C3_SVD_REL_TOL):
    s = np.linalg.svd(J, compute_uv=False)  # descending
    s_max = s[0]
    rank = int(np.sum(s > rel_tol * s_max))
    return rank, s


def _atol_scale(x: np.ndarray) -> float:
    return ROUNDOFF_ATOL * max(1.0, float(np.linalg.norm(x)))


@pytest.mark.parametrize("d", [3, 5, 8])
def test_c3_eps0_instance_rank_d_minus_2(d):
    rng = np.random.default_rng(DEFAULT_SEED + d)
    ln = c3_test_instance(d)
    x = 3.0 + rng.standard_normal(d)  # non-constant, eps=0 precondition satisfied

    J_impl = _build_jacobian_via_backward(ln, x, {})
    J_closed = _closed_form_jacobian(x, eps=0.0)

    scale = max(1.0, np.linalg.norm(J_closed))
    assert np.max(np.abs(J_impl - J_closed)) < ROUNDOFF_ATOL * scale, (
        f"d={d}: J_impl vs closed form max abs diff "
        f"{np.max(np.abs(J_impl - J_closed)):.3e}"
    )

    f = lambda xx: ln.forward(xx, {})[0]
    J_num = numerical_jacobian(f, x)
    rel = np.linalg.norm(J_impl - J_num) / max(np.linalg.norm(J_impl) + np.linalg.norm(J_num), 1e-8)
    assert rel < GRADCHECK_TOL, f"d={d}: J_impl vs FD rel_err={rel:.3e}"

    ones = np.ones(d)
    j1 = J_impl @ ones
    jx = J_impl @ x
    atol = _atol_scale(x)
    assert np.max(np.abs(j1)) < atol, f"d={d}: ||J.1||_inf={np.max(np.abs(j1)):.3e}"
    assert np.max(np.abs(jx)) < atol, f"d={d}: ||J.x||_inf={np.max(np.abs(jx)):.3e}"

    rank, s = _svd_rank(J_impl)
    assert rank == d - 2, f"d={d}: measured rank={rank}, singular values={s}"


def _standardize_to_variance(rng, d, target_var):
    r = rng.standard_normal(d)
    std0 = np.std(r)  # ddof=0, population std
    return r / std0 * np.sqrt(target_var)


def test_c3_production_instance_rank_d_minus_1_at_var_1():
    """Production instance (eps=LN_EPS, gamma=1, beta=0), population variance
    exactly 1 (TL-3 precondition: ratio eps/(var+eps) ~ 1e-5, 10x above
    threshold). Rank is EXACTLY d-1. This is expected per spec section 4 and
    is NOT a bug: the yhat-direction eigenvalue is nonzero for any eps>0, it
    is just numerically small relative to sigma_max.
    """
    d = 8
    rng = np.random.default_rng(DEFAULT_SEED)
    ln = LayerNorm("ln_prod_rank", d)  # eps=LN_EPS, affine=True
    params = {ln.gamma_key: np.ones(d), ln.beta_key: np.zeros(d)}
    x = _standardize_to_variance(rng, d, target_var=1.0)

    J_impl = _build_jacobian_via_backward(ln, x, params)
    rank, s = _svd_rank(J_impl)
    assert rank == d - 1, f"measured rank={rank}, singular values={s}"

    j1 = J_impl @ np.ones(d)
    jx = J_impl @ x
    atol = _atol_scale(x)
    assert np.max(np.abs(j1)) < atol, f"||J.1||_inf={np.max(np.abs(j1)):.3e}"
    assert np.max(np.abs(jx)) > atol, "J.x should be nonzero for eps>0 (not a bug, see TL-3)"

    ratio = s[-2] / s[0]
    expected_ratio = LN_EPS / (1.0 + LN_EPS)
    rel = abs(ratio - expected_ratio) / expected_ratio
    assert rel < 1e-3, f"s[-2]/s[0]={ratio:.6e} vs eps/(var+eps)={expected_ratio:.6e}"


def test_c3_non_vacuity_ranks_differ():
    d = 8
    rng = np.random.default_rng(DEFAULT_SEED + 99)
    ln0 = c3_test_instance(d)
    x0 = 3.0 + rng.standard_normal(d)
    rank0, _ = _svd_rank(_build_jacobian_via_backward(ln0, x0, {}))

    lnp = LayerNorm("ln_prod_nv", d)
    params = {lnp.gamma_key: np.ones(d), lnp.beta_key: np.zeros(d)}
    x1 = _standardize_to_variance(rng, d, target_var=1.0)
    rank1, _ = _svd_rank(_build_jacobian_via_backward(lnp, x1, params))

    assert rank0 != rank1, f"non-vacuity failed: both ranks are {rank0}"
    assert rank0 == d - 2
    assert rank1 == d - 1


def test_tl3_boundary_var_1e3_reads_rank_d_minus_2():
    d = 8
    rng = np.random.default_rng(DEFAULT_SEED + 7)
    ln = LayerNorm("ln_prod_boundary", d)
    params = {ln.gamma_key: np.ones(d), ln.beta_key: np.zeros(d)}
    x = _standardize_to_variance(rng, d, target_var=1e3)

    J_impl = _build_jacobian_via_backward(ln, x, params)
    rank, s = _svd_rank(J_impl)
    assert rank == d - 2, f"measured rank={rank} at var=1e3, singular values={s}"

    ratio = s[-2] / s[0]
    expected_ratio = LN_EPS / (1e3 + LN_EPS)
    rel = abs(ratio - expected_ratio) / expected_ratio
    assert rel < 1e-3, f"s[-2]/s[0]={ratio:.6e} vs eps/(var+eps)={expected_ratio:.6e}"


def test_tl3_ratio_matches_at_both_var_1_and_var_1e3():
    d = 8
    rng = np.random.default_rng(DEFAULT_SEED + 11)
    ln = LayerNorm("ln_prod_ratio", d)
    params = {ln.gamma_key: np.ones(d), ln.beta_key: np.zeros(d)}

    for var in (1.0, 1e3):
        x = _standardize_to_variance(rng, d, target_var=var)
        J_impl = _build_jacobian_via_backward(ln, x, params)
        _, s = _svd_rank(J_impl)
        ratio = s[-2] / s[0]
        expected_ratio = LN_EPS / (var + LN_EPS)
        rel = abs(ratio - expected_ratio) / expected_ratio
        assert rel < 1e-3, f"var={var}: s[-2]/s[0]={ratio:.6e} vs expected={expected_ratio:.6e}"


# ---------------------------------------------------------------------------
# TEST-THE-TEST: dropped-yhat-term backward gives the wrong rank and fails
# the closed-form match on the eps=0 instance.
# ---------------------------------------------------------------------------

def _broken_backward_missing_yhat_term(d_out, cache):
    yhat = cache["yhat"]
    rstd = cache["rstd"]
    gamma = cache["gamma"]
    d = cache["d"]
    affine = cache["affine"]
    g = d_out * gamma if affine else d_out
    mean_g = np.mean(g, axis=-1, keepdims=True)
    dx = rstd * (g - mean_g)  # BUG: missing - yhat * mean(g * yhat)
    d_params = {}
    if affine:
        dbeta = d_out.reshape(-1, d).sum(axis=0)
        dgamma = (d_out * yhat).reshape(-1, d).sum(axis=0)
        d_params = {cache["gamma_key"]: dgamma, cache["beta_key"]: dbeta}
    return dx, d_params


def test_test_the_test_dropped_yhat_term_wrong_rank_and_fails_closed_form():
    d = 8
    rng = np.random.default_rng(DEFAULT_SEED + 3)
    ln = c3_test_instance(d)
    x = 3.0 + rng.standard_normal(d)
    _, cache = ln.forward(x, {})

    J_broken = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        dx, _ = _broken_backward_missing_yhat_term(e, cache)
        J_broken[i, :] = dx

    rank, s = _svd_rank(J_broken)
    assert rank == d - 1, f"broken backward: measured rank={rank} (expected d-1={d-1})"

    J_closed = _closed_form_jacobian(x, eps=0.0)
    scale = max(1.0, np.linalg.norm(J_closed))
    max_diff = np.max(np.abs(J_broken - J_closed))
    assert max_diff > ROUNDOFF_ATOL * scale, "broken backward should NOT match the closed form"

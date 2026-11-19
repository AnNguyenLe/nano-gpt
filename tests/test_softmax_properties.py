"""C4 acceptance tests for nanogpt/ops.py softmax (Engineer B).

C4 requires the Jacobian to be extracted FROM THE IMPLEMENTATION (via
softmax_backward on basis vectors), not only evaluated from the closed-form
formula, then checked for: match to diag(p) - p p^T, symmetric-PSD, rows
summing to 0, shift-invariance, and the expected rank under partial masking.

Most of C4 needs only ops.softmax_forward/backward (no external harness).
Only the "matches nanogpt.gradcheck.numerical_jacobian" clause needs
Engineer A's gradcheck module; it has since landed (see tests/test_ops.py
docstring), so that test now runs for real too. A small local FD
cross-check (``_fd_jacobian_local``) is kept alongside it as an independent
second check, not a substitute.
"""

import numpy as np
import pytest

from nanogpt import ops
from nanogpt.config import C3_SVD_REL_TOL, C4_PSD_TOL, ROUNDOFF_ATOL, GRADCHECK_TOL, rel_err
from nanogpt.gradcheck import gradcheck, numerical_jacobian

SEEDS = [0, 1, 2, 3]
D = 6


def _jacobian_from_backward(z):
    """Build the softmax Jacobian by applying softmax_backward (the actual
    implementation) to each standard basis vector -- this is "extracted from
    the implementation", per C4, not evaluated from the textbook formula."""
    p, cache = ops.softmax_forward(z)
    d = z.shape[-1]
    basis = np.eye(d, dtype=np.float64)
    J = np.zeros((d, d), dtype=np.float64)
    for k in range(d):
        dz, _ = ops.softmax_backward(basis[k], cache)
        J[:, k] = dz
    return p, J


def _fd_jacobian_local(f, x, h=1e-5):
    """Local central-difference Jacobian builder, scoped to this test file.

    This is a small, independent cross-check used only because
    nanogpt.gradcheck.numerical_jacobian was not available yet (see module
    docstring); it is deliberately tiny (D=6) and is not a generic harness.
    The spec-mandated check against the real numerical_jacobian is the
    separate, skip-guarded test below.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    f0 = np.asarray(f(x)).reshape(-1)
    m = f0.size
    J = np.zeros((m, n), dtype=np.float64)
    flat = x.reshape(-1)
    for i in range(n):
        orig = flat[i]
        flat[i] = orig + h
        fp = np.asarray(f(x)).reshape(-1).copy()
        flat[i] = orig - h
        fm = np.asarray(f(x)).reshape(-1).copy()
        flat[i] = orig
        J[:, i] = (fp - fm) / (2.0 * h)
    return J


def _broken_softmax_backward_dropped_term(dp, cache):
    """TEST-THE-TEST (i): a backward that forgets the sum(dp*p) term."""
    p = cache["p"]
    return p * dp, {}


def _non_shift_invariant_softmax(z):
    """TEST-THE-TEST (ii): e^z / (sum(e^z) + 1), not shift-invariant.

    Deliberately skips the max-subtraction stability trick: subtracting the
    row max before exponentiating would make this *accidentally*
    shift-invariant again (since max(z + c) - (z + c) == max(z) - z), which
    would silence the very bug this variant exists to demonstrate.
    """
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_jacobian_matches_closed_form_and_is_symmetric_psd_zero_row_sum(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))

    p, J = _jacobian_from_backward(z)
    J_closed = np.diag(p) - np.outer(p, p)

    max_abs_diff = np.max(np.abs(J - J_closed))
    assert max_abs_diff < ROUNDOFF_ATOL, f"seed={seed}: J vs closed form max_abs_diff={max_abs_diff}"

    asymmetry = np.max(np.abs(J - J.T))
    assert asymmetry < ROUNDOFF_ATOL, f"seed={seed}: asymmetry={asymmetry}"

    Jsym = 0.5 * (J + J.T)
    eigmin = float(np.linalg.eigvalsh(Jsym).min())
    assert eigmin >= -C4_PSD_TOL, f"seed={seed}: eigmin={eigmin}"

    row_sum_inf_norm = float(np.max(np.abs(J @ np.ones(D))))
    assert row_sum_inf_norm < ROUNDOFF_ATOL, f"seed={seed}: ||J@1||_inf={row_sum_inf_norm}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("c", [-5.0, 3.7, 1e3])
def test_c4_shift_invariance(seed, c):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))
    p_base, _ = ops.softmax_forward(z)
    p_shift, _ = ops.softmax_forward(z + c)
    max_abs_diff = float(np.max(np.abs(p_base - p_shift)))
    assert max_abs_diff < ROUNDOFF_ATOL, f"seed={seed}, c={c}: max_abs_diff={max_abs_diff}"


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_rank_under_partial_masking(seed):
    rng = np.random.default_rng(seed)
    for k in range(2, D + 1):  # k finite entries; k == D is the dense case
        z = np.full(D, -np.inf, dtype=np.float64)
        finite_pos = rng.choice(D, size=k, replace=False)
        z[finite_pos] = rng.normal(size=k)

        _, J = _jacobian_from_backward(z)
        s = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(s > C3_SVD_REL_TOL * s[0]))
        assert rank == k - 1, f"seed={seed}, k={k}: measured rank={rank}, s={s}"


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_jacobian_matches_local_fd_jacobian(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))
    _, J = _jacobian_from_backward(z)

    def f(zz):
        p, _ = ops.softmax_forward(zz)
        return p

    J_fd = _fd_jacobian_local(f, z)
    err = rel_err(J, J_fd)
    assert err < GRADCHECK_TOL, f"seed={seed}: rel_err(J, J_fd)={err}"


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_jacobian_matches_gradcheck_numerical_jacobian(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))
    _, J = _jacobian_from_backward(z)

    def f(zz):
        p, _ = ops.softmax_forward(zz)
        return p

    J_num = numerical_jacobian(f, z)
    err = rel_err(J, J_num)
    assert err < GRADCHECK_TOL, f"seed={seed}: rel_err(J, numerical_jacobian)={err}"


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_test_the_test_dropped_term_breaks_row_sum_and_closed_form(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))
    p, cache = ops.softmax_forward(z)
    d = z.shape[-1]
    basis = np.eye(d, dtype=np.float64)
    J_broken = np.zeros((d, d), dtype=np.float64)
    for k in range(d):
        dz, _ = _broken_softmax_backward_dropped_term(basis[k], cache)
        J_broken[:, k] = dz

    row_sum_inf_norm = float(np.max(np.abs(J_broken @ np.ones(d))))
    assert row_sum_inf_norm > ROUNDOFF_ATOL, (
        f"seed={seed}: broken backward unexpectedly satisfied the row-sum "
        f"invariant (row_sum_inf_norm={row_sum_inf_norm})"
    )

    J_closed = np.diag(p) - np.outer(p, p)
    mismatch = float(np.max(np.abs(J_broken - J_closed)))
    assert mismatch > ROUNDOFF_ATOL, (
        f"seed={seed}: broken backward unexpectedly matched the closed form"
    )

    # Section 7.3 test-the-test (i), second half: the dropped-term backward
    # must also FAIL the generic gradcheck harness (added at TL integration).
    report = gradcheck(
        lambda x, params: ops.softmax_forward(x),
        _broken_softmax_backward_dropped_term,
        {},
        rng.normal(size=(2, 3, D)),
    )
    assert not report.passed, str(report)
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_c4_test_the_test_non_shift_invariant_softmax_fails_shift_check(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D,))
    p_base = _non_shift_invariant_softmax(z)
    p_shift = _non_shift_invariant_softmax(z + 3.7)
    max_abs_diff = float(np.max(np.abs(p_base - p_shift)))
    assert max_abs_diff > ROUNDOFF_ATOL, (
        f"seed={seed}: non-shift-invariant variant unexpectedly passed the "
        f"shift-invariance check (max_abs_diff={max_abs_diff})"
    )

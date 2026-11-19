"""nanogpt.gradcheck — a generic central-difference gradient-check harness.

This is the correctness oracle for every differentiable module built in
this project (ops.py, layernorm.py, loss.py, and later attention/mlp/block).
It only perturbs inputs and parameters and compares analytic backward
output against a central-difference numerical estimate. It never builds a
tape, computation graph or autograd-like mechanism of any kind: the whole
point is that the numeric half is dumb and independent of the analytic
implementation under test.

Module contract assumed (see SPRINT_1_PLAN.md §3):
    forward(x, params)     -> (out, cache)
    backward(d_out, cache) -> (d_x, d_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nanogpt.config import (
    DEFAULT_SEED,
    FD_STEP,
    GRADCHECK_TOL,
    check_float64,
    make_rng,
    rel_err,
    rel_err_elementwise,
)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@dataclass
class TensorReport:
    name: str
    shape: tuple
    rel_err: float  # gated: config.rel_err, Frobenius-norm form
    max_elem_rel_err: float  # diagnostic only
    max_abs_err: float  # diagnostic only

    def __str__(self) -> str:
        return (
            f"{self.name:<12s} shape={self.shape!s:<16s} "
            f"rel_err={self.rel_err:.3e} "
            f"max_elem_rel_err={self.max_elem_rel_err:.3e} "
            f"max_abs_err={self.max_abs_err:.3e}"
        )


@dataclass
class GradcheckReport:
    tensors: dict  # name -> TensorReport
    tol: float
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.passed = all(t.rel_err < self.tol for t in self.tensors.values())

    def worst(self) -> TensorReport:
        return max(self.tensors.values(), key=lambda t: t.rel_err)

    def __str__(self) -> str:
        lines = [f"GradcheckReport(tol={self.tol:.1e}, passed={self.passed})"]
        for t in self.tensors.values():
            lines.append("  " + str(t))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------
def _is_float_array(a) -> bool:
    return np.issubdtype(np.asarray(a).dtype, np.floating)


def _scalarize(out, seed: int):
    """Build the random projection R and the scalar objective's "weight"
    tensor used as the upstream gradient fed to backward.

    If `out` is 0-d (a Python float or a 0-d array, e.g. a loss), R = 1.0
    and L = out. Otherwise R is a fixed seeded standard-normal array of
    out's shape, and L = sum(out * R).
    """
    out_arr = np.asarray(out)
    if out_arr.ndim == 0:
        return np.asarray(1.0, dtype=np.float64)
    rng = make_rng(seed)
    return rng.standard_normal(out_arr.shape).astype(np.float64)


def _objective(out, R) -> float:
    out_arr = np.asarray(out)
    if out_arr.ndim == 0:
        return float(out_arr)
    return float(np.sum(out_arr * R))


def gradcheck(
    forward,
    backward,
    params: dict,
    x,
    *,
    wrt_input: bool = True,
    h: float = FD_STEP,
    tol: float = GRADCHECK_TOL,
    seed: int = DEFAULT_SEED,
) -> GradcheckReport:
    """Central-difference gradient check against an analytic `backward`.

    forward(x, params)     -> (out, cache)
    backward(d_out, cache) -> (d_x, d_params)

    params: dict[str, np.ndarray], float64. Never mutated by this function
    beyond the perturb-then-restore cycle (asserted bit-identical at the end).
    x: the input array (float64) or an integer-dtype index array. If x is
    integer, the input is never perturbed, `wrt_input` is ignored, and
    `d_x` is required to be None.
    wrt_input: if True and x is float, `d_x` must be present and checked.
    seed: seeds the upstream projection R. PITFALL (found by Engineer C in
    Sprint 1): do NOT build a fixture's x from `make_rng(seed)` with the same
    seed when x and `out` have the same shape -- R then equals x's own draw,
    and for maps whose Jacobian annihilates x (eps=0 LayerNorm: J.x = 0) the
    true gradient collapses to ~0 and the check fails on pure FD roundoff.
    Use a fixture seed offset from DEFAULT_SEED (see tests/test_layernorm.py).

    Every float parameter array must be C-contiguous: the numeric loop
    perturbs it in place through a flat view, and a non-contiguous array
    would be silently copied by reshape (numeric gradient 0). A ValueError
    is raised instead.
    """
    # --- dtype guards -------------------------------------------------
    float_params = {k: v for k, v in params.items() if _is_float_array(v)}
    check_float64(*float_params.values(), names=list(float_params.keys()))
    for key, p in float_params.items():
        if not np.asarray(p).flags.c_contiguous:
            raise ValueError(
                f"gradcheck: params[{key!r}] is not C-contiguous; in-place FD "
                "perturbation would act on a copy. Pass np.ascontiguousarray(...)."
            )
    x_is_float = _is_float_array(x)
    if x_is_float:
        check_float64(x, names=["x"])

    # --- snapshot params for the bit-identical check at the end --------
    params_snapshot = {k: np.array(v, copy=True) for k, v in params.items()}

    # --- forward + scalarized objective ---------------------------------
    out, cache = forward(x, params)
    R = _scalarize(out, seed)

    # --- analytic backward ----------------------------------------------
    d_x, d_params = backward(R, cache)

    # --- structural validation (before any numeric work) -----------------
    if set(d_params.keys()) != set(params.keys()):
        missing = set(params.keys()) - set(d_params.keys())
        extra = set(d_params.keys()) - set(params.keys())
        raise ValueError(
            "gradcheck: d_params keys mismatch. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    for key, p in params.items():
        dp = d_params[key]
        if np.asarray(dp).shape != np.asarray(p).shape:
            raise ValueError(
                f"gradcheck: d_params[{key!r}] has shape {np.asarray(dp).shape}, "
                f"expected {np.asarray(p).shape}"
            )
        if _is_float_array(p) and not _is_float_array(dp):
            raise ValueError(f"gradcheck: d_params[{key!r}] is not a float array")
        if _is_float_array(p):
            check_float64(dp, names=[f"d_params[{key!r}]"])

    if x_is_float and wrt_input:
        if d_x is None:
            raise ValueError("gradcheck: wrt_input=True but backward returned d_x=None")
        if np.asarray(d_x).shape != np.asarray(x).shape:
            raise ValueError(
                f"gradcheck: d_x has shape {np.asarray(d_x).shape}, expected {np.asarray(x).shape}"
            )
        check_float64(d_x, names=["d_x"])
    if not x_is_float:
        if d_x is not None:
            raise ValueError("gradcheck: x has integer dtype but backward returned d_x is not None")

    # --- numeric central-difference estimates -----------------------------
    # Params and x are perturbed with two small, explicit loops rather than
    # one over-general helper: params is a dict of named arrays (and the
    # perturbed array must be looked up through `params` so `forward` sees
    # the mutation), while x is a single positional array, so the two loops
    # differ slightly in how they rebuild the `forward` call.
    tensors: dict = {}

    def _numeric_grad_for_param(key: str) -> TensorReport:
        arr = params[key]
        analytic = np.asarray(d_params[key], dtype=np.float64)
        numeric = np.zeros(arr.shape, dtype=np.float64)
        flat = arr.reshape(-1)
        numeric_flat = numeric.reshape(-1)
        n = flat.shape[0]
        for i in range(n):
            original = flat[i]
            flat[i] = original + h
            out_plus, _ = forward(x, params)
            L_plus = _objective(out_plus, R)
            flat[i] = original - h
            out_minus, _ = forward(x, params)
            L_minus = _objective(out_minus, R)
            flat[i] = original
            numeric_flat[i] = (L_plus - L_minus) / (2.0 * h)
        r = rel_err(analytic, numeric)
        elem = rel_err_elementwise(analytic, numeric)
        return TensorReport(
            name=key,
            shape=tuple(arr.shape),
            rel_err=r,
            max_elem_rel_err=float(np.max(elem)) if elem.size else 0.0,
            max_abs_err=float(np.max(np.abs(analytic - numeric))) if numeric.size else 0.0,
        )

    def _numeric_grad_for_input() -> TensorReport:
        x_copy = np.array(x, dtype=np.float64, copy=True)
        analytic = np.asarray(d_x, dtype=np.float64)
        numeric = np.zeros(x_copy.shape, dtype=np.float64)
        flat = x_copy.reshape(-1)
        numeric_flat = numeric.reshape(-1)
        n = flat.shape[0]
        for i in range(n):
            original = flat[i]
            flat[i] = original + h
            out_plus, _ = forward(x_copy, params)
            L_plus = _objective(out_plus, R)
            flat[i] = original - h
            out_minus, _ = forward(x_copy, params)
            L_minus = _objective(out_minus, R)
            flat[i] = original
            numeric_flat[i] = (L_plus - L_minus) / (2.0 * h)
        r = rel_err(analytic, numeric)
        elem = rel_err_elementwise(analytic, numeric)
        return TensorReport(
            name="x",
            shape=tuple(x_copy.shape),
            rel_err=r,
            max_elem_rel_err=float(np.max(elem)) if elem.size else 0.0,
            max_abs_err=float(np.max(np.abs(analytic - numeric))) if numeric.size else 0.0,
        )

    for key in params.keys():
        tensors[key] = _numeric_grad_for_param(key)

    if x_is_float and wrt_input:
        tensors["x"] = _numeric_grad_for_input()

    # --- params must be bit-identical to the snapshot after the run --------
    for key, p in params.items():
        if not np.array_equal(p, params_snapshot[key]):
            raise AssertionError(
                f"gradcheck: params[{key!r}] was not restored bit-identically"
            )

    return GradcheckReport(tensors=tensors, tol=tol)


def numerical_jacobian(f, x, h: float = FD_STEP) -> np.ndarray:
    """Central-difference Jacobian of f: R^n -> R^m (any input/output shape,
    internally flattened). Returns an array of shape (m, n), where
    m = f(x).size and n = x.size.

    This does not call backward anywhere: it is a pure black-box numeric
    Jacobian, used as an independent cross-check against both closed-form
    Jacobians and gradcheck's own scalarized estimates.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    x_shape = x.shape
    n = x.size
    out0 = np.asarray(f(x), dtype=np.float64)
    out_shape = out0.shape
    m = out0.size

    J = np.zeros((m, n), dtype=np.float64)
    flat = x.reshape(-1)
    for j in range(n):
        original = flat[j]
        flat[j] = original + h
        out_plus = np.asarray(f(x.reshape(x_shape)), dtype=np.float64).reshape(-1)
        flat[j] = original - h
        out_minus = np.asarray(f(x.reshape(x_shape)), dtype=np.float64).reshape(-1)
        flat[j] = original
        J[:, j] = (out_plus - out_minus) / (2.0 * h)
    return J

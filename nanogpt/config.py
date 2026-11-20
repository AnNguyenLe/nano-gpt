"""nanogpt.config — Frozen Conventions (§0 / §5 of SPRINT_1_PLAN.md).

This module is the SINGLE source of truth for every numeric constant,
dtype, tolerance and seed used anywhere in the `nanogpt` package. No other
module in `nanogpt/` may redefine these values (enforced by
`tests/test_conventions.py`, an AST guard that scans `nanogpt/*.py` except
this file). Every other module imports what it needs from here.

Nothing in this file is differentiable code: it only defines constants,
two small numeric helpers (`rel_err`, `rel_err_elementwise`), an RNG
factory, a dtype guard, and the `ModelConfig` dataclass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------
DTYPE = np.float64
INT_DTYPE = np.int64

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
INIT_STD = 0.02
BIAS_INIT = 0.0
LN_GAMMA_INIT = 1.0
LN_BETA_INIT = 0.0

# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
LN_EPS = 1e-5

# ---------------------------------------------------------------------------
# MLP / attention shape conventions
# ---------------------------------------------------------------------------
MLP_HIDDEN_MULT = 4
LINEAR_HAS_BIAS = True  # Q/K/V/O + MLP linears
LM_HEAD_HAS_BIAS = False  # tied head, no bias
N_HEAD = 4  # default H; must divide d

# ---------------------------------------------------------------------------
# GELU (tanh approximation) constants
# ---------------------------------------------------------------------------
GELU_C = math.sqrt(2.0 / math.pi)
GELU_A = 0.044715

# ---------------------------------------------------------------------------
# Gradcheck / numerical tolerances
# ---------------------------------------------------------------------------
FD_STEP = 1e-5
REL_ERR_FLOOR = 1e-8
GRADCHECK_TOL = 1e-6
C3_SVD_REL_TOL = 1e-6  # singular values < C3_SVD_REL_TOL * s_max are treated as zero
C4_PSD_TOL = 1e-10  # eigenvalues >= -C4_PSD_TOL are treated as non-negative
ROUNDOFF_ATOL = 1e-12  # for "exact identity" checks (row sums, J@1, closed-form vs impl)
INIT_LOSS_TOL = 0.1  # C2 (Phase 3) — defined now, used later
KV_CACHE_ABS_TOL = 1e-10  # C8 (Phase 5)
FLOPS_REL_BAND = 0.25  # C9 (Phase 4)

# ---------------------------------------------------------------------------
# AdamW (Phase 4, TL-24/25/26 -- nanoGPT char-level defaults)
# ---------------------------------------------------------------------------
ADAMW_BETAS = (0.9, 0.99)
ADAMW_EPS = 1e-8  # NOTE: 1e-8 is a FORBIDDEN_FLOATS literal outside this file
ADAMW_WEIGHT_DECAY = 0.1

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
DEFAULT_SEED = 1337
CORPUS_SEED = 1337


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def rel_err(a, b) -> float:
    """Gating relative error, TL-1: the Frobenius-norm form.

    rel_err(a, b) = ||a - b||_F / max(||a||_F + ||b||_F, REL_ERR_FLOOR)

    For 0-d / scalar inputs this is *exactly* the spec's elementwise formula
    |a - b| / max(|a| + |b|, REL_ERR_FLOOR), since the Frobenius norm of a
    scalar is its absolute value. For arrays it is a single norm-wise ratio,
    not an elementwise one: this is deliberate (see TL-1 in SPRINT_1_PLAN.md)
    so that FD roundoff noise (~1e-10 absolute, with h=FD_STEP in float64)
    does not spuriously fail gradient elements that are themselves tiny.

    rel_err(0, 0) == 0.0 by construction (numerator 0, denominator floored
    at REL_ERR_FLOOR).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    num = float(np.linalg.norm((a_arr - b_arr).ravel()))
    denom = float(np.linalg.norm(a_arr.ravel())) + float(np.linalg.norm(b_arr.ravel()))
    return num / max(denom, REL_ERR_FLOOR)


def rel_err_elementwise(a, b) -> np.ndarray:
    """Elementwise relative error, DIAGNOSTIC ONLY (never gated).

    Same floor as `rel_err`, applied per element:
    |a_i - b_i| / max(|a_i| + |b_i|, REL_ERR_FLOOR)
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    num = np.abs(a_arr - b_arr)
    denom = np.maximum(np.abs(a_arr) + np.abs(b_arr), REL_ERR_FLOOR)
    return num / denom


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Factory for a NumPy `Generator`. Nothing in this package should use
    NumPy's legacy global RNG state (`np.random.seed` / `np.random.randn`);
    every random draw takes an explicit `Generator` built from this
    function, so results are reproducible and independent of call order.
    """
    return np.random.default_rng(seed)


def check_float64(*arrays, names=None) -> None:
    """Raise TypeError if any argument is a floating-point ndarray whose
    dtype is not exactly float64 (TL-7: dtype strictness). Non-floating
    arrays (e.g. integer index arrays) are ignored by this check — use it
    only to validate the float tensors a module expects to already be
    float64, so a silent float32 cast never masquerades as a gradcheck bug.

    `names`, if given, must be the same length as `arrays` and is used only
    to produce a more informative error message.
    """
    if names is not None and len(names) != len(arrays):
        raise ValueError("check_float64: len(names) must match len(arrays)")
    for i, arr in enumerate(arrays):
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.floating) and a.dtype != np.float64:
            label = names[i] if names is not None else f"arg[{i}]"
            raise TypeError(
                f"check_float64: {label} has dtype {a.dtype}, expected float64"
            )


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Frozen model shape configuration.

    V: vocabulary size
    T: context length (block size)
    d: model (embedding) dimension
    L: number of transformer blocks
    H: number of attention heads; must evenly divide d
    """

    V: int
    T: int = 64
    d: int = 64
    L: int = 2
    H: int = N_HEAD

    def __post_init__(self) -> None:
        for field_name in ("V", "T", "d", "L", "H"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"ModelConfig.{field_name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"ModelConfig.{field_name} must be positive, got {value}")
        if self.d % self.H != 0:
            raise ValueError(
                f"ModelConfig: d={self.d} must be divisible by H={self.H}"
            )

    @property
    def head_dim(self) -> int:
        return self.d // self.H


def tiny_config(V: int) -> ModelConfig:
    """A deliberately small config for fast gradchecks: d=8, L=2, H=2, T=4."""
    return ModelConfig(V=V, T=4, d=8, L=2, H=2)

"""nanogpt.param_count -- C1: parameter-count formula + live unique-array counter.

See SPRINT_3_PLAN.md section 3.5 (frozen interfaces) and section 2.2 (R16).

Formula derivation (per transformer block, d = model dim):
    attention:  4 Linears (q,k,v,o), each (d,d) weight + (d,) bias
                -> 4*(d*d + d) = 4d^2 + 4d
    MLP:        fc (d,4d)+bias(4d,), proj(4d,d)+bias(d,)
                -> (4d^2 + 4d) + (4d^2 + d) = 8d^2 + 5d
    two LNs:    gamma + beta, each (d,), two LayerNorms per block -> 4d
    block total: (4d^2+4d) + (8d^2+5d) + 4d = 12d^2 + 13d

Plus, once per model (not per block):
    wte:    V*d   (counted ONCE -- TL-17: the tied lm_head reuses the SAME
            array as wte.W, so it must never be added again)
    wpe:    T*d
    ln_f:   2d    (gamma + beta)

    N(V, T, d, L) = V*d + T*d + L*(12*d^2 + 13*d) + 2*d

Preconditions (checked at CALL time, not import time, so a test can
monkeypatch this module's view of a config constant and observe the raise):
MLP_HIDDEN_MULT == 4, LINEAR_HAS_BIAS is True, LM_HEAD_HAS_BIAS is False,
and LN affine (this package's LayerNorm is always affine -- gamma AND beta,
contributing the 2*d-per-LN term the formula depends on).

Owner: Engineer A. Files: nanogpt/param_count.py, tests/test_param_count.py.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import (
    LINEAR_HAS_BIAS,
    LM_HEAD_HAS_BIAS,
    MLP_HIDDEN_MULT,
    ModelConfig,
)

# This package's LayerNorm (nanogpt/layernorm.py) is always affine: every
# instance owns both a gamma and a beta parameter (2*d total), which is what
# the "+ 4d per block" / "+ 2d final LN" terms below assume. There is no
# config flag to disable this, so the check is a named invariant re-affirmed
# at call time (see `_check_preconditions`) rather than a dead branch.
_LN_IS_AFFINE = True


def _check_preconditions() -> None:
    """Raise ValueError if the formula's frozen assumptions are violated.

    Reads the module-level names (not local copies), so monkeypatching
    `param_count.MLP_HIDDEN_MULT` / `param_count.LM_HEAD_HAS_BIAS` is
    observed on the next call (SPRINT_3_PLAN.md section 5.3 test-the-test).
    """
    if MLP_HIDDEN_MULT != 4:
        raise ValueError(
            "param_count preconditions violated: MLP_HIDDEN_MULT must be 4, "
            f"got {MLP_HIDDEN_MULT!r}"
        )
    if LINEAR_HAS_BIAS is not True:
        raise ValueError(
            "param_count preconditions violated: LINEAR_HAS_BIAS must be True, "
            f"got {LINEAR_HAS_BIAS!r}"
        )
    if LM_HEAD_HAS_BIAS is not False:
        raise ValueError(
            "param_count preconditions violated: LM_HEAD_HAS_BIAS must be False, "
            f"got {LM_HEAD_HAS_BIAS!r}"
        )
    if not _LN_IS_AFFINE:
        raise ValueError("param_count preconditions violated: LN must be affine")


def _check_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"formula_param_count: {name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"formula_param_count: {name} must be positive, got {value}")
    return int(value)


def formula_param_count(V: int, T: int, d: int, L: int) -> int:
    """N = V*d + T*d + L*(12*d**2 + 13*d) + 2*d, Python int arithmetic only."""
    _check_preconditions()
    V = _check_positive_int("V", V)
    T = _check_positive_int("T", T)
    d = _check_positive_int("d", d)
    L = _check_positive_int("L", L)

    per_block = 12 * d * d + 13 * d
    return V * d + T * d + L * per_block + 2 * d


def formula_from_config(cfg: ModelConfig) -> int:
    return formula_param_count(cfg.V, cfg.T, cfg.d, cfg.L)


def live_param_count(named_params: dict) -> int:
    """Sum `.size` over the UNIQUE arrays in `named_params`, by identity.

    Two different dict entries that are literally the SAME object (TL-17:
    "lm_head.W" -> params["wte.W"]) are counted once, since the tie is
    real. Two *distinct* objects (different `id`) that nonetheless
    `np.shares_memory` (e.g. one is `the_other.T`, or a slice/view of it)
    raise ValueError: that is exactly the forbidden transposed/sliced-view
    tying pattern TL-17 rules out.
    """
    seen: dict = {}  # id(arr) -> arr, insertion order, for the shares_memory scan
    total = 0
    for key, arr in named_params.items():
        arr = np.asarray(arr)
        aid = id(arr)
        if aid in seen:
            continue  # same object counted once already
        for other_key, other_arr in seen.items():
            if np.shares_memory(arr, other_arr):
                raise ValueError(
                    f"live_param_count: named_params[{key!r}] shares memory with a "
                    "distinct, already-counted array (transposed/sliced-view tie "
                    "is forbidden; tie parameters by identity, not by np.shares_memory)"
                )
        seen[aid] = arr
        total += arr.size
    return total

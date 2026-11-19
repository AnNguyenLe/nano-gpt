"""Affine LayerNorm over the last axis, population variance (ddof=0).

See SPRINT_1_PLAN.md section 8 (TL-2, TL-3, TL-6, TL-8) for the derivation.

Forward (population variance, ddof=0 -- never ddof=1):
    mu     = mean(x, axis=-1)
    var    = mean((x - mu) ** 2, axis=-1)      # ddof=0
    s      = sqrt(var + eps)
    yhat   = (x - mu) / s
    out    = gamma * yhat + beta               (affine=True)
           = yhat                               (affine=False)

Backward -- exact for any eps >= 0 (TL-2), using s = sqrt(var + eps), not the
eps=0 sigma:
    dbeta  = sum_lead d_out
    dgamma = sum_lead d_out * yhat
    g      = d_out * gamma   (or d_out if affine=False)
    dx     = rstd * (g - mean(g) - yhat * mean(g * yhat))   # means over last axis, keepdims

Closed-form Jacobian per row (any eps >= 0), used by the C3 rank tests:
    J = (1/s) * (I - (1/d) * 1 1^T - (1/d) * yhat yhat^T)
This is symmetric. At eps=0, s = sigma, so J.1 = 0 and J.x = 0 (rank d-2).
At eps>0 the yhat-direction eigenvalue is (1/s) * eps / (var + eps), so the
rank is d-1 provided that eigenvalue clears the SVD threshold (TL-3); do not
change eps or the threshold to force a particular answer.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import DTYPE, LN_GAMMA_INIT, LN_BETA_INIT, LN_EPS, check_float64


class LayerNorm:
    """forward(x, params) -> (out, cache); backward(d_out, cache) -> (dx, d_params).

    Parameters live in a flat dict keyed by the module's global names:
    ``f"{prefix}.gamma"`` and ``f"{prefix}.beta"``. A module never mutates
    ``params`` or ``x``.
    """

    def __init__(self, prefix: str, d: int, eps: float = LN_EPS, affine: bool = True):
        self.prefix = prefix
        self.d = int(d)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.gamma_key = f"{prefix}.gamma"
        self.beta_key = f"{prefix}.beta"

    def param_keys(self):
        return [self.gamma_key, self.beta_key] if self.affine else []

    def init_params(self, rng: np.random.Generator) -> dict:
        # rng is accepted for interface uniformity with Linear/Embedding even
        # though gamma/beta are deterministic constants (spec section 8.1).
        del rng
        if not self.affine:
            return {}
        return {
            self.gamma_key: np.full(self.d, LN_GAMMA_INIT, dtype=DTYPE),
            self.beta_key: np.full(self.d, LN_BETA_INIT, dtype=DTYPE),
        }

    def forward(self, x: np.ndarray, params):
        x = np.asarray(x)
        check_float64(x, names=["x"])
        if x.shape[-1] != self.d:
            raise ValueError(f"LayerNorm({self.prefix!r}): expected last axis {self.d}, got {x.shape[-1]}")

        mu = np.mean(x, axis=-1, keepdims=True)
        centered = x - mu
        var = np.mean(centered * centered, axis=-1, keepdims=True)  # ddof=0 -- population variance

        if self.eps == 0.0 and np.any(var == 0.0):
            # TL-8: eps=0 requires non-constant rows; division by s=0 is undefined.
            raise ValueError(
                f"LayerNorm({self.prefix!r}): eps=0 requires every row to be non-constant "
                "(population variance 0 encountered); use eps>0 to tolerate constant rows"
            )

        s = np.sqrt(var + self.eps)
        yhat = centered / s  # eps>0 with a constant row gives yhat=0, no NaN

        if self.affine:
            if params is None:
                raise ValueError(f"LayerNorm({self.prefix!r}): affine=True requires params")
            gamma = params[self.gamma_key]
            beta = params[self.beta_key]
            check_float64(gamma, beta, names=[self.gamma_key, self.beta_key])
            out = gamma * yhat + beta
        else:
            gamma = None
            out = yhat

        cache = {
            "yhat": yhat,
            "rstd": 1.0 / s,
            "gamma": gamma,
            "d": self.d,
            "affine": self.affine,
            "gamma_key": self.gamma_key,
            "beta_key": self.beta_key,
        }
        return np.asarray(out, dtype=DTYPE), cache

    def backward(self, d_out: np.ndarray, cache: dict):
        d_out = np.asarray(d_out)
        check_float64(d_out, names=["d_out"])  # TL-7: raise, never silently cast
        yhat = cache["yhat"]
        rstd = cache["rstd"]
        gamma = cache["gamma"]
        d = cache["d"]
        affine = cache["affine"]

        if affine:
            dbeta = d_out.reshape(-1, d).sum(axis=0)
            dgamma = (d_out * yhat).reshape(-1, d).sum(axis=0)
            g = d_out * gamma
            d_params = {
                cache["gamma_key"]: np.asarray(dgamma, dtype=DTYPE),
                cache["beta_key"]: np.asarray(dbeta, dtype=DTYPE),
            }
        else:
            g = d_out
            d_params = {}

        mean_g = np.mean(g, axis=-1, keepdims=True)
        mean_g_yhat = np.mean(g * yhat, axis=-1, keepdims=True)
        dx = rstd * (g - mean_g - yhat * mean_g_yhat)
        return np.asarray(dx, dtype=DTYPE), d_params


def c3_test_instance(d: int) -> LayerNorm:
    """TEST CONSTRUCTION ONLY -- never used for training.

    ``LayerNorm("ln_c3", d, eps=0.0, affine=False)``: the pure normalization
    map, whose Jacobian is exactly (1/sigma)(I - 11^T/d - yhat yhat^T/d),
    rank d-2 (J.1 = 0 and J.x = 0 exactly).
    """
    return LayerNorm("ln_c3", d, eps=0.0, affine=False)

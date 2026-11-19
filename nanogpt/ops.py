"""nanogpt.ops -- core differentiable primitives (Sprint 1, Engineer B).

Module contract (SPRINT_1_PLAN.md section 3):
    forward(x, params)     -> (out, cache)
    backward(d_out, cache) -> (d_x, d_params)

``d_params`` is always a *new* dict keyed by this module's own global parameter
names (never a reference into ``params``), with float64 arrays shaped like the
corresponding parameter. Stateless ops (``logsumexp``, ``softmax_*``,
``gelu_*``) return ``{}``. Nothing here mutates ``params``, ``x`` or ``d_out``.

Owner: Engineer B. Files: nanogpt/ops.py, tests/test_ops.py,
tests/test_softmax_properties.py.
"""

import numpy as np

from nanogpt.config import (
    DTYPE,
    INIT_STD,
    BIAS_INIT,
    GELU_C,
    GELU_A,
    LINEAR_HAS_BIAS,
    check_float64,
)


# --------------------------------------------------------------------------
# B1: logsumexp + softmax (first commit -- loss.py depends on this)
# --------------------------------------------------------------------------

def logsumexp(z, axis=-1, keepdims=False):
    """m + log(sum(exp(z - m), axis)), m = max(z, axis). Stable, last-axis by default.

    A row containing -inf entries but at least one finite entry is handled
    correctly (the -inf entries contribute exp(-inf - m) = 0). An all--inf
    row is a documented precondition violation (not specially handled).
    """
    check_float64(z, names=["z"])
    m = np.max(z, axis=axis, keepdims=True)
    s = np.sum(np.exp(z - m), axis=axis, keepdims=True)
    out = m + np.log(s)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax_forward(z, params=None):
    """Stable softmax over the last axis. Cache holds p for the backward pass."""
    check_float64(z, names=["z"])
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    s = np.sum(e, axis=-1, keepdims=True)
    p = e / s
    cache = {"p": p}
    return p, cache


def softmax_backward(dp, cache):
    """TL-4: dz = p * (dp - sum(dp * p, -1, keepdims=True)). Returns (dz, {})."""
    p = cache["p"]
    check_float64(dp, names=["dp"])
    row_term = np.sum(dp * p, axis=-1, keepdims=True)
    dz = p * (dp - row_term)
    return dz, {}


# --------------------------------------------------------------------------
# GELU (tanh approximation)
# --------------------------------------------------------------------------

def gelu_forward(x, params=None):
    """y = 0.5*x*(1+tanh(c*(x + a*x**3))). Caches x and t = tanh(u)."""
    check_float64(x, names=["x"])
    u = GELU_C * (x + GELU_A * x ** 3)
    t = np.tanh(u)
    y = 0.5 * x * (1.0 + t)
    cache = {"x": x, "t": t}
    return y, cache


def gelu_backward(dy, cache):
    """dy/dx = 0.5(1+t) + 0.5*x*(1-t**2)*c*(1+3*a*x**2). Returns (dx, {})."""
    x = cache["x"]
    t = cache["t"]
    check_float64(dy, names=["dy"])
    dydx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * GELU_C * (1.0 + 3.0 * GELU_A * x ** 2)
    dx = dy * dydx
    return dx, {}


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------

class Linear:
    """out = x @ W + b, x: (..., d_in), W: (d_in, d_out), b: (d_out,).

    Parameter keys: '{prefix}.W' and, iff bias, '{prefix}.b'. With
    bias=False neither params nor d_params ever contain a '.b' key.
    """

    def __init__(self, prefix, d_in, d_out, bias=LINEAR_HAS_BIAS):
        self.prefix = prefix
        self.d_in = d_in
        self.d_out = d_out
        self.bias = bias

    def init_params(self, rng):
        W = rng.normal(0.0, INIT_STD, size=(self.d_in, self.d_out)).astype(DTYPE)
        params = {f"{self.prefix}.W": W}
        if self.bias:
            b = np.full((self.d_out,), BIAS_INIT, dtype=DTYPE)
            params[f"{self.prefix}.b"] = b
        return params

    def forward(self, x, params):
        W = params[f"{self.prefix}.W"]
        names = [f"{self.prefix}.W", "x"]
        arrs = [W, x]
        if self.bias:
            b = params[f"{self.prefix}.b"]
            arrs.append(b)
            names.append(f"{self.prefix}.b")
        check_float64(*arrs, names=names)
        out = x @ W
        if self.bias:
            out = out + params[f"{self.prefix}.b"]
        cache = {"x": x, "W": W}
        return out, cache

    def backward(self, d_out, cache):
        x = cache["x"]
        W = cache["W"]
        check_float64(d_out, names=["d_out"])
        x2 = x.reshape(-1, self.d_in)
        g2 = d_out.reshape(-1, self.d_out)
        dx = d_out @ W.T
        dW = x2.T @ g2
        d_params = {f"{self.prefix}.W": dW}
        if self.bias:
            d_params[f"{self.prefix}.b"] = g2.sum(axis=0)
        return dx, d_params


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

class Embedding:
    """out = W[idx], W: (num_embeddings, d). Positional embeddings reuse this
    class with idx = np.arange(T). Parameter key: '{prefix}.W'.
    """

    def __init__(self, prefix, num_embeddings, d):
        self.prefix = prefix
        self.num_embeddings = num_embeddings
        self.d = d

    def init_params(self, rng):
        W = rng.normal(0.0, INIT_STD, size=(self.num_embeddings, self.d)).astype(DTYPE)
        return {f"{self.prefix}.W": W}

    def forward(self, idx, params):
        W = params[f"{self.prefix}.W"]
        check_float64(W, names=[f"{self.prefix}.W"])
        idx = np.asarray(idx)
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"idx must have an integer dtype, got {idx.dtype!r}")
        if idx.size > 0 and (int(idx.min()) < 0 or int(idx.max()) >= self.num_embeddings):
            raise ValueError(
                f"idx out of range [0, {self.num_embeddings}): "
                f"min={int(idx.min())}, max={int(idx.max())}"
            )
        out = W[idx]
        cache = {"idx": idx, "W_shape": W.shape}
        return out, cache

    def backward(self, d_out, cache):
        idx = cache["idx"]
        W_shape = cache["W_shape"]
        check_float64(d_out, names=["d_out"])
        dW = np.zeros(W_shape, dtype=DTYPE)
        # Never use dW[idx] += d_out: fancy-index += does not accumulate
        # duplicate indices (last write wins). np.add.at does.
        np.add.at(dW, idx, d_out)
        return None, {f"{self.prefix}.W": dW}


# --------------------------------------------------------------------------
# Cross-module gradient accumulation
# --------------------------------------------------------------------------

def accumulate_grads(grads, d_params):
    """grads[k] = v.copy() on first insert, else grads[k] += v (in place).

    Never rebinds an existing entry, so the accumulator array keeps its
    identity across repeated accumulations (needed for Phase 3's tied
    wte/lm_head gradient, C7).
    """
    for k, v in d_params.items():
        if k not in grads:
            grads[k] = v.copy()
        else:
            grads[k] += v
    return grads

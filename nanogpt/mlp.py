"""nanogpt.mlp -- position-wise MLP (Linear -> GELU-tanh -> Linear).

See SPRINT_3_PLAN.md section 3.2. The hidden width is MLP_HIDDEN_MULT * d
(frozen at 4 by the Phase 0 conventions). Both linears carry a bias
(LINEAR_HAS_BIAS). No GELU or Linear arithmetic is re-implemented here: both
sub-maps are composed strictly from `nanogpt.ops.Linear` and
`ops.gelu_forward` / `ops.gelu_backward`.

Module contract (unchanged): forward(x, params) -> (out, cache);
backward(d_out, cache) -> (d_x, d_params); d_params is a fresh dict keyed by
this module's own global parameter names; nothing mutates params/x/d_out.

Forward:
    h, c_fc   = fc.forward(x, params)        # (..., d) -> (..., 4d)
    a, c_gelu = ops.gelu_forward(h)
    out, c_proj = proj.forward(a, params)    # (..., 4d) -> (..., d)

Backward (hand-derived; restated so nobody re-derives it differently):
    da, g_proj = proj.backward(d_out, c_proj)
    dh, _      = ops.gelu_backward(da, c_gelu)
    dx, g_fc   = fc.backward(dh, c_fc)
    d_params   = {**g_fc, **g_proj}           # exactly the 4 keys

Owner: Engineer A. Files: nanogpt/mlp.py, tests/test_mlp.py.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import LINEAR_HAS_BIAS, MLP_HIDDEN_MULT, check_float64
from nanogpt import ops


class MLP:
    """out = GELU_tanh(x @ W_fc + b_fc) @ W_proj + b_proj.

    Parameter keys (R15 ruling: GPT-2 naming, not W1/b1/W2/b2):
    '{prefix}.fc.W', '{prefix}.fc.b', '{prefix}.proj.W', '{prefix}.proj.b'.
    """

    def __init__(self, prefix: str, d: int):
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"MLP: d must be a positive int, got {d!r}")
        d = int(d)

        self.prefix = prefix
        self.d = d
        self.hidden = MLP_HIDDEN_MULT * d

        self.fc = ops.Linear(f"{prefix}.fc", d, self.hidden, bias=LINEAR_HAS_BIAS)
        self.proj = ops.Linear(f"{prefix}.proj", self.hidden, d, bias=LINEAR_HAS_BIAS)

    def param_keys(self) -> list:
        """[fc.W, fc.b, proj.W, proj.b], each prefixed."""
        return [
            f"{self.fc.prefix}.W", f"{self.fc.prefix}.b",
            f"{self.proj.prefix}.W", f"{self.proj.prefix}.b",
        ]

    def init_params(self, rng) -> dict:
        """fc, then proj, from the SAME rng, in that order."""
        params = {}
        params.update(self.fc.init_params(rng))
        params.update(self.proj.init_params(rng))
        return params

    def forward(self, x: np.ndarray, params: dict):
        x = np.asarray(x)
        check_float64(x, names=["x"])
        if x.shape[-1] != self.d:
            raise ValueError(
                f"MLP.forward: expected last axis {self.d}, got {x.shape[-1]}"
            )
        h, c_fc = self.fc.forward(x, params)
        a, c_gelu = ops.gelu_forward(h)
        out, c_proj = self.proj.forward(a, params)

        cache = {"c_fc": c_fc, "c_gelu": c_gelu, "c_proj": c_proj}
        return out, cache

    def backward(self, d_out: np.ndarray, cache: dict):
        d_out = np.asarray(d_out)
        check_float64(d_out, names=["d_out"])

        c_fc, c_gelu, c_proj = cache["c_fc"], cache["c_gelu"], cache["c_proj"]

        da, g_proj = self.proj.backward(d_out, c_proj)
        dh, _ = ops.gelu_backward(da, c_gelu)
        dx, g_fc = self.fc.backward(dh, c_fc)

        d_params = {**g_fc, **g_proj}
        return dx, d_params

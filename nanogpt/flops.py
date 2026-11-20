"""nanogpt.flops -- instrumented matmul FLOP counter (Phase 4, Engineer B).

SPRINT_4_PLAN.md TL-27..TL-30 and section 3.4.

The counter works by swapping the single production matmul chokepoint,
``nanogpt.ops.matmul`` (TL-27), for a counting wrapper while a
``FlopCounter`` context is active. Every production ``@`` in ``nanogpt/``
goes through that one function (enforced by the AST guard in
tests/test_flops.py), so nothing can be missed.

Prototype finding (TL-27): an "ndarray subclass + __array_ufunc__" counter is
UNSOUND here. ``LayerNorm.forward`` and ``cross_entropy_forward`` both call
``np.asarray(...)``, which strips the subclass, so a downstream plain @ plain
matmul (e.g. ``g_out = d_logits.T @ y``) goes silently uncounted
(7,040 FLOPs/token missing at the default config). It is therefore NOT
re-implemented here.

Only matmuls are counted (TL-28): 2*m*k*n per (broadcast) batch element.
Bias adds, softmax, LayerNorm, GELU, the optimizer and embedding gathers are
not matmuls and contribute nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from nanogpt import ops

_VALID_KINDS = ("dense", "attention")
_VALID_PHASES = ("forward", "backward")


# ---------------------------------------------------------------------------
# TL-28: FLOPs of one matmul call
# ---------------------------------------------------------------------------
def matmul_flops(a_shape, b_shape) -> int:
    """FLOPs of ``a @ b`` for operand shapes (..., m, k) and (..., k, n).

    count = 2 * prod(broadcast(batch_a, batch_b)) * m * k * n. A 1-D operand
    follows NumPy promotion (m = 1 for a 1-D ``a``, n = 1 for a 1-D ``b``).
    A mismatched contraction dimension raises ValueError.
    """
    a_shape = tuple(int(s) for s in a_shape)
    b_shape = tuple(int(s) for s in b_shape)
    if len(a_shape) == 0 or len(b_shape) == 0:
        raise ValueError("matmul_flops: scalar operands are not matmuls")

    a_1d = len(a_shape) == 1
    b_1d = len(b_shape) == 1
    if a_1d:
        a_shape = (1,) + a_shape
    if b_1d:
        b_shape = b_shape + (1,)

    m, k_a = a_shape[-2], a_shape[-1]
    k_b, n = b_shape[-2], b_shape[-1]
    if k_a != k_b:
        raise ValueError(
            f"matmul_flops: contraction mismatch, a has k={k_a} but b has k={k_b}"
        )
    batch = np.broadcast_shapes(a_shape[:-2], b_shape[:-2])
    nb = 1
    for s in batch:
        nb *= int(s)
    return 2 * nb * int(m) * int(k_a) * int(n)


# ---------------------------------------------------------------------------
# The counter (monkeypatches the single chokepoint)
# ---------------------------------------------------------------------------
class FlopCounter:
    """Context manager that counts every ``ops.matmul`` call.

    ``counts[(phase, kind)]`` holds FLOPs; ``calls`` holds
    ``(phase, kind, a.shape, b.shape)`` for diagnostics. Calls made outside
    a ``phase(...)`` block are attributed to ``"unphased"``. A second active
    counter raises RuntimeError. ``__exit__`` always restores the original
    ``ops.matmul`` object (also when the body raised).
    """

    _active = None  # class-level: at most one live counter

    def __init__(self):
        self.counts: dict = {}
        self.calls: list = []
        self._phase = "unphased"
        self._orig = None
        self._entered = False

    def __enter__(self) -> "FlopCounter":
        if FlopCounter._active is not None:
            raise RuntimeError("FlopCounter: a counter is already active (nesting is forbidden)")
        FlopCounter._active = self
        self._orig = ops.matmul
        orig = self._orig
        counter = self

        def counting_matmul(a, b, *, kind="dense"):
            out = orig(a, b, kind=kind)  # validates kind, returns exactly a @ b
            counter._record(kind, np.shape(a), np.shape(b))
            return out

        ops.matmul = counting_matmul
        self._entered = True
        return self

    def __exit__(self, *exc) -> None:
        if self._entered:
            ops.matmul = self._orig
            self._entered = False
        if FlopCounter._active is self:
            FlopCounter._active = None
        return None

    def _record(self, kind, a_shape, b_shape) -> None:
        f = matmul_flops(a_shape, b_shape)
        key = (self._phase, kind)
        self.counts[key] = self.counts.get(key, 0) + f
        self.calls.append((self._phase, kind, tuple(a_shape), tuple(b_shape)))

    @contextmanager
    def phase(self, name: str):
        if name not in _VALID_PHASES:
            raise ValueError(f"FlopCounter.phase: name must be one of {_VALID_PHASES}, got {name!r}")
        prev = self._phase
        self._phase = name
        try:
            yield self
        finally:
            self._phase = prev

    def total(self, phase=None, kind=None) -> int:
        return sum(
            v
            for (p, k), v in self.counts.items()
            if (phase is None or p == phase) and (kind is None or k == kind)
        )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlopReport:
    tokens: int
    fwd_dense: int
    fwd_attn: int
    bwd_dense: int
    bwd_attn: int

    @property
    def fwd_per_token(self) -> float:
        return (self.fwd_dense + self.fwd_attn) / self.tokens

    @property
    def train_per_token(self) -> float:
        return (self.fwd_dense + self.fwd_attn + self.bwd_dense + self.bwd_attn) / self.tokens

    @property
    def attn_share_fwd(self) -> float:
        return self.fwd_attn / (self.fwd_dense + self.fwd_attn)


def measure_train_flops(model, params, idx, targets) -> FlopReport:
    """One forward_loss under phase('forward') and one backward_loss(1.0)
    under phase('backward'), counted at the chokepoint. ``tokens`` is
    ``idx.size`` (= B * T)."""
    idx = np.asarray(idx)
    with FlopCounter() as fc:
        with fc.phase("forward"):
            _, cache = model.forward_loss(idx, targets, params)
        with fc.phase("backward"):
            model.backward_loss(1.0, cache)
    return FlopReport(
        tokens=int(idx.size),
        fwd_dense=fc.total("forward", "dense"),
        fwd_attn=fc.total("forward", "attention"),
        bwd_dense=fc.total("backward", "dense"),
        bwd_attn=fc.total("backward", "attention"),
    )


# ---------------------------------------------------------------------------
# TL-29: exact matmul inventory (the counter's own closed-form anchor)
# ---------------------------------------------------------------------------
def exact_forward_matmul_per_token(cfg) -> dict:
    """Exact per-token forward matmul FLOPs for sequence length cfg.T with the
    full T x T score matrix (masking happens after the matmul):

        dense     = 24*L*d^2 + 2*V*d   (q/k/v/o: 8d^2, MLP: 16d^2, per layer; tied head 2Vd)
        attention = 4*L*T*d            (QK^T and PV)
    """
    L, T, d, V = cfg.L, cfg.T, cfg.d, cfg.V
    return {"dense": 24 * L * d * d + 2 * V * d, "attention": 4 * L * T * d}

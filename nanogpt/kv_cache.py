"""nanogpt.kv_cache -- pre-allocated per-layer K/V cache for incremental decoding.

SPRINT_5_PLAN.md TL-35 (layout and write discipline) and section 3.1.

Layout: float64 buffers ``K`` and ``V`` of shape ``(L, B, H, T, head_dim)``,
allocated once from a ``ModelConfig`` and a batch size ``B``. ``length`` is
the number of fully-written positions. One decoding step is:

    for layer in 0 .. L-1:  kh_all, vh_all = cache.append(layer, kh_step, vh_step)
    cache.advance()

``append`` writes the step at index ``length`` and returns VIEWS covering
``[0, length]`` inclusive, shape ``(B, H, length + 1, head_dim)``. Nothing is
concatenated or reallocated per step.

Write discipline (enforced with ValueError, so stale-cache and layer mix-up bugs
are caught by construction): within one step every layer must be appended
exactly once, in order 0..L-1; ``advance`` is only legal after all L layers
have appended. The hard capacity is ``cfg.T``. There is deliberately NO
sliding window: a window would shift absolute position indices and invalidate
the C8 equivalence the spec requires.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import DTYPE, ModelConfig, check_float64


class KVCache:
    def __init__(self, cfg: ModelConfig, B: int):
        if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or B <= 0:
            raise ValueError(f"KVCache: B must be a positive int, got {B!r}")
        self.cfg = cfg
        self.B = int(B)
        shape = (cfg.L, self.B, cfg.H, cfg.T, cfg.head_dim)
        self.K = np.zeros(shape, dtype=DTYPE)
        self.V = np.zeros(shape, dtype=DTYPE)
        self.length = 0
        self._next_layer = 0  # which layer must be appended next within this step

    @property
    def capacity(self) -> int:
        return self.cfg.T

    def _check_step(self, name: str, a) -> np.ndarray:
        a = np.asarray(a)
        check_float64(a, names=[name])
        want = (self.B, self.cfg.H, 1, self.cfg.head_dim)
        if a.shape != want:
            raise ValueError(f"KVCache.append: {name} has shape {a.shape}, expected {want}")
        return a

    def append(self, layer: int, kh_step, vh_step):
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            raise ValueError(f"KVCache.append: layer must be an int, got {layer!r}")
        if not (0 <= layer < self.cfg.L):
            raise ValueError(f"KVCache.append: layer {layer} out of range [0, {self.cfg.L})")
        if self.length >= self.capacity:
            raise ValueError(
                f"KVCache.append: cache is full (length == capacity == {self.capacity}); "
                "there is no sliding window (it would invalidate C8)"
            )
        if layer != self._next_layer:
            raise ValueError(
                f"KVCache.append: expected layer {self._next_layer} next, got {layer} "
                "(each layer exactly once per step, in order)"
            )
        kh_step = self._check_step("kh_step", kh_step)
        vh_step = self._check_step("vh_step", vh_step)

        pos = self.length
        self.K[layer, :, :, pos : pos + 1, :] = kh_step
        self.V[layer, :, :, pos : pos + 1, :] = vh_step
        self._next_layer += 1
        return self.K[layer, :, :, : pos + 1, :], self.V[layer, :, :, : pos + 1, :]

    def advance(self) -> None:
        if self._next_layer != self.cfg.L:
            raise ValueError(
                f"KVCache.advance: only {self._next_layer} of {self.cfg.L} layers appended this step"
            )
        self.length += 1
        self._next_layer = 0

    def reset(self) -> None:
        """Forget all positions. Buffers are not cleared: only [0, length) is ever read."""
        self.length = 0
        self._next_layer = 0

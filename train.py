"""train.py -- Engineer C, Sprint 4 (Phase 4): batching, AdamW training loop,
and a CLI with profiling.

See SPRINT_4_PLAN.md section 3.5 (frozen interface) and section 7 (scope).

R24 (ruling): this file lives at the REPO ROOT, not inside `nanogpt/`, per
the execution-plan layout. The root `conftest.py` puts the repo root on
`sys.path`, so `import train` resolves under `pytest tests/` with no
installed package or PYTHONPATH hack. Because this file is outside
`nanogpt/`, it is not scanned by `tests/test_conventions.py` (that guard
only walks `nanogpt/*.py`), but it still imports every numeric constant it
needs from `nanogpt.config` rather than re-deriving them, as a matter of
style, not because a guard enforces it here.

R22 (ruling, out of scope this sprint): no checkpointing, no save/load, no
sampling, no generation, and no KV-cache. Loss is logged for visibility but
training loss going down is NEVER asserted anywhere (constant LR only; no
schedule).

Milestone layout (one commit each, in this order):
  [C1] module constants + `get_batch` (THIS milestone; no Sprint 4 deps).
  [C2] `train()`, `TrainResult`, `main()` with the CLI and `--profile`
       (requires nanogpt/optim.py's AdamW, [A2], to be committed).
  [C3] no new production code -- loop-invariant TESTS only.
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import INT_DTYPE

# ---------------------------------------------------------------------------
# Module constants (SPRINT_4_PLAN.md section 3.5). These are `train.py`'s own
# run defaults, not section-0 Frozen Conventions, so they live here rather
# than in `nanogpt/config.py` (TL-31).
# ---------------------------------------------------------------------------
DEFAULT_STEPS = 4000
DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
DEFAULT_LOG_EVERY = 100


def get_batch(data: np.ndarray, batch_size: int, T: int, rng: np.random.Generator):
    """Sample `batch_size` random contiguous (x, y) windows of length T from
    a 1-D token-id array `data`, for next-token prediction.

    Start indices i ~ rng.integers(0, len(data) - T, size=batch_size), so
    every i satisfies 0 <= i <= len(data) - T - 1 and i + T <= len(data) - 1
    (the extreme start len(data) - T - 1 is reachable and valid: then y's
    last index is exactly len(data) - 1).

    x[b] = data[i : i + T]
    y[b] = data[i + 1 : i + T + 1]

    Both returned arrays have shape (batch_size, T) and dtype INT_DTYPE
    (int64). `len(data) <= T` raises ValueError (there would be no valid
    start index at all).
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError(f"get_batch: data must be 1-D, got ndim={data.ndim}")
    n = len(data)
    if n <= T:
        raise ValueError(
            f"get_batch: len(data)={n} must be greater than T={T} "
            "(no valid start index otherwise)"
        )

    starts = rng.integers(0, n - T, size=batch_size)
    x = np.empty((batch_size, T), dtype=INT_DTYPE)
    y = np.empty((batch_size, T), dtype=INT_DTYPE)
    for b, i in enumerate(starts):
        i = int(i)
        x[b] = data[i : i + T]
        y[b] = data[i + 1 : i + T + 1]
    return x, y

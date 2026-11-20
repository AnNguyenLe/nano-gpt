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

R22 (ruling; amended by TL-40 in Sprint 5): no sampling, no generation and no
KV-cache here. The only checkpoint hook is the optional `--save PATH` flag, which
calls `nanogpt.checkpoint.save` after training (so train.py -> generate.py works). Loss is logged for visibility but
training loss going down is NEVER asserted anywhere (constant LR only; no
schedule).

Milestone layout (one commit each, in this order):
  [C1] module constants + `get_batch` (THIS milestone; no Sprint 4 deps).
  [C2] `train()`, `TrainResult`, `main()` with the CLI and `--profile`
       (requires nanogpt/optim.py's AdamW, [A2], to be committed).
  [C3] no new production code -- loop-invariant TESTS only.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats
import time
from dataclasses import dataclass, field

import numpy as np

from nanogpt import checkpoint
from nanogpt.config import (
    ADAMW_BETAS,
    ADAMW_WEIGHT_DECAY,
    DEFAULT_SEED,
    INT_DTYPE,
    ModelConfig,
    make_rng,
)
from nanogpt.model import GPT
from nanogpt.optim import AdamW
from nanogpt.param_count import formula_from_config
from nanogpt.tokenizer import load_corpus_and_tokenizer

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

# ---------------------------------------------------------------------------
# [C2] Training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    params: dict
    optimizer: AdamW
    model: GPT
    losses: list = field(default_factory=list)
    step_times: list = field(default_factory=list)
    wall_time_s: float = 0.0
    init_loss: float = float("nan")


def train(
    cfg: ModelConfig,
    data: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float = ADAMW_WEIGHT_DECAY,
    betas=ADAMW_BETAS,
    seed: int = DEFAULT_SEED,
    log_every: int = DEFAULT_LOG_EVERY,
    log_fn=print,
    profile_steps: int = 0,
) -> TrainResult:
    """Run `steps` AdamW updates of a fresh GPT(cfg) on `data`.

    model = GPT(cfg); params = model.init_params(make_rng(seed)); batches are
    drawn from make_rng(seed + 1). Each step: get_batch -> forward_loss ->
    backward_loss(1.0) -> AdamW.step. A non-finite loss raises
    FloatingPointError (fail loudly; never skip a step).

    `profile_steps` > 0 additionally cProfiles the first that-many steps and
    logs the top 20 functions by cumulative time via `log_fn`. The loss is
    logged for visibility only; nothing here treats a falling loss as
    evidence of correctness.
    """
    if steps < 1:
        raise ValueError(f"train: steps must be >= 1, got {steps}")
    data = np.asarray(data)
    model = GPT(cfg)
    params = model.init_params(make_rng(seed))
    opt = AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    batch_rng = make_rng(seed + 1)

    result = TrainResult(params=params, optimizer=opt, model=model)
    profiler = cProfile.Profile() if profile_steps > 0 else None
    t_start = time.perf_counter()

    for step in range(steps):
        if profiler is not None and step == 0:
            profiler.enable()
        t0 = time.perf_counter()

        x, y = get_batch(data, batch_size, cfg.T, batch_rng)
        loss, cache = model.forward_loss(x, y, params)
        loss = float(loss)
        if not math.isfinite(loss):
            if profiler is not None:
                profiler.disable()
            raise FloatingPointError(f"train: non-finite loss {loss!r} at step {step}")
        _, grads = model.backward_loss(1.0, cache)
        opt.step(grads)

        dt = time.perf_counter() - t0
        result.losses.append(loss)
        result.step_times.append(dt)
        if step == 0:
            result.init_loss = loss

        if profiler is not None and step == profile_steps - 1:
            profiler.disable()
            buf = io.StringIO()
            pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(20)
            log_fn(f"--- cProfile of first {profile_steps} step(s), top 20 by cumtime ---")
            log_fn(buf.getvalue())
            profiler = None

        if log_every and (step % log_every == 0 or step == steps - 1):
            log_fn(
                f"step {step} | loss {loss:.4f} | {dt * 1000:.1f} ms/step | "
                f"elapsed {time.perf_counter() - t_start:.1f}s"
            )

    result.wall_time_s = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# [C2] CLI
# ---------------------------------------------------------------------------
_RANGES = {"L": (2, 4), "d": (64, 128), "T": (64, 128)}  # spec ranges


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the NumPy nanoGPT on the char corpus.")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=ADAMW_WEIGHT_DECAY)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--allow-out-of-range", action="store_true",
                    help="permit L/d/T outside the spec ranges (L 2-4, d 64-128, T 64-128)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="after training, write a checkpoint (params + AdamW state) to PATH")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="cProfile the first N steps and print the top 20 by cumtime")
    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    if not args.allow_out_of_range:
        for name in ("L", "d", "T"):
            lo, hi = _RANGES[name]
            val = getattr(args, name)
            if not (lo <= val <= hi):
                ap.error(f"--{name}={val} is outside the spec range [{lo}, {hi}]; "
                         "pass --allow-out-of-range to override")

    info, tok = load_corpus_and_tokenizer()
    ids = tok.encode(info.text)
    V = tok.vocab_size
    try:
        cfg = ModelConfig(V=V, T=args.T, d=args.d, L=args.L, H=args.H)
    except ValueError as e:
        ap.error(str(e))

    print(f"corpus: source={info.source} sha256={info.sha256[:8]} chars={info.n_chars} V={V}")
    print(f"config: {cfg} N={formula_from_config(cfg)} | batch={args.batch_size} "
          f"lr={args.lr} wd={args.weight_decay} steps={args.steps} seed={args.seed}")

    res = train(
        cfg, ids,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, log_every=args.log_every,
        profile_steps=args.profile,
    )
    if args.save:
        checkpoint.save(args.save, res.model, res.params, res.optimizer)
        print(f"saved checkpoint to {args.save}")
    n = len(res.step_times)
    print(f"step-0 loss {res.init_loss:.4f} vs ln V {math.log(V):.4f} (consistency print only)")
    print(f"done: {n} steps in {res.wall_time_s:.1f}s "
          f"({res.wall_time_s / n:.3f} s/step), final loss {res.losses[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
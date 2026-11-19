"""tests/test_loss_at_init.py -- Engineer A, Sprint 3 milestone A3 (S3-05, C2).

Covers the Sprint 3 "loss at init" invariant: under the frozen
initialization (std=INIT_STD, biases/beta=0, gamma=1), a freshly
initialized GPT's cross-entropy loss on real corpus windows should be close
to ln(V) -- a uniform-over-the-vocabulary prediction, since nothing has
been trained yet. See SPRINT_3_PLAN.md section 5.3 (C2).

V comes from `load_corpus_and_tokenizer()` (must be 55, the Phase 0 value).
Real corpus windows (B=4) with next-char targets; params from
`GPT.init_params(make_rng(DEFAULT_SEED + s))` for s in {0, 1, 2}. The
default config (V=55, T=64, d=64, L=2, H=4) is mandatory; tiny and max are
also run. TL prototype measured worst deltas <= 0.042 / 0.032 / 0.019 for
default / max / tiny (reported, not hard-gated beyond INIT_LOSS_TOL itself
-- the tolerance is frozen; if the check is ever tight, the fix is the
init, never a loosened tolerance).

Fixture seed (per-file, for window selection only -- never DEFAULT_SEED,
and never used for the init seeds s in {0,1,2} themselves):
`DEFAULT_SEED + 303`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt.config import DEFAULT_SEED, INIT_LOSS_TOL, INIT_STD, ModelConfig, N_HEAD, make_rng, tiny_config
from nanogpt.model import GPT
from nanogpt.tokenizer import load_corpus_and_tokenizer

_FIXTURE_SEED = DEFAULT_SEED + 303
_SEEDS = (0, 1, 2)
_B = 4


@pytest.fixture(scope="module")
def corpus_ids():
    """Encode the real corpus once (ints, shape (n_chars,)); assert V == 55."""
    info, tok = load_corpus_and_tokenizer()
    assert tok.vocab_size == 55, tok.vocab_size
    ids = tok.encode(info.text)
    return ids, tok.vocab_size


def _windows(ids: np.ndarray, T: int, B: int, seed: int):
    """B contiguous (input, next-char-target) windows of length T, at fixed
    (seed-determined) start positions. Same windows are reused across the
    three init seeds s in {0,1,2} for a given config, so deltas are
    comparable across seeds."""
    rng = make_rng(seed)
    n = ids.shape[0]
    starts = rng.integers(0, n - T, size=B)
    idx = np.stack([ids[s:s + T] for s in starts]).astype(np.int64)
    targets = np.stack([ids[s + 1:s + T + 1] for s in starts]).astype(np.int64)
    return idx, targets


_CONFIGS = {
    "default": ModelConfig(V=55, T=64, d=64, L=2, H=N_HEAD),  # mandatory
    "tiny": tiny_config(55),
    "max": ModelConfig(V=55, T=128, d=128, L=4),
}

# Fixed, explicit per-config window-seed offsets (never Python's built-in
# `hash()` on a string: str hashing is randomized per-process by default,
# which would make the chosen corpus windows -- and hence the measured
# deltas -- non-reproducible across runs).
_WINDOW_SEED_OFFSET = {"default": 10, "tiny": 11, "max": 12}


# ---------------------------------------------------------------------------
# C2: |loss_init - ln V| < INIT_LOSS_TOL, for every config and seed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["default", "tiny", "max"])
def test_loss_at_init_close_to_ln_v(corpus_ids, cfg_name):
    ids, V = corpus_ids
    cfg = _CONFIGS[cfg_name]
    idx, targets = _windows(ids, cfg.T, _B, _FIXTURE_SEED + _WINDOW_SEED_OFFSET[cfg_name])
    target_loss = math.log(V)

    worst_delta = 0.0
    for s in _SEEDS:
        model = GPT(cfg)
        params = model.init_params(make_rng(DEFAULT_SEED + s))
        loss, _ = model.forward_loss(idx, targets, params)
        delta = abs(loss - target_loss)
        worst_delta = max(worst_delta, delta)
        print(f"C2 [{cfg_name}] seed={s}: loss={loss:.6f} ln(V)={target_loss:.6f} "
              f"delta={delta:.6f}")
        assert delta < INIT_LOSS_TOL, (cfg_name, s, loss, target_loss, delta)

    print(f"C2 [{cfg_name}] worst delta over seeds {_SEEDS} = {worst_delta:.6f}")


# ---------------------------------------------------------------------------
# Init statistics (frozen conventions): biases/beta exactly 0, gamma exactly
# 1, and every 2-D weight with >= 512 entries has sample std within +-10% of
# INIT_STD and |mean| < 0.1*INIT_STD.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["default", "tiny", "max"])
def test_init_statistics(cfg_name):
    cfg = _CONFIGS[cfg_name]
    model = GPT(cfg)
    params = model.init_params(make_rng(DEFAULT_SEED))

    checked_weights = 0
    for key, arr in params.items():
        if key.endswith(".b") or key.endswith(".beta"):
            assert np.all(arr == 0.0), key
        elif key.endswith(".gamma"):
            assert np.all(arr == 1.0), key
        elif key.endswith(".W"):
            if arr.ndim == 2 and arr.size >= 512:
                std = float(np.std(arr))
                mean = float(np.mean(arr))
                assert 0.9 * INIT_STD <= std <= 1.1 * INIT_STD, (key, std)
                assert abs(mean) < 0.1 * INIT_STD, (key, mean)
                checked_weights += 1
    print(f"[{cfg_name}] checked {checked_weights} weight matrices with >=512 entries")


# ---------------------------------------------------------------------------
# Test-the-test: scaling every 2-D weight to std=1.0 breaks the init-loss
# anchor on the default config (measured +12..+17 by the TL prototype).
# Never loosen INIT_LOSS_TOL to make a real regression pass; fix the init.
# ---------------------------------------------------------------------------
def test_tt_std_one_weights_break_loss_at_init(corpus_ids):
    ids, V = corpus_ids
    cfg = _CONFIGS["default"]
    idx, targets = _windows(ids, cfg.T, _B, _FIXTURE_SEED + 50)
    target_loss = math.log(V)

    model = GPT(cfg)
    rng = make_rng(DEFAULT_SEED)
    params = model.init_params(rng)

    bad_params = dict(params)
    noise_rng = make_rng(_FIXTURE_SEED + 51)
    for key, arr in params.items():
        if key.endswith(".W") and arr.ndim == 2:
            bad_params[key] = noise_rng.standard_normal(arr.shape).astype(np.float64)

    loss, _ = model.forward_loss(idx, targets, bad_params)
    delta = abs(loss - target_loss)
    print(f"std=1.0 adversary: loss={loss:.4f} ln(V)={target_loss:.4f} delta={delta:.4f}")
    assert delta > 1.0, delta

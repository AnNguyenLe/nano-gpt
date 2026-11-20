"""tests/test_train.py -- Engineer C, Sprint 4 (Phase 4), milestones C1-C3.

Covers `train.py` at the repo root: batching (`get_batch`), the training
loop + CLI (`train`, `TrainResult`, `main`), and loop invariants (tie
preservation, single optimizer state, determinism, AdamW-is-really-used,
and the non-functional "no loss-decrease assertion" guard).

R22/R24/TL-31 (SPRINT_4_PLAN.md section 2.2, section 3.5, section 7): this
whole file runs on the TINY config, at most 10 steps and batch size <= 4,
and completes in well under 5 s. No assertion anywhere in this file
compares `losses[i] < losses[j]` (a decreasing loss is never evidence of
correctness -- see the module guard near the bottom of this file).

This file is appended to across three milestones (C1, then C2, then C3),
never rewritten: the C1 section below lands first, with no Sprint 4
dependency.

Fixture seed (per-file, never DEFAULT_SEED itself): `_FIXTURE_SEED =
DEFAULT_SEED + 304`.
"""
from __future__ import annotations

import numpy as np
import pytest

import train as train_module
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, make_rng
from nanogpt.tokenizer import load_corpus_and_tokenizer

_FIXTURE_SEED = DEFAULT_SEED + 304


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_ids():
    """The real corpus, tokenized to int ids (module-scoped: resolving the
    corpus + building the vocab is read-only and deterministic, so reusing
    it across tests costs nothing and keeps the file fast)."""
    info, tok = load_corpus_and_tokenizer()
    ids = tok.encode(info.text)
    return info, tok, ids


# ===========================================================================
# [C1] Batching
# ===========================================================================
def test_get_batch_shapes_dtype_and_y_eq_x_plus_1():
    data = np.arange(100)
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(data, batch_size=5, T=4, rng=rng)

    assert x.shape == (5, 4)
    assert y.shape == (5, 4)
    assert x.dtype == INT_DTYPE
    assert y.dtype == INT_DTYPE
    # On np.arange(100), data[i+1] == data[i] + 1 everywhere, so y == x + 1.
    assert np.array_equal(y, x + 1)


def test_get_batch_every_window_stays_in_range():
    data = np.arange(100)
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(data, batch_size=50, T=7, rng=rng)
    assert x.min() >= data[0]
    assert y.max() <= data[-1]
    assert x.max() <= 99
    assert y.max() <= 99


def test_get_batch_extreme_start_len_minus_T_minus_1_is_valid():
    # len(data) - T - 1 must be a reachable, valid start: then y's last
    # index is exactly len(data) - 1. Verify this analytically (the open
    # interval endpoint of rng.integers(0, n - T)) and also hit it directly
    # with a stand-in rng that always returns that exact start.
    data = np.arange(100)
    T = 4
    n = len(data)
    extreme_start = n - T - 1

    class _FixedRNG:
        def integers(self, low, high, size):
            assert low == 0
            assert high == n - T  # exclusive upper bound: max valid i = n - T - 1
            return np.full(size, extreme_start, dtype=np.int64)

    x, y = train_module.get_batch(data, batch_size=3, T=T, rng=_FixedRNG())
    assert np.array_equal(x[0], data[extreme_start : extreme_start + T])
    assert np.array_equal(y[0], data[extreme_start + 1 : extreme_start + T + 1])
    # y's last index is exactly len(data) - 1, the last valid index of data.
    assert y[0, -1] == data[-1]


def test_get_batch_same_seed_gives_identical_batches():
    data = np.arange(1000)
    x1, y1 = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    x2, y2 = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_get_batch_different_seed_gives_different_batches():
    data = np.arange(10000)
    x1, _ = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    x2, _ = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED + 1))
    assert not np.array_equal(x1, x2)


@pytest.mark.parametrize("T", [10, 11, 20])
def test_get_batch_len_data_le_T_raises(T):
    data = np.arange(10)  # len(data) == 10 <= T for every T above
    with pytest.raises(ValueError):
        train_module.get_batch(data, batch_size=2, T=T, rng=make_rng(_FIXTURE_SEED))


def test_get_batch_on_real_corpus_decodes_to_contiguous_slice(real_ids):
    info, tok, ids = real_ids
    T = 16
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(ids, batch_size=3, T=T, rng=rng)

    for b in range(3):
        decoded_x = tok.decode(x[b])
        # x[b] must be a literal, contiguous slice of the corpus text.
        assert decoded_x in info.text
        start = info.text.index(decoded_x)
        # y[b] is the same slice shifted by one character.
        expected_y = info.text[start + 1 : start + 1 + T]
        assert tok.decode(y[b]) == expected_y

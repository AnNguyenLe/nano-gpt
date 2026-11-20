"""tests/test_kv_cache.py -- Engineer A, Sprint 5 (SPRINT_5_PLAN.md section 6.1).

[A1] (this section): KVCache mechanics and write discipline (TL-35).
[A2] and [A3] sections (cached forward, C8, test-the-tests, FLOP evidence) are
appended in later commits, never rewritten.

Fixture seeds are per-file offsets, never DEFAULT_SEED itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt.config import DEFAULT_SEED, ModelConfig, make_rng, tiny_config
from nanogpt.kv_cache import KVCache

_SEED = DEFAULT_SEED + 601


def _step(cfg, B, rng):
    shape = (B, cfg.H, 1, cfg.head_dim)
    return rng.standard_normal(shape), rng.standard_normal(shape)


def _full_step(cache, rng):
    """Append all layers once and advance; return per-layer step arrays."""
    steps = []
    for layer in range(cache.cfg.L):
        k, v = _step(cache.cfg, cache.B, rng)
        cache.append(layer, k, v)
        steps.append((k, v))
    cache.advance()
    return steps


def test_buffers_have_documented_shape_and_dtype():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=3)
    assert c.K.shape == c.V.shape == (cfg.L, 3, cfg.H, cfg.T, cfg.head_dim)
    assert c.K.dtype == c.V.dtype == np.float64
    assert c.length == 0
    assert c.capacity == cfg.T


@pytest.mark.parametrize("bad_B", [0, -1, 1.5, True, None])
def test_constructor_rejects_bad_batch_size(bad_B):
    with pytest.raises(ValueError):
        KVCache(tiny_config(V=11), B=bad_B)


def test_length_grows_only_on_advance():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=2)
    rng = make_rng(_SEED)
    for layer in range(cfg.L):
        k, v = _step(cfg, 2, rng)
        c.append(layer, k, v)
        assert c.length == 0  # appending alone never advances
    c.advance()
    assert c.length == 1


def test_append_returns_views_covering_zero_to_length_inclusive():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=2)
    rng = make_rng(_SEED + 1)
    for pos in range(3):
        for layer in range(cfg.L):
            k, v = _step(cfg, 2, rng)
            ka, va = c.append(layer, k, v)
            assert ka.shape == va.shape == (2, cfg.H, pos + 1, cfg.head_dim)
            assert np.array_equal(ka[:, :, -1:, :], k)
            assert np.array_equal(va[:, :, -1:, :], v)
            assert np.shares_memory(ka, c.K)  # a view, not a copy
        c.advance()


def test_later_steps_do_not_disturb_earlier_ones():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=2)
    rng = make_rng(_SEED + 2)
    first = _full_step(c, rng)
    snap = c.K[:, :, :, :1, :].copy(), c.V[:, :, :, :1, :].copy()
    _full_step(c, rng)
    _full_step(c, rng)
    assert np.array_equal(c.K[:, :, :, :1, :], snap[0])
    assert np.array_equal(c.V[:, :, :, :1, :], snap[1])
    for layer, (k, v) in enumerate(first):
        assert np.array_equal(c.K[layer, :, :, 0:1, :], k)
        assert np.array_equal(c.V[layer, :, :, 0:1, :], v)


def test_layers_are_stored_independently():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    k0 = np.full((1, cfg.H, 1, cfg.head_dim), 1.0)
    k1 = np.full((1, cfg.H, 1, cfg.head_dim), 2.0)
    c.append(0, k0, k0)
    c.append(1, k1, k1)
    c.advance()
    assert np.all(c.K[0, :, :, 0, :] == 1.0) and np.all(c.K[1, :, :, 0, :] == 2.0)


def test_out_of_order_layer_raises():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    k, v = _step(cfg, 1, make_rng(_SEED + 3))
    with pytest.raises(ValueError):
        c.append(1, k, v)  # must start at layer 0


def test_duplicate_layer_in_one_step_raises():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    k, v = _step(cfg, 1, make_rng(_SEED + 4))
    c.append(0, k, v)
    with pytest.raises(ValueError):
        c.append(0, k, v)


def test_advance_before_all_layers_appended_raises():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    with pytest.raises(ValueError):
        c.advance()  # nothing appended
    k, v = _step(cfg, 1, make_rng(_SEED + 5))
    c.append(0, k, v)
    with pytest.raises(ValueError):
        c.advance()  # only 1 of L layers
    assert c.length == 0


def test_layer_index_validation():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    k, v = _step(cfg, 1, make_rng(_SEED + 6))
    for bad in (-1, cfg.L, True, 0.0, None):
        with pytest.raises(ValueError):
            c.append(bad, k, v)


def test_step_shape_and_dtype_validation():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=2)
    k, v = _step(cfg, 2, make_rng(_SEED + 7))
    with pytest.raises(ValueError):
        c.append(0, k[:1], v)  # wrong batch
    with pytest.raises(ValueError):
        c.append(0, np.concatenate([k, k], axis=2), v)  # 2 positions, not 1
    with pytest.raises((ValueError, TypeError)):
        c.append(0, k.astype(np.float32), v)
    assert c._next_layer == 0  # failed appends must not advance the layer cursor


def test_capacity_is_a_hard_ceiling_no_sliding_window():
    cfg = tiny_config(V=11)  # T = 4
    c = KVCache(cfg, B=1)
    rng = make_rng(_SEED + 8)
    for _ in range(cfg.T):
        _full_step(c, rng)
    assert c.length == cfg.T
    k, v = _step(cfg, 1, rng)
    with pytest.raises(ValueError, match="full"):
        c.append(0, k, v)
    assert c.length == cfg.T  # unchanged: nothing was slid out


def test_reset_restarts_from_position_zero():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    rng = make_rng(_SEED + 9)
    _full_step(c, rng)
    _full_step(c, rng)
    c.reset()
    assert c.length == 0
    k, v = _step(cfg, 1, rng)
    ka, va = c.append(0, k, v)
    assert ka.shape[2] == 1 and np.array_equal(ka, k)  # only the new step is visible


def test_reset_mid_step_clears_layer_cursor():
    cfg = tiny_config(V=11)
    c = KVCache(cfg, B=1)
    k, v = _step(cfg, 1, make_rng(_SEED + 10))
    c.append(0, k, v)
    c.reset()
    c.append(0, k, v)  # legal again from layer 0


def test_default_config_cache_dimensions():
    cfg = ModelConfig(V=55)
    c = KVCache(cfg, B=1)
    assert c.K.shape == (2, 1, 4, 64, 16)

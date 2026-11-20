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


# ===========================================================================
# [A2] Cached forward (TL-32/33/36): single-step equivalence + validation
# ===========================================================================
from nanogpt.attention import CausalSelfAttention  # noqa: E402
from nanogpt.block import Block  # noqa: E402
from nanogpt.config import ROUNDOFF_ATOL  # noqa: E402
from nanogpt.model import GPT  # noqa: E402


def _model(cfg, seed_off=0):
    m = GPT(cfg)
    # Non-trivial params: default init has std 0.02 which makes logits tiny, so
    # scale up to make attention/MLP contribute visibly to the comparison.
    p = m.init_params(make_rng(_SEED + 100 + seed_off))
    rng = make_rng(_SEED + 200 + seed_off)
    for k in m.param_keys():
        if not (k.endswith(".gamma") or k.endswith(".beta")):
            p[k] += 0.3 * rng.standard_normal(p[k].shape) if k != "wte.W" else 0.0
    p["wte.W"] += 0.3 * rng.standard_normal(p["wte.W"].shape)
    return m, p


def test_attention_forward_step_without_history_equals_forward_on_one_token():
    cfg = tiny_config(V=11)
    attn = CausalSelfAttention("attn", cfg.d, cfg.H)
    p = attn.init_params(make_rng(_SEED + 11))
    x = make_rng(_SEED + 12).standard_normal((2, 1, cfg.d))
    full, _ = attn.forward(x, p)
    step = attn.forward_step(x, p, kv=None)
    assert np.max(np.abs(full - step)) <= ROUNDOFF_ATOL


def test_block_forward_step_without_history_equals_forward_on_one_token():
    cfg = tiny_config(V=11)
    blk = Block("h.0", cfg.d, cfg.H)
    p = blk.init_params(make_rng(_SEED + 13))
    x = make_rng(_SEED + 14).standard_normal((2, 1, cfg.d))
    full, _ = blk.forward(x, p)
    step = blk.forward_step(x, p, kv=None)
    assert np.max(np.abs(full - step)) <= ROUNDOFF_ATOL


@pytest.mark.parametrize("B", [1, 3])
def test_model_forward_step_first_token_equals_forward_on_one_token(B):
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 15).integers(0, cfg.V, size=(B, 1))
    full, _ = m.forward(idx, p)
    kv = KVCache(cfg, B)
    step = m.forward_step(idx, p, kv)
    assert step.shape == (B, 1, cfg.V)
    assert np.max(np.abs(full - step)) <= ROUNDOFF_ATOL
    assert kv.length == 1  # advanced exactly once per token, not once per layer


def test_forward_step_advances_length_once_per_token():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    kv = KVCache(cfg, 1)
    idx = make_rng(_SEED + 16).integers(0, cfg.V, size=(1, 3))
    for t in range(3):
        m.forward_step(idx[:, t : t + 1], p, kv)
        assert kv.length == t + 1


def test_prefill_stacks_logits_and_matches_stepwise():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 17).integers(0, cfg.V, size=(2, 3))
    kv_a = KVCache(cfg, 2)
    out = m.prefill(idx, p, kv_a)
    assert out.shape == (2, 3, cfg.V) and kv_a.length == 3
    kv_b = KVCache(cfg, 2)
    for t in range(3):
        step = m.forward_step(idx[:, t : t + 1], p, kv_b)
        assert np.array_equal(out[:, t : t + 1], step)


def test_forward_step_validation():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    kv = KVCache(cfg, 1)
    with pytest.raises(ValueError):
        m.forward_step(np.zeros((1, 2), dtype=np.int64), p, kv)  # two tokens
    with pytest.raises(TypeError):
        m.forward_step(np.zeros((1, 1)), p, kv)  # float ids
    attn = CausalSelfAttention("attn", cfg.d, cfg.H)
    ap = attn.init_params(make_rng(_SEED + 18))
    with pytest.raises(ValueError):
        attn.forward_step(np.zeros((1, 2, cfg.d)), ap)  # not a single step
    with pytest.raises(ValueError):
        attn.forward_step(np.zeros((1, 1, cfg.d + 1)), ap)


def test_forward_step_beyond_context_length_raises():
    cfg = tiny_config(V=11)  # T = 4
    m, p = _model(cfg)
    kv = KVCache(cfg, 1)
    idx = make_rng(_SEED + 19).integers(0, cfg.V, size=(1, cfg.T))
    m.prefill(idx, p, kv)
    assert kv.length == cfg.T
    with pytest.raises(ValueError):
        m.forward_step(idx[:, :1], p, kv)


def test_forward_step_does_not_mutate_params():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    before = {k: v.copy() for k, v in p.items()}
    kv = KVCache(cfg, 1)
    m.prefill(make_rng(_SEED + 20).integers(0, cfg.V, size=(1, 3)), p, kv)
    for k in before:
        assert np.array_equal(before[k], p[k]), k


# ===========================================================================
# [A3] C8 (KV-cache equivalence), test-the-tests, FLOP evidence
# ===========================================================================
from nanogpt import flops  # noqa: E402
from nanogpt.attention_mask import shifted_causal_mask  # noqa: E402
from nanogpt.config import KV_CACHE_ABS_TOL  # noqa: E402

_C8_CONFIGS = {"tiny": lambda: tiny_config(V=11), "default": lambda: ModelConfig(V=55)}


def _teacher_forced_cached_logits(m, p, idx, T0):
    """Prefill idx[:, :T0], then decode idx[:, T0:] one token at a time,
    TEACHER-FORCED (the fed token is always the fixed sequence's, never a sample)."""
    cfg = m.cfg
    kv = KVCache(cfg, idx.shape[0])
    parts = [m.prefill(idx[:, :T0], p, kv)]
    for t in range(T0, idx.shape[1]):
        parts.append(m.forward_step(idx[:, t : t + 1], p, kv))
    return np.concatenate(parts, axis=1), kv


@pytest.mark.parametrize("name", list(_C8_CONFIGS))
@pytest.mark.parametrize("T0", [1, 2])
def test_c8_cached_logits_match_full_forward_to_1e_minus_10_absolute(name, T0, capsys):
    cfg = _C8_CONFIGS[name]()
    m, p = _model(cfg)
    # sequence length == cfg.T: the boundary of C8's "length <= T" precondition
    idx = make_rng(_SEED + 300 + T0).integers(0, cfg.V, size=(2, cfg.T))
    full, _ = m.forward(idx, p)
    cached, kv = _teacher_forced_cached_logits(m, p, idx, T0)
    assert kv.length == cfg.T
    assert cached.shape == full.shape == (2, cfg.T, cfg.V)
    max_abs = float(np.max(np.abs(cached - full)))
    print(f"C8 {name} T0={T0}: max abs diff = {max_abs:.3e}")
    assert max_abs <= KV_CACHE_ABS_TOL  # ABSOLUTE, as the spec states it


def test_c8_shorter_than_context_length_also_matches():
    cfg = ModelConfig(V=55)  # T = 64; use a length-20 sequence
    m, p = _model(cfg)
    idx = make_rng(_SEED + 310).integers(0, cfg.V, size=(1, 20))
    full, _ = m.forward(idx, p)
    cached, _ = _teacher_forced_cached_logits(m, p, idx, 5)
    assert np.max(np.abs(cached - full)) <= KV_CACHE_ABS_TOL


def test_c8_logits_are_not_trivially_small_so_the_tolerance_is_meaningful():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 311).integers(0, cfg.V, size=(2, cfg.T))
    full, _ = m.forward(idx, p)
    assert np.max(np.abs(full)) > 1.0  # 1e-10 absolute is a real constraint on O(1) logits


def test_prefix_logits_do_not_depend_on_later_tokens():
    # Cached logits at position i must equal the full forward of just the prefix
    # [0..i]: nothing after i can influence them (the cache cannot see the future).
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 312).integers(0, cfg.V, size=(1, cfg.T))
    cached, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    for i in range(cfg.T):
        prefix_full, _ = m.forward(idx[:, : i + 1], p)
        assert np.max(np.abs(cached[:, i] - prefix_full[:, i])) <= KV_CACHE_ABS_TOL


# ---------------------------------------------------------------------------
# Test-the-tests: each broken variant must be CAUGHT by the C8 comparison
# ---------------------------------------------------------------------------
def test_shifted_mask_makes_cached_and_full_disagree_TL34():
    # Under j <= i+1 the full forward leaks one future token; the cache cannot,
    # because position i+1 does not exist yet when i is decoded. So the two paths
    # legitimately DISAGREE, and C8's 1e-10 claim is made for causal_mask only.
    cfg = tiny_config(V=11)
    m_shift = GPT(cfg, mask_fn=shifted_causal_mask)
    _, p = _model(cfg)
    idx = make_rng(_SEED + 320).integers(0, cfg.V, size=(2, cfg.T))
    full_shift, _ = m_shift.forward(idx, p)
    cached_shift, _ = _teacher_forced_cached_logits(m_shift, p, idx, 1)
    gap = float(np.max(np.abs(cached_shift - full_shift)))
    assert gap > 1e-3

    # ...and the cached path under the shifted mask equals the cached CAUSAL path:
    # mask_fn(pos+1)[pos] is all-True for both, since no key beyond pos exists.
    m_causal = GPT(cfg)
    cached_causal, _ = _teacher_forced_cached_logits(m_causal, p, idx, 1)
    assert np.max(np.abs(cached_shift - cached_causal)) <= KV_CACHE_ABS_TOL


class _StaleKV(KVCache):
    """Adversary: a cache whose advance() never moves the write cursor."""

    def advance(self):
        self._next_layer = 0  # reset the layer cursor but forget to bump `length`


def test_stale_cache_is_caught():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 321).integers(0, cfg.V, size=(1, cfg.T))
    full, _ = m.forward(idx, p)
    kv = _StaleKV(cfg, 1)
    outs = [m.forward_step(idx[:, t : t + 1], p, kv) for t in range(cfg.T)]
    stale = np.concatenate(outs, axis=1)
    assert np.max(np.abs(stale - full)) > 1e-3


def test_wrong_position_embedding_is_caught():
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 322).integers(0, cfg.V, size=(1, cfg.T - 1))
    full, _ = m.forward(idx, p)

    real_forward = m.wpe.forward

    def off_by_one(pos_idx, params):
        return real_forward(np.asarray(pos_idx) + 1, params)  # pos+1: stays < T here

    m.wpe.forward = off_by_one
    try:
        kv = KVCache(cfg, 1)
        outs = [m.forward_step(idx[:, t : t + 1], p, kv) for t in range(cfg.T - 1)]
    finally:
        m.wpe.forward = real_forward
    bad = np.concatenate(outs, axis=1)
    assert np.max(np.abs(bad - full)) > 1e-3


def test_cached_path_ignoring_its_mask_row_is_indistinguishable_under_causal_mask():
    # HONEST RECORD (SPRINT_5_PLAN.md section 6.1 / A3 item 4): under causal_mask the last
    # row of mask_fn(pos+1) is all-True, so a cached path that ignores the mask
    # entirely produces BIT-IDENTICAL logits. C8 therefore cannot detect that bug
    # today. TL-33 ("honour mask_fn") is a robustness / future-proofing ruling,
    # not something this test file can claim to discriminate.
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 323).integers(0, cfg.V, size=(1, cfg.T))
    honest, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    for blk in m.blocks:
        blk.attn.mask_fn = lambda n: np.ones((n, n), dtype=bool)  # "ignore the mask"
    ignoring, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    assert np.array_equal(honest, ignoring)


def test_mask_row_is_actually_consulted_by_the_cached_path():
    # The converse, so the mask hook is not dead code: a restrictive mask (attend
    # only to self) must change the cached logits once there is history to ignore.
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 324).integers(0, cfg.V, size=(1, cfg.T))
    honest, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    for blk in m.blocks:
        blk.attn.mask_fn = lambda n: np.eye(n, dtype=bool)  # self only
    restricted, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    assert np.array_equal(honest[:, 0], restricted[:, 0])  # position 0: no history, identical
    assert np.max(np.abs(honest[:, 1:] - restricted[:, 1:])) > 1e-3


def test_teacher_forcing_matters_a_different_fed_token_changes_later_logits():
    # C8 is defined for a FIXED sequence. Feeding a different token at one position
    # must change every later logit, so "same logits" cannot be an artefact of the
    # cache ignoring its input.
    cfg = tiny_config(V=11)
    m, p = _model(cfg)
    idx = make_rng(_SEED + 325).integers(0, cfg.V, size=(1, cfg.T))
    other = idx.copy()
    other[0, 1] = (other[0, 1] + 1) % cfg.V
    a, _ = _teacher_forced_cached_logits(m, p, idx, 1)
    b, _ = _teacher_forced_cached_logits(m, p, other, 1)
    assert np.array_equal(a[:, 0], b[:, 0])
    assert np.max(np.abs(a[:, 1:] - b[:, 1:])) > 1e-3


# ---------------------------------------------------------------------------
# FLOP evidence for the cache (Sprint 4's chokepoint counter)
# ---------------------------------------------------------------------------
def _cached_step_flops_closed_form(cfg, pos):
    # dense: q,k,v,o = 4 * 2d^2 and MLP = 2 * 2 * d * 4d  -> 24 L d^2 ; tied head 2 V d
    # attention at 0-indexed position `pos` over pos+1 keys: QK^T and PV = 2 * 2 d (pos+1) per layer
    return 24 * cfg.L * cfg.d**2 + 2 * cfg.V * cfg.d + 4 * cfg.L * cfg.d * (pos + 1)


@pytest.mark.parametrize("name", list(_C8_CONFIGS))
def test_cached_step_flops_match_hand_derived_closed_form(name):
    cfg = _C8_CONFIGS[name]()
    m, p = _model(cfg)
    idx = make_rng(_SEED + 330).integers(0, cfg.V, size=(1, cfg.T))
    kv = KVCache(cfg, 1)
    for pos in range(cfg.T):
        with flops.FlopCounter() as fc:
            m.forward_step(idx[:, pos : pos + 1], p, kv)
        assert fc.total() == _cached_step_flops_closed_form(cfg, pos), pos


@pytest.mark.parametrize("name", list(_C8_CONFIGS))
def test_incremental_decoding_costs_strictly_fewer_flops_than_recomputing(name):
    cfg = _C8_CONFIGS[name]()
    m, p = _model(cfg)
    idx = make_rng(_SEED + 331).integers(0, cfg.V, size=(1, cfg.T))
    T0 = 1
    n = cfg.T - T0

    kv = KVCache(cfg, 1)
    m.prefill(idx[:, :T0], p, kv)
    with flops.FlopCounter() as cached_fc:
        for t in range(T0, cfg.T):
            m.forward_step(idx[:, t : t + 1], p, kv)

    uncached = 0
    for s in range(T0 + 1, cfg.T + 1):  # a full forward over the growing prefix, per new token
        with flops.FlopCounter() as fc:
            m.forward(idx[:, :s], p)
        uncached += fc.total()

    assert n > 1
    assert 0 < cached_fc.total() < uncached
    print(f"{name}: cached={cached_fc.total()} uncached={uncached} "
          f"ratio={uncached / cached_fc.total():.2f}x over {n} tokens")

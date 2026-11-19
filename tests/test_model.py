"""tests/test_model.py -- Engineer B, Sprint 3 Phase 3 (S3-03, milestone B2).

Covers `nanogpt.model.GPT`'s forward-only structure (the backward half --
B3's dev gradcheck and the two C7 instrumentation flags -- is added by a
later commit, after this structure is frozen): param_keys order, init,
tied head identity, C-contiguity, shape/validation of `embed`/`forward`,
the mask_fn plumbing (single callable vs. per-block list vs. wrong
length), the exact `forward == forward_from_embedding(embed(...))`
identity, and the tied-head anchor. The formal, full-coverage C6
gradcheck (all 36 tensors, TL-9 for every layer) is Engineer C's
`tests/test_gradients.py`.

Fixture seed (SPRINT_3_PLAN.md section 4): `_FIXTURE_SEED = DEFAULT_SEED + 312`.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt.attention_mask import causal_mask, shifted_causal_mask
from nanogpt.config import DEFAULT_SEED, ROUNDOFF_ATOL, ModelConfig, make_rng, tiny_config
from nanogpt.model import GPT

_FIXTURE_SEED = DEFAULT_SEED + 312


def _tiny_cfg(V=55):
    return tiny_config(V)  # T=4, d=8, L=2, H=2


# ---------------------------------------------------------------------------
# Structure: param_keys order/count, no lm_head/head-bias key.
# ---------------------------------------------------------------------------
def test_param_keys_order_and_count():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    keys = model.param_keys()
    assert len(keys) == 4 + 16 * cfg.L
    assert keys[0] == "wte.W"
    assert keys[1] == "wpe.W"
    expect = ["wte.W", "wpe.W"]
    for l in range(cfg.L):
        expect.extend(model.blocks[l].param_keys())
    expect.extend(["ln_f.gamma", "ln_f.beta"])
    assert keys == expect


def test_no_lm_head_or_head_bias_key():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    for k in model.param_keys():
        assert "lm_head" not in k
        assert k != "ln_f.b"  # sanity: no stray head-bias-like key


def test_default_and_max_config_counts():
    for cfg in (
        ModelConfig(V=55, T=64, d=64, L=2),  # default
        ModelConfig(V=55, T=128, d=128, L=4),  # max
    ):
        model = GPT(cfg)
        assert len(model.param_keys()) == 4 + 16 * cfg.L


# ---------------------------------------------------------------------------
# Init: one rng, same order as param_keys; stats; bit-identical same seed.
# ---------------------------------------------------------------------------
def test_init_params_matches_param_keys_and_stats():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED)
    params = model.init_params(rng)
    assert set(params) == set(model.param_keys())

    for key, v in params.items():
        if key.endswith(".b") or key.endswith(".beta"):
            assert np.all(v == 0.0), key
        elif key.endswith(".gamma"):
            assert np.all(v == 1.0), key
        else:
            assert v.dtype == np.float64
            assert v.flags["C_CONTIGUOUS"]


def test_same_seed_gives_bit_identical_params():
    cfg = _tiny_cfg()
    p1 = GPT(cfg).init_params(make_rng(_FIXTURE_SEED + 1))
    p2 = GPT(cfg).init_params(make_rng(_FIXTURE_SEED + 1))
    assert set(p1) == set(p2)
    for k in p1:
        assert np.array_equal(p1[k], p2[k])


# ---------------------------------------------------------------------------
# Tied head (TL-17): exactly one "wte.W" array; named_parameters identity;
# C-contiguous, owns its data.
# ---------------------------------------------------------------------------
def test_tied_head_structure_and_identity():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    params = model.init_params(make_rng(_FIXTURE_SEED + 2))

    wte_W = params["wte.W"]
    assert wte_W.flags["C_CONTIGUOUS"]
    assert wte_W.base is None

    named = model.named_parameters(params)
    assert "lm_head.W" in named
    assert named["lm_head.W"] is wte_W
    assert "wte.W" in named
    assert named["wte.W"] is wte_W


# ---------------------------------------------------------------------------
# embed / forward: shapes, dtype, validation.
# ---------------------------------------------------------------------------
def test_forward_shape_dtype():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 3)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T)).astype(np.int64)
    logits, cache = model.forward(idx, params)
    assert logits.shape == (2, cfg.T, cfg.V)
    assert logits.dtype == np.float64
    assert np.all(np.isfinite(logits))


def test_T_less_than_cfgT_works():
    cfg = _tiny_cfg(V=11)  # cfg.T == 4
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 4)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, 2)).astype(np.int64)  # T=2 < 4
    logits, _ = model.forward(idx, params)
    assert logits.shape == (2, 2, cfg.V)


def test_T_greater_than_cfgT_raises():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 5)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T + 1)).astype(np.int64)
    with pytest.raises(ValueError):
        model.forward(idx, params)


def test_1d_idx_raises():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 6)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(cfg.T,)).astype(np.int64)
    with pytest.raises(ValueError):
        model.forward(idx, params)


def test_float_idx_raises():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 7)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T)).astype(np.float64)
    with pytest.raises(TypeError):
        model.forward(idx, params)


# ---------------------------------------------------------------------------
# forward == forward_from_embedding(embed(...)), exactly.
# ---------------------------------------------------------------------------
def test_forward_equals_embed_then_forward_from_embedding_exactly():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 8)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T)).astype(np.int64)

    logits_direct, _ = model.forward(idx, params)
    h0, _ = model.embed(idx, params)
    logits_composed, _ = model.forward_from_embedding(h0, params)
    assert np.array_equal(logits_direct, logits_composed)


# ---------------------------------------------------------------------------
# Tied-head anchor: logits == LN_f(h_L) @ wte.W.T, recomputed from cache.
# ---------------------------------------------------------------------------
def test_tied_head_anchor_recomputed_from_cache():
    cfg = _tiny_cfg(V=11)
    model = GPT(cfg)
    rng = make_rng(_FIXTURE_SEED + 9)
    params = model.init_params(rng)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T)).astype(np.int64)
    logits, cache = model.forward(idx, params)

    y = cache["c_fe"]["y"]
    wte_W = cache["c_fe"]["wte_W"]
    recomputed = y @ wte_W.T
    max_abs = float(np.max(np.abs(logits - recomputed)))
    print("tied-head anchor max_abs=", max_abs)
    assert max_abs <= ROUNDOFF_ATOL


# ---------------------------------------------------------------------------
# mask_fn plumbing.
# ---------------------------------------------------------------------------
def test_single_mask_fn_applied_to_every_block():
    cfg = _tiny_cfg()

    def f(T):
        return causal_mask(T)

    model = GPT(cfg, mask_fn=f)
    for b in model.blocks:
        assert b.attn.mask_fn is f


def test_per_block_mask_fn_list_applied_per_block():
    cfg = _tiny_cfg()
    fns = [causal_mask, shifted_causal_mask]
    assert len(fns) == cfg.L
    model = GPT(cfg, mask_fn=fns)
    for l, b in enumerate(model.blocks):
        assert b.attn.mask_fn is fns[l]


def test_wrong_length_mask_fn_list_raises():
    cfg = _tiny_cfg()
    with pytest.raises(ValueError):
        GPT(cfg, mask_fn=[causal_mask])  # cfg.L == 2, length 1


def test_non_callable_non_sequence_mask_fn_raises():
    cfg = _tiny_cfg()
    with pytest.raises(ValueError):
        GPT(cfg, mask_fn="not callable")

"""tests/test_attention.py -- Engineer B, Sprint 2 Phase 2 (S2-01, S2-04..S2-08).

Covers `nanogpt.attention.CausalSelfAttention`: construction, vectorized
forward, the oracle-match hard gate (S2-07), the module gradcheck table
(TL-9 protocol for attn.k.b), the backward-vs-oracle adapter check (P1),
test-the-tests for the head-split convention / fan-out / scale, and
robustness. See SPRINT_2_PLAN.md section 6.

Fixture seed per TL-13 (R17): `_FIXTURE_SEED = DEFAULT_SEED + 202`, never
`make_rng(DEFAULT_SEED)` (that seed is reserved for gradcheck's own internal
upstream-gradient projection R). Every fixture uses one `Generator`, drawn
sequentially (W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, then x), and asserts
`W_q != W_k`.

Milestone B1 (section 6.1): constructor, helpers, init_params, forward.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt.attention import CausalSelfAttention, _merge_heads, _split_heads
from nanogpt.attention_mask import causal_mask, shifted_causal_mask
from nanogpt.attention_reference import reference_attention_forward
from nanogpt.config import (
    DEFAULT_SEED,
    INIT_STD,
    N_HEAD,
    ROUNDOFF_ATOL,
    make_rng,
    tiny_config,
)

_FIXTURE_SEED = DEFAULT_SEED + 202

_PARAM_NAMES = ("q", "k", "v", "o")


# ---------------------------------------------------------------------------
# Fixtures (TL-13): non-init params W = N(0,1)/sqrt(d), b = 0.1*N(0,1), drawn
# sequentially q, k, v, o from one Generator; x drawn next from the same rng.
# ---------------------------------------------------------------------------
def _non_init_params(rng, d, prefix="attn"):
    params = {}
    for name in _PARAM_NAMES:
        W = np.ascontiguousarray((rng.standard_normal((d, d)) / math.sqrt(d)).astype(np.float64))
        b = np.ascontiguousarray((0.1 * rng.standard_normal(d)).astype(np.float64))
        params[f"{prefix}.{name}.W"] = W
        params[f"{prefix}.{name}.b"] = b
    assert not np.array_equal(params[f"{prefix}.q.W"], params[f"{prefix}.k.W"])
    return params


def _tiny_gradcheck_fixture():
    """d=8, H=2, B=2, T=4, non-init fixture, seeded off _FIXTURE_SEED."""
    rng = make_rng(_FIXTURE_SEED + 100)
    d, H, B, T = 8, 2, 2, 4
    attn = CausalSelfAttention("attn", d, H)
    params = _non_init_params(rng, d)
    x = rng.standard_normal((B, T, d)).astype(np.float64)
    return attn, params, x


def _oracle_case(d, H, B, T, seed, mask_fn=causal_mask):
    rng = make_rng(seed)
    attn = CausalSelfAttention("attn", d, H, mask_fn=mask_fn)
    params = _non_init_params(rng, d)
    x = rng.standard_normal((B, T, d)).astype(np.float64)
    out, _ = attn.forward(x, params)
    ref = reference_attention_forward(x, params, H, mask_fn=mask_fn)
    max_abs = float(np.max(np.abs(out - ref)))
    return max_abs


# ---------------------------------------------------------------------------
# Construction (S2-01 / TL-10 / TL-11)
# ---------------------------------------------------------------------------
def test_constructor_rejects_non_dividing_H():
    with pytest.raises(ValueError):
        CausalSelfAttention("attn", 8, 3)


def test_constructor_rejects_non_positive_or_non_int_d_and_H():
    with pytest.raises(ValueError):
        CausalSelfAttention("attn", 0, 2)
    with pytest.raises(ValueError):
        CausalSelfAttention("attn", 8, 0)
    with pytest.raises(ValueError):
        CausalSelfAttention("attn", 8.0, 2)
    with pytest.raises(ValueError):
        CausalSelfAttention("attn", 8, True)


def test_scale_param_keys_and_default_mask():
    attn = CausalSelfAttention("attn", 8, 2)
    assert attn.head_dim == 4
    assert attn.scale == 1.0 / math.sqrt(4)
    assert attn.param_keys() == [
        "attn.q.W", "attn.q.b",
        "attn.k.W", "attn.k.b",
        "attn.v.W", "attn.v.b",
        "attn.o.W", "attn.o.b",
    ]
    assert attn.mask_fn is causal_mask


def test_from_config_tiny_V():
    cfg = tiny_config(11)
    attn = CausalSelfAttention.from_config(cfg)
    assert attn.H == 2
    assert attn.head_dim == 4
    assert attn.d == 8


def test_init_params_shapes_dtype_contiguous_zero_bias_and_distinct_W():
    rng = make_rng(_FIXTURE_SEED)
    attn = CausalSelfAttention("attn", 8, 2)
    params = attn.init_params(rng)
    assert set(params) == set(attn.param_keys())
    for name in _PARAM_NAMES:
        W = params[f"attn.{name}.W"]
        b = params[f"attn.{name}.b"]
        assert W.shape == (8, 8)
        assert b.shape == (8,)
        assert W.dtype == np.float64
        assert b.dtype == np.float64
        assert W.flags["C_CONTIGUOUS"]
        assert b.flags["C_CONTIGUOUS"]
        assert np.all(b == 0.0)
    assert not np.array_equal(params["attn.q.W"], params["attn.k.W"])


def test_init_params_std_band_at_default_d_H():
    rng = make_rng(_FIXTURE_SEED + 1)
    attn = CausalSelfAttention("attn", 64, N_HEAD)
    params = attn.init_params(rng)
    for name in _PARAM_NAMES:
        std = float(np.std(params[f"attn.{name}.W"]))
        assert 0.3 * INIT_STD < std < 3.0 * INIT_STD, (name, std)


def test_split_merge_heads_are_exact_inverses():
    rng = make_rng(_FIXTURE_SEED + 2)
    B, T, H, hd = 2, 4, 2, 3
    a = rng.standard_normal((B, T, H * hd)).astype(np.float64)
    split = _split_heads(a, H)
    assert split.shape == (B, H, T, hd)
    merged = _merge_heads(split)
    assert np.array_equal(merged, a)
    # TL-11: head h owns columns [h*hd, (h+1)*hd)
    for h in range(H):
        assert np.array_equal(split[:, h, :, :], a[:, :, h * hd:(h + 1) * hd])


def test_forward_basic_shape_dtype_and_finite():
    rng = make_rng(_FIXTURE_SEED + 3)
    attn = CausalSelfAttention("attn", 8, 2)
    params = attn.init_params(rng)
    x = rng.standard_normal((2, 4, 8)).astype(np.float64)
    out, cache = attn.forward(x, params)
    assert out.shape == x.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert set(cache) == {"cq", "ck", "cv", "co", "qh", "kh", "vh", "P", "allowed"}
    assert cache["P"].shape == (2, 2, 4, 4)
    assert cache["allowed"].shape == (4, 4)
    assert cache["allowed"].dtype == bool


def test_cache_P_masked_zero_and_rows_sum_to_one():
    attn, params, x = _tiny_gradcheck_fixture()
    _, cache = attn.forward(x, params)
    P = cache["P"]
    allowed = cache["allowed"]

    masked = ~allowed  # (T, T) bool, True strictly above the diagonal
    masked_vals = P[:, :, masked]
    assert np.all(masked_vals == 0.0)

    row_sums = P.sum(axis=-1)
    assert np.max(np.abs(row_sums - 1.0)) <= ROUNDOFF_ATOL


# ---------------------------------------------------------------------------
# Oracle forward match (S2-07, HARD GATE). Milestone B2: this must pass
# before the backward (B3) is trusted (SPRINT_2_PLAN.md section 9).
# ---------------------------------------------------------------------------
def test_oracle_match_tiny():
    m = _oracle_case(8, 2, 2, 4, seed=_FIXTURE_SEED + 10)
    print("oracle match tiny (d=8,H=2,B=2,T=4) max_abs=", m)
    assert m <= ROUNDOFF_ATOL


def test_oracle_match_default_H4_d64():
    m = _oracle_case(64, 4, 2, 7, seed=_FIXTURE_SEED + 11)
    print("oracle match default (d=64,H=4,B=2,T=7) max_abs=", m)
    assert m <= ROUNDOFF_ATOL


@pytest.mark.parametrize("H", [1, 2, 4, 8])
def test_oracle_match_head_sweep_d8_T5(H):
    m = _oracle_case(8, H, 2, 5, seed=_FIXTURE_SEED + 20 + H)
    print(f"oracle match d=8,T=5,H={H} max_abs=", m)
    assert m <= ROUNDOFF_ATOL


def test_oracle_match_T1():
    m = _oracle_case(8, 2, 2, 1, seed=_FIXTURE_SEED + 30)
    print("oracle match T=1 max_abs=", m)
    assert m <= ROUNDOFF_ATOL


def test_oracle_match_B3():
    m = _oracle_case(8, 2, 3, 4, seed=_FIXTURE_SEED + 31)
    print("oracle match B=3 max_abs=", m)
    assert m <= ROUNDOFF_ATOL


def test_oracle_match_tiny_with_shifted_mask():
    """The tiny case also passes with mask_fn=shifted_causal_mask in both
    implementations: shows the mask_fn plumbing is identical in the two
    implementations, independent of masking correctness."""
    m = _oracle_case(8, 2, 2, 4, seed=_FIXTURE_SEED + 32, mask_fn=shifted_causal_mask)
    print("oracle match tiny, shifted mask, max_abs=", m)
    assert m <= ROUNDOFF_ATOL

"""tests/test_attention_reference.py -- Engineer A's tests (Sprint 2, Phase 2).

Covers:
- nanogpt/attention_mask.py: causal_mask, shifted_causal_mask (unit tests
  against an explicit double-loop reference, edge cases, freshness).
- nanogpt/attention_reference.py: the naive loop-over-heads oracle's own
  closed-form anchors (it is the ground truth, so its correctness rests on
  these, never on agreement with Engineer B's vectorized module), plus its
  AST source guard.

Milestone A1 (committed): attention_mask.py + its mask tests below.
Milestone A2 (this commit): the oracle (nanogpt/attention_reference.py) and
its closed-form anchors.
Milestone A3 (later commit): the oracle's AST source guard.

Per TL-13 (SPRINT_2_PLAN.md section 2.2 / 5.3): every fixture tensor is
drawn from one Generator per fixture, seeded with
`_FIXTURE_SEED = DEFAULT_SEED + 201` (never from `make_rng(DEFAULT_SEED)`,
which would collide with `nanogpt.gradcheck`'s internal upstream-gradient
seed -- not itself used in this file, since `reference_attention_forward`
is forward-only and has no backward to gradcheck).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt.attention_mask import causal_mask, shifted_causal_mask
from nanogpt.attention_reference import reference_attention_forward
from nanogpt.config import DEFAULT_SEED, ROUNDOFF_ATOL, make_rng

_FIXTURE_SEED = DEFAULT_SEED + 201
_PREFIX = "attn"

assert _FIXTURE_SEED != DEFAULT_SEED  # TL-13 / R17: never collide with the default seed


# ===========================================================================
# Fixtures (TL-13: one Generator per fixture, tensors drawn sequentially)
# ===========================================================================
def _make_fixture(seed, B, T, d, H, prefix=_PREFIX):
    """Non-init params (TL-13): W = N(0,1)/sqrt(d), b = 0.1*N(0,1), x = N(0,1).

    Draws Wq, bq, Wk, bk, Wv, bv, Wo, bo, then x, all sequentially from a
    single `np.random.Generator` seeded with `seed`. `H` is accepted purely
    so every call site documents the shape it is building a fixture for; it
    plays no part in constructing the (d, d) / (d,) parameter arrays
    themselves.
    """
    rng = make_rng(seed)

    def _W():
        return rng.standard_normal((d, d)) / math.sqrt(d)

    def _b():
        return 0.1 * rng.standard_normal(d)

    Wq, bq = _W(), _b()
    Wk, bk = _W(), _b()
    Wv, bv = _W(), _b()
    Wo, bo = _W(), _b()
    x = rng.standard_normal((B, T, d))

    params = {
        f"{prefix}.q.W": Wq, f"{prefix}.q.b": bq,
        f"{prefix}.k.W": Wk, f"{prefix}.k.b": bk,
        f"{prefix}.v.W": Wv, f"{prefix}.v.b": bv,
        f"{prefix}.o.W": Wo, f"{prefix}.o.b": bo,
    }
    return params, x


def test_fixture_seed_is_not_default_and_q_k_distinct():
    params, _ = _make_fixture(_FIXTURE_SEED, B=2, T=4, d=8, H=2)
    assert not np.array_equal(params[f"{_PREFIX}.q.W"], params[f"{_PREFIX}.k.W"])
    assert not np.array_equal(params[f"{_PREFIX}.q.b"], params[f"{_PREFIX}.k.b"])


# ===========================================================================
# attention_mask.py: causal_mask / shifted_causal_mask
# ===========================================================================
@pytest.mark.parametrize("T", range(1, 7))
def test_causal_mask_matches_explicit_loop(T):
    expected = np.array([[j <= i for j in range(T)] for i in range(T)], dtype=bool)
    actual = causal_mask(T)
    assert actual.dtype == np.bool_
    assert actual.shape == (T, T)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("T", range(1, 7))
def test_causal_mask_diagonal_and_row_counts(T):
    m = causal_mask(T)
    assert np.all(np.diag(m))
    for i in range(T):
        assert int(m[i].sum()) == i + 1


@pytest.mark.parametrize("T", range(1, 7))
def test_shifted_causal_mask_matches_explicit_loop(T):
    expected = np.array([[j <= i + 1 for j in range(T)] for i in range(T)], dtype=bool)
    actual = shifted_causal_mask(T)
    assert actual.dtype == np.bool_
    assert actual.shape == (T, T)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("T", range(1, 7))
def test_shifted_causal_mask_row_counts(T):
    m = shifted_causal_mask(T)
    for i in range(T):
        assert int(m[i].sum()) == min(i + 2, T)


def test_shifted_minus_causal_is_exactly_the_superdiagonal():
    for T in range(1, 7):
        c = causal_mask(T)
        s = shifted_causal_mask(T)
        diff = s & ~c
        expected = np.zeros((T, T), dtype=bool)
        for i in range(T - 1):
            expected[i, i + 1] = True
        assert np.array_equal(diff, expected)


@pytest.mark.parametrize("T", [0, -1, 2.0, True])
def test_mask_functions_reject_invalid_T(T):
    with pytest.raises(ValueError):
        causal_mask(T)
    with pytest.raises(ValueError):
        shifted_causal_mask(T)


def test_mask_functions_return_a_fresh_array_each_call():
    T = 5
    m1 = causal_mask(T)
    m1[:, :] = False
    m2 = causal_mask(T)
    assert m2.any()
    assert np.array_equal(m2, np.array([[j <= i for j in range(T)] for i in range(T)]))

    s1 = shifted_causal_mask(T)
    s1[:, :] = False
    s2 = shifted_causal_mask(T)
    assert s2.any()
    assert np.array_equal(s2, np.array([[j <= i + 1 for j in range(T)] for i in range(T)]))


# ===========================================================================
# attention_reference.py: closed-form oracle anchors
# ===========================================================================
def test_oracle_T1_anchor_F2():
    """T=1 (F2): out == (x Wv + bv) Wo + bo, at B=2."""
    B, T, d, H = 2, 1, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    out = reference_attention_forward(x, params, H, prefix=_PREFIX)
    v = x @ params[f"{_PREFIX}.v.W"] + params[f"{_PREFIX}.v.b"]
    expected = v @ params[f"{_PREFIX}.o.W"] + params[f"{_PREFIX}.o.b"]
    assert np.max(np.abs(out - expected)) <= ROUNDOFF_ATOL


def test_oracle_uniform_attention_cumulative_mean_F6():
    """Uniform attention (F6): Wq=0, bq=0 => out is the running mean of v."""
    B, T, d, H = 2, 5, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    params[f"{_PREFIX}.q.W"] = np.zeros((d, d))
    params[f"{_PREFIX}.q.b"] = np.zeros(d)

    out = reference_attention_forward(x, params, H, prefix=_PREFIX)

    v = x @ params[f"{_PREFIX}.v.W"] + params[f"{_PREFIX}.v.b"]
    cum_mean = np.cumsum(v, axis=1) / np.arange(1, T + 1)[None, :, None]
    expected = cum_mean @ params[f"{_PREFIX}.o.W"] + params[f"{_PREFIX}.o.b"]
    assert np.max(np.abs(out - expected)) <= ROUNDOFF_ATOL


def test_oracle_two_token_hand_computation():
    """d=4, H=2, T=2, Wv=Wo=I, bv=bo=0: row 1 computed by hand with math.exp,
    row 0 == v_0 (per head)."""
    d, H, T, B = 4, 2, 2, 1
    hd = d // H
    scale = 1.0 / math.sqrt(hd)
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    I = np.eye(d)
    params[f"{_PREFIX}.v.W"] = I.copy()
    params[f"{_PREFIX}.v.b"] = np.zeros(d)
    params[f"{_PREFIX}.o.W"] = I.copy()
    params[f"{_PREFIX}.o.b"] = np.zeros(d)

    out = reference_attention_forward(x, params, H, prefix=_PREFIX)

    Wq, bq = params[f"{_PREFIX}.q.W"], params[f"{_PREFIX}.q.b"]
    Wk, bk = params[f"{_PREFIX}.k.W"], params[f"{_PREFIX}.k.b"]
    q = x @ Wq + bq
    k = x @ Wk + bk
    v = x  # Wv = I, bv = 0

    # Row 0 only attends to itself.
    assert np.max(np.abs(out[0, 0, :] - v[0, 0, :])) <= ROUNDOFF_ATOL

    # Row 1, per head, hand-computed with math.exp.
    expected_row1 = np.zeros(d)
    for h in range(H):
        cols = slice(h * hd, (h + 1) * hd)
        s0 = scale * float(np.dot(q[0, 1, cols], k[0, 0, cols]))
        s1 = scale * float(np.dot(q[0, 1, cols], k[0, 1, cols]))
        m = max(s0, s1)
        e0, e1 = math.exp(s0 - m), math.exp(s1 - m)
        p0, p1 = e0 / (e0 + e1), e1 / (e0 + e1)
        expected_row1[cols] = p0 * v[0, 0, cols] + p1 * v[0, 1, cols]
    assert np.max(np.abs(out[0, 1, :] - expected_row1)) <= ROUNDOFF_ATOL


def test_oracle_head_independence_TL11():
    """Wo=I, bo=0 => out == Y. Changing head-1's Wq/Wk columns leaves
    out[...,0:hd] bitwise unchanged and changes out[...,hd:2hd]. Zeroing
    head-1's Wv/bv columns zeroes out[...,hd:2hd]."""
    B, T, d, H = 2, 3, 4, 2
    hd = d // H
    head1 = slice(hd, 2 * hd)
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    params[f"{_PREFIX}.o.W"] = np.eye(d)
    params[f"{_PREFIX}.o.b"] = np.zeros(d)

    out1 = reference_attention_forward(x, params, H, prefix=_PREFIX)

    # Perturb only head 1's Q/K column blocks.
    params2 = {k: v.copy() for k, v in params.items()}
    rng2 = make_rng(_FIXTURE_SEED + 1000)
    params2[f"{_PREFIX}.q.W"][:, head1] = rng2.standard_normal((d, hd))
    params2[f"{_PREFIX}.k.W"][:, head1] = rng2.standard_normal((d, hd))
    assert not np.array_equal(params2[f"{_PREFIX}.q.W"], params[f"{_PREFIX}.q.W"])

    out2 = reference_attention_forward(x, params2, H, prefix=_PREFIX)

    assert np.array_equal(out1[..., 0:hd], out2[..., 0:hd])
    assert not np.allclose(out1[..., head1], out2[..., head1], atol=ROUNDOFF_ATOL)

    # Zero head 1's V columns (and bias) -> head-1 output columns are zero.
    params3 = {k: v.copy() for k, v in params.items()}
    params3[f"{_PREFIX}.v.W"][:, head1] = 0.0
    params3[f"{_PREFIX}.v.b"][head1] = 0.0
    out3 = reference_attention_forward(x, params3, H, prefix=_PREFIX)
    assert np.max(np.abs(out3[..., head1])) <= ROUNDOFF_ATOL


def test_oracle_H1_matches_inline_matrix_form():
    """H=1: equals an inline matrix-form single-head reference,
    softmax(QK^T/sqrt(d) masked) V Wo + bo (written here with ordinary
    NumPy transpose/reshape, which is fine in the TEST file -- only the
    oracle function itself is source-guarded)."""
    B, T, d, H = 2, 4, 6, 1
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    out = reference_attention_forward(x, params, H, prefix=_PREFIX)

    Wq, bq = params[f"{_PREFIX}.q.W"], params[f"{_PREFIX}.q.b"]
    Wk, bk = params[f"{_PREFIX}.k.W"], params[f"{_PREFIX}.k.b"]
    Wv, bv = params[f"{_PREFIX}.v.W"], params[f"{_PREFIX}.v.b"]
    Wo, bo = params[f"{_PREFIX}.o.W"], params[f"{_PREFIX}.o.b"]
    scale = 1.0 / math.sqrt(d)
    mask = causal_mask(T)

    expected = np.zeros((B, T, d))
    for b in range(B):
        Q = x[b] @ Wq + bq
        K = x[b] @ Wk + bk
        V = x[b] @ Wv + bv
        S = (Q @ K.T) * scale
        Z = np.where(mask, S, -np.inf)
        m = np.max(Z, axis=-1, keepdims=True)
        e = np.exp(Z - m)
        P = e / np.sum(e, axis=-1, keepdims=True)
        expected[b] = (P @ V) @ Wo + bo

    assert np.max(np.abs(out - expected)) <= ROUNDOFF_ATOL


def test_oracle_shifted_mask_injection_changes_all_but_last_row():
    """mask_fn=the one-extra-token adversary changes out_t for t < T-1 and
    leaves out_{T-1} identical (the last row already sees every key)."""
    B, T, d, H = 2, 5, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    out_c = reference_attention_forward(x, params, H, prefix=_PREFIX, mask_fn=causal_mask)
    out_s = reference_attention_forward(x, params, H, prefix=_PREFIX, mask_fn=shifted_causal_mask)

    assert np.array_equal(out_c[:, T - 1, :], out_s[:, T - 1, :])
    for t in range(T - 1):
        assert not np.allclose(out_c[:, t, :], out_s[:, t, :], atol=ROUNDOFF_ATOL)


# ===========================================================================
# Validation
# ===========================================================================
def test_oracle_d_mod_H_mismatch_raises_value_error():
    B, T, d = 2, 4, 6
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H=2)
    with pytest.raises(ValueError):
        reference_attention_forward(x, params, H=4, prefix=_PREFIX)  # 6 % 4 != 0


def test_oracle_x_ndim_mismatch_raises_value_error():
    B, T, d, H = 2, 4, 8, 2
    _, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    params, _ = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    with pytest.raises(ValueError):
        reference_attention_forward(x[0], params, H, prefix=_PREFIX)  # 2-D, not 3-D


def test_oracle_float32_x_raises_type_error():
    B, T, d, H = 2, 4, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    x32 = x.astype(np.float32)
    with pytest.raises(TypeError):
        reference_attention_forward(x32, params, H, prefix=_PREFIX)


def test_oracle_float32_param_raises_type_error():
    B, T, d, H = 2, 4, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    params[f"{_PREFIX}.q.W"] = params[f"{_PREFIX}.q.W"].astype(np.float32)
    with pytest.raises(TypeError):
        reference_attention_forward(x, params, H, prefix=_PREFIX)


def test_oracle_inputs_bit_identical_after_call():
    B, T, d, H = 2, 4, 8, 2
    params, x = _make_fixture(_FIXTURE_SEED, B, T, d, H)
    x_snapshot = np.array(x, copy=True)
    params_snapshot = {k: np.array(v, copy=True) for k, v in params.items()}

    _ = reference_attention_forward(x, params, H, prefix=_PREFIX)

    assert np.array_equal(x, x_snapshot)
    for k in params:
        assert np.array_equal(params[k], params_snapshot[k])

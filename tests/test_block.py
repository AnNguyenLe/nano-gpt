"""tests/test_block.py -- Engineer B, Sprint 3 Phase 3 (S3-02, milestone B1).

Covers `nanogpt.block.Block`: construction / param_keys / init_params order,
the module gradcheck table at tiny size (TL-9 protocol for
`blk.attn.k.b`, block level), the residual anchor, mask_fn plumbing
(construction-time only, never monkeypatched), and test-the-tests for the
two residual sums and the LN1/LN2 cache identity.

Fixture seed (SPRINT_3_PLAN.md section 4): `_FIXTURE_SEED = DEFAULT_SEED + 311`,
never bare `DEFAULT_SEED` (reserved for gradcheck's own upstream projection R).

"Non-init fixture" (SPRINT_3_PLAN.md section 3.6, the Sprint-3-wide
convention): take `init_params(rng)`, then W <- W + 0.3*N, b/beta <- 0.3*N,
gamma <- 1 + 0.3*N, every array made C-contiguous.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt import attention, ops
from nanogpt.attention_mask import causal_mask, shifted_causal_mask
from nanogpt.block import Block
from nanogpt.config import (
    DEFAULT_SEED,
    GRADCHECK_TOL,
    REL_ERR_FLOOR,
    ROUNDOFF_ATOL,
    make_rng,
    tiny_config,
)
from nanogpt.gradcheck import gradcheck

_FIXTURE_SEED = DEFAULT_SEED + 311

_D, _H, _B, _T = 8, 2, 2, 4

_ALL_TENSOR_NAMES = (
    "blk.ln1.gamma", "blk.ln1.beta",
    "blk.attn.q.W", "blk.attn.q.b",
    "blk.attn.k.W", "blk.attn.k.b",
    "blk.attn.v.W", "blk.attn.v.b",
    "blk.attn.o.W", "blk.attn.o.b",
    "blk.ln2.gamma", "blk.ln2.beta",
    "blk.mlp.fc.W", "blk.mlp.fc.b",
    "blk.mlp.proj.W", "blk.mlp.proj.b",
    "x",
)


def _non_init_params(module, rng):
    """Sprint-3-wide "non-init fixture" (section 3.6)."""
    params = module.init_params(rng)
    out = {}
    for key, v in params.items():
        v = np.asarray(v, dtype=np.float64)
        if key.endswith(".gamma"):
            out[key] = np.ascontiguousarray(1.0 + 0.3 * rng.standard_normal(v.shape))
        elif key.endswith(".beta") or key.endswith(".b"):
            out[key] = np.ascontiguousarray(0.3 * rng.standard_normal(v.shape))
        elif key.endswith(".W"):
            out[key] = np.ascontiguousarray(v + 0.3 * rng.standard_normal(v.shape))
        else:
            raise AssertionError(f"unexpected param key suffix: {key!r}")
    return out


def _tiny_gradcheck_fixture():
    rng = make_rng(_FIXTURE_SEED)
    blk = Block("blk", _D, _H)
    params = _non_init_params(blk, rng)
    x = rng.standard_normal((_B, _T, _D)).astype(np.float64)
    return blk, params, x


# ---------------------------------------------------------------------------
# Construction / param_keys / init_params order
# ---------------------------------------------------------------------------
def test_param_keys_order_and_count():
    blk = Block("blk", _D, _H)
    assert blk.param_keys() == list(_ALL_TENSOR_NAMES[:-1])  # everything but "x"
    assert len(blk.param_keys()) == 16


def test_from_config_tiny():
    cfg = tiny_config(11)
    blk = Block.from_config(cfg, prefix="h.0")
    assert blk.d == 8
    assert blk.H == 2
    assert blk.attn.prefix == "h.0.attn"


def test_init_params_one_rng_same_order_as_param_keys():
    rng1 = make_rng(_FIXTURE_SEED + 1)
    rng2 = make_rng(_FIXTURE_SEED + 1)
    blk1 = Block("blk", _D, _H)
    blk2 = Block("blk", _D, _H)
    p1 = blk1.init_params(rng1)
    p2 = blk2.init_params(rng2)
    assert set(p1) == set(blk1.param_keys())
    for k in p1:
        assert np.array_equal(p1[k], p2[k])  # same seed -> bit-identical
    # distinct draws: attn.q.W must differ from attn.k.W (same rng, sequential draws)
    assert not np.array_equal(p1["blk.attn.q.W"], p1["blk.attn.k.W"])


def test_no_ln_softmax_gelu_arithmetic_in_block_source():
    """block.py composes the four modules only; it has no normalization,
    softmax or GELU math of its own."""
    import inspect
    import nanogpt.block as block_mod

    src = inspect.getsource(block_mod)
    for forbidden in ("np.mean(", "np.sqrt(", "np.exp(", "np.tanh("):
        assert forbidden not in src, forbidden


# ---------------------------------------------------------------------------
# Module gradcheck (tiny: d=8, H=2, B=2, T=4, non-init fixture). All 16
# tensors + x; blk.attn.k.b gated by TL-9 (a)-(d), not the literal rel_err.
# ---------------------------------------------------------------------------
def test_module_gradcheck_tiny_full_table():
    blk, params, x = _tiny_gradcheck_fixture()
    report = gradcheck(blk.forward, blk.backward, params, x)
    print(report)

    assert set(report.tensors) == set(_ALL_TENSOR_NAMES)
    for name, t in report.tensors.items():
        if name == "blk.attn.k.b":
            continue
        assert t.rel_err < GRADCHECK_TOL, (name, t)

    kb = report.tensors["blk.attn.k.b"]
    print(f"blk.attn.k.b: rel_err={kb.rel_err:.3e} "
          f"max_elem_rel_err={kb.max_elem_rel_err:.3e} max_abs_err={kb.max_abs_err:.3e}")
    assert kb.max_abs_err <= REL_ERR_FLOOR  # TL-9 (b)


def test_tl9_kb_structural_zero_protocol_block_level(monkeypatch):
    """TL-9 (a)-(d) restated at block level for blk.attn.k.b."""
    blk, params, x = _tiny_gradcheck_fixture()
    report = gradcheck(blk.forward, blk.backward, params, x)
    kb = report.tensors["blk.attn.k.b"]

    rng = make_rng(_FIXTURE_SEED + 102)
    out, cache = blk.forward(x, params)
    R = rng.standard_normal(out.shape).astype(np.float64)
    _, d_params = blk.backward(R, cache)

    # (a) analytic
    analytic_max = float(np.max(np.abs(d_params["blk.attn.k.b"])))
    print("TL-9 (a) analytic max|db_k| =", analytic_max)
    assert analytic_max <= ROUNDOFF_ATOL

    # (b) FD-noise bound
    print("TL-9 (b) FD max_abs_err(blk.attn.k.b) =", kb.max_abs_err)
    assert kb.max_abs_err <= REL_ERR_FLOOR

    # (c) invariance: out(b_k + 10*N) == out(b_k)
    delta = 10.0 * rng.standard_normal(params["blk.attn.k.b"].shape).astype(np.float64)
    params_shifted = dict(params)
    params_shifted["blk.attn.k.b"] = params["blk.attn.k.b"] + delta
    out_shifted, _ = blk.forward(x, params_shifted)
    invariance_max = float(np.max(np.abs(out_shifted - out)))
    print("TL-9 (c) invariance max|out(b_k+delta) - out(b_k)| =", invariance_max)
    assert invariance_max <= ROUNDOFF_ATOL

    # (d) dropped-term adversary (softmax backward without the row-sum term)
    def _dropped_softmax_backward(dp, bwd_cache):
        p = bwd_cache["p"]
        return p * dp, {}

    monkeypatch.setattr(attention.ops, "softmax_backward", _dropped_softmax_backward)
    _, d_params_dropped = blk.backward(R, cache)
    dropped_max = float(np.max(np.abs(d_params_dropped["blk.attn.k.b"])))
    print("TL-9 (d) dropped-term max|db_k| =", dropped_max)
    assert dropped_max > 1e-3

    print(f"blk.attn.k.b: rel_err={kb.rel_err:.3e} max_elem_rel_err={kb.max_elem_rel_err:.3e}")


# ---------------------------------------------------------------------------
# Residual anchor: zero attn.o.*/mlp.proj.* -> out == x exactly.
# ---------------------------------------------------------------------------
def test_residual_anchor_zeroed_output_projections_give_identity():
    rng = make_rng(_FIXTURE_SEED + 20)
    blk = Block("blk", _D, _H)
    params = blk.init_params(rng)
    for key in ("blk.attn.o.W", "blk.attn.o.b", "blk.mlp.proj.W", "blk.mlp.proj.b"):
        params[key] = np.zeros_like(params[key])

    x = rng.standard_normal((_B, _T, _D)).astype(np.float64)
    out, _ = blk.forward(x, params)
    assert np.array_equal(out, x)  # exact: zero matmuls/adds introduce no rounding


# ---------------------------------------------------------------------------
# mask_fn plumbing (construction time only; adversary imported only here)
# ---------------------------------------------------------------------------
def test_mask_fn_identity_is_threaded_to_attn():
    def f(T):
        return causal_mask(T)

    blk = Block("blk", _D, _H, mask_fn=f)
    assert blk.attn.mask_fn is f


def test_shifted_mask_changes_block_output():
    rng = make_rng(_FIXTURE_SEED + 21)
    blk_causal = Block("blk", _D, _H, mask_fn=causal_mask)
    blk_shifted = Block("blk", _D, _H, mask_fn=shifted_causal_mask)
    params = _non_init_params(blk_causal, rng)
    x = rng.standard_normal((_B, _T, _D)).astype(np.float64)

    out_causal, _ = blk_causal.forward(x, params)
    out_shifted, _ = blk_shifted.forward(x, params)
    max_abs = float(np.max(np.abs(out_causal - out_shifted)))
    print("causal vs shifted block output max_abs=", max_abs)
    assert max_abs > 1e-3


# ---------------------------------------------------------------------------
# Test-the-tests (monkeypatch only; nothing broken is ever committed in
# nanogpt/block.py itself).
# ---------------------------------------------------------------------------
def test_tt_dropped_identity_residual_term_fails_x():
    blk, params, x = _tiny_gradcheck_fixture()

    def _broken_backward(d_out, cache):
        dz2, g_mlp = blk.mlp.backward(d_out, cache["c_mlp"])
        du2, g_ln2 = blk.ln2.backward(dz2, cache["c_ln2"])
        dx1 = du2  # BUG: dropped the identity term (should be d_out + du2)
        dz1, g_attn = blk.attn.backward(dx1, cache["c_attn"])
        du1, g_ln1 = blk.ln1.backward(dz1, cache["c_ln1"])
        dx = dx1 + du1
        d_params = {}
        ops.accumulate_grads(d_params, g_ln1)
        ops.accumulate_grads(d_params, g_attn)
        ops.accumulate_grads(d_params, g_ln2)
        ops.accumulate_grads(d_params, g_mlp)
        return dx, d_params

    report = gradcheck(blk.forward, _broken_backward, params, x)
    print("dropped identity term gradcheck:", report.tensors["x"])
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL


def test_tt_dropped_second_residual_term_fails_x():
    blk, params, x = _tiny_gradcheck_fixture()

    def _broken_backward(d_out, cache):
        dz2, g_mlp = blk.mlp.backward(d_out, cache["c_mlp"])
        du2, g_ln2 = blk.ln2.backward(dz2, cache["c_ln2"])
        dx1 = d_out + du2
        dz1, g_attn = blk.attn.backward(dx1, cache["c_attn"])
        du1, g_ln1 = blk.ln1.backward(dz1, cache["c_ln1"])
        dx = du1  # BUG: dropped the second residual sum (should be dx1 + du1)
        d_params = {}
        ops.accumulate_grads(d_params, g_ln1)
        ops.accumulate_grads(d_params, g_attn)
        ops.accumulate_grads(d_params, g_ln2)
        ops.accumulate_grads(d_params, g_mlp)
        return dx, d_params

    report = gradcheck(blk.forward, _broken_backward, params, x)
    print("dropped second residual gradcheck:", report.tensors["x"])
    assert report.tensors["x"].rel_err >= GRADCHECK_TOL


def test_tt_swapped_ln_caches_fails_ln_gradcheck():
    blk, params, x = _tiny_gradcheck_fixture()

    def _broken_backward(d_out, cache):
        dz2, g_mlp = blk.mlp.backward(d_out, cache["c_mlp"])
        c_ln2_swapped = dict(cache["c_ln2"])
        c_ln2_swapped["yhat"] = cache["c_ln1"]["yhat"]
        c_ln2_swapped["rstd"] = cache["c_ln1"]["rstd"]
        du2, g_ln2 = blk.ln2.backward(dz2, c_ln2_swapped)  # BUG: ln1's stats
        dx1 = d_out + du2
        dz1, g_attn = blk.attn.backward(dx1, cache["c_attn"])
        c_ln1_swapped = dict(cache["c_ln1"])
        c_ln1_swapped["yhat"] = cache["c_ln2"]["yhat"]
        c_ln1_swapped["rstd"] = cache["c_ln2"]["rstd"]
        du1, g_ln1 = blk.ln1.backward(dz1, c_ln1_swapped)  # BUG: ln2's stats
        dx = dx1 + du1
        d_params = {}
        ops.accumulate_grads(d_params, g_ln1)
        ops.accumulate_grads(d_params, g_attn)
        ops.accumulate_grads(d_params, g_ln2)
        ops.accumulate_grads(d_params, g_mlp)
        return dx, d_params

    report = gradcheck(blk.forward, _broken_backward, params, x)
    print("swapped LN caches gradcheck:",
          report.tensors["blk.ln1.gamma"], report.tensors["blk.ln2.gamma"])
    assert report.tensors["blk.ln1.gamma"].rel_err >= GRADCHECK_TOL
    assert report.tensors["blk.ln2.gamma"].rel_err >= GRADCHECK_TOL


# ---------------------------------------------------------------------------
# Robustness / hygiene
# ---------------------------------------------------------------------------
def test_forward_basic_shape_dtype_finite():
    blk, params, x = _tiny_gradcheck_fixture()
    out, cache = blk.forward(x, params)
    assert out.shape == x.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert set(cache) == {"c_ln1", "c_attn", "c_ln2", "c_mlp"}


def test_no_mutation_of_params_x_and_d_out():
    blk, params, x = _tiny_gradcheck_fixture()
    params_before = {k: v.copy() for k, v in params.items()}
    x_before = x.copy()

    out, cache = blk.forward(x, params)
    for k in params:
        assert np.array_equal(params[k], params_before[k])
    assert np.array_equal(x, x_before)

    rng = make_rng(_FIXTURE_SEED + 400)
    d_out = rng.standard_normal(out.shape).astype(np.float64)
    d_out_before = d_out.copy()
    blk.backward(d_out, cache)
    assert np.array_equal(d_out, d_out_before)
    for k in params:
        assert np.array_equal(params[k], params_before[k])
    assert np.array_equal(x, x_before)

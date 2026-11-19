"""tests/test_param_count.py -- Engineer A, Sprint 3 milestone A2 (S3-04, C1).

Covers `nanogpt.param_count`: the formula (GPT-2 small is formula-only,
R16 -- never instantiated), its preconditions, and the live unique-array
counter cross-checked against a real (tiny/default/max) assembled `GPT`
model from `nanogpt.model` (committed as `[B2]`, "structure final").
See SPRINT_3_PLAN.md section 5.3.

Fixture seed (per-file, never DEFAULT_SEED itself): `DEFAULT_SEED + 302`.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt import param_count as param_count_module
from nanogpt.param_count import formula_from_config, formula_param_count, live_param_count
from nanogpt.config import DEFAULT_SEED, ModelConfig, N_HEAD, make_rng, tiny_config
from nanogpt.model import GPT

_FIXTURE_SEED = DEFAULT_SEED + 302


def _build(cfg: ModelConfig, seed: int):
    rng = make_rng(seed)
    model = GPT(cfg)
    params = model.init_params(rng)
    return model, params


# ---------------------------------------------------------------------------
# Formula values (R16: GPT-2 small is checked by formula only, never
# instantiated -- a live float64 GPT-2-small model is ~1 GB).
# ---------------------------------------------------------------------------
def test_formula_gpt2_small():
    assert formula_param_count(50257, 1024, 768, 12) == 124_439_808


def test_formula_tiny_default_max():
    assert formula_param_count(55, 4, 8, 2) == 2_232
    assert formula_param_count(55, 64, 64, 2) == 107_712
    assert formula_param_count(55, 128, 128, 4) == 816_768


def test_formula_from_config_matches_formula_param_count():
    cfg = ModelConfig(V=55, T=4, d=8, L=2, H=2)
    assert formula_from_config(cfg) == formula_param_count(55, 4, 8, 2)


def test_formula_returns_python_int():
    n = formula_param_count(55, 64, 64, 2)
    assert isinstance(n, int)
    assert not isinstance(n, bool)


def test_formula_per_block_breakdown_matches_12d2_plus_13d_bare_formula():
    for d in (8, 64, 128):
        V, T, L = 55, 4, 1
        # with L=1, N = V*d + T*d + (12d^2+13d) + 2d; isolate the block term.
        n = formula_param_count(V, T, d, L)
        per_block = n - V * d - T * d - 2 * d
        assert per_block == 12 * d * d + 13 * d


def test_formula_rejects_non_positive_or_non_int():
    with pytest.raises(ValueError):
        formula_param_count(0, 4, 8, 2)
    with pytest.raises(ValueError):
        formula_param_count(55, -1, 8, 2)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8.0, 2)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8, True)


# ---------------------------------------------------------------------------
# Preconditions (must raise if violated); monkeypatch this module's view of
# the config constants, per SPRINT_3_PLAN.md section 5.3.
# ---------------------------------------------------------------------------
def test_precondition_mlp_hidden_mult_must_be_4(monkeypatch):
    monkeypatch.setattr(param_count_module, "MLP_HIDDEN_MULT", 3)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8, 2)


def test_precondition_lm_head_has_bias_must_be_false(monkeypatch):
    monkeypatch.setattr(param_count_module, "LM_HEAD_HAS_BIAS", True)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8, 2)


def test_precondition_linear_has_bias_must_be_true(monkeypatch):
    monkeypatch.setattr(param_count_module, "LINEAR_HAS_BIAS", False)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8, 2)


def test_precondition_ln_affine_must_be_true(monkeypatch):
    monkeypatch.setattr(param_count_module, "_LN_IS_AFFINE", False)
    with pytest.raises(ValueError):
        formula_param_count(55, 4, 8, 2)


# ---------------------------------------------------------------------------
# Live vs. formula, on a real assembled GPT (tiny / default / max).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg,expected",
    [
        (tiny_config(55), 2_232),
        (ModelConfig(V=55, T=64, d=64, L=2, H=N_HEAD), 107_712),
        (ModelConfig(V=55, T=128, d=128, L=4), 816_768),
    ],
    ids=["tiny", "default", "max"],
)
def test_live_param_count_matches_formula(cfg, expected):
    model, params = _build(cfg, _FIXTURE_SEED)
    named = model.named_parameters(params)
    formula = formula_from_config(cfg)
    assert formula == expected
    live = live_param_count(named)
    print(f"live={live} formula={formula} (cfg={cfg})")
    assert live == formula


def test_named_parameters_lm_head_is_wte_identity():
    cfg = tiny_config(55)
    model, params = _build(cfg, _FIXTURE_SEED + 1)
    named = model.named_parameters(params)
    assert "lm_head.W" in named
    assert named["lm_head.W"] is params["wte.W"]


def test_per_block_breakdown_matches_12d2_plus_13d_live():
    cfg = tiny_config(55)
    model, params = _build(cfg, _FIXTURE_SEED + 2)
    block0_keys = model.blocks[0].param_keys()
    assert len(block0_keys) == 16
    total = sum(params[k].size for k in block0_keys)
    d = cfg.d
    assert total == 12 * d * d + 13 * d


# ---------------------------------------------------------------------------
# Test-the-tests: naive double-count, transposed-view tie, phantom head bias.
# ---------------------------------------------------------------------------
def test_tt_naive_sum_double_counts_the_tie():
    cfg = tiny_config(55)
    model, params = _build(cfg, _FIXTURE_SEED + 3)
    named = model.named_parameters(params)
    naive = sum(v.size for v in named.values())
    formula = formula_from_config(cfg)
    print("naive sum =", naive, "formula =", formula, "V*d =", cfg.V * cfg.d)
    assert naive == formula + cfg.V * cfg.d


def test_tt_transposed_view_tie_raises_value_error():
    cfg = tiny_config(55)
    model, params = _build(cfg, _FIXTURE_SEED + 4)
    named = dict(model.named_parameters(params))
    named["lm_head.W"] = params["wte.W"].T  # distinct object, shares memory
    with pytest.raises(ValueError):
        live_param_count(named)


def test_tt_extra_head_bias_breaks_live_vs_formula_equality():
    cfg = tiny_config(55)
    model, params = _build(cfg, _FIXTURE_SEED + 5)
    named = dict(model.named_parameters(params))
    named["lm_head.b"] = np.zeros(cfg.V, dtype=np.float64)
    live = live_param_count(named)
    formula = formula_from_config(cfg)
    print("live (with phantom head bias) =", live, "formula =", formula)
    assert live != formula

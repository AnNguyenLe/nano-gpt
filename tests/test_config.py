"""tests/test_config.py — Engineer A, milestone A1.

Verifies nanogpt/config.py is the single source of truth for every §5 name,
with the exact values SPRINT_1_PLAN.md §5 lists, plus the rel_err / dtype /
ModelConfig behaviors §6.4 specifies.
"""

import math

import numpy as np
import pytest

from nanogpt import config


# ---------------------------------------------------------------------------
# Every §5 name exists with the exact value
# ---------------------------------------------------------------------------
def test_dtypes():
    assert config.DTYPE is np.float64
    assert config.INT_DTYPE is np.int64


def test_init_constants():
    assert config.INIT_STD == 0.02
    assert config.BIAS_INIT == 0.0
    assert config.LN_GAMMA_INIT == 1.0
    assert config.LN_BETA_INIT == 0.0
    assert config.LN_EPS == 1e-5


def test_mlp_attention_constants():
    assert config.MLP_HIDDEN_MULT == 4
    assert config.LINEAR_HAS_BIAS is True
    assert config.LM_HEAD_HAS_BIAS is False
    assert config.N_HEAD == 4


def test_gelu_constants():
    assert config.GELU_C == pytest.approx(math.sqrt(2.0 / math.pi), abs=0.0, rel=1e-15)
    assert config.GELU_A == 0.044715


def test_tolerance_constants():
    assert config.FD_STEP == 1e-5
    assert config.REL_ERR_FLOOR == 1e-8
    assert config.GRADCHECK_TOL == 1e-6
    assert config.C3_SVD_REL_TOL == 1e-6
    assert config.C4_PSD_TOL == 1e-10
    assert config.ROUNDOFF_ATOL == 1e-12
    assert config.INIT_LOSS_TOL == 0.1
    assert config.KV_CACHE_ABS_TOL == 1e-10
    assert config.FLOPS_REL_BAND == 0.25


def test_seed_constants():
    assert config.DEFAULT_SEED == 1337
    assert config.CORPUS_SEED == 1337


# ---------------------------------------------------------------------------
# rel_err
# ---------------------------------------------------------------------------
def test_rel_err_zero_zero():
    assert config.rel_err(0.0, 0.0) == 0.0


def test_rel_err_scalar_matches_spec_formula():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = float(rng.normal(scale=10.0))
        b = float(rng.normal(scale=10.0))
        expected = abs(a - b) / max(abs(a) + abs(b), 1e-8)
        assert config.rel_err(a, b) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_rel_err_near_equal_scalars():
    # |1e-9| / max(2.0, 1e-8) = 5e-10
    assert config.rel_err(1.0, 1.0 + 1e-9) == pytest.approx(5e-10, rel=1e-6)


def test_rel_err_array_is_norm_wise_not_elementwise():
    # A tiny element next to a huge one: elementwise ratio could be large,
    # but the norm-wise ratio stays small because the norm is dominated by
    # the huge element.
    a = np.array([1e5, 1e-6])
    b = np.array([1e5, 2e-6])
    r = config.rel_err(a, b)
    expected = np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-8)
    assert r == pytest.approx(expected, rel=1e-12)
    assert r < 1e-8  # norm-wise: the tiny element's absolute diff is negligible


def test_rel_err_elementwise_is_diagnostic_and_elementwise():
    a = np.array([1.0, 2.0, 0.0])
    b = np.array([1.0, 2.0 + 1e-3, 0.0])
    out = config.rel_err_elementwise(a, b)
    assert out.shape == a.shape
    assert out[0] == 0.0
    assert out[2] == 0.0  # 0/0 -> floored, not NaN
    assert out[1] == pytest.approx(1e-3 / max(2.0 + (2.0 + 1e-3), 1e-8))


def test_rel_err_floor_avoids_nan_at_zero():
    assert np.isfinite(config.rel_err(0.0, 0.0))
    assert np.all(np.isfinite(config.rel_err_elementwise(np.zeros(3), np.zeros(3))))


# ---------------------------------------------------------------------------
# make_rng
# ---------------------------------------------------------------------------
def test_make_rng_is_generator_and_deterministic():
    rng1 = config.make_rng(42)
    rng2 = config.make_rng(42)
    assert isinstance(rng1, np.random.Generator)
    np.testing.assert_array_equal(rng1.standard_normal(10), rng2.standard_normal(10))


def test_make_rng_default_seed():
    rng_default = config.make_rng()
    rng_explicit = config.make_rng(config.DEFAULT_SEED)
    np.testing.assert_array_equal(
        rng_default.standard_normal(5), rng_explicit.standard_normal(5)
    )


# ---------------------------------------------------------------------------
# check_float64
# ---------------------------------------------------------------------------
def test_check_float64_passes_on_float64():
    config.check_float64(np.zeros(3, dtype=np.float64), np.ones((2, 2), dtype=np.float64))


def test_check_float64_raises_on_float32():
    with pytest.raises(TypeError):
        config.check_float64(np.zeros(3, dtype=np.float32))


def test_check_float64_raises_on_float16():
    with pytest.raises(TypeError):
        config.check_float64(np.zeros(3, dtype=np.float16))


def test_check_float64_ignores_integer_arrays():
    # Index arrays are int64, not float64, and must not trip this check.
    config.check_float64(np.zeros(3, dtype=np.int64))


def test_check_float64_names_in_message():
    with pytest.raises(TypeError, match="my_tensor"):
        config.check_float64(np.zeros(3, dtype=np.float32), names=["my_tensor"])


# ---------------------------------------------------------------------------
# ModelConfig / tiny_config
# ---------------------------------------------------------------------------
def test_model_config_defaults():
    cfg = config.ModelConfig(V=100)
    assert cfg.V == 100
    assert cfg.T == 64
    assert cfg.d == 64
    assert cfg.L == 2
    assert cfg.H == config.N_HEAD
    assert cfg.head_dim == 64 // config.N_HEAD


def test_model_config_rejects_non_dividing_heads():
    with pytest.raises(ValueError):
        config.ModelConfig(V=10, d=64, H=5)


def test_model_config_rejects_non_positive_fields():
    with pytest.raises(ValueError):
        config.ModelConfig(V=0)
    with pytest.raises(ValueError):
        config.ModelConfig(V=10, T=-1)
    with pytest.raises(ValueError):
        config.ModelConfig(V=10, d=0)
    with pytest.raises(ValueError):
        config.ModelConfig(V=10, L=0)
    with pytest.raises(ValueError):
        config.ModelConfig(V=10, H=0)


def test_model_config_is_frozen():
    cfg = config.ModelConfig(V=10)
    with pytest.raises(Exception):
        cfg.V = 20


def test_tiny_config_shape():
    cfg = config.tiny_config(V=17)
    assert cfg.V == 17
    assert cfg.d == 8
    assert cfg.L == 2
    assert cfg.H == 2
    assert cfg.T == 4
    assert cfg.head_dim == 4

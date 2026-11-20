"""tests/test_adamw.py -- Engineer A, Sprint 4 (Phase 4), milestones A1-A3.

See SPRINT_4_PLAN.md section 5 for the acceptance criteria this file
implements exactly. Must run in < 2 s (no real training, tiny/dev configs
only). No test in this file may assert that a loss decreases.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt import optim
from nanogpt.config import ADAMW_EPS, ROUNDOFF_ATOL, tiny_config
from nanogpt.model import GPT


# ---------------------------------------------------------------------------
# A1: decay / no-decay group membership
# ---------------------------------------------------------------------------
def _block_keys(l: int) -> list:
    return [
        f"h.{l}.ln1.gamma", f"h.{l}.ln1.beta",
        f"h.{l}.attn.q.W", f"h.{l}.attn.q.b",
        f"h.{l}.attn.k.W", f"h.{l}.attn.k.b",
        f"h.{l}.attn.v.W", f"h.{l}.attn.v.b",
        f"h.{l}.attn.o.W", f"h.{l}.attn.o.b",
        f"h.{l}.ln2.gamma", f"h.{l}.ln2.beta",
        f"h.{l}.mlp.fc.W", f"h.{l}.mlp.fc.b",
        f"h.{l}.mlp.proj.W", f"h.{l}.mlp.proj.b",
    ]


def _expected_decay_nodecay(L: int):
    decay = set()
    no_decay = {"wte.W", "wpe.W", "ln_f.gamma", "ln_f.beta"}
    for l in range(L):
        decay.update({
            f"h.{l}.attn.q.W", f"h.{l}.attn.k.W",
            f"h.{l}.attn.v.W", f"h.{l}.attn.o.W",
            f"h.{l}.mlp.fc.W", f"h.{l}.mlp.proj.W",
        })
        no_decay.update({
            f"h.{l}.ln1.gamma", f"h.{l}.ln1.beta",
            f"h.{l}.attn.q.b", f"h.{l}.attn.k.b",
            f"h.{l}.attn.v.b", f"h.{l}.attn.o.b",
            f"h.{l}.ln2.gamma", f"h.{l}.ln2.beta",
            f"h.{l}.mlp.fc.b", f"h.{l}.mlp.proj.b",
        })
    return decay, no_decay


@pytest.mark.parametrize("cfg_kind", ["tiny", "default"])
def test_decay_groups_match_param_keys_exactly(cfg_kind):
    if cfg_kind == "tiny":
        cfg = tiny_config(V=17)
    else:
        from nanogpt.config import ModelConfig
        cfg = ModelConfig(V=55)
    model = GPT(cfg)
    keys = model.param_keys()

    expected_decay, expected_no_decay = _expected_decay_nodecay(cfg.L)
    assert set(keys) == expected_decay | expected_no_decay  # sanity on the fixture itself

    actual_decay = {k for k in keys if optim.is_decay_param(k)}
    actual_no_decay = {k for k in keys if not optim.is_decay_param(k)}
    assert actual_decay == expected_decay
    assert actual_no_decay == expected_no_decay
    assert len(actual_decay) == 6 * cfg.L
    assert len(actual_no_decay) == 4 + 10 * cfg.L


def test_decay_element_counts_at_default_config():
    from nanogpt.config import ModelConfig
    from nanogpt.param_count import formula_from_config

    cfg = ModelConfig(V=55)
    model = GPT(cfg)
    params = model.init_params(np.random.default_rng(0))
    keys = model.param_keys()

    decay_elems = sum(params[k].size for k in keys if optim.is_decay_param(k))
    no_decay_elems = sum(params[k].size for k in keys if not optim.is_decay_param(k))

    assert decay_elems == 98_304
    assert no_decay_elems == 9_408
    assert decay_elems + no_decay_elems == formula_from_config(cfg)


def test_gamma_wte_wpe_are_no_decay():
    assert optim.is_decay_param("ln_f.gamma") is False
    assert optim.is_decay_param("h.0.ln1.gamma") is False
    assert optim.is_decay_param("wte.W") is False
    assert optim.is_decay_param("wpe.W") is False


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        optim.is_decay_param("totally.unknown.key")


def test_lm_head_alias_raises():
    with pytest.raises(ValueError):
        optim.is_decay_param("lm_head.W")


def test_near_miss_key_raises():
    with pytest.raises(ValueError):
        optim.is_decay_param("h.0.attn.q.Wx")


# ---------------------------------------------------------------------------
# A1: canonicalize_params
# ---------------------------------------------------------------------------
def test_canonicalize_params_same_keys_as_original_and_identity_preserved():
    cfg = tiny_config(V=11)
    model = GPT(cfg)
    p = model.init_params(np.random.default_rng(0))
    named = model.named_parameters(p)  # p plus "lm_head.W" aliasing p["wte.W"]

    canon = optim.canonicalize_params(named)
    assert set(canon.keys()) == set(p.keys())
    assert "lm_head.W" not in canon
    assert canon["wte.W"] is p["wte.W"]
    for k in p:
        assert canon[k] is p[k]


def test_canonicalize_params_passthrough_on_plain_params():
    cfg = tiny_config(V=11)
    model = GPT(cfg)
    p = model.init_params(np.random.default_rng(0))
    canon = optim.canonicalize_params(p)
    assert set(canon.keys()) == set(p.keys())
    for k in p:
        assert canon[k] is p[k]


def test_canonicalize_params_rejects_transposed_view_pair():
    X = np.ascontiguousarray(np.arange(12.0).reshape(3, 4))
    with pytest.raises(ValueError):
        optim.canonicalize_params({"a": X, "b": X.T})


def test_canonicalize_params_rejects_slice_view_partial_overlap():
    Y = np.ascontiguousarray(np.arange(20.0).reshape(4, 5))
    Z = Y[:2]  # distinct object, C-contiguous, shares memory with Y
    assert Z.flags["C_CONTIGUOUS"]
    assert Z is not Y
    with pytest.raises(ValueError):
        optim.canonicalize_params({"full": Y, "part": Z})


def test_canonicalize_params_rejects_float32():
    bad = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(Exception):
        optim.canonicalize_params({"a": bad})


def test_canonicalize_params_rejects_fortran_order():
    bad = np.asfortranarray(np.zeros((2, 3)))
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        optim.canonicalize_params({"a": bad})


# ---------------------------------------------------------------------------
# A2: closed-form AdamW anchors (SPRINT_4_PLAN.md section 3.2, literal tables)
# ---------------------------------------------------------------------------
DECAY_KEY = "h.0.mlp.fc.W"       # a real decay-group key
NO_DECAY_KEY = "h.0.mlp.fc.b"    # a real no-decay-group key
ETA = 0.1
BETAS_ANCHOR = (0.9, 0.999)
LAMBDA = 0.01


def _make_opt(key, theta0):
    params = {key: theta0.copy()}
    return optim.AdamW(params, lr=ETA, betas=BETAS_ANCHOR, eps=ADAMW_EPS, weight_decay=LAMBDA)


def test_first_step_anchor_decay_key():
    theta0 = np.array([0.7, -1.2, 0.5], dtype=np.float64)
    g = np.array([3.0, -2e-3, 0.0], dtype=np.float64)
    opt = _make_opt(DECAY_KEY, theta0)

    opt.step({DECAY_KEY: g})
    theta1 = opt.params[DECAY_KEY]
    delta = theta1 - theta0

    closed_form = -ETA * (g / (np.abs(g) + ADAMW_EPS) + LAMBDA * theta0)
    np.testing.assert_allclose(delta, closed_form, atol=ROUNDOFF_ATOL, rtol=0.0)

    # within eta*eps/|g| of -eta*sign(g) - eta*lambda*theta0, for g != 0
    nonzero = g != 0.0
    approx = -ETA * np.sign(g) - ETA * LAMBDA * theta0
    # eps/(|g|+eps) < eps/|g| exactly, so this bound holds mathematically;
    # allow a hair of float64 roundoff slack on top of it.
    tol = ETA * ADAMW_EPS / np.abs(g[nonzero]) + ROUNDOFF_ATOL
    assert np.all(np.abs(delta[nonzero] - approx[nonzero]) <= tol)

    # the g == 0 element gets exactly -eta*lambda*theta0
    zero = ~nonzero
    np.testing.assert_allclose(
        delta[zero], -ETA * LAMBDA * theta0[zero], atol=ROUNDOFF_ATOL, rtol=0.0
    )

    b1, b2 = BETAS_ANCHOR
    np.testing.assert_allclose(opt.state[DECAY_KEY]["m"], (1 - b1) * g, atol=ROUNDOFF_ATOL, rtol=0.0)
    np.testing.assert_allclose(opt.state[DECAY_KEY]["v"], (1 - b2) * g * g, atol=ROUNDOFF_ATOL, rtol=0.0)
    assert opt.t == 1


def test_first_step_no_decay_key_ignores_lambda():
    theta0 = np.array([0.7, -1.2, 0.5], dtype=np.float64)
    g = np.array([3.0, -2e-3, 0.0], dtype=np.float64)
    opt = _make_opt(NO_DECAY_KEY, theta0)

    opt.step({NO_DECAY_KEY: g})
    theta1 = opt.params[NO_DECAY_KEY]
    delta = theta1 - theta0

    closed_form = -ETA * g / (np.abs(g) + ADAMW_EPS)
    np.testing.assert_allclose(delta, closed_form, atol=ROUNDOFF_ATOL, rtol=0.0)


# §3.2 hand-computed anchor table: f(theta) = 1/2 sum a_i theta_i^2, so
# g_i = a_i * theta_i, computed from the CURRENT theta at every step (never
# recomputed by the implementation -- these are literal Decimal-derived
# values typed in from the plan, not values this test recomputes).
THETA0 = np.array([1.0, -0.5, 2.0], dtype=np.float64)
A_COEFFS = np.array([2.0, 4.0, 0.5], dtype=np.float64)

TRAJECTORY_DECAY = {
    1: {
        "g": [2.0, -2.0, 1.0],
        "m": [0.2, -0.2, 0.1],
        "v": [0.004, 0.004, 0.001],
        "m_hat": [2.0, -2.0, 1.0],
        "v_hat": [4.0, 4.0, 1.0],
        "theta": [0.89900000049999995, -0.39950000050000001, 1.898000001],
    },
    2: {
        "g": [1.7980000009999999, -1.598000002, 0.9490000005],
        "m": [0.35980000010000002, -0.33980000020000001, 0.18490000005000001],
        "v": [0.0072288040035960004, 0.0065496040063920003, 0.0018996010009489999],
        "m_hat": [1.8936842110526315, -1.7884210536842104, 0.97315789500000005],
        "v_hat": [3.6162101068514256, 3.2764402233076537, 0.95027563829364681],
        "theta": [0.79851902716852097, -0.30029783822405853, 1.7962725901605623],
    },
    3: {
        "g": [1.5970380543370419, -1.2011913528962341, 0.89813629508028114],
        "m": [0.48352380552370416, -0.42593913546962342, 0.25622362955302813],
        "v": [0.0097721057465930475, 0.007985915068658294, 0.0027043502044885848],
        "m_hat": [1.7842206845893143, -1.5717311271941823, 0.94547464779715173],
        "v_hat": [3.2606281234450867, 2.6646354367777301, 0.90235211949832006],
        "theta": [0.69891118315823153, -0.20371238138538447, 1.6949445166806532],
    },
}

TRAJECTORY_NO_DECAY_THETA = {
    1: [0.90000000049999995, -0.40000000050000001, 1.900000001],
    2: [0.80041222869179218, -0.30118742062342935, 1.8001664876318755],
    3: [0.7015862729460296, -0.20487125097072312, 1.7006233943434106],
}


def test_three_step_trajectory_decay_case_matches_table_at_every_step():
    opt = _make_opt(DECAY_KEY, THETA0)
    theta = THETA0.copy()
    for t in (1, 2, 3):
        g = A_COEFFS * theta  # computed from the CURRENT theta, before this step
        row = TRAJECTORY_DECAY[t]
        np.testing.assert_allclose(g, row["g"], atol=ROUNDOFF_ATOL, rtol=0.0)

        opt.step({DECAY_KEY: g})
        theta = opt.params[DECAY_KEY]

        b1, b2 = BETAS_ANCHOR
        m_hat = opt.state[DECAY_KEY]["m"] / (1 - b1 ** t)
        v_hat = opt.state[DECAY_KEY]["v"] / (1 - b2 ** t)

        np.testing.assert_allclose(theta, row["theta"], atol=ROUNDOFF_ATOL, rtol=0.0)
        np.testing.assert_allclose(opt.state[DECAY_KEY]["m"], row["m"], atol=ROUNDOFF_ATOL, rtol=0.0)
        np.testing.assert_allclose(opt.state[DECAY_KEY]["v"], row["v"], atol=ROUNDOFF_ATOL, rtol=0.0)
        np.testing.assert_allclose(m_hat, row["m_hat"], atol=ROUNDOFF_ATOL, rtol=0.0)
        np.testing.assert_allclose(v_hat, row["v_hat"], atol=ROUNDOFF_ATOL, rtol=0.0)


def test_three_step_trajectory_no_decay_case_matches_table_theta_only():
    opt = _make_opt(NO_DECAY_KEY, THETA0)
    theta = THETA0.copy()
    for t in (1, 2, 3):
        g = A_COEFFS * theta
        opt.step({NO_DECAY_KEY: g})
        theta = opt.params[NO_DECAY_KEY]
        np.testing.assert_allclose(
            theta, TRAJECTORY_NO_DECAY_THETA[t], atol=ROUNDOFF_ATOL, rtol=0.0
        )


def test_decoupled_decay_zero_gradient_geometric_decay():
    theta0 = np.array([1.0, -0.5, 2.0], dtype=np.float64)
    opt = _make_opt(DECAY_KEY, theta0)
    g = np.zeros(3, dtype=np.float64)

    expected = theta0.copy()
    for t in (1, 2, 3):
        opt.step({DECAY_KEY: g})
        assert np.all(opt.state[DECAY_KEY]["m"] == 0.0)
        assert np.all(opt.state[DECAY_KEY]["v"] == 0.0)
        expected = expected * (1.0 - ETA * LAMBDA)
        np.testing.assert_allclose(opt.params[DECAY_KEY], expected, atol=ROUNDOFF_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# A3: tied single optimizer state (PO P0 dedicated test)
# ---------------------------------------------------------------------------
from nanogpt.config import ADAMW_BETAS, ADAMW_WEIGHT_DECAY, DEFAULT_SEED, make_rng  # noqa: E402

_A3_SEED = DEFAULT_SEED + 401  # per-file fixture offset (never DEFAULT_SEED itself)
_A3_LR = 0.01


def _tiny_model_batch(V=11):
    cfg = tiny_config(V=V)
    model = GPT(cfg)
    params = model.init_params(make_rng(_A3_SEED))
    rng = make_rng(_A3_SEED + 1)
    idx = rng.integers(0, V, size=(2, cfg.T))
    tgt = rng.integers(0, V, size=(2, cfg.T))
    loss, cache = model.forward_loss(idx, tgt, params)
    _, grads = model.backward_loss(1.0, cache)
    return model, params, grads


def test_tied_single_state_on_real_model():
    model, p, grads = _tiny_model_batch()
    wte_obj = p["wte.W"]
    wte0 = wte_obj.copy()
    named = model.named_parameters(p)
    assert named["lm_head.W"] is p["wte.W"]

    opt = optim.AdamW(named, lr=_A3_LR)
    assert len(opt.state) == len(model.param_keys())
    assert "lm_head.W" not in opt.state
    assert "lm_head.W" not in opt.params
    assert opt.params["wte.W"] is p["wte.W"]

    opt.step(grads)

    b1, b2 = ADAMW_BETAS
    g = grads["wte.W"]
    # one accumulation (not two): m == (1 - b1) * g exactly once
    np.testing.assert_allclose(opt.state["wte.W"]["m"], (1 - b1) * g, atol=ROUNDOFF_ATOL, rtol=0.0)
    np.testing.assert_allclose(opt.state["wte.W"]["v"], (1 - b2) * g * g, atol=ROUNDOFF_ATOL, rtol=0.0)

    # wte.W is no-decay: single-step closed form is -lr * g / (|g| + eps)
    expected = wte0 - _A3_LR * g / (np.abs(g) + ADAMW_EPS)
    np.testing.assert_allclose(p["wte.W"], expected, atol=ROUNDOFF_ATOL, rtol=0.0)

    assert p["wte.W"] is wte_obj
    assert model.named_parameters(p)["lm_head.W"] is p["wte.W"]
    assert p["wte.W"].flags["C_CONTIGUOUS"]
    assert opt.t == 1


def test_tied_two_state_adversary_moves_wte_about_twice_as_far():
    model, p_real, grads = _tiny_model_batch()
    wte0 = p_real["wte.W"].copy()

    # real optimizer
    opt = optim.AdamW(model.named_parameters(p_real), lr=_A3_LR)
    opt.step(grads)
    real_delta = p_real["wte.W"] - wte0

    # test-local adversary: one (m, v) per named_parameters() key => two
    # states and two updates of the SAME array.
    p_bad = {k: v.copy() for k, v in p_real.items()}
    p_bad["wte.W"] = wte0.copy()
    named_bad = model.named_parameters(p_bad)
    b1, b2 = ADAMW_BETAS
    g = grads["wte.W"]
    for key in ("wte.W", "lm_head.W"):
        arr = named_bad[key]
        m = (1 - b1) * g
        v = (1 - b2) * g * g
        arr -= _A3_LR * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + ADAMW_EPS)
    bad_delta = p_bad["wte.W"] - wte0

    assert np.linalg.norm(bad_delta - real_delta) > 1e-3 * _A3_LR
    # ... and specifically about 2x the real move
    ratio = np.linalg.norm(bad_delta) / np.linalg.norm(real_delta)
    assert 1.9 < ratio < 2.1


def test_step_rejects_lm_head_alias_in_grads():
    model, p, grads = _tiny_model_batch()
    opt = optim.AdamW(model.named_parameters(p), lr=_A3_LR)
    with pytest.raises(ValueError):
        opt.step({**grads, "lm_head.W": grads["wte.W"]})
    assert opt.t == 0  # failed validation must not advance the counter


def test_adamw_default_config_single_state():
    from nanogpt.config import ModelConfig

    model = GPT(ModelConfig(V=55))
    p = model.init_params(make_rng(_A3_SEED))
    opt = optim.AdamW(model.named_parameters(p), lr=_A3_LR)
    assert len(opt.state) == len(model.param_keys()) == 4 + 16 * 2
    assert "lm_head.W" not in opt.state


# ---------------------------------------------------------------------------
# A3: test-the-tests -- each buggy variant must MISS the section 3.2 tables
# ---------------------------------------------------------------------------
def _ref_run(variant, theta0, steps=3, decay=True):
    """Test-local scalar-loop AdamW with an injectable bug. 'correct' is the
    spec and must reproduce the tables; the others must not."""
    b1, b2 = BETAS_ANCHOR
    m = np.zeros_like(theta0)
    v = np.zeros_like(theta0)
    theta = theta0.copy()
    out = []
    with np.errstate(all="ignore"):
        for t in range(1, steps + 1):
            g = A_COEFFS * theta
            tt = t + 1 if variant == "off_by_one_t" else t
            if variant == "t_from_zero":
                tt = t - 1
            if variant == "coupled_l2" and decay:
                g = g + LAMBDA * theta
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            if variant == "no_bias_correction":
                m_hat, v_hat = m, v
            else:
                m_hat = m / (1 - b1 ** tt)
                v_hat = v / (1 - b2 ** tt)
            u = m_hat / (np.sqrt(v_hat) + ADAMW_EPS)
            if (variant == "correct" and decay) or variant == "decay_on_excluded":
                u = u + LAMBDA * theta
            theta = theta - ETA * u
            out.append(theta.copy())
    return out


def _max_miss(thetas, table):
    return max(float(np.max(np.abs(thetas[t - 1] - np.asarray(table[t])))) for t in (1, 2, 3))


def test_reference_loop_correct_variant_matches_both_tables():
    assert _max_miss(_ref_run("correct", THETA0, decay=True),
                     {t: TRAJECTORY_DECAY[t]["theta"] for t in (1, 2, 3)}) < 1e-12
    assert _max_miss(_ref_run("correct", THETA0, decay=False),
                     TRAJECTORY_NO_DECAY_THETA) < 1e-12


@pytest.mark.parametrize("variant", ["off_by_one_t", "no_bias_correction", "coupled_l2"])
def test_buggy_variants_miss_decay_table(variant):
    table = {t: TRAJECTORY_DECAY[t]["theta"] for t in (1, 2, 3)}
    assert _max_miss(_ref_run(variant, THETA0, decay=True), table) > 1e-6


def test_decay_on_excluded_key_misses_no_decay_table():
    assert _max_miss(_ref_run("decay_on_excluded", THETA0, decay=False),
                     TRAJECTORY_NO_DECAY_THETA) > 1e-6


def test_t_from_zero_gives_non_finite_first_update():
    first = _ref_run("t_from_zero", THETA0, steps=1, decay=True)[0]
    assert not np.all(np.isfinite(first))


def test_real_optimizer_matches_where_every_bug_fails():
    opt = _make_opt(DECAY_KEY, THETA0)
    got = {}
    for t in (1, 2, 3):
        opt.step({DECAY_KEY: A_COEFFS * opt.params[DECAY_KEY]})
        got[t] = opt.params[DECAY_KEY].copy()
    assert _max_miss([got[1], got[2], got[3]],
                     {t: TRAJECTORY_DECAY[t]["theta"] for t in (1, 2, 3)}) < 1e-12


# ---------------------------------------------------------------------------
# A3: validation, non-mutation, determinism
# ---------------------------------------------------------------------------
def _small_opt():
    return optim.AdamW({"h.0.mlp.fc.W": np.ones((2, 3)), "h.0.mlp.fc.b": np.zeros(3)}, lr=0.1)


def test_step_rejects_missing_extra_wrong_shape_and_float32_grads():
    good = {"h.0.mlp.fc.W": np.ones((2, 3)), "h.0.mlp.fc.b": np.ones(3)}
    for bad in (
        {"h.0.mlp.fc.W": np.ones((2, 3))},                                   # missing
        {**good, "h.0.mlp.fc.extra": np.ones(1)},                            # extra
        {**good, "h.0.mlp.fc.W": np.ones((3, 2))},                           # wrong shape
        {**good, "h.0.mlp.fc.b": np.ones(3, dtype=np.float32)},              # float32
    ):
        opt = _small_opt()
        before = {k: v.copy() for k, v in opt.params.items()}
        with pytest.raises((ValueError, TypeError)):
            opt.step(bad)
        assert opt.t == 0
        for k in before:
            assert np.array_equal(opt.params[k], before[k])


@pytest.mark.parametrize("kwargs", [
    {"lr": 0.0}, {"lr": -1.0},
    {"lr": 0.1, "betas": (1.0, 0.9)}, {"lr": 0.1, "betas": (0.9, 1.0)},
    {"lr": 0.1, "betas": (-0.1, 0.9)},
    {"lr": 0.1, "eps": 0.0}, {"lr": 0.1, "eps": -1e-8},
    {"lr": 0.1, "weight_decay": -0.1},
])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        optim.AdamW({"h.0.mlp.fc.b": np.zeros(2)}, **kwargs)


def test_step_does_not_mutate_grads():
    opt = _small_opt()
    grads = {"h.0.mlp.fc.W": np.full((2, 3), 0.5), "h.0.mlp.fc.b": np.full(3, -0.25)}
    snap = {k: v.copy() for k, v in grads.items()}
    opt.step(grads)
    for k in grads:
        assert np.array_equal(grads[k], snap[k])


def test_two_identical_runs_are_bit_identical():
    def run():
        model, p, grads = _tiny_model_batch()
        opt = optim.AdamW(model.named_parameters(p), lr=_A3_LR, weight_decay=ADAMW_WEIGHT_DECAY)
        for _ in range(3):
            opt.step(grads)
        return {k: v.copy() for k, v in p.items()}

    a, b = run(), run()
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k])

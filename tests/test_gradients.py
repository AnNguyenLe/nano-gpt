"""tests/test_gradients.py -- Engineer C, Sprint 3: C6 full-model gradcheck
(SPRINT_3_PLAN.md S3-06, section 7.2/7.3).

Reserved for Phase 3 since Sprint 1 (R6/R13); this is its first content.

G1: gradcheck(model.forward, model.backward, params, idx) -- logits objective.
G2: gradcheck of forward_loss/backward_loss (wrapped), fixed targets -- loss objective.
G3: gradcheck of forward_from_embedding/backward_from_embedding on params minus
    "wpe.W", w.r.t. the float embedding-stream input h0 (TL-18).

Every non-exempt tensor must clear GRADCHECK_TOL (1e-6). The TWO "h.{l}.attn.k.b"
tensors are a structural zero (softmax shift-invariance) and are gated instead by
the TL-9 (a)-(d) protocol, never the literal rel_err gate (PO ratification
condition, carried from Sprint 2).

Fixture seed: DEFAULT_SEED + 321 (never DEFAULT_SEED itself for float fixtures --
Sprint 1 seed-collision watch item). Tiny config `tiny_config(55)`, non-init
fixture (SPRINT_3_PLAN.md section 3.6): init params, then W += 0.3*N, b/beta +=
0.3*N, gamma = 1 + 0.3*N, every array made C-contiguous.

Only `nanogpt.gradcheck.gradcheck` / `numerical_jacobian` are used -- no bespoke
FD loops. Expensive reports are module-scoped fixtures so each gradcheck runs
once per file (<10s file budget).
"""
from __future__ import annotations

import numpy as np
import pytest

from nanogpt.config import (
    DEFAULT_SEED,
    GRADCHECK_TOL,
    REL_ERR_FLOOR,
    ROUNDOFF_ATOL,
    make_rng,
    tiny_config,
)
from nanogpt import attention
from nanogpt.gradcheck import gradcheck
from nanogpt.loss import cross_entropy_forward, cross_entropy_backward
from nanogpt.model import GPT

_FIXTURE_SEED = DEFAULT_SEED + 321
V = 55


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    return tiny_config(V)


@pytest.fixture(scope="module")
def model(cfg):
    return GPT(cfg)


@pytest.fixture(scope="module")
def exempt_keys(cfg):
    return {f"h.{l}.attn.k.b" for l in range(cfg.L)}


@pytest.fixture(scope="module")
def non_init_params(model, cfg):
    """Non-init fixture (section 3.6): init, then perturb every tensor so
    gradients are not degenerate (beta=0/gamma=1/bias=0 at pure init)."""
    rng = make_rng(_FIXTURE_SEED)
    params = model.init_params(rng)
    out = {}
    for k, v in params.items():
        if k.endswith(".gamma"):
            out[k] = 1.0 + 0.3 * rng.standard_normal(v.shape)
        else:
            out[k] = v + 0.3 * rng.standard_normal(v.shape)
        out[k] = np.ascontiguousarray(out[k].astype(np.float64))
    return out


@pytest.fixture(scope="module")
def idx(cfg):
    rng = make_rng(_FIXTURE_SEED + 1)
    return rng.integers(0, V, size=(2, cfg.T))


@pytest.fixture(scope="module")
def targets(cfg):
    rng = make_rng(_FIXTURE_SEED + 2)
    return rng.integers(0, V, size=(2, cfg.T))


# ---------------------------------------------------------------------------
# G1: logits objective -- model.forward / model.backward
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def g1_report(model, non_init_params, idx):
    return gradcheck(model.forward, model.backward, non_init_params, idx, seed=_FIXTURE_SEED + 3)


def test_g1_key_coverage(g1_report, non_init_params):
    assert set(g1_report.tensors.keys()) == set(non_init_params.keys())
    assert len(g1_report.tensors) == 36


def test_g1_non_exempt_rel_err(g1_report, exempt_keys):
    print("\n[G1 logits-objective gradcheck]")
    for name, t in g1_report.tensors.items():
        print("  ", t)
    for name, t in g1_report.tensors.items():
        if name in exempt_keys:
            continue
        assert t.rel_err < GRADCHECK_TOL, f"{name}: rel_err={t.rel_err:.3e} >= {GRADCHECK_TOL:.1e}"


# ---------------------------------------------------------------------------
# G2: loss objective -- forward_loss / backward_loss wrapped to the harness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def g2_report(model, non_init_params, idx, targets):
    def forward_wrapped(ii, pp):
        return model.forward_loss(ii, targets, pp)

    def backward_wrapped(d_loss, cache):
        return model.backward_loss(d_loss, cache)

    return gradcheck(forward_wrapped, backward_wrapped, non_init_params, idx, seed=_FIXTURE_SEED + 4)


def test_g2_key_coverage(g2_report, non_init_params):
    assert set(g2_report.tensors.keys()) == set(non_init_params.keys())
    assert len(g2_report.tensors) == 36


def test_g2_non_exempt_rel_err(g2_report, exempt_keys):
    print("\n[G2 loss-objective gradcheck]")
    for name, t in g2_report.tensors.items():
        print("  ", t)
    for name, t in g2_report.tensors.items():
        if name in exempt_keys:
            continue
        assert t.rel_err < GRADCHECK_TOL, f"{name}: rel_err={t.rel_err:.3e} >= {GRADCHECK_TOL:.1e}"


# ---------------------------------------------------------------------------
# G3: continuous input coverage (TL-18) -- forward_from_embedding w.r.t. h0,
# params minus "wpe.W" (that sub-map never reads it).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def params_minus_wpe(non_init_params):
    return {k: v for k, v in non_init_params.items() if k != "wpe.W"}


@pytest.fixture(scope="module")
def h0(model, non_init_params, idx):
    h, _ = model.embed(idx, non_init_params)
    return np.ascontiguousarray(h)


@pytest.fixture(scope="module")
def g3_report(model, params_minus_wpe, h0):
    return gradcheck(
        model.forward_from_embedding,
        model.backward_from_embedding,
        params_minus_wpe,
        h0,
        seed=_FIXTURE_SEED + 5,
    )


def test_g3_key_coverage(g3_report, params_minus_wpe):
    expected = set(params_minus_wpe.keys()) | {"x"}
    assert set(g3_report.tensors.keys()) == expected
    assert len(g3_report.tensors) == 35 + 1


def test_g3_non_exempt_rel_err(g3_report, exempt_keys):
    print("\n[G3 forward_from_embedding gradcheck, w.r.t. h0]")
    for name, t in g3_report.tensors.items():
        print("  ", t)
    for name, t in g3_report.tensors.items():
        if name in exempt_keys:
            continue
        assert t.rel_err < GRADCHECK_TOL, f"{name}: rel_err={t.rel_err:.3e} >= {GRADCHECK_TOL:.1e}"


# ---------------------------------------------------------------------------
# TL-9 (a)-(d), applied to EVERY layer's h.{l}.attn.k.b (PO ratification
# condition, restated section 7.2). Report but do not gate rel_err /
# max_elem_rel_err for these tensors.
# ---------------------------------------------------------------------------
def test_tl9_exempt_set_is_exact(exempt_keys, cfg):
    assert exempt_keys == {f"h.{l}.attn.k.b" for l in range(cfg.L)}
    assert len(exempt_keys) == cfg.L == 2


def test_tl9_a_analytic_zero(model, non_init_params, idx, exempt_keys, cfg):
    logits, cache = model.forward(idx, non_init_params)
    R = make_rng(_FIXTURE_SEED + 6).standard_normal(logits.shape)
    _, grads = model.backward(R, cache)
    for l in range(cfg.L):
        key = f"h.{l}.attn.k.b"
        m = float(np.max(np.abs(grads[key])))
        print(f"TL-9(a) layer {l}: analytic max|d{key}|={m:.3e}")
        assert m <= ROUNDOFF_ATOL, f"TL-9(a) layer {l}: {m:.3e} > ROUNDOFF_ATOL"


def test_tl9_b_fd_noise_bound(g1_report, g2_report, exempt_keys, cfg):
    for l in range(cfg.L):
        key = f"h.{l}.attn.k.b"
        m1 = g1_report.tensors[key].max_abs_err
        m2 = g2_report.tensors[key].max_abs_err
        print(f"TL-9(b) layer {l}: FD max_abs_err logits={m1:.3e} loss={m2:.3e}")
        assert m1 <= REL_ERR_FLOOR, f"TL-9(b) layer {l} (logits): {m1:.3e} > REL_ERR_FLOOR"
        assert m2 <= REL_ERR_FLOOR, f"TL-9(b) layer {l} (loss): {m2:.3e} > REL_ERR_FLOOR"


def test_tl9_c_invariance(model, non_init_params, idx, cfg):
    logits0, _ = model.forward(idx, non_init_params)
    for l in range(cfg.L):
        key = f"h.{l}.attn.k.b"
        rng = make_rng(_FIXTURE_SEED + 7 + l)
        shifted = dict(non_init_params)
        shifted[key] = np.ascontiguousarray(
            non_init_params[key] + 10.0 * rng.standard_normal(non_init_params[key].shape)
        )
        logits1, _ = model.forward(idx, shifted)
        m = float(np.max(np.abs(logits1 - logits0)))
        print(f"TL-9(c) layer {l}: invariance max|delta logits|={m:.3e}")
        assert m <= ROUNDOFF_ATOL, f"TL-9(c) layer {l}: {m:.3e} > ROUNDOFF_ATOL"


def test_tl9_d_dropped_term_adversary(model, non_init_params, idx, exempt_keys, cfg, monkeypatch):
    logits, cache = model.forward(idx, non_init_params)
    R = make_rng(_FIXTURE_SEED + 20).standard_normal(logits.shape)
    monkeypatch.setattr(attention.ops, "softmax_backward", lambda dp, cc: (cc["p"] * dp, {}))
    _, grads = model.backward(R, cache)
    for l in range(cfg.L):
        key = f"h.{l}.attn.k.b"
        m = float(np.max(np.abs(grads[key])))
        print(f"TL-9(d) layer {l}: dropped-term max|d{key}|={m:.3e}")
        assert m > 1e-3, f"TL-9(d) layer {l}: dropped-term adversary did not discriminate ({m:.3e})"


# ---------------------------------------------------------------------------
# Test-the-test: a broken full-model backward (ln_f's backward replaced by an
# identity-like broken variant) must make G1 fail on at least one tensor.
# ---------------------------------------------------------------------------
def test_test_the_test_broken_ln_f_backward_fails_g1(model, non_init_params, idx, monkeypatch):
    def broken_ln_f_backward(d_out, cache):
        # Drops the gamma-scaling / mean-centering terms entirely: passes
        # d_out straight through as dx, and zeros for the affine grads.
        return d_out, {
            model.ln_f.gamma_key: np.zeros_like(non_init_params[model.ln_f.gamma_key]),
            model.ln_f.beta_key: np.zeros_like(non_init_params[model.ln_f.beta_key]),
        }

    monkeypatch.setattr(model.ln_f, "backward", broken_ln_f_backward)
    report = gradcheck(model.forward, model.backward, non_init_params, idx, seed=_FIXTURE_SEED + 30)
    assert not report.passed, "test-the-test: broken ln_f.backward should make the full-model gate fail"
    print("[test-the-test] broken ln_f.backward -> worst:", report.worst())

"""tests/test_weight_tying.py -- Engineer C, Sprint 3: C7 weight tying
(SPRINT_3_PLAN.md S3-07, section 7.2/7.3, TL-17, TL-20).

Structure: params holds exactly "wte.W" (no "lm_head.W" / head-bias key);
named_parameters()["lm_head.W"] is params["wte.W"] (identity); wte.W is
C-contiguous with base None.

Independent decomposition (TL-20, not tautological): g_full, g_in
(include_output_side=False), g_out (include_input_side=False) for the SAME
upstream R. The partial-sum identity is checked, AND each partial is checked
against an INDEPENDENT finite-difference estimate built from a different
sub-map (forward_from_embedding / forward), never from the production
backward's own decomposition.

Accumulation (structural): an AST scan of nanogpt/model.py finds no
assignment whose target is a subscript with the literal "wte.W".

Fixture seed: DEFAULT_SEED + 322 (never DEFAULT_SEED itself). Tiny config,
non-init fixture (section 3.6). Only nanogpt.gradcheck.gradcheck /
numerical_jacobian -- no bespoke FD loops.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from nanogpt.config import (
    DEFAULT_SEED,
    GRADCHECK_TOL,
    ROUNDOFF_ATOL,
    make_rng,
    rel_err,
    tiny_config,
)
from nanogpt import model as model_module
from nanogpt import ops
from nanogpt.gradcheck import gradcheck, numerical_jacobian
from nanogpt.model import GPT

_FIXTURE_SEED = DEFAULT_SEED + 322
V = 55


@pytest.fixture(scope="module")
def cfg():
    return tiny_config(V)


@pytest.fixture(scope="module")
def model(cfg):
    return GPT(cfg)


@pytest.fixture(scope="module")
def non_init_params(model):
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


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_no_lm_head_key_in_params(non_init_params):
    assert "wte.W" in non_init_params
    assert not any("lm_head" in k for k in non_init_params)


def test_named_parameters_lm_head_is_wte_identity(model, non_init_params):
    named = model.named_parameters(non_init_params)
    assert "lm_head.W" in named
    assert named["lm_head.W"] is non_init_params["wte.W"]


def test_wte_w_contiguous_no_base(non_init_params):
    W = non_init_params["wte.W"]
    assert W.flags.c_contiguous
    assert W.base is None


# ---------------------------------------------------------------------------
# Partials (production decomposition) + independent FD (TL-20)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def upstream_r(model, non_init_params, idx):
    logits, cache = model.forward(idx, non_init_params)
    R = make_rng(_FIXTURE_SEED + 2).standard_normal(logits.shape)
    return R, cache


@pytest.fixture(scope="module")
def partials(model, upstream_r):
    R, cache = upstream_r
    _, g_full = model.backward(R, cache)
    _, g_in = model.backward(R, cache, include_output_side=False)
    _, g_out = model.backward(R, cache, include_input_side=False)
    return g_full, g_in, g_out


def test_partial_sum_identity_and_nonvacuity(partials):
    g_full, g_in, g_out = partials
    diff = float(np.max(np.abs(g_in["wte.W"] + g_out["wte.W"] - g_full["wte.W"])))
    n_in = float(np.linalg.norm(g_in["wte.W"]))
    n_out = float(np.linalg.norm(g_out["wte.W"]))
    print(f"[C7] sum-identity max diff={diff:.3e}  ||g_in||={n_in:.3e}  ||g_out||={n_out:.3e}")
    assert diff <= ROUNDOFF_ATOL
    assert n_in > 1e-3
    assert n_out > 1e-3


@pytest.fixture(scope="module")
def fd_out_full(model, non_init_params, idx, upstream_r):
    """TL-20: independent finite differences, never built from the
    production backward's own decomposition.

    FD_out: W -> sum(R * forward_from_embedding(h0, params[wte.W<-W])),
    with h0 = embed(idx) held FIXED.
    FD_full: W -> sum(R * forward(idx, params[wte.W<-W])), through the full
    forward (embed recomputed each perturbation too).
    """
    R, _ = upstream_r
    h0, _ = model.embed(idx, non_init_params)
    h0 = np.ascontiguousarray(h0)

    def f_out(W):
        p2 = dict(non_init_params)
        p2["wte.W"] = W
        logits, _ = model.forward_from_embedding(h0, p2)
        return np.asarray(np.sum(logits * R))

    def f_full(W):
        p2 = dict(non_init_params)
        p2["wte.W"] = W
        logits, _ = model.forward(idx, p2)
        return np.asarray(np.sum(logits * R))

    W0 = non_init_params["wte.W"]
    FD_out = numerical_jacobian(f_out, W0).reshape(W0.shape)
    FD_full = numerical_jacobian(f_full, W0).reshape(W0.shape)
    return FD_out, FD_full


def test_independent_fd_matches_partials(partials, fd_out_full):
    g_full, g_in, g_out = partials
    FD_out, FD_full = fd_out_full

    r_out = rel_err(g_out["wte.W"], FD_out)
    r_full = rel_err(g_full["wte.W"], FD_full)
    r_in = rel_err(g_in["wte.W"], FD_full - FD_out)
    print(f"[C7 independent FD] rel_err(g_out,FD_out)={r_out:.3e} "
          f"rel_err(g_full,FD_full)={r_full:.3e} rel_err(g_in,FD_full-FD_out)={r_in:.3e}")

    assert r_out < GRADCHECK_TOL
    assert r_full < GRADCHECK_TOL
    assert r_in < GRADCHECK_TOL


# ---------------------------------------------------------------------------
# Accumulation (structural, AST scan) + test-the-test
# ---------------------------------------------------------------------------
def test_no_wte_w_subscript_assignment_in_model_py():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nanogpt", "model.py"
    )
    with open(model_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    tree = ast.parse(text, filename="model.py")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "wte.W"
                ):
                    offenders.append(node.lineno)
    assert offenders == [], f"found direct ['wte.W'] = assignment(s) at lines {offenders}"


def test_test_the_test_overwriting_accumulate_grads_breaks_sum_identity(
    model, non_init_params, idx, upstream_r, monkeypatch
):
    """Monkeypatch model.ops.accumulate_grads to an OVERWRITE variant
    (grads[k] = v.copy() always, never +=). This breaks the C7 sum identity
    and must fail the wte.W gradcheck."""
    R, cache = upstream_r

    def overwrite_accumulate(grads, d_params):
        for k, v in d_params.items():
            grads[k] = v.copy()
        return grads

    monkeypatch.setattr(model_module.ops, "accumulate_grads", overwrite_accumulate)

    _, g_full_broken = model.backward(R, cache)
    _, g_in_broken = model.backward(R, cache, include_output_side=False)
    _, g_out_broken = model.backward(R, cache, include_input_side=False)
    diff = float(np.max(np.abs(g_in_broken["wte.W"] + g_out_broken["wte.W"] - g_full_broken["wte.W"])))
    print(f"[test-the-test] overwrite accumulate_grads sum-identity diff={diff:.3e}")
    assert diff > ROUNDOFF_ATOL, "overwrite accumulate_grads should break the C7 sum identity"

    report = gradcheck(model.forward, model.backward, non_init_params, idx, seed=_FIXTURE_SEED + 3)
    assert report.tensors["wte.W"].rel_err >= GRADCHECK_TOL, (
        "test-the-test: broken accumulate_grads should fail the wte.W gradcheck"
    )


# ---------------------------------------------------------------------------
# Contiguity trap: gradcheck must refuse a non-contiguous / transposed wte.W.
# ---------------------------------------------------------------------------
def test_transposed_or_fortran_wte_raises_valueerror(model, non_init_params, idx):
    # Fortran-order copy: same shape and values, not C-contiguous.
    bad_fortran = dict(non_init_params)
    bad_fortran["wte.W"] = np.asfortranarray(non_init_params["wte.W"])
    assert not bad_fortran["wte.W"].flags.c_contiguous
    with pytest.raises(ValueError):
        gradcheck(model.forward, model.backward, bad_fortran, idx, seed=_FIXTURE_SEED + 4)

    # A transposed VIEW taken back to the original (V, d) shape via a second
    # transpose is still, in general, not C-contiguous (it is a view of a
    # view, not a fresh array) -- this is the "transposed-view tie" shape
    # TL-17 forbids for lm_head and the harness must refuse.
    bad_view = dict(non_init_params)
    bad_view["wte.W"] = non_init_params["wte.W"].T.copy().T
    assert not bad_view["wte.W"].flags.c_contiguous
    assert bad_view["wte.W"].base is not None
    with pytest.raises(ValueError):
        gradcheck(model.forward, model.backward, bad_view, idx, seed=_FIXTURE_SEED + 5)


def test_production_wte_passes_contiguity_guard(model, non_init_params, idx):
    # Sanity: the production (contiguous) array does NOT raise.
    gradcheck(model.forward, model.backward, non_init_params, idx, seed=_FIXTURE_SEED + 6)

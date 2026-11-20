"""tests/test_numerical_stability.py -- Phase 6 (SPRINT_6 scope; execution plan Phase 6).

The remaining edge case from the build requirements section 4: logits of order 1e3 must not
overflow, at the FULL-MODEL level (not just the unit-level smoke in test_loss.py), through
the loss, its gradients, the probability/sampling path and the cached decoding path.
T=1 and the shifted mask were resolved in Sprint 2 and re-confirmed on the full model in
Sprint 3; here they are only RE-CONFIRMED on TRAINED parameters, not re-derived.

Every claim has a discriminating counterpart (test-the-test): the naive
exp(z)/sum(exp(z)) really does overflow on these exact logits.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nanogpt import ops, sampling
from nanogpt.attention_mask import shifted_causal_mask
from nanogpt.config import DEFAULT_SEED, FD_STEP, ModelConfig, make_rng, tiny_config
from nanogpt.kv_cache import KVCache
from nanogpt.loss import cross_entropy_backward, cross_entropy_forward
from nanogpt.model import GPT

_SEED = DEFAULT_SEED + 1001


def _big_logit_model(cfg, target=1e3, seed_off=0):
    """A model whose logits have max |logit| ~ `target`: tied head scaled up."""
    m = GPT(cfg)
    p = m.init_params(make_rng(_SEED + seed_off))
    r = make_rng(_SEED + 500 + seed_off)
    for k in m.param_keys():
        if not (k.endswith(".gamma") or k.endswith(".beta")):
            p[k] += 0.3 * r.standard_normal(p[k].shape)
    idx = r.integers(0, cfg.V, size=(2, cfg.T))
    tgt = r.integers(0, cfg.V, size=(2, cfg.T))
    logits, _ = m.forward(idx, p)
    p["wte.W"] *= target / np.abs(logits).max()  # in place: keeps the tie / contiguity
    return m, p, idx, tgt


def _reference_loss(logits, tgt):
    """Independent, pure-Python log-sum-exp (math.fsum), shares no code with loss.py."""
    z = logits.reshape(-1, logits.shape[-1])
    t = tgt.reshape(-1)
    tot = []
    for row, ti in zip(z, t):
        mx = max(row)
        tot.append(mx + math.log(math.fsum(math.exp(v - mx) for v in row)) - row[ti])
    return math.fsum(tot) / len(tot)


@pytest.fixture(scope="module")
def big():
    cfg = tiny_config(V=11)
    m, p, idx, tgt = _big_logit_model(cfg)
    logits, _ = m.forward(idx, p)
    return cfg, m, p, idx, tgt, logits


def test_fixture_really_produces_order_1e3_logits(big):
    *_, logits = big
    assert 5e2 < np.abs(logits).max() < 2e3


def test_naive_softmax_overflows_on_these_exact_logits_but_production_does_not(big):
    *_, logits = big
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(logits)
        naive = e / e.sum(axis=-1, keepdims=True)
    assert not np.all(np.isfinite(naive))  # the discriminator: naive FAILS here
    p, _ = ops.softmax_forward(logits)
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(p >= 0)


def test_full_model_loss_is_finite_and_matches_independent_reference(big):
    cfg, m, p, idx, tgt, logits = big
    loss, _ = m.forward_loss(idx, tgt, p)
    assert np.isfinite(loss)
    assert abs(loss - _reference_loss(logits, tgt)) <= 1e-9 * max(1.0, abs(loss))


def test_loss_has_the_closed_form_margin_value_when_one_logit_dominates():
    # logits row = [M, 0, 0]; target 1 -> loss = M + log(1 + 2e^-M) ~ M exactly for M = 1e3.
    z = np.array([[1000.0, 0.0, 0.0]])
    loss, _ = cross_entropy_forward(z, np.array([1]))
    assert loss == pytest.approx(1000.0 + math.log1p(2 * math.exp(-1000.0)), abs=1e-9)
    loss0, _ = cross_entropy_forward(z, np.array([0]))
    assert loss0 == pytest.approx(math.log1p(2 * math.exp(-1000.0)), abs=1e-12)  # ~0: a saturated, correct prediction


def test_cross_entropy_gradient_is_finite_rows_sum_to_zero_and_matches_fd_at_1e3_scale():
    rng = make_rng(_SEED + 5)
    z = 1e3 * rng.standard_normal((6, 9))
    tgt = rng.integers(0, 9, size=6)
    loss, cache = cross_entropy_forward(z, tgt)
    dz, _ = cross_entropy_backward(1.0, cache)
    assert np.all(np.isfinite(dz)) and np.isfinite(loss)
    assert np.max(np.abs(dz.sum(axis=-1))) <= 1e-12  # (p - onehot) rows sum to 0
    # central-difference check directly on the 1e3-scale logits (curvature of CE in z is O(1),
    # so the frozen FD step is valid here)
    fd = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zp, zm = z.copy(), z.copy()
            zp[i, j] += FD_STEP
            zm[i, j] -= FD_STEP
            fd[i, j] = (cross_entropy_forward(zp, tgt)[0] - cross_entropy_forward(zm, tgt)[0]) / (2 * FD_STEP)
    assert np.max(np.abs(fd - dz)) <= 1e-6


def test_full_model_gradients_are_finite_at_1e3_logits(big):
    cfg, m, p, idx, tgt, _ = big
    _, cache = m.forward_loss(idx, tgt, p)
    _, g = m.backward_loss(1.0, cache)
    assert set(g) == set(m.param_keys())
    for k, v in g.items():
        assert np.all(np.isfinite(v)), k
    assert max(np.abs(v).max() for v in g.values()) > 1.0  # non-trivial: saturated softmax, large grads


@pytest.mark.parametrize("target", [1e2, 1e3, 1e4, 1e5])
def test_finite_loss_and_gradients_across_a_logit_scale_sweep(target):
    cfg = tiny_config(V=11)
    m, p, idx, tgt = _big_logit_model(cfg, target=target, seed_off=3)
    loss, cache = m.forward_loss(idx, tgt, p)
    _, g = m.backward_loss(1.0, cache)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(v)) for v in g.values())


def test_default_config_at_1e3_logits_is_finite_too():
    cfg = ModelConfig(V=55)
    m, p, idx, tgt = _big_logit_model(cfg, seed_off=4)
    loss, cache = m.forward_loss(idx[:1], tgt[:1], p)
    _, g = m.backward_loss(1.0, cache)
    assert np.isfinite(loss) and all(np.all(np.isfinite(v)) for v in g.values())


def test_tie_still_holds_and_wte_gradient_is_one_accumulated_array_at_1e3(big):
    cfg, m, p, idx, tgt, _ = big
    assert m.named_parameters(p)["lm_head.W"] is p["wte.W"]
    _, cache = m.forward_loss(idx, tgt, p)
    _, g = m.backward_loss(1.0, cache)
    assert g["wte.W"].shape == p["wte.W"].shape and "lm_head.W" not in g


# --------------------------------------------------------------------------
# sampling at 1e3 logits
# --------------------------------------------------------------------------
def test_sampling_pipeline_is_finite_and_sane_on_1e3_logits(big):
    *_, logits = big
    flat = logits.reshape(-1, logits.shape[-1])
    for t in (1e-3, 1.0, 1e3):
        pr = sampling.probabilities(flat, temperature=t)
        assert np.all(np.isfinite(pr)) and np.allclose(pr.sum(axis=-1), 1.0, atol=1e-12)
    ids = sampling.sample(flat, make_rng(_SEED + 6), temperature=1.0)
    assert np.array_equal(ids, sampling.greedy(flat))  # a 1e3 margin makes sampling deterministic
    hot = sampling.probabilities(flat, temperature=1e6)
    assert np.max(np.abs(hot - 1.0 / flat.shape[-1])) < 1e-3  # t -> infinity: uniform
    assert np.all(np.isfinite(sampling.probabilities(flat, top_k=3)))


# --------------------------------------------------------------------------
# cached decoding at 1e3 logits
# --------------------------------------------------------------------------
def test_cached_decoding_matches_full_forward_at_1e3_logits_in_relative_terms(big):
    # C8 itself is an ABSOLUTE 1e-10 claim at the spec's init scale (tests/test_kv_cache.py).
    # At 1e3 logits one ulp of a logit is ~1e-13, so the right invariant here is relative.
    cfg, m, p, idx, tgt, logits = big
    kv = KVCache(cfg, idx.shape[0])
    cached = np.concatenate([m.forward_step(idx[:, t : t + 1], p, kv) for t in range(cfg.T)], axis=1)
    assert np.max(np.abs(cached - logits)) <= 1e-10 * np.abs(logits).max()


# --------------------------------------------------------------------------
# T=1 and shifted mask: RE-CONFIRMED on trained parameters (not re-derived)
# --------------------------------------------------------------------------
def _trained_tiny(steps=5):
    import train as train_module
    from nanogpt.tokenizer import load_corpus_and_tokenizer

    info, tok = load_corpus_and_tokenizer()
    ids = tok.encode(info.text)
    res = train_module.train(tiny_config(tok.vocab_size), ids, steps=steps, batch_size=4, lr=1e-2,
                             log_fn=lambda *_a, **_k: None)
    return res


@pytest.fixture(scope="module")
def trained():
    return _trained_tiny()


def test_T1_forward_backward_finite_with_exact_zero_q_k_gradients_on_trained_params(trained):
    m, p = trained.model, trained.params
    idx = make_rng(_SEED + 7).integers(0, m.cfg.V, size=(2, 1))
    tgt = make_rng(_SEED + 8).integers(0, m.cfg.V, size=(2, 1))
    loss, cache = m.forward_loss(idx, tgt, p)
    _, g = m.backward_loss(1.0, cache)
    assert np.isfinite(loss) and all(np.all(np.isfinite(v)) for v in g.values())
    for l in range(m.cfg.L):
        for key in (f"h.{l}.attn.q.W", f"h.{l}.attn.q.b", f"h.{l}.attn.k.W", f"h.{l}.attn.k.b"):
            assert np.all(g[key] == 0.0), key  # one position: softmax over one key is constant 1
    assert np.abs(g["h.0.attn.v.W"]).max() > 0  # and V/O are NOT degenerate


def test_shifted_mask_still_leaks_on_trained_params_and_causal_does_not(trained):
    # A change at position s can reach position i iff s <= i + 1 per attention layer, and the
    # leak COMPOUNDS across layers: with L layers the shifted mask reaches s <= i + L. (The
    # single-attention test in Sprint 2 sees exactly one extra token because there L == 1.)
    cfg, p = trained.model.cfg, trained.params
    assert cfg.L == 2 and cfg.T == 4
    m_ok, m_bad = GPT(cfg), GPT(cfg, mask_fn=shifted_causal_mask)
    a = make_rng(_SEED + 9).integers(0, cfg.V, size=(1, cfg.T))
    s = cfg.T - 1  # change only the LAST token
    b = a.copy()
    b[0, s] = (b[0, s] + 1) % cfg.V
    ok_a, _ = m_ok.forward(a, p)
    ok_b, _ = m_ok.forward(b, p)
    bad_a, _ = m_bad.forward(a, p)
    bad_b, _ = m_bad.forward(b, p)
    # causal: every position strictly before s is bit-identical
    assert np.array_equal(ok_a[:, :s], ok_b[:, :s])
    # shifted, L=2: positions i with s <= i + 2, i.e. i >= 1, are affected ...
    for i in range(1, s):
        assert np.max(np.abs(bad_a[:, i] - bad_b[:, i])) > 1e-6, i
    # ... while position 0 (s = 3 > 0 + 2) still cannot see it
    assert np.array_equal(bad_a[:, 0], bad_b[:, 0])
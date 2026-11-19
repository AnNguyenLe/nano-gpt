"""tests/test_causal_mask.py -- Engineer C: independent causality verifier
(SPRINT_2_PLAN.md S2-09, S2-10).

Engineer C writes **no production code**. This file checks causality from
outside both attention implementations, coding against the frozen
interfaces in SPRINT_2_PLAN.md S3:

- A's `nanogpt.attention_mask.causal_mask` / `shifted_causal_mask` and
  `nanogpt.attention_reference.reference_attention_forward(x, params, H,
  prefix="attn", mask_fn=...)` (the naive loop-over-heads oracle).
- B's `nanogpt.attention.CausalSelfAttention(prefix, d, H, mask_fn=...)`
  with `forward(x, params) -> (out, cache)` and
  `backward(d_out, cache) -> (d_x, d_params)`. `cache["P"]` has shape
  (B, H, T, T) and `cache["allowed"]` is a frozen, documented cache key.

Milestones (committed together with their shifted-mask failure test,
per S7.1 / S9):
  C1 (needs A1 masks + A2 oracle): the C5 helper; C5 + shifted
     test-the-test on the ORACLE; the strict-mask documentation test.
  C2 (needs B1 vectorized forward): C5 + shifted on the VECTORIZED
     module; regression guards; the d=64 spot check.
  C3 (needs B3 backward + gradcheck): T=1 suite; backward causality;
     masked-softmax rank.

Fixtures (TL-13): `_FIXTURE_SEED = DEFAULT_SEED + 203`. No fixture
tensor is ever drawn from `make_rng(DEFAULT_SEED)`. One `Generator` per
fixture, tensors drawn sequentially from it, never re-created per tensor
(which would make W_q == W_k). Non-init parameters: W = N(0,1)/sqrt(d),
b = 0.1*N(0,1), x = N(0,1) (TL-13's non-init analogue of TL-6): the
default-init (std=0.02, biases 0) fixture would make attention
near-uniform and Q/K gradients ~1e-12, which discriminates far less.
"""

from __future__ import annotations

import ast
import math
import os

import numpy as np
import pytest

from nanogpt.config import (
    C3_SVD_REL_TOL,
    DEFAULT_SEED,
    GRADCHECK_TOL,
    REL_ERR_FLOOR,
    ROUNDOFF_ATOL,
    make_rng,
    rel_err,
)
from nanogpt import ops
from nanogpt.gradcheck import gradcheck, numerical_jacobian

from nanogpt.attention_mask import causal_mask, shifted_causal_mask
from nanogpt.attention_reference import reference_attention_forward
from nanogpt.attention import CausalSelfAttention


# ---------------------------------------------------------------------------
# Test-local constants (S7.2)
# ---------------------------------------------------------------------------
_FIXTURE_SEED = DEFAULT_SEED + 203  # TL-13: C's offset, never DEFAULT_SEED itself
NONVACUITY_MIN = 1e-6  # s <= t blocks (and non-zero gradients) must exceed this


# ---------------------------------------------------------------------------
# Fixture builders (TL-13: one rng per fixture, drawn sequentially)
# ---------------------------------------------------------------------------
def _make_params(rng, d, prefix="attn"):
    """Non-init Q/K/V/O params (TL-10 keys), drawn sequentially from `rng`.

    W = rng.standard_normal((d, d)) / sqrt(d), b = 0.1 * rng.standard_normal(d).
    Every array is made C-contiguous (gradcheck's hardened contiguity
    check, Sprint 1 watch item b / TL-10).
    """
    params = {}
    for k in ("q", "k", "v", "o"):
        W = (rng.standard_normal((d, d)) / math.sqrt(d)).astype(np.float64)
        b = (0.1 * rng.standard_normal(d)).astype(np.float64)
        params[f"{prefix}.{k}.W"] = np.ascontiguousarray(W)
        params[f"{prefix}.{k}.b"] = np.ascontiguousarray(b)
    assert not np.array_equal(
        params[f"{prefix}.q.W"], params[f"{prefix}.k.W"]
    ), "TL-13: fixture draw must not yield W_q == W_k"
    return params


def _make_x(rng, B, T, d):
    return np.ascontiguousarray(rng.standard_normal((B, T, d)).astype(np.float64))


def _assert_nonvacuous(name, grad):
    """Shared paired-test helper (S7.3 'paired non-zero helper'): raises if
    `grad`'s Frobenius norm does not clear NONVACUITY_MIN. Used both to gate
    the paired T=4 non-zero check and, in the disconnection adversary test,
    to show it correctly *rejects* a structurally-disconnected gradient.
    """
    n = float(np.linalg.norm(grad))
    if n <= NONVACUITY_MIN:
        raise AssertionError(
            f"{name}: ||grad||_F={n:.3e} <= NONVACUITY_MIN={NONVACUITY_MIN:.1e} "
            "(looks disconnected, not merely small)"
        )
    return n


# ---------------------------------------------------------------------------
# C5 helper (S7.2, shared by the oracle- and vectorized-side checks)
# ---------------------------------------------------------------------------
def _c5_blocks(f, x_td):
    """J = numerical_jacobian(f, x_td) for x_td of shape (T, d); returns
    J.reshape(T, d, T, d), so J4[t, :, s, :] = d out_t / d x_s.
    """
    T, d = x_td.shape
    J = numerical_jacobian(f, x_td)
    return J.reshape(T, d, T, d)


def _assert_c5(J4):
    """Raises AssertionError naming (t, s, value) if:
      - max_{s>t} |J4[t,:,s,:]| > ROUNDOFF_ATOL (a future leak), or
      - min_{s<=t} ||J4[t,:,s,:]||_F <= NONVACUITY_MIN (non-vacuity failure:
        an all-zero Jacobian must not pass).
    """
    T = J4.shape[0]
    for t in range(T):
        for s in range(T):
            block = J4[t, :, s, :]
            if s > t:
                m = float(np.max(np.abs(block))) if block.size else 0.0
                if m > ROUNDOFF_ATOL:
                    raise AssertionError(
                        f"C5 violated: future block (t={t}, s={s}) has "
                        f"max|J4[t,:,s,:]|={m:.3e} > ROUNDOFF_ATOL={ROUNDOFF_ATOL:.1e}"
                    )
            else:
                fro = float(np.linalg.norm(block))
                if fro <= NONVACUITY_MIN:
                    raise AssertionError(
                        f"C5 non-vacuity violated: block (t={t}, s={s}), s<=t, has "
                        f"||J4[t,:,s,:]||_F={fro:.3e} <= NONVACUITY_MIN={NONVACUITY_MIN:.1e}"
                    )


def _future_max_past_min(J4):
    T = J4.shape[0]
    future = [
        float(np.max(np.abs(J4[t, :, s, :])))
        for t in range(T)
        for s in range(T)
        if s > t
    ]
    past = [
        float(np.linalg.norm(J4[t, :, s, :]))
        for t in range(T)
        for s in range(T)
        if s <= t
    ]
    future_max = max(future) if future else 0.0
    past_min = min(past)
    return future_max, past_min


# ===========================================================================
# C1: C5 + shifted-mask test-the-test on the ORACLE
# ===========================================================================
def test_c5_oracle_correct_mask():
    """C5, correct mask (T=4, d=8, H=2, B=1): _assert_c5 passes on the
    oracle. Records the measured future max (expected 0.0) and past min
    (prototype 0.13).
    """
    rng = make_rng(_FIXTURE_SEED)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)

    def f(z):
        return reference_attention_forward(z[None], params, H, mask_fn=causal_mask)[0]

    J4 = _c5_blocks(f, x[0])
    _assert_c5(J4)
    future_max, past_min = _future_max_past_min(J4)
    assert future_max <= ROUNDOFF_ATOL
    assert past_min > NONVACUITY_MIN
    print(f"[C5 oracle, correct mask] future_max={future_max:.3e} past_min={past_min:.3e}")


def test_c5_shifted_mask_oracle_test_the_test():
    """Shifted test-the-test (the required adversary), oracle side: the bug
    is actually substituted via mask_fn=shifted_causal_mask, and
    _assert_c5 must RAISE. The leak sits exactly at s=t+1; s>=t+2 stays 0.
    """
    rng = make_rng(_FIXTURE_SEED + 1)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)

    def f(z):
        return reference_attention_forward(z[None], params, H, mask_fn=shifted_causal_mask)[0]

    J4 = _c5_blocks(f, x[0])
    with pytest.raises(AssertionError):
        _assert_c5(J4)

    leaks = []
    for t in range(T - 1):
        leak = float(np.linalg.norm(J4[t, :, t + 1, :]))
        assert leak > NONVACUITY_MIN, f"shifted-mask leak missing at (t={t}, s={t+1})"
        leaks.append(leak)
    for t in range(T):
        for s in range(t + 2, T):
            m = float(np.max(np.abs(J4[t, :, s, :])))
            assert m <= ROUNDOFF_ATOL, f"unexpected leak at (t={t}, s={s}): {m:.3e}"
    print(f"[C5 oracle, shifted mask] leak(t,t+1) min={min(leaks):.3e} max={max(leaks):.3e}")


def test_strict_mask_j_lt_i_is_not_the_c5_adversary_and_produces_nan():
    """Wrong adversary, kept separate (P0 per PO): the diagonal-masking
    interpretation `j < i` (as opposed to the frozen `j <= i+1` shifted
    mask) masks row 0 entirely -- every key is disallowed for query 0, so
    softmax over an all--inf row is exactly NaN (under
    np.errstate(invalid="ignore")). This is a DIFFERENT bug from the
    shifted-mask adversary (TL-12) and must NEVER be used as the C5
    test-the-test: it documents why j<i is rejected, it does not replace
    the j<=i+1 check above.

    This NaN failure mode is specific to the VECTORIZED module's -inf
    representation (Z = where(allowed, S, -inf); an all--inf row gives
    exp(-inf - (-inf)) = exp(nan) = nan). The oracle has no -inf concept at
    all (S3.3: it softmaxes only over the allowed subset of keys) and so
    would instead raise on an empty subset for row 0 -- a different, also
    non-C5 failure, not exercised here.
    """

    def strict_mask(T):
        return np.arange(T)[None, :] < np.arange(T)[:, None]

    T, d, H = 4, 8, 2
    rng = make_rng(_FIXTURE_SEED + 2)
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H, mask_fn=strict_mask)

    assert not strict_mask(T)[0, :].any(), "row 0 of j<i must be all-False"

    with np.errstate(invalid="ignore"):
        out, _ = attn.forward(x, params)
    assert np.all(np.isnan(out[0, 0])), "row 0 under j<i must be NaN (all keys masked)"


# ===========================================================================
# C2: C5 + shifted-mask test-the-test on the VECTORIZED module;
#     regression guards; d=64 spot check.
# ===========================================================================
def test_c5_vectorized_correct_mask():
    """C5, correct mask: _assert_c5 passes on the vectorized module too."""
    rng = make_rng(_FIXTURE_SEED + 3)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H)

    def f(z):
        out, _ = attn.forward(z[None], params)
        return out[0]

    J4 = _c5_blocks(f, x[0])
    _assert_c5(J4)
    future_max, past_min = _future_max_past_min(J4)
    assert future_max <= ROUNDOFF_ATOL
    assert past_min > NONVACUITY_MIN
    print(f"[C5 vectorized, correct mask] future_max={future_max:.3e} past_min={past_min:.3e}")


def test_c5_shifted_mask_vectorized_test_the_test():
    """Shifted test-the-test, vectorized side: CausalSelfAttention(...,
    mask_fn=shifted_causal_mask) makes _assert_c5 raise. The bug is
    substituted through the frozen `mask_fn` constructor argument, with no
    monkeypatching (TL-12).
    """
    rng = make_rng(_FIXTURE_SEED + 4)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H, mask_fn=shifted_causal_mask)

    def f(z):
        out, _ = attn.forward(z[None], params)
        return out[0]

    J4 = _c5_blocks(f, x[0])
    with pytest.raises(AssertionError):
        _assert_c5(J4)

    leaks = []
    for t in range(T - 1):
        leak = float(np.linalg.norm(J4[t, :, t + 1, :]))
        assert leak > NONVACUITY_MIN, f"shifted-mask leak missing at (t={t}, s={t+1})"
        leaks.append(leak)
    for t in range(T):
        for s in range(t + 2, T):
            m = float(np.max(np.abs(J4[t, :, s, :])))
            assert m <= ROUNDOFF_ATOL, f"unexpected leak at (t={t}, s={s}): {m:.3e}"
    print(f"[C5 vectorized, shifted mask] leak(t,t+1) min={min(leaks):.3e} max={max(leaks):.3e}")


def test_c5_spot_check_d64_H4_T6():
    """d=64, H=4, T=6 spot check (P2): _assert_c5 passes on the vectorized
    module at the default head count/dimension.
    """
    rng = make_rng(_FIXTURE_SEED + 5)
    T, d, H = 6, 64, 4
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H)

    def f(z):
        out, _ = attn.forward(z[None], params)
        return out[0]

    J4 = _c5_blocks(f, x[0])
    _assert_c5(J4)


def test_batch_independence_cross_batch_derivative_zero():
    """Batch independence (P2): with B=2, d out[b] / d x[b'] is exactly 0
    for b != b'.
    """
    rng = make_rng(_FIXTURE_SEED + 6)
    T, d, H, B = 4, 8, 2, 2
    params = _make_params(rng, d)
    x = _make_x(rng, B, T, d)
    attn = CausalSelfAttention("attn", d, H)

    def f(z):
        out, _ = attn.forward(z, params)
        return out

    J = numerical_jacobian(f, x)
    J6 = J.reshape(B, T, d, B, T, d)
    for b in range(B):
        for bp in range(B):
            if b == bp:
                continue
            m = float(np.max(np.abs(J6[b, :, :, bp, :, :])))
            assert m <= ROUNDOFF_ATOL, f"cross-batch leak b={b} <- b'={bp}: {m:.3e}"


def test_default_mask_fn_is_causal_mask():
    """Regression guard: the default mask_fn is causal_mask (identity, not
    merely equal)."""
    attn = CausalSelfAttention("attn", 8, 2)
    assert attn.mask_fn is causal_mask


def test_shifted_causal_mask_referenced_only_in_attention_mask_module():
    """Regression guard: `shifted_causal_mask` is never *used as code*
    anywhere in nanogpt/*.py except its home module attention_mask.py, so a
    future regression that silently makes the adversary a default is
    caught automatically.

    Implemented as an AST scan for real identifier occurrences (a
    `def shifted_causal_mask`, an `import`/`from import` of that name, or a
    `Name` reference such as `mask_fn=shifted_causal_mask`), not a naive
    substring/text scan. Deviation, recorded for the TL: a plain text scan
    also flags prose that merely *names* the adversary while explaining the
    sanctioned mask_fn substitution path (observed in
    attention_reference.py's docstring), which is not itself a usage and
    would make this guard fail on harmless documentation. The AST form
    still catches the thing the guard exists for: any real code path that
    imports, calls or defaults to `shifted_causal_mask` outside its own
    module.
    """
    nanogpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nanogpt"
    )
    target = "shifted_causal_mask"
    hits = []
    for fname in sorted(os.listdir(nanogpt_dir)):
        if not fname.endswith(".py") or fname == "attention_mask.py":
            continue
        path = os.path.join(nanogpt_dir, fname)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text, filename=fname)
        used = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == target:
                used = True
            elif isinstance(node, ast.Attribute) and node.attr == target:
                # TL fixup: `attention_mask.shifted_causal_mask` (module
                # attribute access) is an ast.Attribute, not an ast.Name, and
                # was previously invisible to this guard.
                used = True
            elif isinstance(node, ast.alias) and target in (node.name, node.asname):
                used = True
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
                used = True
            if used:
                break
        if used:
            hits.append(fname)
    assert hits == [], f"'{target}' used as code outside attention_mask.py: {hits}"

    # TL ruling (Sprint 2 integration): the literal SPRINT_2_PLAN.md §7.3
    # criterion is a TEXT scan, and it is enforced too. Engineer A rephrased
    # the one docstring that named the adversary, so prose mentions are no
    # longer needed anywhere outside attention_mask.py; keeping the text scan
    # also catches string-based lookups such as getattr(m, "shifted_causal_mask").
    text_hits = []
    for fname in sorted(os.listdir(nanogpt_dir)):
        if not fname.endswith(".py") or fname == "attention_mask.py":
            continue
        with open(os.path.join(nanogpt_dir, fname), "r", encoding="utf-8") as fh:
            if target in fh.read():
                text_hits.append(fname)
    assert text_hits == [], f"'{target}' appears in nanogpt/ outside attention_mask.py: {text_hits}"

    # Non-vacuity: the name must actually be defined where it belongs.
    mask_path = os.path.join(nanogpt_dir, "attention_mask.py")
    with open(mask_path, "r", encoding="utf-8") as fh:
        mask_text = fh.read()
    assert target in mask_text, "attention_mask.py must define shifted_causal_mask"


# ===========================================================================
# C3: T=1 degenerate case, backward causality, masked-softmax rank.
# ===========================================================================
def test_backward_causality_matches_numeric_and_zero_future():
    """Backward causality (P1): the analytic Jacobian, built from `backward`
    with one-hot upstream gradients e_{t,i}, has d_x[s] <= ROUNDOFF_ATOL for
    all s > t, and matches the numerical Jacobian with rel_err < GRADCHECK_TOL.
    """
    rng = make_rng(_FIXTURE_SEED + 7)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H)

    out, cache = attn.forward(x, params)
    analytic = np.zeros((T, d, T, d), dtype=np.float64)
    for t in range(T):
        for c in range(d):
            d_out = np.zeros_like(out)
            d_out[0, t, c] = 1.0
            d_x, _ = attn.backward(d_out, cache)
            analytic[t, c, :, :] = d_x[0]
            for s in range(t + 1, T):
                m = float(np.max(np.abs(d_x[0, s])))
                assert m <= ROUNDOFF_ATOL, (
                    f"backward causality violated: d_x[s={s}] from upstream "
                    f"e_(t={t},c={c}) has max|.|={m:.3e}"
                )

    def f(z):
        out2, _ = attn.forward(z[None], params)
        return out2[0]

    J4 = _c5_blocks(f, x[0])
    r = rel_err(analytic, J4)
    assert r < GRADCHECK_TOL, f"analytic vs numeric backward Jacobian rel_err={r:.3e}"
    print(f"[backward causality] analytic-vs-numeric rel_err={r:.3e}")


def test_masked_softmax_rank_matches_row_support():
    """Masked-softmax rank (P2, F5): for each causal row i of
    cache['P'][0, h], the Jacobian built from ops.softmax_backward on basis
    vectors has SVD rank exactly i, at threshold C3_SVD_REL_TOL*s_max. Row
    0 has J == 0, so rank 0.
    """
    rng = make_rng(_FIXTURE_SEED + 8)
    T, d, H = 4, 8, 2
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H)
    _, cache = attn.forward(x, params)
    P = cache["P"]  # (1, H, T, T), frozen cache key

    for h in range(H):
        for i in range(T):
            p_row = P[0, h, i, :]
            J = np.zeros((T, T), dtype=np.float64)
            for j in range(T):
                dp = np.zeros((1, T), dtype=np.float64)
                dp[0, j] = 1.0
                dz, _ = ops.softmax_backward(dp, {"p": p_row[None, :]})
                J[:, j] = dz[0]
            s = np.linalg.svd(J, compute_uv=False)
            s_max = float(s[0]) if s.size else 0.0
            rank = int(np.sum(s > C3_SVD_REL_TOL * s_max)) if s_max > 0.0 else 0
            assert rank == i, f"head={h} row={i}: expected rank {i}, got {rank}"


def test_t1_degenerate_case():
    """T=1 (B=2): forward/backward finite; closed-form anchor; exact-zero
    analytic Q/K gradients; gradcheck reports rel_err==0.0 exactly for
    those four tensors and report.passed is True over all 9 tensors (F2).
    """
    rng = make_rng(_FIXTURE_SEED + 9)
    T, d, H, B = 1, 8, 2, 2
    params = _make_params(rng, d)
    x = _make_x(rng, B, T, d)
    attn = CausalSelfAttention("attn", d, H)

    out, cache = attn.forward(x, params)
    assert np.all(np.isfinite(out))

    Wv, bv = params["attn.v.W"], params["attn.v.b"]
    Wo, bo = params["attn.o.W"], params["attn.o.b"]
    expected = (x @ Wv + bv) @ Wo + bo
    assert float(np.max(np.abs(out - expected))) <= ROUNDOFF_ATOL

    d_out = make_rng(_FIXTURE_SEED + 90).standard_normal(out.shape)
    d_x, d_params = attn.backward(d_out, cache)
    assert np.all(np.isfinite(d_x))
    for v in d_params.values():
        assert np.all(np.isfinite(v))

    for key in ("attn.q.W", "attn.q.b", "attn.k.W", "attn.k.b"):
        assert np.all(d_params[key] == 0.0), f"{key}: expected EXACT zero at T=1 (F2)"

    report = gradcheck(attn.forward, attn.backward, params, x)
    for key in ("attn.q.W", "attn.q.b", "attn.k.W", "attn.k.b"):
        assert report.tensors[key].rel_err == 0.0, (
            f"{key}: expected rel_err == 0.0 exactly at T=1, got "
            f"{report.tensors[key].rel_err:.3e}"
        )
    assert report.passed
    assert len(report.tensors) == 9  # 8 params + x
    print(str(report))


def test_t1_paired_t4_same_weights_nonzero_except_bk():
    """Paired T=4 (same weights as the T=1 test, new x): ||dW_q||, ||db_q||,
    ||dW_k|| > NONVACUITY_MIN. b_k stays <= ROUNDOFF_ATOL (TL-9 / R14): its
    true gradient is identically zero at EVERY T, by softmax
    shift-invariance (b_k shifts every allowed score in a row by the same
    constant, which softmax is invariant to), not merely small at T=4.
    """
    d, H = 8, 2
    # Same seed as test_t1_degenerate_case's param draw -> identical weights,
    # reconstructed independently so the two tests don't depend on order.
    rng_w = make_rng(_FIXTURE_SEED + 9)
    params = _make_params(rng_w, d)
    rng_x4 = make_rng(_FIXTURE_SEED + 10)
    x4 = _make_x(rng_x4, 1, 4, d)
    attn = CausalSelfAttention("attn", d, H)

    out4, cache4 = attn.forward(x4, params)
    d_out4 = make_rng(_FIXTURE_SEED + 11).standard_normal(out4.shape)
    _, d_params4 = attn.backward(d_out4, cache4)

    _assert_nonvacuous("attn.q.W", d_params4["attn.q.W"])
    _assert_nonvacuous("attn.q.b", d_params4["attn.q.b"])
    _assert_nonvacuous("attn.k.W", d_params4["attn.k.W"])

    bk_max = float(np.max(np.abs(d_params4["attn.k.b"])))
    assert bk_max <= ROUNDOFF_ATOL, f"TL-9: db_k max|.|={bk_max:.3e} > ROUNDOFF_ATOL"


def test_disconnection_adversary_rejected_by_paired_helper():
    """Disconnection adversary: a test-local forward substitutes zeros for
    params['attn.q.W'] and ['attn.q.b'] unconditionally (ignoring whatever
    is passed), so the FD gradient of W_q through it is exactly 0 at T=4 --
    not because softmax is constant over one key (F2), but because Q is
    disconnected from the output entirely. The paired `_assert_nonvacuous`
    helper used above must REJECT this zero gradient.
    """
    d, H, T = 8, 2, 4
    rng = make_rng(_FIXTURE_SEED + 12)
    params = _make_params(rng, d)
    x = _make_x(rng, 1, T, d)
    attn = CausalSelfAttention("attn", d, H)
    R = make_rng(_FIXTURE_SEED + 13).standard_normal((1, T, d))

    def f(Wq):
        p2 = dict(params)
        p2["attn.q.W"] = np.zeros_like(params["attn.q.W"])  # disconnected: Wq arg unused
        p2["attn.q.b"] = np.zeros_like(params["attn.q.b"])
        out, _ = attn.forward(x, p2)
        return np.asarray(np.sum(out * R))

    Wq_grad = numerical_jacobian(f, params["attn.q.W"]).reshape(d, d)
    assert float(np.max(np.abs(Wq_grad))) <= ROUNDOFF_ATOL

    with pytest.raises(AssertionError):
        _assert_nonvacuous("attn.q.W (disconnected adversary)", Wq_grad)

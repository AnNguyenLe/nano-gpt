"""tests/test_train.py -- Engineer C, Sprint 4 (Phase 4), milestones C1-C3.

Covers `train.py` at the repo root: batching (`get_batch`), the training
loop + CLI (`train`, `TrainResult`, `main`), and loop invariants (tie
preservation, single optimizer state, determinism, AdamW-is-really-used,
and the non-functional "no loss-decrease assertion" guard).

R22/R24/TL-31 (SPRINT_4_PLAN.md section 2.2, section 3.5, section 7): this
whole file runs on the TINY config, at most 10 steps and batch size <= 4,
and completes in well under 5 s. No assertion anywhere in this file
compares `losses[i] < losses[j]` (a decreasing loss is never evidence of
correctness -- see the module guard near the bottom of this file).

This file is appended to across three milestones (C1, then C2, then C3),
never rewritten: the C1 section below lands first, with no Sprint 4
dependency.

Fixture seed (per-file, never DEFAULT_SEED itself): `_FIXTURE_SEED =
DEFAULT_SEED + 304`.
"""
from __future__ import annotations

import numpy as np
import pytest

import train as train_module
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, make_rng
from nanogpt.tokenizer import load_corpus_and_tokenizer

_FIXTURE_SEED = DEFAULT_SEED + 304


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_ids():
    """The real corpus, tokenized to int ids (module-scoped: resolving the
    corpus + building the vocab is read-only and deterministic, so reusing
    it across tests costs nothing and keeps the file fast)."""
    info, tok = load_corpus_and_tokenizer()
    ids = tok.encode(info.text)
    return info, tok, ids


# ===========================================================================
# [C1] Batching
# ===========================================================================
def test_get_batch_shapes_dtype_and_y_eq_x_plus_1():
    data = np.arange(100)
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(data, batch_size=5, T=4, rng=rng)

    assert x.shape == (5, 4)
    assert y.shape == (5, 4)
    assert x.dtype == INT_DTYPE
    assert y.dtype == INT_DTYPE
    # On np.arange(100), data[i+1] == data[i] + 1 everywhere, so y == x + 1.
    assert np.array_equal(y, x + 1)


def test_get_batch_every_window_stays_in_range():
    data = np.arange(100)
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(data, batch_size=50, T=7, rng=rng)
    assert x.min() >= data[0]
    assert y.max() <= data[-1]
    assert x.max() <= 99
    assert y.max() <= 99


def test_get_batch_extreme_start_len_minus_T_minus_1_is_valid():
    # len(data) - T - 1 must be a reachable, valid start: then y's last
    # index is exactly len(data) - 1. Verify this analytically (the open
    # interval endpoint of rng.integers(0, n - T)) and also hit it directly
    # with a stand-in rng that always returns that exact start.
    data = np.arange(100)
    T = 4
    n = len(data)
    extreme_start = n - T - 1

    class _FixedRNG:
        def integers(self, low, high, size):
            assert low == 0
            assert high == n - T  # exclusive upper bound: max valid i = n - T - 1
            return np.full(size, extreme_start, dtype=np.int64)

    x, y = train_module.get_batch(data, batch_size=3, T=T, rng=_FixedRNG())
    assert np.array_equal(x[0], data[extreme_start : extreme_start + T])
    assert np.array_equal(y[0], data[extreme_start + 1 : extreme_start + T + 1])
    # y's last index is exactly len(data) - 1, the last valid index of data.
    assert y[0, -1] == data[-1]


def test_get_batch_same_seed_gives_identical_batches():
    data = np.arange(1000)
    x1, y1 = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    x2, y2 = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_get_batch_different_seed_gives_different_batches():
    data = np.arange(10000)
    x1, _ = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED))
    x2, _ = train_module.get_batch(data, batch_size=8, T=16, rng=make_rng(_FIXTURE_SEED + 1))
    assert not np.array_equal(x1, x2)


@pytest.mark.parametrize("T", [10, 11, 20])
def test_get_batch_len_data_le_T_raises(T):
    data = np.arange(10)  # len(data) == 10 <= T for every T above
    with pytest.raises(ValueError):
        train_module.get_batch(data, batch_size=2, T=T, rng=make_rng(_FIXTURE_SEED))


def test_get_batch_on_real_corpus_decodes_to_contiguous_slice(real_ids):
    info, tok, ids = real_ids
    T = 16
    rng = make_rng(_FIXTURE_SEED)
    x, y = train_module.get_batch(ids, batch_size=3, T=T, rng=rng)

    for b in range(3):
        decoded_x = tok.decode(x[b])
        # x[b] must be a literal, contiguous slice of the corpus text.
        assert decoded_x in info.text
        start = info.text.index(decoded_x)
        # y[b] is the same slice shifted by one character.
        expected_y = info.text[start + 1 : start + 1 + T]
        assert tok.decode(y[b]) == expected_y


# ===========================================================================
# [C2] Training loop + CLI smoke tests (tiny config, <= 10 steps, B <= 4)
# ===========================================================================
import math  # noqa: E402

from nanogpt.config import INIT_LOSS_TOL, tiny_config  # noqa: E402


def _tiny_train(ids, V, steps=5, **kw):
    cfg = tiny_config(V)
    kw.setdefault("batch_size", 4)
    kw.setdefault("lr", 1e-2)
    kw.setdefault("log_fn", lambda *_a, **_k: None)
    return cfg, train_module.train(cfg, ids, steps=steps, **kw)


def test_train_smoke_completes_with_finite_losses(real_ids):
    info, tok, ids = real_ids
    cfg, res = _tiny_train(ids, tok.vocab_size, steps=5)
    assert len(res.losses) == 5
    assert len(res.step_times) == 5
    assert all(math.isfinite(x) for x in res.losses)
    assert res.wall_time_s > 0.0
    assert res.init_loss == res.losses[0]


def test_train_init_loss_matches_ln_V_through_real_loop(real_ids):
    # Consistency of the C2 anchor through the real loop (not optimizer evidence).
    info, tok, ids = real_ids
    cfg, res = _tiny_train(ids, tok.vocab_size, steps=1)
    assert abs(res.init_loss - math.log(tok.vocab_size)) < INIT_LOSS_TOL


def test_train_changes_every_param_except_attention_k_bias(real_ids):
    info, tok, ids = real_ids
    cfg = tiny_config(tok.vocab_size)
    from nanogpt.model import GPT

    m = GPT(cfg)
    before = {k: v.copy() for k, v in m.init_params(make_rng(DEFAULT_SEED)).items()}
    _, res = _tiny_train(ids, tok.vocab_size, steps=5)  # same default seed => same init
    exempt = {f"h.{l}.attn.k.b" for l in range(cfg.L)}
    assert exempt <= set(res.params)
    for k, v0 in before.items():
        if k in exempt:
            # TL-9: structurally-zero gradient, so AdamW leaves it ~fixed
            assert np.max(np.abs(res.params[k] - v0)) <= 1e-6, k
        else:
            assert not np.array_equal(res.params[k], v0), k


def test_main_runs_tiny_and_prints_step0_and_summary(capsys):
    rc = train_module.main([
        "--L", "2", "--d", "8", "--T", "4", "--H", "2", "--allow-out-of-range",
        "--steps", "3", "--batch-size", "4", "--lr", "0.01", "--log-every", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "step 0 |" in out
    assert "step-0 loss" in out and "ln V" in out
    assert "done: 3 steps" in out
    assert "sha256=7afc055a" in out


def test_main_out_of_range_config_without_flag_exits_nonzero():
    with pytest.raises(SystemExit) as ei:
        train_module.main(["--L", "2", "--d", "8", "--T", "4", "--H", "2", "--steps", "1"])
    assert ei.value.code not in (0, None)


def test_train_rejects_zero_steps(real_ids):
    info, tok, ids = real_ids
    with pytest.raises(ValueError):
        train_module.train(tiny_config(tok.vocab_size), ids, steps=0, batch_size=2, lr=1e-2)


def test_train_profile_steps_logs_a_profile(real_ids):
    info, tok, ids = real_ids
    lines = []
    train_module.train(tiny_config(tok.vocab_size), ids, steps=2, batch_size=2, lr=1e-2,
                       log_fn=lines.append, profile_steps=2)
    assert any("cProfile" in s for s in lines)


# ===========================================================================
# [C3] Loop invariants
# ===========================================================================
import ast  # noqa: E402
from pathlib import Path  # noqa: E402

import nanogpt.model as model_module  # noqa: E402
import nanogpt.optim as optim_module  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_two_runs_with_same_seed_are_bit_identical(real_ids):
    info, tok, ids = real_ids
    _, a = _tiny_train(ids, tok.vocab_size, steps=4)
    _, b = _tiny_train(ids, tok.vocab_size, steps=4)
    assert a.losses == b.losses
    assert set(a.params) == set(b.params)
    for k in a.params:
        assert np.array_equal(a.params[k], b.params[k]), k


def test_different_seed_gives_different_run(real_ids):
    info, tok, ids = real_ids
    _, a = _tiny_train(ids, tok.vocab_size, steps=2, seed=DEFAULT_SEED)
    _, b = _tiny_train(ids, tok.vocab_size, steps=2, seed=DEFAULT_SEED + 1)
    assert a.losses != b.losses


def test_tie_preserved_through_training(real_ids):
    info, tok, ids = real_ids
    cfg = tiny_config(tok.vocab_size)
    seen = {}
    orig_init = model_module.GPT.init_params

    def spy(self, rng):
        p = orig_init(self, rng)
        seen["wte"] = p["wte.W"]
        return p

    model_module.GPT.init_params = spy
    try:
        _, res = _tiny_train(ids, tok.vocab_size, steps=3)
    finally:
        model_module.GPT.init_params = orig_init

    assert res.params["wte.W"] is seen["wte"]  # same object as at init
    assert res.params["wte.W"].flags["C_CONTIGUOUS"]
    named = res.model.named_parameters(res.params)
    assert named["lm_head.W"] is res.params["wte.W"]
    assert cfg.V == res.params["wte.W"].shape[0]


def test_single_optimizer_state_and_step_count(real_ids):
    info, tok, ids = real_ids
    _, res = _tiny_train(ids, tok.vocab_size, steps=4)
    opt = res.optimizer
    assert len(opt.state) == len(res.model.param_keys())
    assert "lm_head.W" not in opt.state
    assert opt.t == 4


def test_adamw_step_is_really_used_exactly_steps_times(real_ids, monkeypatch):
    info, tok, ids = real_ids
    calls = {"n": 0}
    orig = optim_module.AdamW.step

    def counting(self, grads):
        calls["n"] += 1
        return orig(self, grads)

    monkeypatch.setattr(optim_module.AdamW, "step", counting)
    _tiny_train(ids, tok.vocab_size, steps=6)
    assert calls["n"] == 6


def test_non_finite_loss_raises_floating_point_error(real_ids, monkeypatch):
    info, tok, ids = real_ids
    orig = model_module.GPT.forward_loss

    def nan_loss(self, idx, targets, params):
        _, cache = orig(self, idx, targets, params)
        return float("nan"), cache

    monkeypatch.setattr(model_module.GPT, "forward_loss", nan_loss)
    with pytest.raises(FloatingPointError):
        _tiny_train(ids, tok.vocab_size, steps=2)


def _imported_roots(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
    return names


# TL-40 (Sprint 5): "checkpoint" is now permitted (train.py --save); the rest of R22 stands.
_PHASE5_WORDS = ("kv_cache", "sampling", "generate")


def _scope_violations(source: str):
    """R22: train.py must contain no checkpoint/kv-cache/sampling/generate
    import and no np.save / pickle use."""
    tree = ast.parse(source)
    bad = [n for n in _imported_roots(tree) if any(w in n for w in _PHASE5_WORDS) or "pickle" in n]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("save", "savez", "savez_compressed") \
                and isinstance(node.value, ast.Name) and node.value.id == "np":
            bad.append(f"np.{node.attr}")
        if isinstance(node, ast.Name) and node.id == "pickle":
            bad.append("pickle")
    return bad


def test_train_py_has_no_phase5_scope_creep():
    src = (_REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert _scope_violations(src) == []


@pytest.mark.parametrize("snippet", [
    "import nanogpt.kv_cache",
    "from nanogpt import sampling",
    "import pickle",
    "import numpy as np\nnp.save('x.npy', 1)",
])
def test_scope_guard_catches_synthetic_violations(snippet):
    assert _scope_violations(snippet) != []


def _loss_decrease_comparisons(source: str):
    """Compare nodes of the form losses[...] < losses[...] (any ordering op)."""
    def is_losses_subscript(n):
        if not isinstance(n, ast.Subscript):
            return False
        v = n.value
        return (isinstance(v, ast.Name) and v.id.endswith("losses")) or \
               (isinstance(v, ast.Attribute) and v.attr.endswith("losses"))

    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
            operands = [node.left] + list(node.comparators)
            if sum(is_losses_subscript(o) for o in operands) >= 2:
                hits.append(node.lineno)
    return hits


def test_no_loss_decrease_assertion_in_this_file():
    # Non-functional rule: a decreasing loss is never evidence of correctness.
    src = Path(__file__).read_text(encoding="utf-8")
    assert _loss_decrease_comparisons(src) == []


def test_loss_decrease_guard_catches_the_forbidden_pattern():
    assert _loss_decrease_comparisons("assert res.losses[-1] < res.losses[0]") != []
    assert _loss_decrease_comparisons("assert losses[0] >= losses[-1]") != []
    assert _loss_decrease_comparisons("assert losses[0] == losses[1]") == []

def test_train_py_may_import_checkpoint_but_nothing_else_from_phase5():
    # TL-40: --save needs checkpoint; the remaining R22 scope limits are unchanged.
    assert _scope_violations("from nanogpt import checkpoint") == []
    assert _scope_violations("from nanogpt import kv_cache") != []
    assert _scope_violations("import generate") != []


def test_main_save_writes_a_loadable_checkpoint_that_matches_the_run(tmp_path, capsys):
    from nanogpt import checkpoint as ckpt

    path = tmp_path / "t.ckpt"
    rc = train_module.main([
        "--L", "2", "--d", "8", "--T", "4", "--H", "2", "--allow-out-of-range",
        "--steps", "2", "--batch-size", "2", "--lr", "0.01", "--log-every", "1", "--save", str(path),
    ])
    assert rc == 0 and "saved checkpoint" in capsys.readouterr().out
    cfg, params, opt_state = ckpt.load(path)
    assert (cfg.L, cfg.d, cfg.T, cfg.H) == (2, 8, 4, 2) and cfg.V == 55
    assert opt_state["t"] == 2
    assert "lm_head.W" not in params and params["wte.W"].flags["C_CONTIGUOUS"]
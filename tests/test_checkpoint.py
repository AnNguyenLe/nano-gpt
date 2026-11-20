"""tests/test_checkpoint.py -- Engineer C, Sprint 5 (SPRINT_5_PLAN.md section 6.3, milestone [C1]).

Round-trip of model params and AdamW state, tie restored as ONE array, resume
equivalence, and refusal of malformed / pickled checkpoints. Tiny config only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nanogpt import checkpoint
from nanogpt.config import DEFAULT_SEED, make_rng, tiny_config
from nanogpt.model import GPT
from nanogpt.optim import AdamW

_SEED = DEFAULT_SEED + 801
_V = 11


def _setup(seed_off=0):
    cfg = tiny_config(_V)
    m = GPT(cfg)
    p = m.init_params(make_rng(_SEED + seed_off))
    rng = make_rng(_SEED + 50 + seed_off)
    idx = rng.integers(0, _V, size=(2, cfg.T))
    tgt = rng.integers(0, _V, size=(2, cfg.T))
    return cfg, m, p, idx, tgt


def _grads(m, p, idx, tgt):
    _, cache = m.forward_loss(idx, tgt, p)
    return m.backward_loss(1.0, cache)[1]


def _train(m, p, opt, idx, tgt, n):
    for _ in range(n):
        opt.step(_grads(m, p, idx, tgt))


def test_params_round_trip_bit_identically(tmp_path):
    cfg, m, p, idx, tgt = _setup()
    path = tmp_path / "a.npz"
    checkpoint.save(path, m, p)
    cfg2, p2, opt_state = checkpoint.load(path)
    assert cfg2 == cfg and opt_state is None
    assert list(p2) == m.param_keys()
    for k in p:
        assert np.array_equal(p[k], p2[k]) and p2[k].dtype == np.float64, k
        assert p2[k] is not p[k]  # a copy: loading never aliases the file buffer or the source


def test_loaded_params_keys_are_exactly_param_keys_and_tie_is_one_array(tmp_path):
    cfg, m, p, *_ = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)
    cfg2, p2, _ = checkpoint.load(tmp_path / "a.npz")
    m2 = GPT(cfg2)
    assert set(p2) == set(m2.param_keys())
    assert "lm_head.W" not in p2
    assert p2["wte.W"].flags["C_CONTIGUOUS"]
    assert m2.named_parameters(p2)["lm_head.W"] is p2["wte.W"]


def test_loaded_model_gives_identical_loss_and_gradients(tmp_path):
    cfg, m, p, idx, tgt = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)
    cfg2, p2, _ = checkpoint.load(tmp_path / "a.npz")
    m2 = GPT(cfg2)
    l1, c1 = m.forward_loss(idx, tgt, p)
    l2, c2 = m2.forward_loss(idx, tgt, p2)
    assert l1 == l2
    g1, g2 = m.backward_loss(1.0, c1)[1], m2.backward_loss(1.0, c2)[1]
    for k in g1:
        assert np.array_equal(g1[k], g2[k]), k


def test_optimizer_state_round_trips_exactly(tmp_path):
    cfg, m, p, idx, tgt = _setup()
    opt = AdamW(m.named_parameters(p), lr=3e-3, betas=(0.8, 0.95), weight_decay=0.05)
    _train(m, p, opt, idx, tgt, 3)
    checkpoint.save(tmp_path / "a.npz", m, p, opt)
    _, p2, st = checkpoint.load(tmp_path / "a.npz")
    assert st["t"] == 3 and st["lr"] == 3e-3 and st["betas"] == (0.8, 0.95)
    assert st["weight_decay"] == 0.05 and st["eps"] == opt.eps
    for k in p:
        assert np.array_equal(st["m"][k], opt.state[k]["m"])
        assert np.array_equal(st["v"][k], opt.state[k]["v"])

    opt2 = checkpoint.restore_optimizer(p2, st)
    assert opt2.t == 3 and opt2.lr == 3e-3 and opt2.betas == (0.8, 0.95)
    for k in p2:
        assert opt2.params[k] is p2[k]  # the SAME array objects as the loaded params
        assert np.array_equal(opt2.state[k]["m"], opt.state[k]["m"])
        assert np.array_equal(opt2.state[k]["v"], opt.state[k]["v"])
    assert len(opt2.state) == len(m.param_keys()) and "lm_head.W" not in opt2.state


def test_resume_equivalence_is_bit_identical(tmp_path):
    # train 3, save, load, step once more  ==  train 4 without stopping
    cfg, m, p, idx, tgt = _setup()
    opt = AdamW(m.named_parameters(p), lr=1e-2)
    _train(m, p, opt, idx, tgt, 3)
    checkpoint.save(tmp_path / "mid.npz", m, p, opt)
    _train(m, p, opt, idx, tgt, 1)  # the uninterrupted 4th step

    cfg2, p2, st = checkpoint.load(tmp_path / "mid.npz")
    m2 = GPT(cfg2)
    opt2 = checkpoint.restore_optimizer(p2, st)
    _train(m2, p2, opt2, idx, tgt, 1)

    assert opt.t == opt2.t == 4
    for k in p:
        assert np.array_equal(p[k], p2[k]), k
        assert np.array_equal(opt.state[k]["m"], opt2.state[k]["m"]), k
        assert np.array_equal(opt.state[k]["v"], opt2.state[k]["v"]), k


def test_resume_equivalence_would_fail_without_the_optimizer_state(tmp_path):
    # Test-the-test: a "resume" that restores params but a FRESH optimizer must differ.
    cfg, m, p, idx, tgt = _setup()
    opt = AdamW(m.named_parameters(p), lr=1e-2)
    _train(m, p, opt, idx, tgt, 3)
    checkpoint.save(tmp_path / "mid.npz", m, p, opt)
    _train(m, p, opt, idx, tgt, 1)

    cfg2, p2, _ = checkpoint.load(tmp_path / "mid.npz")
    fresh = AdamW(p2, lr=1e-2)  # t = 0, m = v = 0
    _train(GPT(cfg2), p2, fresh, idx, tgt, 1)
    assert any(not np.array_equal(p[k], p2[k]) for k in p)


def test_save_requires_optimizer_to_share_the_param_arrays(tmp_path):
    cfg, m, p, *_ = _setup()
    other = {k: v.copy() for k, v in p.items()}
    opt = AdamW(other, lr=1e-2)
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path / "a.npz", m, p, opt)


def test_save_rejects_wrong_param_keys_and_dtypes(tmp_path):
    cfg, m, p, *_ = _setup()
    bad = dict(p)
    bad["lm_head.W"] = p["wte.W"]
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path / "a.npz", m, bad)
    bad = dict(p)
    del bad["wpe.W"]
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path / "a.npz", m, bad)
    bad = dict(p)
    bad["wte.W"] = p["wte.W"].astype(np.float32)
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path / "a.npz", m, bad)
    bad = dict(p)
    bad["wte.W"] = np.asfortranarray(p["wte.W"])
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path / "a.npz", m, bad)


def _rewrite(src, dst, mutate):
    with np.load(src, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    mutate(d)
    with open(dst, "wb") as f:
        np.savez(f, **d)


def test_load_rejects_lm_head_key(tmp_path):
    cfg, m, p, *_ = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)
    _rewrite(tmp_path / "a.npz", tmp_path / "b.npz", lambda d: d.update({"param/lm_head.W": d["param/wte.W"]}))
    with pytest.raises(ValueError, match="lm_head"):
        checkpoint.load(tmp_path / "b.npz")


def test_load_rejects_wrong_or_missing_format(tmp_path):
    cfg, m, p, *_ = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)
    _rewrite(tmp_path / "a.npz", tmp_path / "b.npz", lambda d: d.update({"format": np.array("something-else")}))
    with pytest.raises(ValueError, match="format"):
        checkpoint.load(tmp_path / "b.npz")
    _rewrite(tmp_path / "a.npz", tmp_path / "c.npz", lambda d: d.pop("format"))
    with pytest.raises(ValueError, match="format"):
        checkpoint.load(tmp_path / "c.npz")


def test_load_rejects_missing_and_extra_param_keys(tmp_path):
    cfg, m, p, *_ = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)
    _rewrite(tmp_path / "a.npz", tmp_path / "b.npz", lambda d: d.pop("param/h.0.mlp.fc.W"))
    with pytest.raises(ValueError, match="missing"):
        checkpoint.load(tmp_path / "b.npz")
    _rewrite(tmp_path / "a.npz", tmp_path / "c.npz", lambda d: d.update({"param/bogus": np.zeros(2)}))
    with pytest.raises(ValueError, match="extra"):
        checkpoint.load(tmp_path / "c.npz")


def test_load_rejects_incomplete_or_misshapen_optimizer_state(tmp_path):
    cfg, m, p, idx, tgt = _setup()
    opt = AdamW(p, lr=1e-2)
    _train(m, p, opt, idx, tgt, 1)
    checkpoint.save(tmp_path / "a.npz", m, p, opt)
    _rewrite(tmp_path / "a.npz", tmp_path / "b.npz", lambda d: d.pop("opt_t"))
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint.load(tmp_path / "b.npz")
    _rewrite(tmp_path / "a.npz", tmp_path / "c.npz", lambda d: d.pop("opt_m/wte.W"))
    with pytest.raises(ValueError, match="opt_m/wte.W"):
        checkpoint.load(tmp_path / "c.npz")
    _rewrite(tmp_path / "a.npz", tmp_path / "d.npz", lambda d: d.update({"opt_v/wpe.W": np.zeros(3)}))
    with pytest.raises(ValueError, match="shape"):
        checkpoint.load(tmp_path / "d.npz")


def test_load_refuses_pickled_object_arrays(tmp_path):
    cfg, m, p, *_ = _setup()
    checkpoint.save(tmp_path / "a.npz", m, p)

    def poison(d):
        d["param/wte.W"] = np.array([{"evil": 1}], dtype=object)  # np.savez WILL pickle this

    _rewrite_obj = tmp_path / "evil.npz"
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    poison(d)
    with open(_rewrite_obj, "wb") as f:
        np.savez(f, **d)
    with pytest.raises(ValueError):
        checkpoint.load(_rewrite_obj)


def test_every_np_load_in_the_module_passes_allow_pickle_false():
    src = (Path(checkpoint.__file__)).read_text(encoding="utf-8")
    loads = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "load" and isinstance(n.func.value, ast.Name) and n.func.value.id == "np"]
    assert loads, "expected at least one np.load call"
    for call in loads:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("allow_pickle"), ast.Constant) and kw["allow_pickle"].value is False
    assert "pickle.load" not in src and "import pickle" not in src


def test_save_writes_exactly_the_given_path_no_extension_appended(tmp_path):
    cfg, m, p, *_ = _setup()
    target = tmp_path / "model.ckpt"
    checkpoint.save(target, m, p)
    assert target.exists() and not (tmp_path / "model.ckpt.npz").exists()
    assert checkpoint.load(target)[0] == cfg

# ===========================================================================
# PO-review additions: a params-only checkpoint must be validated too
# ===========================================================================
def _saved(tmp_path, with_opt=False):
    cfg, m, p, idx, tgt = _setup()
    opt = None
    if with_opt:
        opt = AdamW(p, lr=1e-2)
        _train(m, p, opt, idx, tgt, 1)
    checkpoint.save(tmp_path / "a.npz", m, p, opt)
    return tmp_path / "a.npz"


@pytest.mark.parametrize("key,shape", [
    ("param/h.0.attn.q.b", (1,)),
    ("param/ln_f.gamma", (1,)),
    ("param/h.0.mlp.fc.W", (24, 1)),
    ("param/wpe.W", (3, 8)),
    ("param/wte.W", (11, 9)),
])
def test_load_rejects_wrong_param_shapes_even_without_optimizer_state(tmp_path, key, shape):
    path = _saved(tmp_path)
    _rewrite(path, tmp_path / "bad.npz", lambda d: d.update({key: np.zeros(shape)}))
    with pytest.raises(ValueError, match="shape"):
        checkpoint.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.float16])
def test_load_rejects_non_float64_params_instead_of_silently_coercing(tmp_path, dtype):
    path = _saved(tmp_path)

    def mutate(d):
        d["param/wte.W"] = d["param/wte.W"].astype(dtype)

    _rewrite(path, tmp_path / "bad.npz", mutate)
    with pytest.raises(ValueError, match="dtype"):
        checkpoint.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_load_rejects_non_finite_params_and_optimizer_state(tmp_path, bad):
    path = _saved(tmp_path, with_opt=True)

    def poison_param(d):
        a = d["param/wpe.W"].copy()
        a[0, 0] = bad
        d["param/wpe.W"] = a

    def poison_state(d):
        a = d["opt_m/wpe.W"].copy()
        a[0, 0] = bad
        d["opt_m/wpe.W"] = a

    _rewrite(path, tmp_path / "p.npz", poison_param)
    with pytest.raises(ValueError, match="NaN or infinity"):
        checkpoint.load(tmp_path / "p.npz")
    _rewrite(path, tmp_path / "s.npz", poison_state)
    with pytest.raises(ValueError, match="NaN or infinity"):
        checkpoint.load(tmp_path / "s.npz")


def test_load_rejects_negative_step_count(tmp_path):
    path = _saved(tmp_path, with_opt=True)
    _rewrite(path, tmp_path / "bad.npz", lambda d: d.update({"opt_t": np.array(-3, dtype=np.int64)}))
    with pytest.raises(ValueError, match="opt_t"):
        checkpoint.load(tmp_path / "bad.npz")


def test_load_still_accepts_a_valid_checkpoint_after_hardening(tmp_path):
    path = _saved(tmp_path, with_opt=True)
    cfg, params, st = checkpoint.load(path)
    assert st["t"] == 1 and set(params) == set(GPT(cfg).param_keys())
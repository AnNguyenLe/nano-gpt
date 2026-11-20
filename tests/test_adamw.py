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

"""tests/test_generate.py -- Engineer C, Sprint 5 (SPRINT_5_PLAN.md section 6.3, milestone [C2]).

Covers root-level ``generate.py``: determinism, shapes, cache-vs-recompute
agreement, scope limits, and the CLI. NO test asserts anything about the CONTENT of
generated text (that would be "looks like English" evidence, which is inadmissible).
"""
from __future__ import annotations

import numpy as np
import pytest

import generate as gen
from nanogpt import checkpoint
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, ModelConfig, make_rng
from nanogpt.kv_cache import KVCache
from nanogpt.model import GPT
from nanogpt.tokenizer import load_corpus_and_tokenizer

_SEED = DEFAULT_SEED + 901
_V = 55
_CFG = ModelConfig(V=_V, T=16, d=8, L=2, H=2)


@pytest.fixture(scope="module")
def corpus():
    info, tok = load_corpus_and_tokenizer()
    return info, tok


def _model(seed_off=0):
    m = GPT(_CFG)
    p = m.init_params(make_rng(_SEED + seed_off))
    rng = make_rng(_SEED + 77 + seed_off)
    for k in m.param_keys():  # non-degenerate logits so argmax / sampling are meaningful
        if not (k.endswith(".gamma") or k.endswith(".beta")):
            p[k] += 0.4 * rng.standard_normal(p[k].shape)
    return m, p


def _gen(m, p, prompt, n, seed, **kw):
    return gen.generate(m, p, KVCache(m.cfg, 1), np.asarray(prompt, dtype=INT_DTYPE), n, make_rng(seed), **kw)


def test_generate_returns_requested_length_and_int_dtype_in_vocab():
    m, p = _model()
    out = _gen(m, p, [1, 2, 3], 7, _SEED)
    assert out.shape == (7,) and out.dtype == INT_DTYPE
    assert out.min() >= 0 and out.max() < _V


def test_generate_is_deterministic_for_a_fixed_seed_and_differs_for_another():
    m, p = _model()
    a = _gen(m, p, [4, 5], 12, 11)
    b = _gen(m, p, [4, 5], 12, 11)
    c = _gen(m, p, [4, 5], 12, 12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_greedy_equals_top_k_one_and_ignores_rng():
    m, p = _model()
    g1 = _gen(m, p, [3, 9], 10, 1, greedy_decode=True)
    g2 = _gen(m, p, [3, 9], 10, 999, greedy_decode=True)
    k1 = _gen(m, p, [3, 9], 10, 5, top_k=1)
    assert np.array_equal(g1, g2) and np.array_equal(g1, k1)


def test_cached_greedy_generation_equals_recomputing_the_full_forward_every_step():
    # Integration check beyond C8: the cached loop must pick the same tokens as a
    # naive loop that re-runs the whole (non-cached) forward on the growing sequence.
    m, p = _model()
    prompt = [7, 2, 40]
    n = 9
    cached = _gen(m, p, prompt, n, 0, greedy_decode=True)
    seq = list(prompt)
    for _ in range(n):
        logits, _ = m.forward(np.asarray([seq], dtype=INT_DTYPE), p)
        seq.append(int(np.argmax(logits[0, -1])))
    assert np.array_equal(cached, seq[len(prompt):])


def test_generate_fills_the_context_window_exactly_without_error():
    m, p = _model()
    out = _gen(m, p, [1, 2, 3, 4], _CFG.T - 4, 3)
    assert out.shape == (_CFG.T - 4,)


def test_generate_refuses_to_exceed_context_length():
    m, p = _model()
    with pytest.raises(ValueError, match="context length"):
        _gen(m, p, [1, 2, 3, 4], _CFG.T - 4 + 1, 3)


@pytest.mark.parametrize("bad_prompt", [[], [-1], [_V], [[1, 2]]])
def test_generate_rejects_bad_prompts(bad_prompt):
    m, p = _model()
    with pytest.raises(ValueError):
        gen.generate(m, p, KVCache(m.cfg, 1), np.asarray(bad_prompt), 2, make_rng(0))


def test_generate_rejects_float_prompt_and_bad_token_counts():
    m, p = _model()
    with pytest.raises(TypeError):
        gen.generate(m, p, KVCache(m.cfg, 1), np.asarray([1.0, 2.0]), 2, make_rng(0))
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            _gen(m, p, [1], bad, 0)


def test_generate_requires_an_empty_batch_one_cache():
    m, p = _model()
    used = KVCache(m.cfg, 1)
    m.forward_step(np.array([[1]]), p, used)
    with pytest.raises(ValueError):
        gen.generate(m, p, used, np.array([1]), 2, make_rng(0))
    with pytest.raises(ValueError):
        gen.generate(m, p, KVCache(m.cfg, 2), np.array([1]), 2, make_rng(0))


def test_generate_does_not_mutate_params():
    m, p = _model()
    before = {k: v.copy() for k, v in p.items()}
    _gen(m, p, [1, 2], 6, 4)
    for k in before:
        assert np.array_equal(before[k], p[k])


def test_temperature_and_top_k_are_forwarded():
    m, p = _model()
    hot = _gen(m, p, [1, 2], 14, 21, temperature=50.0)
    cold = _gen(m, p, [1, 2], 14, 21, temperature=0.01)
    greedy_ = _gen(m, p, [1, 2], 14, 21, greedy_decode=True)
    assert np.array_equal(cold, greedy_)  # t -> 0 collapses onto argmax
    assert not np.array_equal(hot, greedy_)
    with pytest.raises(ValueError):
        _gen(m, p, [1, 2], 3, 0, top_k=_V + 1)
    with pytest.raises(ValueError):
        _gen(m, p, [1, 2], 3, 0, temperature=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_TINY = ["--random-init", "--L", "2", "--d", "8", "--T", "16", "--H", "2"]


def test_main_random_init_prints_prompt_plus_requested_number_of_characters(corpus, capsys):
    info, tok = corpus
    prompt = info.text[:3]
    rc = gen.main(_TINY + ["--prompt", prompt, "--tokens", "8", "--seed", "5"])
    assert rc == 0
    out = capsys.readouterr().out.rstrip("\n")  # only the trailing newline we printed
    assert len(out) == len(prompt) + 8 and out.startswith(prompt)


def test_main_is_deterministic_for_a_fixed_seed(corpus, capsys):
    info, tok = corpus
    args = _TINY + ["--prompt", info.text[:2], "--tokens", "6", "--seed", "9", "--top-k", "5"]
    gen.main(args)
    a = capsys.readouterr().out
    gen.main(args)
    b = capsys.readouterr().out
    gen.main(args[:-4] + ["--seed", "10"])
    c = capsys.readouterr().out
    assert a == b
    assert a != c  # --seed also seeds the random init, so a new seed gives a different run


def test_main_default_prompt_is_the_first_corpus_character(corpus, capsys):
    info, tok = corpus
    gen.main(_TINY + ["--tokens", "4"])
    out = capsys.readouterr().out.rstrip("\n")
    assert out[0] == info.text[0] and len(out) == 5


def test_main_greedy_flag_ignores_the_sampling_seed(corpus, tmp_path, capsys):
    info, tok = corpus
    m, p = _model(5)
    path = tmp_path / "g.ckpt"
    checkpoint.save(path, m, p)  # fixed weights, so only the sampling seed varies
    base = ["--checkpoint", str(path), "--prompt", info.text[:2], "--tokens", "8", "--greedy"]
    gen.main(base + ["--seed", "1"])
    a = capsys.readouterr().out
    gen.main(base + ["--seed", "2"])
    b = capsys.readouterr().out
    assert a == b

def test_main_loads_a_checkpoint_and_matches_the_in_memory_model(corpus, tmp_path, capsys):
    info, tok = corpus
    m, p = _model(3)
    path = tmp_path / "m.ckpt"
    checkpoint.save(path, m, p)
    prompt = info.text[:2]
    rc = gen.main(["--checkpoint", str(path), "--prompt", prompt, "--tokens", "9", "--seed", "4", "--top-k", "7"])
    assert rc == 0
    out = capsys.readouterr().out.rstrip("\n")
    want = _gen(m, p, tok.encode(prompt), 9, 4 + 2, top_k=7)
    assert out == prompt + tok.decode(want)


def test_main_rejects_out_of_vocab_prompt(capsys):
    with pytest.raises(SystemExit) as e:
        gen.main(_TINY + ["--prompt", "\u00e9\u00e9\u2603"])
    assert e.value.code not in (0, None)


def test_main_rejects_exceeding_context_length(corpus):
    info, tok = corpus
    with pytest.raises(SystemExit) as e:
        gen.main(_TINY + ["--prompt", info.text[:4], "--tokens", "13"])  # 4 + 13 > 16
    assert e.value.code not in (0, None)


def test_main_requires_exactly_one_model_source():
    with pytest.raises(SystemExit):
        gen.main(["--tokens", "3"])
    with pytest.raises(SystemExit):
        gen.main(["--random-init", "--checkpoint", "x.ckpt"])


def test_main_rejects_vocab_mismatch_with_the_checkpoint(tmp_path):
    from nanogpt.config import tiny_config

    cfg = tiny_config(11)  # V=11 != corpus V=55
    m = GPT(cfg)
    p = m.init_params(make_rng(0))
    path = tmp_path / "wrongV.ckpt"
    checkpoint.save(path, m, p)
    with pytest.raises(SystemExit):
        gen.main(["--checkpoint", str(path), "--tokens", "2"])

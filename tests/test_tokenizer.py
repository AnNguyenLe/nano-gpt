"""tests/test_tokenizer.py — Engineer A, milestone A3 (Phase 0 hard gate)."""

import socket

import numpy as np
import pytest

from nanogpt.tokenizer import (
    CharTokenizer,
    CorpusInfo,
    generate_synthetic_corpus,
    load_corpus_and_tokenizer,
    resolve_corpus,
)


# ---------------------------------------------------------------------------
# generate_synthetic_corpus: determinism
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_corpus():
    a = generate_synthetic_corpus(seed=1337, n_chars=5000)
    b = generate_synthetic_corpus(seed=1337, n_chars=5000)
    assert a == b


def test_different_seed_gives_different_corpus():
    a = generate_synthetic_corpus(seed=1337, n_chars=5000)
    b = generate_synthetic_corpus(seed=7, n_chars=5000)
    assert a != b


def test_synthetic_corpus_exact_length():
    for n in (0, 1, 17, 500, 12345):
        text = generate_synthetic_corpus(seed=42, n_chars=n)
        assert len(text) == n


def test_synthetic_corpus_structure_sanity():
    text = generate_synthetic_corpus(seed=42, n_chars=20000)
    # Should contain paragraph separators and sentence-ending punctuation.
    assert "\n\n" in text
    assert any(c in text for c in ".!?")
    # No non-ASCII surprises (the word list + punctuation is all ASCII).
    assert all(ord(c) < 128 for c in text)


def test_synthetic_generator_touches_no_network(monkeypatch):
    def _raise(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _raise)
    text = generate_synthetic_corpus(seed=1337, n_chars=2000)
    assert len(text) == 2000


# ---------------------------------------------------------------------------
# resolve_corpus: bundled vs synthetic branches
# ---------------------------------------------------------------------------
def test_resolve_corpus_bundled_branch(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("hello world, this is a bundled corpus!\n", encoding="utf-8")

    info = resolve_corpus(tmp_path)
    assert isinstance(info, CorpusInfo)
    assert info.source == "bundled"
    assert info.generator is None
    assert info.seed is None
    assert info.text == "hello world, this is a bundled corpus!\n"
    assert info.n_chars == len(info.text)


def test_resolve_corpus_synthetic_branch_on_empty_dir(tmp_path):
    info = resolve_corpus(tmp_path)  # empty dir, no corpus.txt
    assert info.source == "synthetic"
    assert info.generator == "synthetic-v1"
    assert info.seed == 1337
    assert info.n_chars == len(info.text) == 1_000_000


def test_resolve_corpus_generator_never_writes_corpus_txt(tmp_path):
    resolve_corpus(tmp_path)
    assert not (tmp_path / "corpus.txt").exists()


def test_resolve_corpus_rejects_empty_bundled_file(tmp_path):
    (tmp_path / "corpus.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_corpus(tmp_path)


def test_resolve_corpus_rejects_oversized_bundled_file(tmp_path):
    big_path = tmp_path / "corpus.txt"
    # Write just over 5 MB.
    with open(big_path, "w", encoding="utf-8") as f:
        f.write("x" * (5 * 1024 * 1024 + 10))
    with pytest.raises(ValueError):
        resolve_corpus(tmp_path)


def test_resolve_corpus_sha256_matches_text(tmp_path):
    import hashlib

    corpus_path = tmp_path / "corpus.txt"
    text = "a small bundled corpus for hashing\n"
    corpus_path.write_text(text, encoding="utf-8")
    info = resolve_corpus(tmp_path)
    assert info.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_resolve_corpus_no_network(monkeypatch, tmp_path):
    def _raise(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _raise)
    # synthetic branch
    info_synth = resolve_corpus(tmp_path)
    assert info_synth.source == "synthetic"
    # bundled branch
    (tmp_path / "corpus.txt").write_text("bundled text\n", encoding="utf-8")
    info_bundled = resolve_corpus(tmp_path)
    assert info_bundled.source == "bundled"


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------
def test_decode_encode_roundtrip_small():
    text = "Hello, world! This is a test.\n\nAnother paragraph."
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_decode_encode_roundtrip_full_corpus():
    text = generate_synthetic_corpus(seed=1337, n_chars=50_000)
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_encode_returns_int64():
    tok = CharTokenizer.from_text("abc")
    ids = tok.encode("abc")
    assert ids.dtype == np.int64


def test_encode_raises_on_unknown_char():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abcd")


def test_vocab_size_and_roundtrip_indices():
    tok = CharTokenizer.from_text("banana")
    assert tok.vocab_size == len(set("banana"))
    for ch, i in tok.stoi.items():
        assert tok.itos[i] == ch


# ---------------------------------------------------------------------------
# load_corpus_and_tokenizer: V stability across fresh loads
# ---------------------------------------------------------------------------
def test_vocab_size_identical_across_two_fresh_loads(tmp_path):
    info1, tok1 = load_corpus_and_tokenizer(tmp_path)
    info2, tok2 = load_corpus_and_tokenizer(tmp_path)
    assert tok1.vocab_size == tok2.vocab_size
    assert info1.sha256 == info2.sha256
    assert info1.text == info2.text


def test_load_corpus_and_tokenizer_bundled(tmp_path):
    (tmp_path / "corpus.txt").write_text("abcabcabc\n", encoding="utf-8")
    info, tok = load_corpus_and_tokenizer(tmp_path)
    assert info.source == "bundled"
    assert tok.decode(tok.encode(info.text)) == info.text

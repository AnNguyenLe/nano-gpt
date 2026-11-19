"""nanogpt.tokenizer — corpus resolution, a seeded synthetic corpus
generator, and a character-level tokenizer.

This is the Phase 0 hard gate (SPRINT_1_PLAN.md §6.3 / §10): no network
access is ever made, and the corpus is either read from `data/corpus.txt`
(if present) or generated deterministically from a local `random.Random`
seed. The generator never writes `data/corpus.txt`, so provenance ("did a
corpus.txt get added since the last run?") can never silently flip.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanogpt.config import CORPUS_SEED, INT_DTYPE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GENERATOR_ID = "synthetic-v1"
SYNTHETIC_N_CHARS = 1_000_000

_MAX_BUNDLED_BYTES = 5 * 1024 * 1024  # 5 MB

# A fixed, in-file word list (~180 common lowercase English words). This is
# a learnable, structured synthetic source, not a text-synthesis project:
# keep it simple and deterministic.
_WORDS = [
    "the", "a", "an", "of", "and", "to", "in", "that", "it", "is",
    "was", "for", "on", "with", "as", "at", "by", "from", "this", "but",
    "not", "are", "be", "or", "have", "had", "has", "his", "her", "its",
    "they", "we", "you", "i", "he", "she", "which", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its",
    "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "water", "light", "mountain", "river", "forest", "city", "garden",
    "book", "story", "letter", "music", "picture", "table", "window", "door", "road", "house",
    "friend", "family", "child", "teacher", "student", "doctor", "farmer", "artist", "writer", "traveler",
    "morning", "evening", "night", "summer", "winter", "spring", "autumn", "season", "weather", "cloud",
    "ocean", "island", "valley", "desert", "forest", "field", "garden", "village", "town", "country",
    "happy", "quiet", "bright", "dark", "gentle", "strong", "simple", "ancient", "modern", "distant",
    "slowly", "quickly", "softly", "clearly", "often", "always", "never", "sometimes", "again", "still",
    "walk", "run", "speak", "listen", "write", "read", "build", "grow", "learn", "teach",
    "dream", "remember", "wonder", "imagine", "travel", "explore", "discover", "create", "share", "return",
    "river", "mountain", "star", "moon", "sun", "sky", "sea", "wind", "rain", "snow",
    "stone", "tree", "leaf", "flower", "bird", "animal", "fish", "horse", "dog", "cat",
]


def generate_synthetic_corpus(seed: int = CORPUS_SEED, n_chars: int = SYNTHETIC_N_CHARS) -> str:
    """Deterministically generate a synthetic text corpus of exactly
    `n_chars` characters, using a LOCAL `random.Random(seed)` instance only
    (no global RNG state touched, no numpy, no network). Same seed ->
    byte-identical output; different seed -> (almost certainly) different
    output.

    Structure: paragraphs of 3-7 sentences joined by "\n\n"; sentences of
    4-12 words, first word capitalized, an occasional ", " between words,
    ending in ". ", "! " or "? ".
    """
    rnd = random.Random(seed)
    sentence_enders = [". ", "! ", "? "]

    def make_sentence() -> str:
        n_words = rnd.randint(4, 12)
        words = [rnd.choice(_WORDS) for _ in range(n_words)]
        pieces = [words[0].capitalize()]
        for w in words[1:]:
            if rnd.random() < 0.12:
                pieces[-1] = pieces[-1] + ","
            pieces.append(w)
        sentence = " ".join(pieces)
        sentence += rnd.choice(sentence_enders)
        return sentence

    def make_paragraph() -> str:
        n_sentences = rnd.randint(3, 7)
        return "".join(make_sentence() for _ in range(n_sentences))

    chunks = []
    total_len = 0
    while total_len < n_chars:
        para = make_paragraph()
        chunks.append(para)
        total_len += len(para) + 2  # account for the "\n\n" joiner
        chunks.append("\n\n")

    text = "".join(chunks)
    if len(text) < n_chars:
        # Extremely unlikely given the while-loop above, but be exact.
        while len(text) < n_chars:
            text += make_paragraph() + "\n\n"
    return text[:n_chars]


@dataclass(frozen=True)
class CorpusInfo:
    text: str
    source: str  # "bundled" | "synthetic"
    generator: str  # None for bundled, GENERATOR_ID for synthetic
    seed: int  # None for bundled, the seed used for synthetic
    n_chars: int
    sha256: str


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_corpus(data_dir: Path = DATA_DIR) -> CorpusInfo:
    """Resolve the training corpus with NO network access, ever.

    If `data_dir/corpus.txt` exists, it is read as UTF-8, required to be
    non-empty and under 5 MB, and returned with source="bundled". The
    generator never writes `data/corpus.txt`: if it doesn't exist, a
    synthetic corpus is generated in memory and returned with
    source="synthetic", but the file is never created on disk.
    """
    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.txt"

    if corpus_path.exists():
        size = corpus_path.stat().st_size
        if size == 0:
            raise ValueError(f"resolve_corpus: {corpus_path} is empty")
        if size >= _MAX_BUNDLED_BYTES:
            raise ValueError(
                f"resolve_corpus: {corpus_path} is {size} bytes, must be under 5 MB"
            )
        text = corpus_path.read_text(encoding="utf-8")
        if len(text) == 0:
            raise ValueError(f"resolve_corpus: {corpus_path} is empty")
        return CorpusInfo(
            text=text,
            source="bundled",
            generator=None,
            seed=None,
            n_chars=len(text),
            sha256=_sha256_of(text),
        )

    text = generate_synthetic_corpus(CORPUS_SEED, SYNTHETIC_N_CHARS)
    return CorpusInfo(
        text=text,
        source="synthetic",
        generator=GENERATOR_ID,
        seed=CORPUS_SEED,
        n_chars=len(text),
        sha256=_sha256_of(text),
    )


class CharTokenizer:
    """A simple character-level tokenizer over a fixed, sorted vocabulary."""

    def __init__(self, stoi: dict, itos: list):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = list(vocab)
        return cls(stoi, itos)

    def encode(self, s: str) -> np.ndarray:
        try:
            ids = [self.stoi[ch] for ch in s]
        except KeyError as e:
            raise ValueError(f"CharTokenizer.encode: unknown character {e.args[0]!r}") from None
        return np.array(ids, dtype=INT_DTYPE)

    def decode(self, ids) -> str:
        arr = np.asarray(ids)
        return "".join(self.itos[int(i)] for i in arr.reshape(-1))


def load_corpus_and_tokenizer(data_dir: Path = DATA_DIR):
    """Convenience: resolve the corpus and build a CharTokenizer from it
    in one call. Returns (CorpusInfo, CharTokenizer).
    """
    info = resolve_corpus(data_dir)
    tok = CharTokenizer.from_text(info.text)
    return info, tok

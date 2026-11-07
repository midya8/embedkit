"""Skip-gram with negative sampling (SGNS), pure Python.

Word2Vec's SGNS objective trained with plain SGD: for each (center, context)
pair, push the center embedding toward the context embedding and away from k
negatives drawn from the unigram^0.75 distribution. All randomness comes from
a single seeded generator: weight init, pair shuffling and negative sampling
- the same corpus and seed always produce the same embeddings.
"""

from __future__ import annotations

import math
import random
from collections import Counter


class Embeddings:
    """A trained embedding table with similarity queries."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            raise ValueError("vectors must be non-empty")
        dims = {len(v) for v in vectors.values()}
        if len(dims) != 1:
            raise ValueError("all vectors must share one dimension")
        self._vectors = {w: tuple(v) for w, v in vectors.items()}
        self.dim = dims.pop()
        self._norms = {w: math.sqrt(sum(x * x for x in v)) for w, v in self._vectors.items()}

    @property
    def vocab(self) -> frozenset[str]:
        return frozenset(self._vectors)

    def vector(self, word: str) -> tuple[float, ...]:
        if word not in self._vectors:
            raise KeyError(f"unknown word: {word}")
        return self._vectors[word]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity; zero vectors score 0."""
        va, vb = self.vector(a), self.vector(b)
        na, nb = self._norms[a], self._norms[b]
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(va, vb)) / (na * nb)

    def nearest(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """k nearest words by cosine similarity, excluding the word itself.
        Ties break on the lexicographically smaller word."""
        if k < 1:
            raise ValueError("k must be >= 1")
        self.vector(word)
        scored = sorted(
            ((w, self.similarity(word, w)) for w in self._vectors if w != word),
            key=lambda ws: (-ws[1], ws[0]),
        )
        return scored[:k]


def _negative_sampler(counts: Counter, rng: random.Random) -> "callable":
    adjusted = {w: c ** 0.75 for w, c in counts.items()}
    words = sorted(adjusted)
    total = sum(adjusted.values())
    cumulative: list[float] = []
    running = 0.0
    for w in words:
        running += adjusted[w] / total
        cumulative.append(running)

    def draw(exclude: set[str]) -> str:
        for _ in range(20):
            r = rng.random()
            lo, hi = 0, len(words) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if r < cumulative[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            if words[lo] not in exclude:
                return words[lo]
        return next(w for w in words if w not in exclude)

    return draw


def train(
    sentences: list[list[str]],
    dim: int = 16,
    window: int = 2,
    negatives: int = 5,
    epochs: int = 30,
    lr: float = 0.05,
    seed: int = 0,
    min_count: int = 1,
) -> Embeddings:
    """Train SGNS embeddings. Deterministic for a fixed seed and corpus."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")
    if negatives < 1:
        raise ValueError("negatives must be >= 1")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    counts: Counter = Counter()
    for sent in sentences:
        counts.update(sent)
    vocab = sorted(w for w, c in counts.items() if c >= min_count)
    if len(vocab) < 2:
        raise ValueError("vocabulary too small to train")
    vocab_set = set(vocab)

    rng = random.Random(seed)
    init = lambda: (rng.random() - 0.5) / dim
    center_w = {w: [init() for _ in range(dim)] for w in vocab}
    context_w = {w: [0.0] * dim for w in vocab}
    sampler = _negative_sampler(Counter({w: counts[w] for w in vocab}), rng)

    def step(center: str, context: str, label: int) -> None:
        """One sigmoid-gradient update toward (label 1) or away from (0) context."""
        cv, xv = center_w[center], context_w[context]
        dot = sum(a * b for a, b in zip(cv, xv))
        # clamp for numerical stability of the gradient factor
        dot = max(-15.0, min(15.0, dot))
        grad = (1.0 / (1.0 + math.exp(-dot)) - label) * lr
        for i in range(dim):
            c_i = cv[i]
            cv[i] -= grad * xv[i]
            xv[i] -= grad * c_i

    for _ in range(epochs):
        pairs: list[tuple[str, str]] = []
        for sent in sentences:
            toks = [t for t in sent if t in vocab_set]
            for i, center in enumerate(toks):
                lo = max(0, i - window)
                hi = min(len(toks), i + window + 1)
                for j in range(lo, hi):
                    if j != i:
                        pairs.append((center, toks[j]))
        rng.shuffle(pairs)
        for center, context in pairs:
            step(center, context, 1)
            negatives_drawn: set[str] = set()
            while len(negatives_drawn) < min(negatives, len(vocab) - 1):
                neg = sampler({context} | negatives_drawn)
                negatives_drawn.add(neg)
                step(center, neg, 0)

    return Embeddings({w: center_w[w] for w in vocab})

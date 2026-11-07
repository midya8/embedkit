"""Regenerate corpus.txt deterministically (seeded; output is checked in).

Each sentence stays inside one topic's vocabulary with a small amount of
shared function words, so SGNS has a learnable cluster structure without
the task being trivially bag-of-words separable.
"""

import random
from pathlib import Path

TOPICS = {
    "ml": ["gradient", "descent", "optimizer", "loss", "epoch", "backprop", "weights", "tensor"],
    "nlp": ["token", "corpus", "vocabulary", "embedding", "language", "parser", "grammar", "sentence"],
    "systems": ["server", "latency", "deploy", "cluster", "cache", "throughput", "replica", "failover"],
}
FUNCTION = ["the", "a", "with", "of", "and", "to", "in", "on"]


def generate(seed: int = 1234, sentences_per_topic: int = 100) -> str:
    rng = random.Random(seed)
    sentences = []
    for _ in range(sentences_per_topic):
        for words in TOPICS.values():
            size = rng.randint(4, 7)
            toks = []
            for _ in range(size):
                toks.append(rng.choice(words) if rng.random() < 0.8 else rng.choice(FUNCTION))
            rng.shuffle(toks)
            sentences.append(" ".join(toks) + ".")
    rng.shuffle(sentences)
    return "\n".join(sentences) + "\n"


if __name__ == "__main__":
    out = Path(__file__).with_name("corpus.txt")
    out.write_text(generate(), encoding="utf-8")
    print(f"wrote {out}")

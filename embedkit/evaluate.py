"""Cluster-coherence evaluation: do nearest neighbors stay in topic?"""

from __future__ import annotations

from .model import Embeddings


def coherence(emb: Embeddings, clusters: dict[str, list[str]], k: int = 3) -> dict:
    """For every word in every cluster, the fraction of its k nearest
    neighbors that share its cluster. Returns per-word detail plus means."""
    if k < 1:
        raise ValueError("k must be >= 1")
    word_to_cluster: dict[str, str] = {}
    for name, words in clusters.items():
        for w in words:
            if w in word_to_cluster:
                raise ValueError(f"word in two clusters: {w}")
            word_to_cluster[w] = name
    missing = [w for w in word_to_cluster if w not in emb.vocab]
    if missing:
        raise ValueError(f"eval words missing from vocabulary: {sorted(missing)}")

    per_word: dict[str, dict] = {}
    for name, words in clusters.items():
        for w in words:
            near = emb.nearest(w, k)
            hits = sum(1 for n, _ in near if word_to_cluster.get(n) == name)
            per_word[w] = {
                "cluster": name,
                "coherence": round(hits / k, 4),
                "nearest": [n for n, _ in near],
            }
    scores = [d["coherence"] for d in per_word.values()]
    return {
        "k": k,
        "words": len(scores),
        "mean_coherence": round(sum(scores) / len(scores), 4),
        "per_word": per_word,
    }


def separation(emb: Embeddings, clusters: dict[str, list[str]]) -> dict:
    """Mean intra-cluster minus mean inter-cluster similarity: the margin
    between words that share a topic and words that do not."""
    word_to_cluster = {w: name for name, words in clusters.items() for w in words}
    words = sorted(word_to_cluster)
    intra: list[float] = []
    inter: list[float] = []
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            sim = emb.similarity(a, b)
            (intra if word_to_cluster[a] == word_to_cluster[b] else inter).append(sim)
    if not intra or not inter:
        raise ValueError("need at least two clusters with at least two words between them")
    mean_intra = sum(intra) / len(intra)
    mean_inter = sum(inter) / len(inter)
    return {
        "mean_intra": round(mean_intra, 4),
        "mean_inter": round(mean_inter, 4),
        "margin": round(mean_intra - mean_inter, 4),
        "pairs_intra": len(intra),
        "pairs_inter": len(inter),
    }

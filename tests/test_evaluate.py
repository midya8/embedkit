import unittest

from embedkit.evaluate import coherence, separation
from embedkit.model import Embeddings


def axis_embeddings():
    # two clean clusters along different axes
    return Embeddings({
        "a1": [1.0, 0.0, 0.0], "a2": [0.9, 0.1, 0.0], "a3": [0.8, 0.0, 0.2],
        "b1": [0.0, 1.0, 0.0], "b2": [0.1, 0.9, 0.0], "b3": [0.0, 0.8, 0.2],
    })


CLUSTERS = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"]}


class TestCoherence(unittest.TestCase):
    def test_perfect_clusters_score_one(self):
        result = coherence(axis_embeddings(), CLUSTERS, k=2)
        self.assertEqual(result["mean_coherence"], 1.0)
        self.assertEqual(result["words"], 6)
        for detail in result["per_word"].values():
            self.assertEqual(len(detail["nearest"]), 2)

    def test_missing_vocab_word_raises(self):
        with self.assertRaises(ValueError):
            coherence(axis_embeddings(), {"a": ["nope"], "b": ["b1"]}, k=1)

    def test_word_in_two_clusters_raises(self):
        with self.assertRaises(ValueError):
            coherence(axis_embeddings(), {"a": ["a1"], "b": ["a1"]}, k=1)

    def test_k_validation(self):
        with self.assertRaises(ValueError):
            coherence(axis_embeddings(), CLUSTERS, k=0)


class TestSeparation(unittest.TestCase):
    def test_margin_positive_for_clean_clusters(self):
        result = separation(axis_embeddings(), CLUSTERS)
        self.assertGreater(result["margin"], 0.5)
        self.assertGreater(result["mean_intra"], result["mean_inter"])
        self.assertEqual(result["pairs_intra"], 6)  # C(3,2) per cluster
        self.assertEqual(result["pairs_inter"], 9)  # 3 x 3

    def test_single_cluster_raises(self):
        with self.assertRaises(ValueError):
            separation(axis_embeddings(), {"a": ["a1", "a2"]})


if __name__ == "__main__":
    unittest.main()

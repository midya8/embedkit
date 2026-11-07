import unittest

from embedkit.model import Embeddings, train


def small_corpus():
    return [["alpha", "beta", "gamma"], ["beta", "gamma", "delta"]] * 5


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.emb = Embeddings({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Embeddings({})

    def test_mismatched_dims_raise(self):
        with self.assertRaises(ValueError):
            Embeddings({"a": [1.0], "b": [1.0, 2.0]})

    def test_vector_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.emb.vector("zzz")

    def test_vocab(self):
        self.assertEqual(self.emb.vocab, frozenset({"a", "b", "c"}))

    def test_similarity_self_is_one(self):
        self.assertAlmostEqual(self.emb.similarity("c", "c"), 1.0)

    def test_similarity_symmetric(self):
        self.assertAlmostEqual(self.emb.similarity("a", "c"), self.emb.similarity("c", "a"))

    def test_similarity_orthogonal_zero(self):
        self.assertAlmostEqual(self.emb.similarity("a", "b"), 0.0)

    def test_zero_vector_similarity_zero(self):
        emb = Embeddings({"z": [0.0, 0.0], "a": [1.0, 0.0]})
        self.assertEqual(emb.similarity("z", "a"), 0.0)

    def test_nearest_excludes_self(self):
        near = self.emb.nearest("a", 2)
        self.assertNotIn("a", [w for w, _ in near])

    def test_nearest_tie_break_lexicographic(self):
        emb = Embeddings({"q": [1.0, 0.0], "zz": [2.0, 0.0], "aa": [2.0, 0.0]})
        near = emb.nearest("q", 2)
        self.assertEqual([w for w, _ in near], ["aa", "zz"])

    def test_nearest_k_validation(self):
        with self.assertRaises(ValueError):
            self.emb.nearest("a", 0)

    def test_nearest_capped_at_vocab(self):
        self.assertEqual(len(self.emb.nearest("a", 99)), 2)


class TestTrain(unittest.TestCase):
    def test_deterministic_same_seed(self):
        e1 = train(small_corpus(), dim=4, epochs=3, seed=5)
        e2 = train(small_corpus(), dim=4, epochs=3, seed=5)
        for w in e1.vocab:
            self.assertEqual(e1.vector(w), e2.vector(w))

    def test_different_seed_differs(self):
        e1 = train(small_corpus(), dim=4, epochs=3, seed=5)
        e2 = train(small_corpus(), dim=4, epochs=3, seed=6)
        self.assertNotEqual(e1.vector("alpha"), e2.vector("alpha"))

    def test_min_count_filters(self):
        corpus = [["common", "other"], ["common", "other"], ["rare"]]
        emb = train(corpus, dim=4, epochs=1, seed=1, min_count=2)
        self.assertNotIn("rare", emb.vocab)
        self.assertIn("common", emb.vocab)

    def test_vocab_too_small_raises(self):
        with self.assertRaises(ValueError):
            train([["only"]], dim=4, epochs=1, seed=1)

    def test_param_validation(self):
        for kwargs in ({"dim": 0}, {"window": 0}, {"negatives": 0},
                       {"epochs": 0}, {"lr": 0}, {"lr": -1.0}, {"min_count": 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                train(small_corpus(), seed=1, **kwargs)


class TestTrainedQuality(unittest.TestCase):
    """SGNS actually separates topics on the checked-in clustered corpus."""

    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        from embedkit.tokenize import tokenize_sentences
        data = Path(__file__).parent.parent / "examples" / "data"
        cls.sentences = tokenize_sentences((data / "corpus.txt").read_text())
        cls.clusters = json.loads((data / "eval.json").read_text())["clusters"]
        cls.emb = train(cls.sentences, dim=8, window=2, negatives=5,
                        epochs=15, seed=7)

    def test_same_topic_closer_than_cross_topic(self):
        sim_in = self.emb.similarity("gradient", "optimizer")
        sim_out = self.emb.similarity("gradient", "corpus")
        self.assertGreater(sim_in, sim_out + 0.2)

    def test_neighbors_stay_in_cluster(self):
        word_to_cluster = {w: name for name, ws in self.clusters.items() for w in ws}
        clean = 0
        for name, words in self.clusters.items():
            for w in words:
                near = self.emb.nearest(w, 3)
                if all(word_to_cluster.get(n) == name for n, _ in near):
                    clean += 1
        self.assertGreaterEqual(clean / sum(len(ws) for ws in self.clusters.values()), 0.9)


if __name__ == "__main__":
    unittest.main()

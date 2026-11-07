import json
import unittest

from embedkit.report import build_report, to_json, to_markdown, to_terminal


def sample_report():
    coh = {"k": 2, "words": 2, "mean_coherence": 0.75,
           "per_word": {"a": {"cluster": "x", "coherence": 1.0, "nearest": ["b", "c"]},
                        "d": {"cluster": "y", "coherence": 0.5, "nearest": ["e", "a"]}}}
    sep = {"mean_intra": 0.9, "mean_inter": 0.2, "margin": 0.7,
           "pairs_intra": 2, "pairs_inter": 4}
    return build_report(coh, sep, {"dim": 8, "seed": 7})


class TestReport(unittest.TestCase):
    def test_json_round_trip(self):
        report = sample_report()
        parsed = json.loads(to_json(report))
        self.assertEqual(parsed["coherence"]["mean_coherence"], 0.75)
        self.assertEqual(parsed["params"]["dim"], 8)

    def test_markdown_shape(self):
        md = to_markdown(sample_report())
        self.assertIn("# EmbedKit report", md)
        self.assertIn("| Word | Cluster | Coherence | Nearest |", md)
        self.assertIn("| a | x | 1.0000 | b, c |", md)
        self.assertIn("Separation margin: 0.7", md)

    def test_terminal_shape(self):
        text = to_terminal(sample_report())
        self.assertIn("mean_coherence=0.75", text)
        self.assertIn("margin=0.7", text)
        self.assertIn("a ", text)


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "examples" / "data" / "corpus.txt"
EVAL = ROOT / "examples" / "data" / "eval.json"


def run_cli(*argv):
    return subprocess.run([sys.executable, "-m", "embedkit", *argv],
                          cwd=ROOT, capture_output=True, text=True)


class TestCli(unittest.TestCase):
    def test_nearest_prints_neighbors(self):
        proc = run_cli("nearest", "--corpus", str(CORPUS), "--word", "gradient",
                       "--dim", "8", "--epochs", "10", "--seed", "7")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("loss", proc.stdout)

    def test_nearest_unknown_word_errors(self):
        proc = run_cli("nearest", "--corpus", str(CORPUS), "--word", "zzz",
                       "--dim", "4", "--epochs", "1", "--seed", "1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown word", proc.stderr)

    def test_evaluate_passes_easy_gates(self):
        proc = run_cli("evaluate", "--corpus", str(CORPUS), "--eval", str(EVAL),
                       "--dim", "8", "--epochs", "10", "--seed", "7",
                       "--min-coherence", "0.8", "--min-margin", "0.3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("mean_coherence=", proc.stdout)

    def test_evaluate_fails_impossible_gate(self):
        proc = run_cli("evaluate", "--corpus", str(CORPUS), "--eval", str(EVAL),
                       "--dim", "8", "--epochs", "10", "--seed", "7",
                       "--min-coherence", "1.01")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("GATE FAIL", proc.stderr)

    def test_evaluate_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, m = Path(tmp) / "r.json", Path(tmp) / "r.md"
            proc = run_cli("evaluate", "--corpus", str(CORPUS), "--eval", str(EVAL),
                           "--dim", "8", "--epochs", "10", "--seed", "7",
                           "--json", str(j), "--markdown", str(m))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = json.loads(j.read_text())
            self.assertEqual(parsed["coherence"]["words"], 24)
            self.assertIn("# EmbedKit report", m.read_text())

    def test_missing_corpus_errors(self):
        proc = run_cli("nearest", "--corpus", "/no/such/file", "--word", "x",
                       "--dim", "4", "--epochs", "1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("cannot read corpus", proc.stderr)

    def test_bad_eval_json_errors(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
            proc = run_cli("evaluate", "--corpus", str(CORPUS), "--eval", fh.name,
                           "--dim", "4", "--epochs", "1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid JSON", proc.stderr)


if __name__ == "__main__":
    unittest.main()

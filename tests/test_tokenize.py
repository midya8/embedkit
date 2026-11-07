import unittest

from embedkit.tokenize import tokenize, tokenize_sentences


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", ",", "world", "!"])

    def test_keeps_apostrophes(self):
        self.assertEqual(tokenize("it's fine"), ["it's", "fine"])

    def test_sentence_split_drops_terminal_punct(self):
        self.assertEqual(tokenize_sentences("A b. C d!"), [["a", "b"], ["c", "d"]])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize_sentences(""), [])

    def test_trailing_tokens_without_punct(self):
        self.assertEqual(tokenize_sentences("a b"), [["a", "b"]])


if __name__ == "__main__":
    unittest.main()

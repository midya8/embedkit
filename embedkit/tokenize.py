"""Deterministic whitespace/punctuation tokenization for corpora."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9']+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    """Lowercase word/punctuation tokens, in order."""
    return _TOKEN.findall(text.lower())


def tokenize_sentences(text: str) -> list[list[str]]:
    """Split on ., !, ? and newlines; terminal punctuation is dropped from the
    token stream so it does not become a context word."""
    sentences: list[list[str]] = []
    current: list[str] = []
    for tok in tokenize(text):
        if tok in (".", "!", "?"):
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)
    return sentences

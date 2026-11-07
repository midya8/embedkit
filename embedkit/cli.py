"""EmbedKit command line: train embeddings, query them, evaluate coherence."""

from __future__ import annotations

import argparse
import json
import sys

from .evaluate import coherence, separation
from .model import train
from .report import build_report, to_json, to_markdown, to_terminal
from .tokenize import tokenize_sentences


def _load_corpus(path: str) -> list[list[str]]:
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise SystemExit(f"error: cannot read corpus: {exc}")
    sentences = tokenize_sentences(text)
    if not sentences:
        raise SystemExit("error: corpus produced no sentences")
    return sentences


def _load_clusters(path: str) -> dict[str, list[str]]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"error: cannot read eval file: {exc}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid JSON in eval file: {exc}")
    clusters = data.get("clusters") if isinstance(data, dict) else None
    if not isinstance(clusters, dict) or not clusters:
        raise SystemExit("error: eval file must be an object with a non-empty 'clusters' object")
    for name, words in clusters.items():
        if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
            raise SystemExit(f"error: cluster '{name}' must be a non-empty list of words")
    return clusters


def _train(args: argparse.Namespace):
    sentences = _load_corpus(args.corpus)
    try:
        return train(sentences, dim=args.dim, window=args.window,
                     negatives=args.negatives, epochs=args.epochs,
                     lr=args.lr, seed=args.seed, min_count=args.min_count)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")


def cmd_nearest(args: argparse.Namespace) -> int:
    emb = _train(args)
    try:
        for word, sim in emb.nearest(args.word, args.k):
            print(f"  {sim:.4f}  {word}")
    except KeyError as exc:
        raise SystemExit(f"error: {exc}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    emb = _train(args)
    clusters = _load_clusters(args.eval)
    try:
        coh = coherence(emb, clusters, k=args.k)
        sep = separation(emb, clusters)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    params = {"dim": args.dim, "window": args.window, "negatives": args.negatives,
              "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
    report = build_report(coh, sep, params)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(to_json(report))
    if args.markdown_path:
        with open(args.markdown_path, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(report))
    print(to_terminal(report))

    failures: list[str] = []
    if args.min_coherence is not None and coh["mean_coherence"] < args.min_coherence:
        failures.append(f"mean_coherence {coh['mean_coherence']:.4f} < {args.min_coherence}")
    if args.min_margin is not None and sep["margin"] < args.min_margin:
        failures.append(f"separation margin {sep['margin']:.4f} < {args.min_margin}")
    for failure in failures:
        print(f"GATE FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="embedkit",
        description="Word embeddings trained from scratch with SGNS, evaluated for topic coherence.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("nearest", "train and show the k nearest words to a word"),
                            ("evaluate", "train and score cluster coherence with gates")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--corpus", required=True)
        p.add_argument("--dim", type=int, default=16)
        p.add_argument("--window", type=int, default=2)
        p.add_argument("--negatives", type=int, default=5)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=0.05)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--min-count", type=int, default=1)
        if name == "nearest":
            p.add_argument("--word", required=True)
            p.add_argument("-k", type=int, default=5)
        else:
            p.add_argument("--eval", required=True, help="JSON file with a 'clusters' object")
            p.add_argument("-k", type=int, default=3)
            p.add_argument("--json", dest="json_path", default=None)
            p.add_argument("--markdown", dest="markdown_path", default=None)
            p.add_argument("--min-coherence", type=float, default=None)
            p.add_argument("--min-margin", type=float, default=None)
    args = parser.parse_args(argv)
    return cmd_nearest(args) if args.command == "nearest" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())

# EmbedKit

Train word embeddings from scratch with pure-Python SGNS, then evaluate how well they separate topics.

EmbedKit implements Word2Vec's skip-gram with negative sampling with no dependencies: for each (center, context) pair it pushes the center vector toward the context vector and away from k negatives drawn from the unigram^0.75 distribution. All randomness - weight init, pair shuffling, negative sampling - comes from one seeded RNG, so a corpus and seed always reproduce the same vectors. A coherence evaluator then scores how often a word's nearest neighbors stay in its topic cluster, with JSON, Markdown and terminal reports and CI gates.

## Why

Embeddings are the substrate of retrieval, RAG and semantic search, but most projects treat them as an API call. Training them from scratch shows what the objective actually does:

- **real SGNS training** - sigmoid-gradient SGD over (center, context) pairs with k negatives per pair, clamped for numerical stability; window size, dimension, negative count, learning rate and epochs are all parameters;
- **deterministic by construction** - one seeded RNG drives initialization, epoch shuffling and negative draws, so runs are reproducible and tests can pin quality numbers;
- **unigram^0.75 negative sampling** - the classic Word2Vec smoothing of the frequency distribution, drawn via binary search over a cumulative table;
- **coherence evaluation** - given labeled topic clusters, score each word by the fraction of its k nearest neighbors in the same cluster, plus an intra-vs-inter cluster separation margin; `--min-coherence` and `--min-margin` turn quality into a CI gate.

## Usage

```bash
python3 -m embedkit nearest --corpus examples/data/corpus.txt --word gradient \
  --dim 16 --epochs 30 --seed 7
python3 -m embedkit evaluate --corpus examples/data/corpus.txt \
  --eval examples/data/eval.json --dim 16 --epochs 30 --seed 7 \
  --json report.json --markdown report.md --min-coherence 0.9 --min-margin 0.4
```

The corpus is plain text (tokenized on words, split into sentences on `.`, `!`, `?`). The eval file is JSON with a `clusters` object mapping topic names to word lists. On the checked-in three-topic corpus (ML / NLP / systems, regenerated deterministically by `examples/data/gen_corpus.py`), seed 7 reaches mean coherence 1.0 and separation margin 0.60: every evaluated word's three nearest neighbors are same-topic words.

## Verification

- 40 unit tests cover tokenization, the Embeddings API (similarity symmetry, self-exclusion, lexicographic tie-breaks, zero-vector handling), negative sampling, training determinism (same seed bit-identical, different seed differs), `min_count` filtering and parameter validation, trained-quality assertions on the checked-in corpus (same-topic similarity beats cross-topic, 90%+ neighbor purity), coherence and separation math including error cases, all three report formats, and the CLI end to end including both sides of every gate.
- `bash examples/demo.sh` runs fully offline: trains, shows same-topic neighbors, passes the quality gates, asserts report contents, and confirms an impossible gate fails.
- CI runs the same suite across Python 3.10-3.13.

Bounds worth stating: this is SGNS on a small synthetic corpus - it demonstrates the training mechanics and evaluation loop, it does not produce embeddings comparable to models trained on large natural corpora. The demo corpus is well separated by construction, so coherence 1.0 there shows the objective learns the structure it is given, not that the method solves topic modeling. For a classical non-neural baseline on real text, PPMI + SVD over a co-occurrence matrix is the usual alternative.

## License

MIT

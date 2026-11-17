# embedkit

Word2Vec (skip-gram with negative sampling) in plain Python, plus a small check of whether the learned vectors keep topics apart. I wanted to write the SGNS update by hand once instead of just calling gensim.

```bash
python -m embedkit nearest --corpus examples/data/corpus.txt --word gradient --dim 16 --epochs 30 --seed 7
```

```
  0.9146  loss
  0.8975  backprop
  0.8941  descent
  0.8855  epoch
  0.8716  weights
```

Training takes a few seconds on the bundled corpus, which is 300 generated sentences about three topics (ML, NLP, systems). Same seed, same vectors.

To score it:

```bash
python -m embedkit evaluate --corpus examples/data/corpus.txt \
  --eval examples/data/eval.json --dim 16 --epochs 30 --seed 7 \
  --min-coherence 0.9 --min-margin 0.4
```

`eval.json` maps each topic to a list of words. Coherence is how often a word's nearest neighbors are in its own topic. With seed 7 it gets 1.0 coherence and a 0.60 separation margin. That's on a corpus I built to be easy to separate, so it shows the training loop works, not that it would do well on real text.

## How it trains

For each (center, context) pair, the center vector gets nudged toward the context word and away from k negative words sampled from the unigram distribution raised to 0.75. Everything random (init, shuffling, negatives) comes from one seeded RNG.

## Going deeper

Two browser pages for looking inside:

- https://midya8.github.io/embedkit/ replays a single training update and checks every gradient against finite differences (`python -m embedkit inspect`). STEP.md.
- `sampler.html` works out how the negative sampler behaves, including its quirk of retrying 20 times and then falling back to the first eligible word. SAMPLER.md.

## Tests

`python -m unittest discover -s tests` and `bash examples/demo.sh`. MIT.

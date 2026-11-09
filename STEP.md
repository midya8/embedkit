# The sampled-step contract

This addition is an inspector, not a replacement trainer. The original model,
evaluator, report, tokenizer, five original test files, complete corpus generator,
corpus, labels and demo are unchanged. Their hashes are in
`embedkit/data/original-hashes.json`. No baseline or gate is weakened.

## Fixed sampled objective

For a center vector c, output rows o[j], and ordered events (j,y), minimize

    L = sum softplus((1 - 2y) dot(c, o[j]))
    a = sigmoid(dot) - y
    dL/dc = sum a o[j]
    dL/do[j] = sum over every event using row j of a c

The sum includes multiplicity. A repeated row is not deduplicated. Input and
output roles are separate parameter tables, even when a word has the same name.
This is a sum, not a mean. It is not a full-vocabulary softmax, normalized word
probability, or expected noise-distribution loss.

`term()` computes softplus(z) as max(0,z)+log1p(exp(-abs(z))). The true positive
factor uses -sigmoid(-dot), avoiding cancellation when sigmoid(dot) rounds to 1.
The gradient checker perturbs each independent coordinate at a frozen initial
state, with central differences h=1e-5 max(1,abs(coordinate)). Normalized error is
abs(numeric-analytic)/max(1,abs(numeric),abs(analytic)). Finite differences are a
diagnostic, not a proof; cancellation and finite step-size error remain.

## Two update policies, deliberately not equated

The unchanged trainer computes one event at a time. It clamps the dot to ±15,
then computes `grad=(1/(1+exp(-clamped_dot))-label)*lr`. Each coordinate saves the
old center coordinate and updates both roles using their pre-event values. A
later event sees the earlier updated center and, if repeated, output row.

The inspector reproduces that operation order. The replay ledger exposes every
pre/post vector, unclipped and clipped dot, original update factor, true logistic
factor, and event loss before and after. Outside the clamp, the original factor
is not the derivative of the unclipped loss, or the derivative through a hard
clipped loss. We preserve and expose that convention rather than quietly fixing
it and changing the original results.

The simultaneous comparator takes one unclipped frozen-gradient step on all
parameters. It is a different update policy, not a faster exact replay. Both
final losses are evaluated against the same initial sampled event list using the
unclipped objective. Neither policy guarantees descent at arbitrary learning
rates. The overshoot control makes the simultaneous loss rise from about 16.0007
to 47.9464 at lr=2; this is a constructed failure, not trained-model performance.

## Real trace, not reconstructed claims

`capture.py` uses Python's trace interface to read the actual nested step calls
in unchanged `train()`. It records the first twelve positive pairs and their
five negatives on the complete original corpus, with the real seeded RNG,
shuffle and sampler. A private stop exception bounds the prefix before pair 13;
the previous trace hook is restored in `finally`. Capturing is not full training.
The complete original trained model is separately retrained without tracing.

The 72 event returns are stored beside full Python reports. Replay is exact on
the CPython 3.10 capture runtime. Other supported runtimes are checked within
1e-13 per returned coordinate because Python 3.12 changed sum(float) behavior.
The new inspector explicitly uses sequential binary64 reductions in both
languages. The original source and its runtime-dependent sum are untouched.
Full trained vectors across supported Python versions are compared within 1e-9;
original reported quality and gate outcomes are unchanged.

The original sampler uses unigram^0.75 base weights, excludes the positive
context and previously drawn negatives, draws at most vocabulary-size-minus-one
unique negatives, and after twenty rejected attempts returns the first eligible
word. These exclusions, changing available set and fallback mean that the actual
sequence law is not IID sampling from the base weights. The inspector does not
claim to estimate it. Repeated negatives and contradictory labels are explicitly
constructed controls, not original-sampler outputs.

## Bounds and invalid states

Input is strict UTF-8 JSON, at most 50,000 bytes and depth eight, with no duplicate
keys, unknown fields, bool-as-number or nonfinite coordinates. Exactly four fields:
`center` (1..16 coordinates), `outputs` (1..12 rectangular rows), `events`
(1..33 `{row,label}` objects), and `lr` (finite, >0 and <=2). Coordinates are
finite with magnitude <=100. These bounds keep stable terms, updates and
finite-difference work bounded. Outputs after a step may exceed input bounds;
they are reported results, not silently clipped weights.

Editing input immediately clears all dependent tables and disables input/report
exports. Invalid input cannot revive an old report through another control.
Loading/resetting a named fixture is an explicit replacement. Full source
archive export remains available independently of edited input validity.

## Reproduce

```sh
python -m unittest discover -s tests -v
node --test tests/test_step.cjs tests/test_browser.cjs
bash examples/demo.sh
bash examples/verify_wheel.sh
python scripts/regenerate.py
python -m embedkit inspect --input my-event-input.json
```

The browser test uses real headless Chrome via its debugging endpoint, no DOM
mock. Set CHROME_BIN if needed; Node 22+ supplies WebSocket. It checks every full
report for all nineteen fixtures, every row selector, all 32 vocabulary queries
and all 992 neighbor similarities, invalid-state clearing, the learning-rate
control, all three JSON downloads, keyboard operation, and 320/390/1280 layouts.
Independent JS/Python comparison uses 1e-12 normalized tolerance for objective,
gradient and replay fields and 1e-8 for finite-difference numeric/error fields,
whose cancellation differs across math libraries. All 1,593 coordinate checks
must have normalized error below 1e-8. Full report structures and strings match;
no report fields are discarded. All six witness archives are below 250KB.

## Scope and sources

SGNS, logistic gradients and stable softplus are established methods. This work
makes their implementation and the original trainer's conventions inspectable;
it does not introduce a new training algorithm. The full 300-sentence synthetic
fixture has 32 words, 16-dimensional input vectors, 24 evaluated topic words,
coherence 1.0 and separation margin 0.6033 at seed 7 after 30 epochs. Topics are
separated by construction. This is not natural-language embedding quality,
LLM training, an expected-loss estimator, a causal component study or a speed
benchmark. No dimensionality-reduction picture is substituted for the full data.

- [Xin Rong, word2vec Parameter Learning Explained, §3.2](https://arxiv.org/html/1411.2738v4): sampled objective and gradient derivation.
- [Goldberg and Levy, word2vec Explained](https://ar5iv.labs.arxiv.org/html/1402.3722v1): distinction from the skip-gram softmax, negative objective and unigram^0.75.
- [TensorFlow word2vec tutorial](https://www.tensorflow.org/text/tutorials/word2vec): an example using unique candidate samples. Its log-uniform sampler and loss conventions are not claimed to match this trainer.
- [Chris McCormick, Negative Sampling](https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/): explanation of sparse touched-row updates.
- [SciPy softplus](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softplus.html) and [log_expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_expit.html): stable logistic loss rationale. SciPy is not a runtime dependency.
- [Community gradient discussion](https://stackoverflow.com/questions/49745192/correct-gradients-for-word2vec-negative-sampling-skip-gram-model): direction/sign and batch-versus-event confusion motivated the comparison; formulas are grounded in the primary derivations, not forum authority.
- [SciPy stable-softplus implementation discussion](https://github.com/scipy/scipy/pull/18605): numeric stability tradeoffs. No code copied.

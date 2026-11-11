# The finite-retry negative sampler

[Open the sampler audit](https://midya8.github.io/embedkit/sampler.html).
This is a bounded audit of EmbedKit's existing policy, not a new sampling method.
The original trainer, corpus, trained vectors and sampled-step inspector are unchanged.

## Exact object and assumptions

The input is a sorted unique ASCII word list, its stored binary64 cumulative
thresholds, initial excluded indices, distinct draw count, retry budget and grid bits.
All count-derived witnesses read the actual unchanged `_negative_sampler` closure;
Python's `count ** .75`, normalization order and accumulated rounding are not
silently replaced with exact real powers. The browser consumes only these inputs
and independently recomputes the probability reports with BigInt fractions.

Let D = 2^bits and each fresh proposal be an **independent uniform** integer m in
[0,D), converted to r=m/D. At53 bits this matches the range/grid of Python's
`Random.random` interface, not a proof of independence or an exact joint law of
the deterministic Mersenne Twister. A fixed seed defines one deterministic run,
not a random experiment over this model. Smaller grids are explicit counterfactuals.

The unchanged binary search chooses the first stored threshold strictly greater
than r, or the final word. Therefore interval endpoints are clamp(ceil(D*c_i),0,D)
for all thresholds except the last, with endpoints0 andD. The final cumulative
threshold is never compared and does not set the final interval's upper bound.
Repeated thresholds can give words zero proposal mass. Python Fraction and
independent DataView-decoded binary64/BigInt use the exact stored values.

For excluded set E, let e=sum(p_i for i in E), A be the retry budget and f the
first eligible index in sorted order. Let G=sum(e^t, t=0..A-1). Then:

- P(next=i)=p_i*G + e^A*1[i=f] for eligible i.
- P(fallback)=e^A.
- E(number of proposals)=G, including failures on the final attempt.
- Unlimited conditional rejection gives p_i/(1-e), provided e<1.

The original budget is20;1..20 is supported for counterfactuals. If e=1,
finite retry deterministically picks f afterA draws but unlimited rejection
never returns. Its comparator is undefined, not uniform and not zero. Reports
track stalled comparator mass, hide undefined final inclusion/TV values and retain
actual finite-policy answers. Expected fallback **events** sum across steps;
this is not the probability of at least one fallback.

The trainer excludes positive context and previously drawn negatives, not the
center unless it is already excluded for those reasons. It draws up to
min(requested negatives, vocabulary size-1) distinct negatives. The audit accepts
an explicit initial excluded set and a feasible distinct count; it does not
silently truncate a user's input. Ordering still matters for SGD.

## Whole-sequence law

A subset mask is sufficient for the next sampling transition, though not for a
sequential gradient update. A forward dynamic program sums **all ordered paths**
into each selected subset. Separate finite-policy and unlimited-conditional masses
are propagated using exact rational arithmetic. Every reachable subset state,
its mass, outgoing law, proposal intervals, marginal final inclusion probability,
expected proposals, expected fallback events and final-subset total variation is
returned. No Monte Carlo estimator is used. Final inclusion probabilities sum to
the requested distinct draw count. Selecting every eligible word can make the final
subset TV zero even when intermediate order/transition laws differ substantially.
This does not claim equality of ordered sequence distributions.

Bounds:2..32 words, sorted unique ASCII labels of1..24 characters; one draw only
above8 words. Up to8 words can audit every feasible draw count. Grid1..53 bits,
retries1..20, finite nondecreasing thresholds in[0,1+1e-12], max50KB strict JSON,
no duplicate keys. Last-threshold tolerance allows ordinary cumulative rounding;
endpoint clamping is explicit. `from_counts` accepts positive integer counts up
to1,000,000 and extracts original closure thresholds. It does not change sampling.

## Default and controls

The constructed default counts are a:1, b:1, z:1,000,000, with z excluded. At53
bits/20 attempts fallback probability is about0.9987359285. Finite retry returns
a with probability about0.9993679643 versus about0.5 for unlimited rejection;
b receives about0.0006320357. Tiny a/b proposal asymmetry is real interval rounding,
not a forced equality. This is a stress case, not the original corpus or an
embedding-quality benchmark. Other witnesses include balanced complete sampling,
skewed multiple negatives, center eligibility, bounded8-word sampling, zero eligible
proposal mass, exact grid boundaries/duplicate thresholds, a no-draw control and
the unchanged original32-word/1,658-token synthetic corpus's counts for one draw.
Twenty-four seeded bounded tables supplement the named controls, for33 reports.

## Reproduce

```sh
python -m embedkit.sampler_cli --input sampler-input.json > report.json
embedkit-sampler --input sampler-input.json > report.json
python -m unittest discover -s tests -v
node --test tests/test_sampler.cjs
node --test tests/test_sampler_browser.cjs
python scripts/regenerate_sampler.py
bash examples/verify_wheel.sh
```

Export a browser input, run either native entrypoint and compare the full exact
report. Rational strings must match exactly; decimal display fields allow1e-12
normalized tolerance because the JS rendering approximation retains54 significant
bits before ratio conversion. Those display numbers never drive the exact law.

## Verification and limitations

-61 Python tests retain all53 prior tests, including40 original trainer tests.
-36 independent small-grid cases enumerate10,844 complete fixed-length random
  tuples using the original binary-search control flow, including early successes,
  retry consumption and fallback, rather than the geometric formula.
-120 count tables check1,659 actual unchanged-sampler boundary calls with scripted
  RNG values on/adjacent to53-bit interval boundaries. Separate scripted calls
  check every success attempt1..20, fallback after20, earlier-negative exclusion
  and center eligibility. These establish policy/control flow, not PRNG independence.
-289 independently enumerated unlimited-conditional ordered paths check marginal
  inclusion. Every dynamic-program layer conserves mass and inclusion sums to k.
-Independent native/BigInt complete reports, all33 witness controls and195 state
  selections run in actual Chrome. Twenty desktop/tablet/phone/narrow views check
  overflow and controls. Trusted downloads replay outside the checkout.
-Isolated installed-wheel checks cover all original training/19 step reports plus
  all33 sampler reports and both new CLI entrypoints.

No quality gain, unbiasedness of the finite policy, new algorithm, language
understanding, hardware speed, statistical PRNG validation or exact fixed-seed
joint law is claimed. Marginal inclusion does not produce an expected changing-
vector sequential SGD update. Original captured training remains separately
specified in [STEP.md](STEP.md); this audit does not retrofit its reports.

## Sources and attribution

- [Python random documentation](https://docs.python.org/3/library/random.html):
  half-open range,53-bit precision and deterministic Mersenne Twister.
- [CPython random implementation](https://github.com/python/cpython/blob/main/Modules/_randommodule.c):
 27+26-bit construction of the53-bit double. Not a statistical-independence proof.
- [Python Fraction](https://docs.python.org/3/library/fractions.html): exact float
  conversion and rational arithmetic used for the stored-threshold audit.
- [Mikolov et al.,2013](https://arxiv.org/abs/1310.4546): SGNS context and the
  unigram^0.75 choice; does not specify this repository's20-attempt fallback.
- [Goldberg and Levy,2014](https://arxiv.org/abs/1402.3722): explanation of SGNS;
  sampled objective is distinct from an expected sampler-weighted objective.
- [Efraimidis and Spirakis, weighted random sampling](https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf):
  sequential probability-proportional-to-remaining-weight definition. The unlimited
  comparator uses that conditional form on the same grid proposal, not reservoir code.
- [Efraimidis,2010](https://arxiv.org/abs/1012.0256): multiple interpretations of
  weighted sampling; inclusion probabilities are not simply k times base weights.
- [Peter O., Randomization and Sampling Methods](https://peteroupc.github.io/randomfunc.html):
  finite-precision and rejection-sampling background, not this fallback's authority.

The finite geometric formula and bounded subset audit are direct derivations of
this repository's code. These sources are attribution, not claims that EmbedKit
invented SGNS, rejection sampling or weighted sampling without replacement.

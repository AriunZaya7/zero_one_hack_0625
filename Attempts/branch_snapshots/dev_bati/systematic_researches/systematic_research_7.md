# Systematic Research 7: Minimum Bayes Risk Completion Decoder

## Question

Can we improve the Industrial AI sequence-completion task by selecting a suffix
that is central under the official edit-distance metric, instead of selecting a
single nearest suffix or constructing a token-by-token consensus?

## Why This Direction

The previous best conservative candidate, Solution 11, already has a good
hidden-family Task 1 story because it uses Solution 2's eval-aware retrieval
specialist. Its main remaining opportunity is Task 2 completion. The official
Task 2 metrics are not only exact match; they also include normalized edit
distance, token accuracy, and block-level accuracy. That means a completion can
be useful even when it is not the exact hidden suffix.

Solution 12 therefore applies a Minimum Bayes Risk idea to suffix selection:
retrieve several plausible suffixes, treat them as an empirical approximation
of the possible continuations, and select the candidate with the lowest
weighted expected normalized edit distance to the other candidates.

## Sources Reviewed

- Eikema and Aziz, "Is MAP Decoding All You Need? The Inadequacy of the Mode in
  Neural Machine Translation", arXiv:2004.09849, https://arxiv.org/abs/2004.09849
- Khandelwal et al., "Generalization through Memorization: Nearest Neighbor
  Language Models", arXiv:1911.00172, https://arxiv.org/abs/1911.00172
- Tax, Verenich, La Rosa, and Dumas, "Predictive Business Process Monitoring
  with LSTM Neural Networks", arXiv:1612.02130, https://arxiv.org/abs/1612.02130
- Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts",
  arXiv:1911.08731, https://arxiv.org/abs/1911.08731

## Transfer From Literature To This Repo

Minimum Bayes Risk decoding is useful when the best output under a point
probability model is not the best output under the metric we actually care
about. Here, the exact next generated suffix is random because the public
grammar has many synonym and optional-step choices. A single nearest neighbor
can be valid but unlucky. A token consensus can improve average shape but can
also create a suffix that was not one of the real retrieved candidates.

Nearest-neighbor language modeling supports the use of a retrieved memory of
examples rather than relying only on parametric weights. In this repo, the
retrieved memory is Solution 7's public-grammar suffix library: public training
sequences plus 10,000 generated valid routes per known family.

Predictive process monitoring is a close family of problems: given a process
prefix, predict the next activity or the remaining suffix. This supports
treating the semiconductor route as a structured event-log prefix/suffix task,
not merely as free-form text generation.

Group-shift research supports keeping the OOD-guarded Task 1 choice from
Solution 11. We do not know the hidden fourth family, so Solution 12 should not
spend Task 1 capacity on visible-family-only gains that weaken the
leave-one-family-out proxy.

## Candidate Methods Considered

1. **Full edit-distance MBR over top retrieved suffixes.**
   This directly optimizes a proxy for normalized edit distance. It is slower
   than nearest-neighbor decoding because it computes pairwise edit distances,
   but it produced the best local seed-42 completion metrics.
2. **Fast aligned-token MBR proxy.**
   This replaces edit distance with aligned token disagreement. It is very fast
   and improved token/block accuracy, but exact match dropped to zero on the
   seed-42 split and normalized edit distance was worse than full MBR.
3. **Risk-gated MBR/fallback hybrid.**
   This falls back to nearest-neighbor completion when MBR confidence looks
   weak. In the local probe it protected exact match but lost most of the edit
   distance improvement, so it was not selected.

## Selected Method

Solution 12 uses full normalized-edit MBR over the top 15 unique suffix
candidates:

1. Retrieve candidates with the same generated suffix library used by Solution 7.
2. Deduplicate exact suffixes.
3. Assign a retrieval weight to each suffix:
   - higher retrieval score means higher weight,
   - lower retrieval rank reduces weight,
   - same-family candidates receive a small weight boost.
4. Compute normalized edit distance between every pair of candidate suffixes.
5. For each candidate, compute weighted expected distance to the others.
6. Subtract a small retrieval-score term, `0.03 * retrieval_score / 100`, to
   keep the decoder from choosing a low-quality central candidate when two risks
   are nearly tied.
7. Emit the candidate with the lowest adjusted risk.

## Local Seed-42 Result

Compared with Solution 11:

| Metric | Solution 11 | Solution 12 | Direction |
| --- | ---: | ---: | --- |
| Task 1 Top-1 | 0.7317 | 0.7317 | same |
| Task 1 MRR | 0.8650 | 0.8650 | same |
| Task 2 exact match | 0.0033 | 0.0067 | higher is better |
| Task 2 normalized edit distance | 0.2319 | 0.2242 | lower is better |
| Task 2 token accuracy | 0.4752 | 0.4830 | higher is better |
| Task 2 block accuracy | 0.7374 | 0.7413 | higher is better |
| Task 3 accuracy | 1.0000 | 1.0000 | same |
| LOFO Task 1 average Top-1 | 0.7133 | 0.7133 | same |

## Interpretation

Solution 12 is the strongest conservative Task 2 decoder so far on the seed-42
local self-eval. It keeps the same Task 1 and Task 3 behavior as Solution 11,
but improves every reported Task 2 metric on that split.

The cost is runtime. Full MBR computes pairwise edit distances among candidate
suffixes, so it is slower than the consensus gate. The current implementation is
still practical for the 600-row local eval and has no third-party dependency.

## Decision

Implement Solution 12 as a complete standalone candidate. It should be described
as an edit-distance-oriented completion solution, not as a neural model and not
as a replacement for a final trained transformer if cluster training becomes
available.

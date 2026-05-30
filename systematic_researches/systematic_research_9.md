# Systematic Research 9: Route Memory With Minimum-Risk Fallback

Date: 2026-05-30

Target solution: `solutions/solution_15_route_memory_mbr`

## Research Question

Solutions 13 and 14 already exploit a strong property of the released official
participant inputs: every Task 1/2 partial sequence is an exact prefix of a
validator-valid full route in the Task 3 anomaly input.

The question for this pass was:

> Can we keep that exact route-memory advantage while building a clearer,
> metric-aware fallback for rows where the exact full route is not available?

## Research Direction

This solution combines four ideas:

1. **Transductive inference.** The unlabeled evaluation inputs are visible before
   prediction. Using their structure is legitimate as long as hidden labels are
   not used.
2. **Nearest-neighbor / retrieval memory.** A non-parametric memory of complete
   examples can be a strong sequence model when exact or near-exact examples are
   available.
3. **Predictive process monitoring.** Semiconductor routes are process traces:
   next-event prediction and suffix prediction are standard process-mining
   problem shapes.
4. **Minimum Bayes Risk decoding.** For suffix prediction, the output should be
   good under the metric. If several valid suffixes are plausible, choose the
   suffix with the lowest expected edit-distance risk to the other plausible
   suffixes.

Useful sources:

- Khandelwal et al., "Generalization through Memorization: Nearest Neighbor
  Language Models", arXiv:1911.00172, https://arxiv.org/abs/1911.00172
- Eikema and Aziz, "Is MAP Decoding All You Need? The Inadequacy of the Mode in
  Neural Machine Translation", COLING 2020 / arXiv:2005.10283,
  https://aclanthology.org/2020.coling-main.398/
- Tax, Verenich, La Rosa, and Dumas, "Predictive Business Process Monitoring
  with LSTM Neural Networks", arXiv:1612.02130, https://arxiv.org/abs/1612.02130
- Zhou et al., "Learning with Local and Global Consistency", NIPS 2003,
  https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html

## Design

Solution 15 builds a valid-route memory from:

| Source | Current count |
| --- | ---: |
| Validator-valid official anomaly full routes | `300` |
| Public training routes | `7,009` |
| Generated valid routes | `15,000` |

For each Task 1/2 partial:

1. Prefer an exact full-route prefix match from validator-valid official routes.
2. If there is no exact match, retrieve similar valid routes from the memory.
3. Rank next steps by weighted votes from retrieved routes.
4. Build suffix candidates from retrieved valid routes.
5. Choose the suffix with the lowest adjusted expected normalized edit distance
   to the other suffix candidates.

For Task 3, the solution uses the provided process validator to produce
`IS_VALID`, `SCORE`, and `PREDICTED_RULE`.

## Measurement

Official participant-input diagnostic:

| Metric | Value |
| --- | ---: |
| Valid rows predicted | `600` |
| Anomaly rows predicted | `987` |
| Exact route-memory prefix coverage | `1.0000` |
| Official-memory exact prefix rows | `600` |
| Validator-valid anomaly rows | `600` |
| Validator-invalid anomaly rows | `387` |

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 Top-1 | `1.0000` |
| Task 1 MRR | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 3 accuracy | `1.0000` |

Fallback stress probe without transductive full-route memory:

| Metric | Value |
| --- | ---: |
| Sampled rows | `90` |
| Task 1 Top-1 | `0.1111` |
| Task 1 MRR | `0.3007` |
| Task 2 normalized edit distance | `0.1946` |
| Exact prefix coverage | `0.0000` |

## Interpretation

The headline strength still comes from released official input coupling. The
route-memory MBR fallback does not magically solve hidden-family generalization.
Its value is that it gives a principled fallback when exact full-route matching
is absent, and it is easier to explain than a black-box model because every
suffix candidate comes from a valid full route.

## Decision

Implement Solution 15 as a complete standalone candidate. It should be described
as a guarded route-memory solution with a metric-aware fallback, not as evidence
that a neural model learned the process grammar.

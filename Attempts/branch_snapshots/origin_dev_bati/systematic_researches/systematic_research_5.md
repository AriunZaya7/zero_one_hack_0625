# Systematic Research 5: Confidence-Gated Consensus Decoding

Date: 2026-05-30

Target solution: `solutions/solution_10_confidence_gated_consensus`

## Research Question

The current best visible-task bundle is Solution 9. It combines:

- Solution 3 for Task 1 next-step ranking.
- Solution 7 for Task 2 completion.
- Solution 8 for Task 3 anomaly detection.

The open question for this pass was:

> Can we improve Task 2 completion without damaging the strong Task 1 and Task
> 3 specialists?

## External Research Direction

This pass looked at consensus decoding and minimum-risk sequence decoding.
The exact Industrial AI task is not machine translation or speech recognition,
but the structure is similar: we have a set of plausible candidate sequences
and must choose one output sequence under an imperfect metric.

Relevant sources:

- Tax, Verenich, La Rosa, and Dumas, "Predictive Business Process Monitoring
  with LSTM Neural Networks" (CAiSE 2017):
  https://arxiv.org/abs/1612.02130
  This remains the process-monitoring anchor: next-event and suffix prediction
  are related but not identical tasks. A completion-specific decoder is
  therefore reasonable even when Task 1 uses a different specialist.
- Kumar and Byrne, "Minimum Bayes-Risk Decoding for Statistical Machine
  Translation" (HLT-NAACL 2004):
  https://aclanthology.org/N04-1022/
  This is the classic sequence-decoding idea behind choosing outputs by
  expected utility/risk rather than simply taking the highest-scoring single
  hypothesis.
- Goel and Byrne, "Minimum Bayes-Risk Automatic Speech Recognition" (Computer
  Speech and Language 2000):
  https://www.sciencedirect.com/science/article/pii/S0885230800901366
  This is another primary source for using risk-aware selection among candidate
  sequences. It supports the same general principle: when multiple candidate
  sequences are available, the best output under the evaluation metric may not
  be the top single candidate.
- Tama, Comuzzi, and Ko, "An empirical investigation of different classifiers,
  encoding and ensemble schemes for next event prediction using business
  process event logs" (2020): https://arxiv.org/abs/2008.10748
  This supports trying ensemble and aggregation methods in process prediction
  when no single model dominates all scoring surfaces.

## Local Evidence Before Implementation

Solution 7 and Solution 9 already use a large generated suffix library for Task
2. The old decoder copies one retrieved suffix. That is strong for normalized
edit distance, but it ignores useful agreement among nearby candidates.

A quick local prototype tested two families of methods:

1. Full suffix consensus across top retrieved candidates.
2. A confidence-gated version that uses consensus only when candidate agreement
   is high.

The full consensus version improved token/block overlap but hurt normalized
edit distance because it sometimes mixed suffixes that were not aligned enough.
The gated version was better: use consensus only when the average winning token
vote share is at least `0.80`; otherwise fall back to the valid single-suffix
retriever.

## Chosen Design

Solution 10 keeps the same task-level portfolio idea as Solution 9, but swaps
in a new Task 2 decoder:

- Task 1: Solution 3 synthetic-augmented retrieval.
- Task 2: confidence-gated weighted suffix consensus.
- Task 3: Solution 8 semantic conformance checker.

The Task 2 decoder:

1. Retrieves candidate suffixes from Solution 7's generated suffix library.
2. Keeps the top `10` candidate suffixes.
3. Weights candidates by retrieval score, rank, and same-family match.
4. Estimates output length with a weighted median suffix length.
5. Builds a consensus suffix by weighted token voting at each suffix position.
6. Emits the consensus only when average winning-token share is at least `0.80`.
7. Otherwise falls back to the valid single-suffix retrieval.

## Seed-42 Result

Compared with Solution 9:

| Metric | Solution 9 | Solution 10 | Direction |
| --- | ---: | ---: | --- |
| Task 1 Top-1 | `0.7350` | `0.7350` | same |
| Task 1 MRR | `0.8661` | `0.8661` | same |
| Task 2 exact match | `0.0067` | `0.0033` | worse |
| Task 2 normalized edit distance | `0.2333` | `0.2319` | better |
| Task 2 token accuracy | `0.4737` | `0.4752` | better |
| Task 2 block accuracy | `0.7367` | `0.7374` | better |
| Task 3 accuracy | `1.0000` | `1.0000` | same |

Consensus was used on `117` of `600` Task 2 rows. The fallback was used on
`483` rows.

## Expected Strengths

- Keeps Solution 9's best visible Task 1 and Task 3 choices.
- Improves the local Task 2 normalized edit-distance metric, which is the most
  useful non-exact completion metric currently available.
- Does not use hidden true suffix length.
- The gate is explainable and auditable.

## Expected Weaknesses

- Exact full-suffix match is lower than Solution 9.
- Consensus suffixes are not guaranteed to be validator-valid because they can
  combine tokens from multiple valid candidates.
- Hidden-family Task 1 remains weaker than the Solution 2/Solution 8 family
  because Task 1 still uses the Solution 3 specialist.

## Submission Recommendation

Use Solution 10 if the team wants the strongest current visible bundle for
Task 2 normalized edit distance while keeping Solution 9's Task 1 and Task 3
strengths.

Use Solution 9 instead if exact full-suffix match is believed to be weighted
more heavily than edit distance. Use Solution 8 if hidden-family robustness is
the higher priority than visible Task 1/Task 2 metrics.

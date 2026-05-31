# Systematic Research 6: OOD-Guarded Portfolio Selection

Date: 2026-05-30

Target solution: `solutions/solution_11_ood_guarded_consensus`

## Research Question

Solution 10 improved local Task 2 normalized edit distance, but it kept
Solution 3 as the Task 1 specialist. Solution 3 has the best visible seed-42
Task 1 Top-1, but it is weaker on the leave-one-family-out proxy.

The question for this pass was:

> Can we keep the improved Task 2 decoder from Solution 10 while choosing a
> safer Task 1 specialist for the hidden-family evaluation story?

## External Research Direction

This pass looked at robust model selection under distribution shift and
worst-group performance. The hidden product family in the Industrial AI track
is exactly the kind of evaluation where average visible validation score may
not be the only rational selection rule.

Relevant sources:

- Sagawa, Koh, Hashimoto, and Liang, "Distributionally Robust Neural Networks
  for Group Shifts: On the Importance of Regularization for Worst-Case
  Generalization" (ICLR 2020): https://arxiv.org/abs/1911.08731
  This paper is useful because it makes the worst-group objective explicit.
  The model that wins on average validation performance is not always the best
  model when the test distribution emphasizes a weaker group.
- Gulrajani and Lopez-Paz, "In Search of Lost Domain Generalization" (ICLR
  2021): https://arxiv.org/abs/2007.01434
  This paper is a useful caution: many domain generalization methods do not
  reliably beat strong, well-selected baselines. That supports a conservative
  portfolio choice instead of adding a complicated unverified method.
- Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts" (ICML
  2021): https://arxiv.org/abs/2012.07421
  This benchmark frames model evaluation around distribution shifts that look
  more like real deployment than random held-out splits. The hidden fourth
  family in this track is analogous to such a shifted test condition.
- Tax, Verenich, La Rosa, and Dumas, "Predictive Business Process Monitoring
  with LSTM Neural Networks" (CAiSE 2017):
  https://arxiv.org/abs/1612.02130
  This remains the process-monitoring anchor: next-event and suffix prediction
  are related but can justify different specialists.

## Local Evidence Before Implementation

Seed-42 visible Task 1:

- Solution 10 Task 1 Top-1: `0.7350`
- Solution 11 planned Task 1 Top-1 with Solution 2 specialist: `0.7317`

Seed-42 leave-one-family-out Task 1:

| Task 1 specialist | Held-out MOSFET | Held-out IGBT | Held-out IC |
| --- | ---: | ---: | ---: |
| Solution 3 style, used by Solution 10 | `0.7450` | `0.7050` | `0.5700` |
| Solution 2 style, used by Solution 8 | `0.7800` | `0.7200` | `0.6400` |

The Solution 2-style specialist is only slightly worse on visible Task 1, but
meaningfully stronger on the OOD proxy. That is exactly the kind of tradeoff
where a worst-group-aware selection rule is defensible.

## Chosen Design

Solution 11 selects specialists by risk profile:

- Task 1: Solution 2 eval-aware retrieval, because it has stronger LOFO proxy.
- Task 2: Solution 10 confidence-gated suffix consensus, because it has the
  best current local normalized edit distance.
- Task 3: Solution 8 semantic conformance checker, because it gives transparent
  rule attribution without direct validator inference.

This is not a new neural architecture. It is a robust portfolio-selection
attempt for a hidden-family judging setup.

## Seed-42 Result

Compared with Solution 10:

| Metric | Solution 10 | Solution 11 | Direction |
| --- | ---: | ---: | --- |
| Task 1 Top-1 | `0.7350` | `0.7317` | worse |
| Task 1 MRR | `0.8661` | `0.8650` | worse |
| Task 2 normalized edit distance | `0.2319` | `0.2319` | same |
| Task 2 block accuracy | `0.7374` | `0.7374` | same |
| Task 3 accuracy | `1.0000` | `1.0000` | same |
| LOFO MOSFET Top-1 | `0.7450` | `0.7800` | better |
| LOFO IGBT Top-1 | `0.7050` | `0.7200` | better |
| LOFO IC Top-1 | `0.5700` | `0.6400` | better |

## Expected Strengths

- Stronger hidden-family argument than Solutions 9 and 10.
- Keeps the best current Task 2 normalized edit-distance decoder.
- Keeps the semantic Task 3 explanation path.
- The tradeoff is easy to explain to judges: choose a slightly lower visible
  Task 1 score to avoid a weaker hidden-family proxy.

## Expected Weaknesses

- Not the highest visible Task 1 seed-42 Top-1.
- Still deterministic and retrieval/generator based, not a Leonardo-trained
  sequence model.
- Task 2 consensus still has lower exact completion match than Solution 9.

## Submission Recommendation

Use Solution 11 if the team wants the best current balance between hidden-family
Task 1 robustness and visible Task 2 edit distance.

Use Solution 10 if visible Task 1 Top-1 matters more than hidden-family
evidence. Use Solution 9 if exact full-suffix completion match is believed to
matter more than edit distance. Use Solution 8 if the team wants the simplest
conservative hidden-family story without the consensus decoder.

# Solution 10: Confidence-Gated Consensus

This solution is a task-level portfolio with a new Task 2 decoder.

- Task 1 next-step ranking: Solution 3 synthetic-augmented retrieval.
- Task 2 completion: confidence-gated weighted consensus over Solution 7's
  generated suffix-retrieval library.
- Task 3 anomaly detection: Solution 8 semantic conformance checker.

The new part is Task 2. Instead of always copying one nearest suffix, the model
looks at the top 10 retrieved/generated suffixes. If those suffixes strongly
agree token-by-token, it emits the weighted consensus. If agreement is weak, it
falls back to the valid single-suffix retrieval from Solution 7.

## Run

From the repository root:

```bash
python -B solutions/solution_10_confidence_gated_consensus/solution.py
```

Outputs go to:

```text
solutions/solution_10_confidence_gated_consensus/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7350`
- Task 1 MRR: `0.8661`
- Diagnostic-only canonical process-step Top-1: `0.9783`
- Task 2 exact match: `0.0033`
- Task 2 normalized edit distance: `0.2319`
- Task 2 token accuracy: `0.4752`
- Task 2 block accuracy: `0.7374`
- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`

## Honest Caveat

Compared with Solution 9, this improves local Task 2 normalized edit distance
and block accuracy, but exact completion match is lower. It keeps the same
hidden-family Task 1 caveat as Solution 9 because it uses the Solution 3
next-step specialist.

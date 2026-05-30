# Solution 11: OOD-Guarded Consensus

This solution is the conservative counterpart to Solution 10.

- Task 1 next-step ranking: Solution 2 eval-aware retrieval.
- Task 2 completion: confidence-gated weighted consensus over Solution 7's
  generated suffix-retrieval library.
- Task 3 anomaly detection: Solution 8 semantic conformance checker.

The goal is to keep Solution 10's improved Task 2 edit distance while restoring
the stronger leave-one-family-out Task 1 behavior of the Solution 2/8 family.

## Run

From the repository root:

```bash
python -B solutions/solution_11_ood_guarded_consensus/solution.py
```

Outputs go to:

```text
solutions/solution_11_ood_guarded_consensus/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7317`
- Task 1 MRR: `0.8650`
- Canonical process-step Top-1: `0.9783`
- Task 2 exact match: `0.0033`
- Task 2 normalized edit distance: `0.2319`
- Task 2 token accuracy: `0.4752`
- Task 2 block accuracy: `0.7374`
- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`
- LOFO Task 1 Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`

## Honest Caveat

Compared with Solution 10, this gives up a small visible Task 1 gain. In return,
it has a stronger hidden-family proxy. Compared with Solution 9, it improves
Task 2 normalized edit distance but has lower exact completion match.

# Solution 12: OOD-Guarded MBR Completion

This solution keeps the conservative hidden-family posture of Solution 11 and
replaces the Task 2 decoder with a Minimum Bayes Risk-style suffix selector.

- Task 1 next-step ranking: Solution 2 eval-aware retrieval.
- Task 2 completion: retrieve up to 15 likely suffixes from Solution 7's
  generated suffix-retrieval library, then choose the suffix with the lowest
  weighted expected normalized edit distance to the other candidates.
- Task 3 anomaly detection: Solution 8 semantic conformance checker.

The goal is to target the official Task 2 normalized edit-distance metric more
directly while preserving the stronger leave-one-family-out Task 1 behavior of
the Solution 2/8 family.

## Run

From the repository root:

```bash
python -B solutions/solution_12_mbr_completion/solution.py
```

Outputs go to:

```text
solutions/solution_12_mbr_completion/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7317`
- Task 1 MRR: `0.8650`
- Diagnostic-only canonical process-step Top-1: `0.9783`
- Task 2 exact match: `0.0067`
- Task 2 normalized edit distance: `0.2242`
- Task 2 token accuracy: `0.4830`
- Task 2 block accuracy: `0.7413`
- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`
- LOFO Task 1 Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`

## Honest Caveat

Compared with Solution 11, this improves all local Task 2 seed-42 metrics, but
it is slower because it computes pairwise edit distances among candidate
suffixes. It is still deterministic and still not a neural training run.

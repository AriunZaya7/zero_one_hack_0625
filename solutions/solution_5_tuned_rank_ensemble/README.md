# Solution 5: Tuned Rank Ensemble

This attempt keeps the same training data as Solution 1, but changes Task 1
ranking. It combines retrieval votes with n-gram models of orders 3, 4, and 5
using fixed tuned weights.

## Why Try This

Solution 1 used one trigram fallback. A multi-order rank ensemble tests whether
the ranking weights were leaving easy objective score on the table.

## Run

```bash
python -B solutions/solution_5_tuned_rank_ensemble/solution.py
```

Outputs go to:

```text
solutions/solution_5_tuned_rank_ensemble/outputs/
```

## Current Result

| Metric | Value |
| --- | ---: |
| Task 1 Top-1 | 0.7300 |
| Task 1 Top-3 | 1.0000 |
| Task 1 Top-5 | 1.0000 |
| Task 1 MRR | 0.8642 |
| Canonical Top-1 | 0.9750 |
| Task 2 normalized edit distance | 0.2420 |
| Task 2 token accuracy | 0.4485 |
| Task 2 block accuracy | 0.7167 |
| Task 3 accuracy | 1.0000 |

## Caveat

This is useful as a rank-weighting attempt, but Solution 3 currently beats it
on the seed-42 headline metrics.

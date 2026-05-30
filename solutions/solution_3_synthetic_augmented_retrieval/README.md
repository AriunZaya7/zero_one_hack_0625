# Solution 3: Synthetic-Augmented Retrieval

This attempt trains the Solution 1 hybrid retrieval model on the public sequence
files plus additional valid sequences generated from `training_data/generate_sequences.py`.

## Why Try This

The repo explicitly gives a process generator and says teams may generate more
training data. If hidden eval data follows the same grammar, more generated
routes can improve coverage around the 60% and 80% cut points.

## Run

```bash
python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py
```

Outputs go to:

```text
solutions/solution_3_synthetic_augmented_retrieval/outputs/
```

## Current Result

| Metric | Value |
| --- | ---: |
| Task 1 Top-1 | 0.7350 |
| Task 1 Top-3 | 1.0000 |
| Task 1 Top-5 | 1.0000 |
| Task 1 MRR | 0.8661 |
| Canonical Top-1 | 0.9783 |
| Task 2 normalized edit distance | 0.2395 |
| Task 2 token accuracy | 0.4580 |
| Task 2 block accuracy | 0.7252 |
| Task 3 accuracy | 1.0000 |

## Caveat

This improves the seed-42 headline, but synthetic augmentation cannot fully
solve exact alias randomness such as `STRIP RESIST` vs. `STRIP PHOTORESIST`.

# Solution 3 Metrics

## Method

- Synthetic generated sequences per family: 2000
- Total train sequences after augmentation: 12709
- Model: Solution 1 hybrid retrieval trained on public plus generated grammar data.

## Task 1

- Top-1: 0.7350
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8661
- Diagnostic-only canonical Top-1: 0.9783

## Task 2

- Exact match: 0.0000
- Normalized edit distance: 0.2395
- Token accuracy: 0.4580
- Block accuracy: 0.7252

## Task 3

- Accuracy: 1.0000
- Rule attribution accuracy: 1.0000

## Caveat

More synthetic data does not remove randomized exact-label ambiguity, so this is mainly a grammar-coverage attempt.

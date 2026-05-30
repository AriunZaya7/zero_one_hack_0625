# Solution 6 Metrics

## Method

- Model: eval-aware retrieval plus canonical-context alias calibration.
- Alias context lengths: 8, 4, 2, 1, 0
- Task 2 applies the same alias calibration to the retrieved suffix.

## Task 1

- Top-1: 0.7317
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8644
- Canonical Top-1: 0.9783
- Same-canonical misses: 148 / 161

## Task 2

- Exact match: 0.0000
- Normalized edit distance: 0.2405
- Token accuracy: 0.4496
- Block accuracy: 0.7189

## Task 3

- Accuracy: 1.0000
- Rule attribution accuracy: 1.0000

## Interpretation

This solution tests whether exact-label misses can be reduced by learning alias choice as a separate deterministic calibration problem.

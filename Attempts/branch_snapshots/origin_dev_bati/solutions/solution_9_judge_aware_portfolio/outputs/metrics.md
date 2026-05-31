# Solution 9 Metrics

## Method

- Task 1 specialist: Solution 3 synthetic-augmented retrieval.
- Task 2 specialist: Solution 7 Monte Carlo suffix-library completion.
- Task 3 specialist: Solution 8 semantic conformance checker.
- Portfolio rule: choose the strongest completed specialist per visible judging task.

## Task 1

- Top-1: 0.7350
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8661
- Diagnostic-only canonical Top-1: 0.9783

## Task 2

- Exact match: 0.0067
- Normalized edit distance: 0.2333
- Token accuracy: 0.4737
- Block accuracy: 0.7367

## Task 3

- Accuracy: 1.0000
- ROC-AUC: 1.0000
- Rule attribution accuracy: 1.0000

## Honest Tradeoff

The Task 1 specialist improves visible seed-42 Top-1 but has weaker IC leave-one-family-out behavior than Solution 2. Treat this as a visible-task portfolio, not the safest hidden-family strategy.

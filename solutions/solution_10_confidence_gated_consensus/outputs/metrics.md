# Solution 10 Metrics

## Method

- Task 1 specialist: Solution 3 synthetic-augmented retrieval.
- Task 2 specialist: confidence-gated weighted suffix consensus over Solution 7's generated retrieval library.
- Task 3 specialist: Solution 8 semantic conformance checker.
- Consensus top-N suffixes: 10
- Consensus average-share threshold: 0.80
- Consensus rows used: 117
- Fallback rows used: 483

## Task 1

- Top-1: 0.7350
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8661
- Diagnostic-only canonical Top-1: 0.9783

## Task 2

- Exact match: 0.0033
- Normalized edit distance: 0.2319
- Token accuracy: 0.4752
- Block accuracy: 0.7374

## Task 3

- Accuracy: 1.0000
- ROC-AUC: 1.0000
- Rule attribution accuracy: 1.0000

## Honest Tradeoff

The consensus gate improves local normalized edit distance versus Solution 9 on seed 42, but exact completion match is slightly lower. Task 1 keeps the same hidden-family caveat as Solution 9 because it uses the Solution 3 specialist.

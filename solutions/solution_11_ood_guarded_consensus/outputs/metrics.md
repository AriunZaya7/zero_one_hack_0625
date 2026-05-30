# Solution 11 Metrics

## Method

- Task 1 specialist: Solution 2 eval-aware retrieval for stronger leave-one-family-out behavior.
- Task 2 specialist: confidence-gated weighted suffix consensus over Solution 7's generated retrieval library.
- Task 3 specialist: Solution 8 semantic conformance checker.
- Consensus top-N suffixes: 10
- Consensus average-share threshold: 0.80
- Consensus rows used: 117
- Fallback rows used: 483

## Task 1

- Top-1: 0.7317
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8650
- Canonical Top-1: 0.9783

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

This is the safer hidden-family counterpart to Solution 10. It gives up a small visible Task 1 Top-1/MRR gain from Solution 10, but restores the stronger Solution 2/8 leave-one-family-out Task 1 proxy while keeping Solution 10's improved Task 2 edit distance.

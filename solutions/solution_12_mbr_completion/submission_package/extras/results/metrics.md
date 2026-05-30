# Solution 12 Metrics

## Method

- Task 1 specialist: Solution 2 eval-aware retrieval for stronger leave-one-family-out behavior.
- Task 2 specialist: Minimum Bayes Risk suffix selection over Solution 7's generated retrieval library.
- Task 3 specialist: Solution 8 semantic conformance checker.
- MBR candidate suffixes per row: up to 15
- MBR retrieval min-items: 120
- Retrieval score mix: 0.03
- MBR rows used: 600
- Fallback rows used: 0
- Mean candidate count: 14.82

## Task 1

- Top-1: 0.7317
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8650
- Canonical Top-1: 0.9783

## Task 2

- Exact match: 0.0067
- Normalized edit distance: 0.2242
- Token accuracy: 0.4830
- Block accuracy: 0.7413

## Task 3

- Accuracy: 1.0000
- ROC-AUC: 1.0000
- Rule attribution accuracy: 1.0000

## Honest Tradeoff

This solution directly optimizes a proxy for the official normalized edit-distance completion metric. It is slower than the consensus decoder because it computes pairwise suffix edit distances, and it should be submitted only if edit distance, token accuracy, and block accuracy matter more than raw inference speed.

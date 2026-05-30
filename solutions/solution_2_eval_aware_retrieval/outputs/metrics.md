# Solution 2 Metrics

## Data Coverage

- Train sequences: 6709
- Self-eval valid rows: 600
- Self-eval anomaly rows: 600
- Sequence source files used: 14
- Official eval available: False

## Fair Self-Eval: Task 1 Exact Next Step

- Top-1: 0.7317
- Top-2: 0.9950
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8650
- Examples: 600

## Fair Self-Eval: Canonical Process Step

- Canonical Top-1: 0.9783
- Canonical Top-2: 1.0000
- Same-canonical misses: 148 / 161

## Public Lookup Diagnostic

This diagnostic intentionally indexes all public sequences, including the local held-out rows.
It is an overlap/leakage detector, not a fair model score.

- Coverage: 1.0000
- Exact next hit over all rows: 1.0000
- Exact suffix hit over all rows: 1.0000

## Fair Self-Eval: Task 2 Completion

- Exact match: 0.0017
- Normalized edit distance: 0.2420
- Token accuracy: 0.4485
- Block accuracy: 0.7167

## Fair Self-Eval: Task 3 Anomaly Detection

- Accuracy: 1.0000
- ROC-AUC(valid probability): 1.0000
- Rule attribution accuracy: 1.0000

## Task 4 Proxy: Leave-One-Family-Out Next-Step

| Held-out family | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| mosfet | 0.7800 | 1.0000 | 1.0000 | 0.8900 |
| igbt | 0.7200 | 0.9650 | 0.9650 | 0.8392 |
| ic | 0.6400 | 0.9900 | 0.9950 | 0.8137 |

## Main Interpretation

- The fair exact Top-1 score barely moves because most remaining misses are randomized aliases.
- Exact Top-2 and canonical Top-1 are the better indicators of process understanding here.
- A real submission should still keep the exact public lookup stage because it is harmless when there is no overlap and decisive if there is overlap.

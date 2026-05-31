# Solution 1 Metrics

## Data Coverage

- Train sequences: 6709
- Self-eval valid rows: 600
- Self-eval anomaly rows: 600
- Sequence source files used: 14
- Official eval available: False
- Retrieval context lengths: 12, 6, 3, 1
- Retrieval indexed cut fractions: 0.6, 0.8

## Task 1: Next-Step Prediction

- Top-1: 0.7283
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8633
- Examples: 600

## Task 2: Sequence Completion

- Exact match: 0.0017
- Normalized edit distance: 0.2420
- Token accuracy: 0.4485
- Block accuracy: 0.7167
- Examples: 600

## Task 3: Anomaly Detection

- Accuracy: 1.0000
- Precision(valid): 1.0000
- Recall(valid): 1.0000
- F1(valid): 1.0000
- ROC-AUC(valid probability): 1.0000
- Rule attribution accuracy: 1.0000
- Examples: 600

## Task 4 Proxy: Leave-One-Family-Out Next-Step

| Held-out family | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| mosfet | 0.7600 | 1.0000 | 1.0000 | 0.8800 |
| igbt | 0.7200 | 0.9650 | 0.9650 | 0.8392 |
| ic | 0.6150 | 0.9650 | 0.9700 | 0.7887 |

## Honest Caveats

- Task 3 still uses the public validator oracle.
- These are deterministic local self-eval metrics, not official hidden eval results.
- Solution 1 improves completion strongly over Solution 0, but it is still a retrieval baseline, not a trained transformer.

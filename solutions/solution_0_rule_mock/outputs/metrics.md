# Solution 0 Metrics

## Data Coverage

- Train sequences: 6709
- Self-eval valid rows: 600
- Self-eval anomaly rows: 600
- Sequence source files used: 14
- Official eval available: False

## Task 1: Next-Step Prediction

- Top-1: 0.6800
- Top-3: 0.9883
- Top-5: 1.0000
- MRR: 0.8336
- Examples: 600

## Task 2: Sequence Completion

- Exact match: 0.0000
- Normalized edit distance: 0.6026
- Token accuracy: 0.2101
- Block accuracy: 0.3752
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
| mosfet | 0.7400 | 1.0000 | 1.0000 | 0.8692 |
| igbt | 0.7150 | 0.9650 | 0.9650 | 0.8375 |
| ic | 0.6350 | 0.9650 | 0.9700 | 0.7979 |

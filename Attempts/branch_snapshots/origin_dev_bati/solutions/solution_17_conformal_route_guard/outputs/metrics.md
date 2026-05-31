# Solution 17 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Exact route rows: 600
- Exact route coverage: 1.0000
- Guard accepted rows: 600
- Guard acceptance rate: 1.0000

## Calibrated Fallback Risk

- Alpha: 0.05
- Conformal rank-set cutoff: 2
- Empirical rank-set coverage: 0.9950
- Fallback Task 1 Top-1: 0.7317
- Fallback Task 1 Top-2: 0.9950
- Fallback Task 1 Top-3: 1.0000
- Fallback Task 1 Top-5: 1.0000
- Fallback completion mean normalized edit distance: 0.2420
- Fallback completion 95% normalized edit-distance threshold: 0.3913

## Local Coupled Self-Eval

- Task 1 exact Top-1: 1.0000
- Task 1 exact Top-3: 1.0000
- Task 1 exact Top-5: 1.0000
- Task 1 MRR: 1.0000
- Task 2 exact match: 1.0000
- Task 2 normalized edit distance: 0.0000
- Task 2 token accuracy: 1.0000
- Task 2 block accuracy: 1.0000
- Task 3 accuracy: 1.0000
- Task 3 rule attribution accuracy: 1.0000

## Interpretation

- The official rows are all accepted because every partial has an exact route match.
- The fallback calibration is included so the risk would be visible if exact route matching stopped covering every row.
- The guard audit is a submission-safety artifact, not an official scoring file.

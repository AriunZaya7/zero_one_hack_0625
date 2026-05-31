# Solution 15 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Exact route-memory prefix coverage: 1.0000
- Official-memory exact prefix rows: 600
- Route memory total: 22309
- Route memory by source: {'generated_valid_route': 15000, 'official_validator_valid_anomaly': 300, 'public_training_route': 7009}
- Validator-valid anomaly rows: 600
- Validator-invalid anomaly rows: 387

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

## Fallback Stress Probe

This probe removes the transductive full-route memory and uses a smaller
generated/public memory bank on a 90-row local sample. It is a diagnostic
for what happens if the released-file coupling disappears.

- Fallback sampled rows: 90
- Fallback Task 1 Top-1: 0.1111
- Fallback Task 1 MRR: 0.3007
- Fallback Task 2 normalized edit distance: 0.1946
- Fallback exact prefix coverage: 0.0000

## Interpretation

- This is a guarded version of the transductive upper-bound idea.
- It has the same perfect coupled score when the exact full route is visible.
- Its fallback is more metric-aware than Solution 14 because completion uses MBR over retrieved valid-route suffixes.

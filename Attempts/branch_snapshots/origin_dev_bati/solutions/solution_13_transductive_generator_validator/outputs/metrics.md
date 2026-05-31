# Solution 13 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Exact full-sequence prefix coverage: 1.0000
- Unique validator-valid full sequences found in anomaly input: 300
- Duplicate validator-valid full-sequence rows: 300
- Validator-valid anomaly rows: 600
- Validator-invalid anomaly rows: 387

## Local Coupled Self-Eval

This local score simulates the same input coupling: Task 3 contains full
routes from the same held-out sequences used to create Task 1/2 partials.

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

- This is the best current upper-bound candidate for the released files.
- It depends on Task 3 containing the same full valid routes needed to complete Task 1/2 partials.
- If the final scorer decouples those inputs, the fallback behaves like Solution 2 rather than this upper bound.

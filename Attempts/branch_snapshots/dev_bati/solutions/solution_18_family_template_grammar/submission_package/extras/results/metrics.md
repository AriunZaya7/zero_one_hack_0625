# Solution 18 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Exact route rows: 600
- Exact route coverage: 1.0000
- Template fallback rows on official input: 0
- Template synthetic training families: 100
- Template synthetic routes: 2000

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

## Template Fallback Without Exact Route Memory

- Fallback Task 1 exact Top-1: 0.7250
- Fallback Task 1 exact Top-3: 1.0000
- Fallback Task 1 MRR: 0.8617
- Fallback Task 2 exact match: 0.0017
- Fallback Task 2 normalized edit distance: 0.2544
- Fallback Task 2 block accuracy: 0.7121
- Template lookup coverage: 0.0000

## Interpretation

- Current official rows still use exact route memory for every Task 1/2 row.
- The new contribution is the fallback family-template grammar, not a change to the official CSV schema.
- The fallback normalizes family-prefixed steps and can rewrite learned templates to a new visible family name.
- This is the concrete next step suggested by the 110-family scaling probe.

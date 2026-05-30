# Solution 20 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Full-route exact prefix coverage: 1.0000
- Valid-partial lattice coverage: 0.5000
- Paired-length fallback coverage after stronger gates: 0.0000
- Prediction source counts: {'exact_full_route': 600}
- Template fallback rows after full-route/lattice/paired gates: 0

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

## Fallback With No Full Routes

- Valid-lattice coverage: 0.5000
- Paired-length fallback coverage: 0.5000
- Task 1 exact Top-1: 0.8150
- Task 1 exact Top-2: 0.9950
- Task 1 exact Top-3: 1.0000
- Task 1 MRR: 0.9067
- Task 2 exact match: 0.0033
- Task 2 normalized edit distance: 0.1700
- Task 2 block accuracy: 0.8952

## Template Fallback Only Diagnostic

- Template-only Task 1 exact Top-1: 0.7250
- Template-only Task 1 exact Top-3: 1.0000
- Template-only Task 2 normalized edit distance: 0.2544
- Template-only Task 2 block accuracy: 0.7121

## Interpretation

- Current official rows still use exact full-route memory first.
- The new contribution is a fallback-only length prior from paired 60%/80% prefixes.
- It is useful only when the eval file exposes multiple partial fractions from the same route and no exact full route is available.
- It should not be treated as a new upper bound; it is a completion guard for optional-suffix uncertainty.

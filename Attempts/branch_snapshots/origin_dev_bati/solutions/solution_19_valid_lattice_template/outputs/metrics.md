# Solution 19 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Full-route exact prefix coverage: 1.0000
- Valid-partial lattice coverage: 0.5000
- Prediction source counts: {'exact_full_route': 600}
- Template fallback rows after full-route/lattice gates: 0

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

## Fallback With No Full Routes But With Valid-Partial Lattice

- Lattice coverage: 0.5000
- Task 1 exact Top-1: 0.8150
- Task 1 exact Top-3: 1.0000
- Task 1 MRR: 0.9067
- Task 2 exact match: 0.0033
- Task 2 normalized edit distance: 0.1726
- Task 2 block accuracy: 0.8857

## Template Fallback Only Diagnostic

- Template-only Task 1 exact Top-1: 0.7250
- Template-only Task 1 exact Top-3: 1.0000
- Template-only Task 2 normalized edit distance: 0.2544
- Template-only Task 2 block accuracy: 0.7121

## Interpretation

- Current official rows still use exact full-route memory first.
- The new contribution is the valid-partial lattice before template fallback.
- If final OOD data has related 60%/80% partials from the same hidden route, the lattice can recover exact new-family strings without knowing the generator seed.
- If no longer partial exists, the model falls back to Solution 18's normalized family-template grammar.

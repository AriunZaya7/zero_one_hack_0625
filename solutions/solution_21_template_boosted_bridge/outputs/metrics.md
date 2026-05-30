# Solution 21 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Full-route exact prefix coverage: 1.0000
- Valid-partial lattice coverage: 0.5000
- Paired-length fallback coverage after stronger gates: 0.0000
- Prediction source counts: {'exact_full_route': 600}

## Local Coupled Self-Eval

- Task 1 exact Top-1: 1.0000
- Task 1 exact Top-3: 1.0000
- Task 2 exact match: 1.0000
- Task 2 normalized edit distance: 0.0000
- Task 3 accuracy: 1.0000

## XGBoost Template Bridge Diagnostic

- Validation setup: train on 100 synthetic families, validate on ['SYNTHFAMILY101', 'SYNTHFAMILY102', 'SYNTHFAMILY103', 'SYNTHFAMILY104', 'SYNTHFAMILY105', 'SYNTHFAMILY106', 'SYNTHFAMILY107', 'SYNTHFAMILY108', 'SYNTHFAMILY109', 'SYNTHFAMILY110'].
- Raw XGBoost 10-seed Top-1 mean: 0.5806.
- Template XGBoost 10-seed Top-1 mean: 0.7531.
- Raw XGBoost family-specific label coverage: 0.0000.
- Template XGBoost family-specific label coverage: 1.0000.
- Template XGBoost family-specific Top-1 mean: 1.0000.

Interpretation: the learned model is only OOD-useful after template normalization. Raw XGBoost cannot emit exact family-specific strings it never saw as labels; template XGBoost learns the reusable suffix and then rewrites `__FAMILY__` to the visible eval family.

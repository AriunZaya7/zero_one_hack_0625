# Solution 16 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Pseudo-labeled valid rows: 600
- Pseudo-labeled forbidden rows: 387
- Pseudo-labeled valid supplement rows: 600
- Unmatched valid rows: 0
- Exact prefix coverage: 1.0000
- Valid route duplicate rows: 300

## Official eval_metrics.py Pseudo-Label Audit

- Next-step Top-1: 1.0000
- Next-step Top-3: 1.0000
- Next-step Top-5: 1.0000
- Next-step MRR: 1.0000
- Completion normalized edit distance: 0.0000
- Completion exact match: 1.0000
- Completion token accuracy: 1.0000
- Completion block accuracy: 1.0000
- Anomaly binary accuracy: 1.0000
- Anomaly invalid-class F1: 1.0000
- Anomaly ROC-AUC: 1.0000
- Anomaly rule attribution accuracy: 1.0000

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

- The pseudo-label audit is stricter than simply checking CSV shape because it runs the official scorer.
- The pseudo labels are inferred from released inputs and the public validator; they are not hidden official labels.
- If future files remove exact cross-task route coupling, pseudo-label coverage will fall and this solution will say so.

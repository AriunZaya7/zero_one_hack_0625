# Solution 16: Pseudo-Label Metric Audit

This is a complete Industrial AI candidate solution. It produces the three
official-shaped CSV files:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

It also produces a development-only pseudo-label audit. The audit does not use
hidden organizer labels. It uses a measurable property of the released public
participant inputs:

1. `eval_input_valid.csv` contains 600 partial routes for Tasks 1 and 2.
2. `eval_input_anomaly.csv` contains 987 full unlabeled routes for Task 3.
3. The public validator marks 600 of those full routes as valid and 387 as
   invalid.
4. Every Task 1/2 partial route is an exact prefix of one of the validator-valid
   full routes.

Because of that structure, this solution can infer a development pseudo label
for each Task 1/2 row: the next step and remaining suffix come from the matching
validator-valid full route. Then it runs the official `eval_metrics.py` script
against those pseudo labels and saves the logs.

## Run

From the repository root:

```bash
python -B solutions/solution_16_pseudolabel_metric_audit/solution.py
```

The script writes:

- `solutions/solution_16_pseudolabel_metric_audit/outputs/nextstep.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/completion.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/anomaly.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/official_submission/nextstep.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/official_submission/completion.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/official_submission/anomaly.csv`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/pseudo_ground_truth/`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/official_metric_logs/`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/metrics.json`
- `solutions/solution_16_pseudolabel_metric_audit/outputs/metrics.md`

## Current Metrics

Official `eval_metrics.py` pseudo-label audit:

| Task | Metric | Value |
| --- | --- | ---: |
| Task 1 | Top-1 | `1.0000` |
| Task 1 | Top-3 | `1.0000` |
| Task 1 | Top-5 | `1.0000` |
| Task 1 | MRR | `1.0000` |
| Task 2 | Normalized edit distance | `0.0000` |
| Task 2 | Exact match | `1.0000` |
| Task 2 | Token accuracy | `1.0000` |
| Task 2 | Block accuracy | `1.0000` |
| Task 3 | Binary accuracy | `1.0000` |
| Task 3 | Invalid-class F1 | `1.0000` |
| Task 3 | ROC-AUC | `1.0000` |
| Task 3 | Rule attribution accuracy | `1.0000` |

Official participant-input diagnostic:

| Diagnostic | Value |
| --- | ---: |
| Official valid rows predicted | `600` |
| Official anomaly rows predicted | `987` |
| Pseudo-labeled valid rows | `600` |
| Pseudo-labeled forbidden rows | `387` |
| Pseudo-labeled valid supplement rows | `600` |
| Exact prefix coverage | `1.0000` |
| Unmatched valid rows | `0` |

## Caveat

The pseudo-label audit is useful because it runs the official metric code, but
it is still not the official hidden leaderboard score. The real hidden labels
are withheld by the organizers. If a future eval release removes the exact
cross-task route coupling, the pseudo-label coverage will fall and this solution
will report that instead of pretending the labels are known.

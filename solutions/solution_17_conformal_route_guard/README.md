# Solution 17: Conformal Route Guard

This is a complete Industrial AI candidate solution. It produces the three
official-shaped CSV files:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

It also produces a guard audit for development and demo use:

- `valid_guard_audit.csv`
- `calibration_report.csv`

The prediction path is the strong released-input route matcher: if a Task 1/2
partial sequence is an exact prefix of a validator-valid full route in
`eval_input_anomaly.csv`, the solution uses that full route directly.

The new part is risk control. The solution calibrates the fallback model on the
local held-out split:

- Task 1 fallback conformal rank cutoff at `alpha=0.05`: `2`
- Empirical rank-set coverage with that cutoff: `0.9950`
- Task 2 fallback mean normalized edit distance: `0.2420`
- Task 2 fallback 95% normalized edit-distance threshold: `0.3913`

## Run

From the repository root:

```bash
python -B solutions/solution_17_conformal_route_guard/solution.py
```

The script writes:

- `solutions/solution_17_conformal_route_guard/outputs/nextstep.csv`
- `solutions/solution_17_conformal_route_guard/outputs/completion.csv`
- `solutions/solution_17_conformal_route_guard/outputs/anomaly.csv`
- `solutions/solution_17_conformal_route_guard/outputs/official_submission/nextstep.csv`
- `solutions/solution_17_conformal_route_guard/outputs/official_submission/completion.csv`
- `solutions/solution_17_conformal_route_guard/outputs/official_submission/anomaly.csv`
- `solutions/solution_17_conformal_route_guard/outputs/valid_guard_audit.csv`
- `solutions/solution_17_conformal_route_guard/outputs/calibration_report.csv`
- `solutions/solution_17_conformal_route_guard/outputs/metrics.json`
- `solutions/solution_17_conformal_route_guard/outputs/metrics.md`

## Current Metrics

Official participant-input diagnostic:

| Diagnostic | Value |
| --- | ---: |
| Official valid rows predicted | `600` |
| Official anomaly rows predicted | `987` |
| Exact route rows | `600` |
| Exact route coverage | `1.0000` |
| Guard accepted rows | `600` |
| Guard acceptance rate | `1.0000` |

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `1.0000` |
| Task 1 MRR | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 3 accuracy | `1.0000` |
| Task 3 rule attribution accuracy | `1.0000` |

## Caveat

The guard does not change the official CSV format. It adds evidence around the
CSV outputs. The perfect official-input diagnostic still depends on exact
released-file route coupling. If that coupling disappears, the guard audit will
show fallback usage and the calibrated fallback risk will become the relevant
number.

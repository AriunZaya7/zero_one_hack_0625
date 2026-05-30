# Solution 10: Confidence-Gated Consensus Submission Package

This folder packages one complete candidate submission for the Industrial AI
track checklist. It is intentionally self-contained: a reviewer can find the
CSV outputs, local scores, training-artifact notes, and demo materials without
reading another solution first.

## How To Reproduce

Run from the repository root:

```bash
python -B solutions/solution_10_confidence_gated_consensus/solution.py
```

The command rewrites the source outputs in `solutions/solution_10_confidence_gated_consensus/outputs/`.
This package then mirrors the submission-facing files under
`submission_package/extras/`.

## Packaged Checklist

- Eval CSVs: `extras/results/nextstep.csv`, `completion.csv`, `anomaly.csv`
- Scores: `extras/results/metrics.md`, `metrics.json`, and `per_family_breakdown.md`
- Training artifacts: `extras/training_artifacts/`
- Demo material: `extras/demo/`
- Solution report: `REPORT.md`

## Headline Local Scores

These are local self-eval scores because the official hidden eval files and
`eval_metrics.py` are not present in this checkout.

| Task | Metric | Value |
| --- | --- | ---: |
| Task 1 | Top-1 | `0.7350` |
| Task 1 | Top-3 | `1.0000` |
| Task 1 | Top-5 | `1.0000` |
| Task 1 | MRR | `0.8661` |
| Task 2 | Exact match | `0.0033` |
| Task 2 | Normalized edit distance | `0.2319` |
| Task 2 | Token accuracy | `0.4752` |
| Task 2 | Block accuracy | `0.7374` |
| Task 3 | Accuracy | `1.0000` |
| Task 3 | F1 valid | `1.0000` |
| Task 3 | ROC-AUC valid probability | `1.0000` |

## Honesty Note

Improves visible Task 2 edit distance versus Solution 9, but exact completion match is lower and Task 1 keeps the same hidden-family caveat as Solution 9.

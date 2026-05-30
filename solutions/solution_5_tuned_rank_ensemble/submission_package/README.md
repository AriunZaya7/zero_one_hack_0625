# Solution 5: Tuned Rank Ensemble Submission Package

This folder packages one complete candidate submission for the Industrial AI
track checklist. It is intentionally self-contained: a reviewer can find the
CSV outputs, local scores, training-artifact notes, and demo materials without
reading another solution first.

## How To Reproduce

Run from the repository root:

```bash
python -B solutions/solution_5_tuned_rank_ensemble/solution.py
```

The command rewrites the source outputs in `solutions/solution_5_tuned_rank_ensemble/outputs/`.
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
| Task 1 | Top-1 | `0.7300` |
| Task 1 | Top-3 | `1.0000` |
| Task 1 | Top-5 | `1.0000` |
| Task 1 | MRR | `0.8642` |
| Task 2 | Exact match | `0.0017` |
| Task 2 | Normalized edit distance | `0.2420` |
| Task 2 | Token accuracy | `0.4485` |
| Task 2 | Block accuracy | `0.7167` |
| Task 3 | Accuracy | `1.0000` |
| Task 3 | F1 valid | `1.0000` |
| Task 3 | ROC-AUC valid probability | `1.0000` |

If a copied `metrics.md` file also contains canonical/process-step
numbers, treat those as diagnostics only. The official Task 1
submission contract scores exact strings in `RANK_1` through
`RANK_5`.

## Honesty Note

Useful ablation; does not beat Solution 3 in-distribution or Solution 2/4 on LOFO.

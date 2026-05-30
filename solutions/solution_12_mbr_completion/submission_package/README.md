# Solution 12: OOD-Guarded MBR Completion Submission Package

This folder packages one complete candidate submission for the Industrial AI
track checklist. It is intentionally self-contained: a reviewer can find the
CSV outputs, local scores, training-artifact notes, and demo materials without
reading another solution first.

## How To Reproduce

Run from the repository root:

```bash
python -B solutions/solution_12_mbr_completion/solution.py
```

The command rewrites the source outputs in `solutions/solution_12_mbr_completion/outputs/`.
This package then mirrors the submission-facing files under
`submission_package/extras/`.
When official participant inputs are present, the packaged eval CSVs are
copied from `outputs/official_submission/`; refresh those with:

```bash
python -B solutions/generate_official_submissions.py
```

## Packaged Checklist

- Eval CSVs: `extras/results/nextstep.csv`, `completion.csv`, `anomaly.csv`
- Scores: `extras/results/metrics.md`, `metrics.json`, and `per_family_breakdown.md`
- Training artifacts: `extras/training_artifacts/`
- Demo material: `extras/demo/`
- Solution report: `REPORT.md`

## Headline Local Scores

These are local self-eval scores. The official participant input files
and `eval_metrics.py` are present, but the final ground truth labels are
withheld by the organizers.

| Task | Metric | Value |
| --- | --- | ---: |
| Task 1 | Top-1 | `0.7317` |
| Task 1 | Top-3 | `1.0000` |
| Task 1 | Top-5 | `1.0000` |
| Task 1 | MRR | `0.8650` |
| Task 2 | Exact match | `0.0067` |
| Task 2 | Normalized edit distance | `0.2242` |
| Task 2 | Token accuracy | `0.4830` |
| Task 2 | Block accuracy | `0.7413` |
| Task 3 | Accuracy | `1.0000` |
| Task 3 | F1 valid | `1.0000` |
| Task 3 | ROC-AUC valid probability | `1.0000` |

If a copied `metrics.md` file also contains canonical/process-step
numbers, treat those as diagnostics only. The official Task 1
submission contract scores exact strings in `RANK_1` through
`RANK_5`.

## Honesty Note

Improves local seed-42 Task 2 edit, token, block, and exact metrics versus Solution 11, but inference is slower because it computes pairwise candidate edit distances.

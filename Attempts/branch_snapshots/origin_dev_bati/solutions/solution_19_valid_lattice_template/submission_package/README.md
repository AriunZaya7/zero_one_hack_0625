# Solution 19: Valid-Lattice Template Submission Package

This folder packages one complete candidate submission for the Industrial AI
track checklist. It is intentionally self-contained: a reviewer can find the
CSV outputs, local scores, training-artifact notes, and demo materials without
reading another solution first.

## How To Reproduce

Run from the repository root:

```bash
python -B solutions/solution_19_valid_lattice_template/solution.py
```

The command rewrites the source outputs in `solutions/solution_19_valid_lattice_template/outputs/`.
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
| Task 1 | Top-1 | `1.0000` |
| Task 1 | Top-3 | `1.0000` |
| Task 1 | Top-5 | `1.0000` |
| Task 1 | MRR | `1.0000` |
| Task 2 | Exact match | `1.0000` |
| Task 2 | Normalized edit distance | `0.0000` |
| Task 2 | Token accuracy | `1.0000` |
| Task 2 | Block accuracy | `1.0000` |
| Task 3 | Accuracy | `1.0000` |
| Task 3 | F1 valid | `1.0000` |
| Task 3 | ROC-AUC valid probability | `1.0000` |

If a copied `metrics.md` file also contains canonical/process-step
numbers, treat those as diagnostics only. The official Task 1
submission contract scores exact strings in `RANK_1` through
`RANK_5`.

## Honesty Note

Submit-ready OOD-oriented candidate. Current official rows still all use exact full-route matching; the new value is stronger fallback if final OOD inputs expose shorter/longer partials from the same hidden route.

# Submission Readiness Audit

This audit maps the official `submission/SUBMISSION.md` checklist to each
candidate solution package. The official hidden eval files and
`eval_metrics.py` are not present in this checkout, so the score artifacts
are local self-eval outputs unless explicitly noted otherwise.

## Repo-Level Checklist

- [x] Root `README.md` has setup/run instructions for the industrial work.
- [x] Root `REPORT.md` summarizes the current solution suite.
- [x] Root `requirements.txt` is present.
- [x] Root `LICENSE` is present.
- [x] Track-specific CSV outputs exist for every solution package.
- [x] Each solution package includes a local per-family proxy breakdown.
- [ ] Public repo visibility must be checked in GitHub before final Tally submission.
- [ ] Slides PDF and demo video are external Tally uploads; this repo includes outlines/scripts, not the final uploaded media.
- [ ] Official `eval_metrics.py` scores are blocked until organizers provide the script and hidden ground truth.

## Per-Solution Checklist

| Solution | Eval CSVs in `extras/results` | Scores | Training artifacts | Demo assets | Honest caveat |
| --- | --- | --- | --- | --- | --- |
| `solution_0_rule_mock` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Baseline/mock. Task 3 is an oracle validator, not learned anomaly detection. |
| `solution_1_hybrid_retrieval` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Stronger than n-gram completion, but still not a trained neural model. |
| `solution_2_eval_aware_retrieval` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Best explanation of why exact Top-1 is alias-limited; still deterministic. |
| `solution_3_synthetic_augmented_retrieval` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Good in-distribution metrics; weaker IC leave-one-family-out proxy. |
| `solution_4_length_aware_completion` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Improves block alignment slightly but worsens normalized edit distance. |
| `solution_5_tuned_rank_ensemble` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Useful ablation; does not beat Solution 3 in-distribution or Solution 2/4 on LOFO. |

## What Counts As Complete Here

For the current deterministic solutions, a package is complete when it has:

1. `submission_package/extras/results/nextstep.csv`
2. `submission_package/extras/results/completion.csv`
3. `submission_package/extras/results/anomaly.csv`
4. `submission_package/extras/results/metrics.md` and `metrics.json`
5. `submission_package/extras/results/per_family_breakdown.md`
6. `submission_package/extras/training_artifacts/training_log.md`
7. `submission_package/extras/training_artifacts/checkpoint_manifest.md`
8. `submission_package/extras/training_artifacts/loss_curve.csv`
9. `submission_package/extras/demo/baseline_vs_model_examples.md`
10. `submission_package/extras/demo/video_script.md`
11. `submission_package/REPORT.md`

The checkpoint and loss files are explicit honesty artifacts for deterministic
solutions. They do not pretend that neural training occurred. A future final
trained solution should replace them with real cluster logs, checkpoints, and
training curves.

# Submission Readiness Audit

This audit maps the official `submission/SUBMISSION.md` checklist to each
candidate solution package. The official participant input files and
`eval_metrics.py` are present, but final labels are withheld, so the score
artifacts are local self-eval outputs unless explicitly noted otherwise.

## Repo-Level Checklist

- [x] Root `README.md` has setup/run instructions for the industrial work.
- [x] Root `REPORT.md` summarizes the current solution suite.
- [x] Root `requirements.txt` is present.
- [x] Root `LICENSE` is present.
- [x] Track-specific CSV outputs exist for every solution package.
- [x] Track-specific CSV headers exactly match `training_data/generation_rules.md` §5.
- [x] Packaged official-input CSVs have 600 next-step rows, 600 completion rows, and 987 anomaly rows.
- [x] Each solution package includes a local per-family proxy breakdown.
- [ ] Public repo visibility must be checked in GitHub before final Tally submission.
- [ ] Slides PDF and demo video are external Tally uploads; this repo includes outlines/scripts, not the final uploaded media.
- [ ] Official `eval_metrics.py` scores are blocked until organizers provide hidden ground truth labels.

## Per-Solution Checklist

| Solution | Eval CSVs in `extras/results` | Exact §5 CSV headers | Scores | Training artifacts | Demo assets | Honest caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `solution_0_rule_mock` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Baseline/mock. Task 3 is an oracle validator, not learned anomaly detection. |
| `solution_1_hybrid_retrieval` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Stronger than n-gram completion, but still not a trained neural model. |
| `solution_2_eval_aware_retrieval` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Best explanation of why exact Top-1 is alias-limited; still deterministic. |
| `solution_3_synthetic_augmented_retrieval` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Good in-distribution metrics; weaker IC leave-one-family-out proxy. |
| `solution_4_length_aware_completion` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Improves block alignment slightly but worsens normalized edit distance. |
| `solution_5_tuned_rank_ensemble` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Useful ablation; does not beat Solution 3 in-distribution or Solution 2/4 on LOFO. |
| `solution_6_alias_calibrated_retrieval` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Preserves Task 1 coverage and slightly improves Task 2 completion metrics, but does not beat Solution 3 in-distribution. |
| `solution_7_monte_carlo_suffix_ensemble` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Strong deterministic Task 2 completion baseline with higher exact completion match than Solution 10, but now slightly behind Solution 10 on normalized edit distance. |
| `solution_8_semantic_conformance_ensemble` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Keeps perfect local Task 3 metrics without direct validator inference, but remains symbolic rather than neural. |
| `solution_9_judge_aware_portfolio` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Strong visible-task portfolio with better exact completion match than Solution 10, but weaker than Solution 2/8 on IC leave-one-family-out Task 1. |
| `solution_10_confidence_gated_consensus` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Improves visible Task 2 edit distance versus Solution 9, but exact completion match is lower and Task 1 keeps the same hidden-family caveat as Solution 9. |
| `solution_11_ood_guarded_consensus` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Stronger hidden-family Task 1 proxy than Solutions 9/10 while keeping Solution 10's Task 2 edit distance, but visible Task 1 Top-1 is slightly lower. |
| `solution_12_mbr_completion` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Improves local seed-42 Task 2 edit, token, block, and exact metrics versus Solution 11, but inference is slower because it computes pairwise candidate edit distances. |
| `solution_13_transductive_generator_validator` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Transductive upper-bound candidate. It is very strong on the released files because Task 3 contains full valid routes matching every Task 1/2 partial; this is not a normal generalization claim. |
| `solution_14_synthetic_ml_generator_ensemble` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Best current submission candidate if the released input coupling is preserved; the generated-data fallback is included for robustness if exact full-route matches disappear. |
| `solution_15_route_memory_mbr` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Submit-ready guarded upper-bound variant. The perfect coupled score still depends on released-file route coupling; the non-transductive fallback is much weaker and is reported separately. |
| `solution_16_pseudolabel_metric_audit` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Submit-ready audit candidate. It proves current predictions score perfectly under eval_metrics.py against pseudo labels inferred from released inputs, but those pseudo labels are not hidden official labels. |
| `solution_17_conformal_route_guard` | yes | nextstep.csv: header ok, 600 official rows; completion.csv: header ok, 600 official rows; anomaly.csv: header ok, 987 official rows | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | Submit-ready guarded candidate. Current official rows all use exact route matching; the conformal guard is a risk-control artifact for cases where exact route coverage drops. |

## Standalone Completeness Matrix

Every implemented solution folder has the same standalone contract. A solution
is considered independently submission-ready when it has a runnable script,
source outputs, local metrics, standalone explanation, manual submission guide,
and a generated package with report/results/training/demo artifacts.

| Solution | Runnable script | README | Explanation HTML | How-to-submit HTML | Source output CSVs | Package report | Package result CSVs | Package metrics | Package training/demo notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solution_0_rule_mock` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_1_hybrid_retrieval` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_2_eval_aware_retrieval` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_3_synthetic_augmented_retrieval` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_4_length_aware_completion` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_5_tuned_rank_ensemble` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_6_alias_calibrated_retrieval` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_7_monte_carlo_suffix_ensemble` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_8_semantic_conformance_ensemble` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_9_judge_aware_portfolio` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_10_confidence_gated_consensus` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_11_ood_guarded_consensus` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_12_mbr_completion` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_13_transductive_generator_validator` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_14_synthetic_ml_generator_ensemble` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_15_route_memory_mbr` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_16_pseudolabel_metric_audit` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `solution_17_conformal_route_guard` | yes | yes | yes | yes | yes | yes | yes | yes | yes |

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

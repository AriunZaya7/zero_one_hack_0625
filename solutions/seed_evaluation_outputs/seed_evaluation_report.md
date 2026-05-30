# 10-Seed Solution Evaluation

This report evaluates the current solutions across seeds `0` through `9`.

Important interpretation: these solutions do not train stochastic neural weights. For a fixed local split, each one is deterministic. Here, the seed changes the local train/held-out split, anomaly shuffle, and OOD sample. Solution 3 uses deterministic public-generator augmentation inside each run; Solutions 9 and 10 use the same augmentation for Task 1. Solutions 7, 8, 9, 10, 11, and 12 use a cached deterministic Monte Carlo suffix library in this evaluator to avoid regenerating the same 30,000 suffix candidates for every split seed. Metrics marked `constant_across_split_seeds` did not change at all across the 10 runs.

## Main Judging Metrics

| Solution | Task 1 Top-1 mean | best | worst | Task 1 MRR mean | Task 2 block mean | Task 2 edit mean | Task 3 acc mean | OOD avg Top-1 mean | Determinism note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `solution_0_rule_mock` | 0.6722 | 0.6983 (seed 1) | 0.6550 (seed 5) | 0.8309 | 0.3723 | 0.6045 | 1.0000 | 0.6743 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | 0.6918 | 0.7150 (seed 9) | 0.6650 (seed 6) | 0.8433 | 0.7150 | 0.2387 | 1.0000 | 0.6627 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7150 | 0.2387 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | 0.6970 | 0.7150 (seed 4) | 0.6767 (seed 0) | 0.8462 | 0.7263 | 0.2359 | 1.0000 | 0.6622 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7172 | 0.2438 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | 0.6918 | 0.7117 (seed 9) | 0.6650 (seed 6) | 0.8433 | 0.7150 | 0.2387 | 1.0000 | 0.6635 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8456 | 0.7160 | 0.2385 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7347 | 0.2334 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7347 | 0.2334 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | 0.6970 | 0.7150 (seed 4) | 0.6767 (seed 0) | 0.8462 | 0.7347 | 0.2334 | 1.0000 | 0.6622 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | 0.6970 | 0.7150 (seed 4) | 0.6767 (seed 0) | 0.8462 | 0.7348 | 0.2327 | 1.0000 | 0.6622 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7348 | 0.2327 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7248 | 0.2260 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |

## Alias / Canonical Diagnostic

This is diagnostic-only, not an official judging metric. Official Task 1
scoring uses exact strings; this section exists only to explain why exact Top-1
is much lower than process understanding.

| Solution | Diagnostic-only canonical Top-1 mean | best | worst | Diagnostic-only canonical Top-2 mean | Same-canonical miss-rate mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `solution_0_rule_mock` | 0.9450 | 0.9517 (seed 8) | 0.9400 (seed 4) | 0.9918 | 0.8320 |
| `solution_1_hybrid_retrieval` | 0.9687 | 0.9733 (seed 0) | 0.9600 (seed 1) | 0.9992 | 0.8981 |
| `solution_2_eval_aware_retrieval` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |
| `solution_3_synthetic_augmented_retrieval` | 0.9747 | 0.9817 (seed 0) | 0.9683 (seed 1) | 0.9993 | 0.9160 |
| `solution_4_length_aware_completion` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |
| `solution_5_tuned_rank_ensemble` | 0.9687 | 0.9733 (seed 0) | 0.9600 (seed 1) | 0.9992 | 0.8982 |
| `solution_6_alias_calibrated_retrieval` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9990 | 0.9110 |
| `solution_7_monte_carlo_suffix_ensemble` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |
| `solution_8_semantic_conformance_ensemble` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |
| `solution_9_judge_aware_portfolio` | 0.9747 | 0.9817 (seed 0) | 0.9683 (seed 1) | 0.9993 | 0.9160 |
| `solution_10_confidence_gated_consensus` | 0.9747 | 0.9817 (seed 0) | 0.9683 (seed 1) | 0.9993 | 0.9160 |
| `solution_11_ood_guarded_consensus` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |
| `solution_12_mbr_completion` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |

## Full Summary Table

| Solution | Metric | Direction | Mean | Std | Best | Best seed | Worst | Worst seed | Determinism note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `solution_0_rule_mock` | `ood_avg_top1` | higher_is_better | 0.6743 | 0.0152 | 0.6983 | 0 | 0.6433 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_ic_mrr` | higher_is_better | 0.8041 | 0.0155 | 0.8350 | 0 | 0.7796 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_ic_top1` | higher_is_better | 0.6385 | 0.0276 | 0.6950 | 0 | 0.5850 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_igbt_mrr` | higher_is_better | 0.8303 | 0.0132 | 0.8608 | 9 | 0.8133 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_igbt_top1` | higher_is_better | 0.6935 | 0.0256 | 0.7400 | 9 | 0.6500 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_mosfet_mrr` | higher_is_better | 0.8440 | 0.0165 | 0.8692 | 8 | 0.8096 | 9 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `ood_mosfet_top1` | higher_is_better | 0.6910 | 0.0328 | 0.7400 | 2 | 0.6250 | 9 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_0_rule_mock` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_0_rule_mock` | `task1_canonical_top1` | higher_is_better | 0.9450 | 0.0037 | 0.9517 | 8 | 0.9400 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_canonical_top2` | higher_is_better | 0.9918 | 0.0037 | 1.0000 | 2 | 0.9867 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_mrr` | higher_is_better | 0.8309 | 0.0064 | 0.8441 | 1 | 0.8214 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_same_canonical_miss_rate` | higher_is_better | 0.8320 | 0.0130 | 0.8557 | 8 | 0.8075 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top1` | higher_is_better | 0.6722 | 0.0131 | 0.6983 | 1 | 0.6550 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top2` | higher_is_better | 0.9760 | 0.0058 | 0.9850 | 8 | 0.9667 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top3` | higher_is_better | 0.9895 | 0.0043 | 0.9967 | 8 | 0.9833 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task2_block_accuracy` | higher_is_better | 0.3723 | 0.0019 | 0.3763 | 9 | 0.3697 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_exact_match` | higher_is_better | 0.0002 | 0.0005 | 0.0017 | 1 | 0.0000 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_normalized_edit_distance` | lower_is_better | 0.6045 | 0.0009 | 0.6031 | 9 | 0.6061 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_token_accuracy` | higher_is_better | 0.2054 | 0.0031 | 0.2114 | 9 | 0.2003 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_10_confidence_gated_consensus` | `ood_avg_top1` | higher_is_better | 0.6622 | 0.0080 | 0.6717 | 0 | 0.6417 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_ic_mrr` | higher_is_better | 0.7977 | 0.0157 | 0.8229 | 1 | 0.7700 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_ic_top1` | higher_is_better | 0.6235 | 0.0194 | 0.6550 | 1 | 0.5950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_igbt_mrr` | higher_is_better | 0.8233 | 0.0086 | 0.8408 | 9 | 0.8117 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_igbt_top1` | higher_is_better | 0.6800 | 0.0110 | 0.7000 | 9 | 0.6600 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_mosfet_mrr` | higher_is_better | 0.8406 | 0.0113 | 0.8617 | 5 | 0.8237 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `ood_mosfet_top1` | higher_is_better | 0.6830 | 0.0220 | 0.7250 | 5 | 0.6500 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_10_confidence_gated_consensus` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_10_confidence_gated_consensus` | `task1_canonical_top1` | higher_is_better | 0.9747 | 0.0041 | 0.9817 | 0 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_canonical_top2` | higher_is_better | 0.9993 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_mrr` | higher_is_better | 0.8462 | 0.0078 | 0.8556 | 9 | 0.8347 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9160 | 0.0151 | 0.9433 | 0 | 0.8927 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_top1` | higher_is_better | 0.6970 | 0.0145 | 0.7150 | 4 | 0.6767 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_top2` | higher_is_better | 0.9877 | 0.0038 | 0.9950 | 5 | 0.9817 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_top3` | higher_is_better | 0.9968 | 0.0024 | 1.0000 | 5 | 0.9917 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_10_confidence_gated_consensus` | `task2_block_accuracy` | higher_is_better | 0.7348 | 0.0051 | 0.7392 | 9 | 0.7234 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task2_exact_match` | higher_is_better | 0.0027 | 0.0013 | 0.0050 | 9 | 0.0000 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task2_normalized_edit_distance` | lower_is_better | 0.2327 | 0.0029 | 0.2273 | 2 | 0.2365 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task2_token_accuracy` | higher_is_better | 0.4724 | 0.0070 | 0.4796 | 7 | 0.4551 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_10_confidence_gated_consensus` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_10_confidence_gated_consensus` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_10_confidence_gated_consensus` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_10_confidence_gated_consensus` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_11_ood_guarded_consensus` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_11_ood_guarded_consensus` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_11_ood_guarded_consensus` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_11_ood_guarded_consensus` | `task2_block_accuracy` | higher_is_better | 0.7348 | 0.0051 | 0.7392 | 9 | 0.7234 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task2_exact_match` | higher_is_better | 0.0027 | 0.0013 | 0.0050 | 9 | 0.0000 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task2_normalized_edit_distance` | lower_is_better | 0.2327 | 0.0029 | 0.2273 | 2 | 0.2365 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task2_token_accuracy` | higher_is_better | 0.4724 | 0.0070 | 0.4796 | 7 | 0.4551 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_11_ood_guarded_consensus` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_11_ood_guarded_consensus` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_11_ood_guarded_consensus` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_11_ood_guarded_consensus` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_12_mbr_completion` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_12_mbr_completion` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_12_mbr_completion` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_12_mbr_completion` | `task2_block_accuracy` | higher_is_better | 0.7248 | 0.0040 | 0.7308 | 6 | 0.7178 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task2_exact_match` | higher_is_better | 0.0038 | 0.0018 | 0.0067 | 2 | 0.0000 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task2_normalized_edit_distance` | lower_is_better | 0.2260 | 0.0017 | 0.2234 | 6 | 0.2294 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task2_token_accuracy` | higher_is_better | 0.4638 | 0.0056 | 0.4771 | 6 | 0.4577 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_12_mbr_completion` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_12_mbr_completion` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_12_mbr_completion` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_12_mbr_completion` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_1_hybrid_retrieval` | `ood_avg_top1` | higher_is_better | 0.6627 | 0.0177 | 0.7067 | 0 | 0.6367 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_ic_mrr` | higher_is_better | 0.7984 | 0.0147 | 0.8167 | 7 | 0.7758 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_ic_top1` | higher_is_better | 0.6250 | 0.0263 | 0.6600 | 7 | 0.5750 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_mosfet_mrr` | higher_is_better | 0.8365 | 0.0183 | 0.8762 | 0 | 0.8121 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `ood_mosfet_top1` | higher_is_better | 0.6750 | 0.0363 | 0.7550 | 0 | 0.6300 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_1_hybrid_retrieval` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_1_hybrid_retrieval` | `task1_canonical_top1` | higher_is_better | 0.9687 | 0.0050 | 0.9733 | 0 | 0.9600 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_mrr` | higher_is_better | 0.8433 | 0.0077 | 0.8560 | 9 | 0.8307 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_same_canonical_miss_rate` | higher_is_better | 0.8981 | 0.0167 | 0.9154 | 6 | 0.8686 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_top1` | higher_is_better | 0.6918 | 0.0149 | 0.7150 | 9 | 0.6650 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_1_hybrid_retrieval` | `task2_block_accuracy` | higher_is_better | 0.7150 | 0.0055 | 0.7247 | 9 | 0.7082 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task2_exact_match` | higher_is_better | 0.0022 | 0.0021 | 0.0067 | 4 | 0.0000 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task2_normalized_edit_distance` | lower_is_better | 0.2387 | 0.0015 | 0.2359 | 2 | 0.2409 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task2_token_accuracy` | higher_is_better | 0.4478 | 0.0071 | 0.4612 | 9 | 0.4388 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_1_hybrid_retrieval` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_1_hybrid_retrieval` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_1_hybrid_retrieval` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `public_lookup_coverage` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `public_lookup_exact_next_when_covered` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `task2_block_accuracy` | higher_is_better | 0.7150 | 0.0055 | 0.7247 | 9 | 0.7082 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task2_exact_match` | higher_is_better | 0.0022 | 0.0021 | 0.0067 | 4 | 0.0000 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task2_normalized_edit_distance` | lower_is_better | 0.2387 | 0.0015 | 0.2359 | 2 | 0.2409 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task2_token_accuracy` | higher_is_better | 0.4478 | 0.0071 | 0.4612 | 9 | 0.4388 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_2_eval_aware_retrieval` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_3_synthetic_augmented_retrieval` | `ood_avg_top1` | higher_is_better | 0.6622 | 0.0080 | 0.6717 | 0 | 0.6417 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_ic_mrr` | higher_is_better | 0.7977 | 0.0157 | 0.8229 | 1 | 0.7700 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_ic_top1` | higher_is_better | 0.6235 | 0.0194 | 0.6550 | 1 | 0.5950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_igbt_mrr` | higher_is_better | 0.8233 | 0.0086 | 0.8408 | 9 | 0.8117 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_igbt_top1` | higher_is_better | 0.6800 | 0.0110 | 0.7000 | 9 | 0.6600 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_mosfet_mrr` | higher_is_better | 0.8406 | 0.0113 | 0.8617 | 5 | 0.8237 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `ood_mosfet_top1` | higher_is_better | 0.6830 | 0.0220 | 0.7250 | 5 | 0.6500 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_3_synthetic_augmented_retrieval` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_3_synthetic_augmented_retrieval` | `task1_canonical_top1` | higher_is_better | 0.9747 | 0.0041 | 0.9817 | 0 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_canonical_top2` | higher_is_better | 0.9993 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_mrr` | higher_is_better | 0.8462 | 0.0078 | 0.8556 | 9 | 0.8347 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9160 | 0.0151 | 0.9433 | 0 | 0.8927 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_top1` | higher_is_better | 0.6970 | 0.0145 | 0.7150 | 4 | 0.6767 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_top2` | higher_is_better | 0.9877 | 0.0038 | 0.9950 | 5 | 0.9817 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_top3` | higher_is_better | 0.9968 | 0.0024 | 1.0000 | 5 | 0.9917 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_3_synthetic_augmented_retrieval` | `task2_block_accuracy` | higher_is_better | 0.7263 | 0.0057 | 0.7348 | 9 | 0.7147 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task2_exact_match` | higher_is_better | 0.0015 | 0.0012 | 0.0033 | 2 | 0.0000 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task2_normalized_edit_distance` | lower_is_better | 0.2359 | 0.0026 | 0.2318 | 2 | 0.2392 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task2_token_accuracy` | higher_is_better | 0.4593 | 0.0083 | 0.4708 | 9 | 0.4408 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_3_synthetic_augmented_retrieval` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_3_synthetic_augmented_retrieval` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_3_synthetic_augmented_retrieval` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_3_synthetic_augmented_retrieval` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_4_length_aware_completion` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_4_length_aware_completion` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_4_length_aware_completion` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_4_length_aware_completion` | `task2_block_accuracy` | higher_is_better | 0.7172 | 0.0053 | 0.7261 | 9 | 0.7104 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task2_exact_match` | higher_is_better | 0.0022 | 0.0021 | 0.0067 | 4 | 0.0000 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task2_normalized_edit_distance` | lower_is_better | 0.2438 | 0.0017 | 0.2414 | 2 | 0.2464 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task2_token_accuracy` | higher_is_better | 0.4488 | 0.0070 | 0.4616 | 9 | 0.4398 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_4_length_aware_completion` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_4_length_aware_completion` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_4_length_aware_completion` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_4_length_aware_completion` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_5_tuned_rank_ensemble` | `ood_avg_top1` | higher_is_better | 0.6635 | 0.0174 | 0.7033 | 0 | 0.6350 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_ic_mrr` | higher_is_better | 0.7989 | 0.0151 | 0.8167 | 7 | 0.7758 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_ic_top1` | higher_is_better | 0.6260 | 0.0271 | 0.6600 | 7 | 0.5750 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_igbt_mrr` | higher_is_better | 0.8292 | 0.0145 | 0.8583 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_igbt_top1` | higher_is_better | 0.6915 | 0.0245 | 0.7350 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_mosfet_mrr` | higher_is_better | 0.8355 | 0.0183 | 0.8738 | 0 | 0.8096 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `ood_mosfet_top1` | higher_is_better | 0.6730 | 0.0361 | 0.7500 | 0 | 0.6250 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_5_tuned_rank_ensemble` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_5_tuned_rank_ensemble` | `task1_canonical_top1` | higher_is_better | 0.9687 | 0.0050 | 0.9733 | 0 | 0.9600 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_mrr` | higher_is_better | 0.8433 | 0.0072 | 0.8543 | 9 | 0.8307 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_same_canonical_miss_rate` | higher_is_better | 0.8982 | 0.0166 | 0.9154 | 6 | 0.8693 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_top1` | higher_is_better | 0.6918 | 0.0140 | 0.7117 | 9 | 0.6650 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_top2` | higher_is_better | 0.9858 | 0.0043 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_5_tuned_rank_ensemble` | `task2_block_accuracy` | higher_is_better | 0.7150 | 0.0055 | 0.7247 | 9 | 0.7082 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task2_exact_match` | higher_is_better | 0.0022 | 0.0021 | 0.0067 | 4 | 0.0000 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task2_normalized_edit_distance` | lower_is_better | 0.2387 | 0.0015 | 0.2359 | 2 | 0.2409 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task2_token_accuracy` | higher_is_better | 0.4478 | 0.0071 | 0.4612 | 9 | 0.4388 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_5_tuned_rank_ensemble` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_5_tuned_rank_ensemble` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_5_tuned_rank_ensemble` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_5_tuned_rank_ensemble` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_6_alias_calibrated_retrieval` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_ic_mrr` | higher_is_better | 0.8175 | 0.0131 | 0.8375 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_mosfet_mrr` | higher_is_better | 0.8511 | 0.0190 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_6_alias_calibrated_retrieval` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_6_alias_calibrated_retrieval` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_canonical_top2` | higher_is_better | 0.9990 | 0.0011 | 1.0000 | 0 | 0.9967 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_mrr` | higher_is_better | 0.8456 | 0.0078 | 0.8582 | 9 | 0.8318 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_top2` | higher_is_better | 0.9860 | 0.0044 | 0.9917 | 6 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_top3` | higher_is_better | 0.9977 | 0.0013 | 1.0000 | 8 | 0.9950 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_6_alias_calibrated_retrieval` | `task2_block_accuracy` | higher_is_better | 0.7160 | 0.0064 | 0.7270 | 9 | 0.7088 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task2_exact_match` | higher_is_better | 0.0028 | 0.0021 | 0.0067 | 4 | 0.0000 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task2_normalized_edit_distance` | lower_is_better | 0.2385 | 0.0020 | 0.2346 | 0 | 0.2408 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task2_token_accuracy` | higher_is_better | 0.4481 | 0.0066 | 0.4584 | 9 | 0.4400 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_6_alias_calibrated_retrieval` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_6_alias_calibrated_retrieval` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_6_alias_calibrated_retrieval` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_6_alias_calibrated_retrieval` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_7_monte_carlo_suffix_ensemble` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_7_monte_carlo_suffix_ensemble` | `task2_block_accuracy` | higher_is_better | 0.7347 | 0.0052 | 0.7393 | 5 | 0.7231 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task2_exact_match` | higher_is_better | 0.0042 | 0.0015 | 0.0067 | 2 | 0.0017 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task2_normalized_edit_distance` | lower_is_better | 0.2334 | 0.0028 | 0.2272 | 2 | 0.2377 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task2_token_accuracy` | higher_is_better | 0.4710 | 0.0074 | 0.4800 | 7 | 0.4530 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_7_monte_carlo_suffix_ensemble` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_7_monte_carlo_suffix_ensemble` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_7_monte_carlo_suffix_ensemble` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_7_monte_carlo_suffix_ensemble` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_8_semantic_conformance_ensemble` | `ood_avg_top1` | higher_is_better | 0.6787 | 0.0181 | 0.7183 | 0 | 0.6550 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_ic_mrr` | higher_is_better | 0.8174 | 0.0130 | 0.8367 | 7 | 0.7929 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_ic_top1` | higher_is_better | 0.6440 | 0.0266 | 0.6800 | 7 | 0.5950 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_igbt_mrr` | higher_is_better | 0.8274 | 0.0148 | 0.8558 | 9 | 0.8092 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_igbt_top1` | higher_is_better | 0.6880 | 0.0244 | 0.7300 | 9 | 0.6550 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_mosfet_mrr` | higher_is_better | 0.8510 | 0.0189 | 0.8862 | 0 | 0.8196 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `ood_mosfet_top1` | higher_is_better | 0.7040 | 0.0374 | 0.7750 | 0 | 0.6450 | 7 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_8_semantic_conformance_ensemble` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_8_semantic_conformance_ensemble` | `task1_canonical_top1` | higher_is_better | 0.9730 | 0.0031 | 0.9783 | 9 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_canonical_top2` | higher_is_better | 0.9992 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_mrr` | higher_is_better | 0.8455 | 0.0081 | 0.8585 | 9 | 0.8315 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9110 | 0.0105 | 0.9226 | 9 | 0.8941 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_top1` | higher_is_better | 0.6962 | 0.0156 | 0.7200 | 9 | 0.6667 | 6 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_top2` | higher_is_better | 0.9860 | 0.0042 | 0.9917 | 9 | 0.9750 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_top3` | higher_is_better | 0.9970 | 0.0012 | 0.9983 | 1 | 0.9950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_8_semantic_conformance_ensemble` | `task2_block_accuracy` | higher_is_better | 0.7347 | 0.0052 | 0.7393 | 5 | 0.7231 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task2_exact_match` | higher_is_better | 0.0042 | 0.0015 | 0.0067 | 2 | 0.0017 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task2_normalized_edit_distance` | lower_is_better | 0.2334 | 0.0028 | 0.2272 | 2 | 0.2377 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task2_token_accuracy` | higher_is_better | 0.4710 | 0.0074 | 0.4800 | 7 | 0.4530 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_8_semantic_conformance_ensemble` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_8_semantic_conformance_ensemble` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_8_semantic_conformance_ensemble` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_8_semantic_conformance_ensemble` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_9_judge_aware_portfolio` | `ood_avg_top1` | higher_is_better | 0.6622 | 0.0080 | 0.6717 | 0 | 0.6417 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_ic_mrr` | higher_is_better | 0.7977 | 0.0157 | 0.8229 | 1 | 0.7700 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_ic_top1` | higher_is_better | 0.6235 | 0.0194 | 0.6550 | 1 | 0.5950 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_igbt_mrr` | higher_is_better | 0.8233 | 0.0086 | 0.8408 | 9 | 0.8117 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_igbt_top1` | higher_is_better | 0.6800 | 0.0110 | 0.7000 | 9 | 0.6600 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_mosfet_mrr` | higher_is_better | 0.8406 | 0.0113 | 0.8617 | 5 | 0.8237 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `ood_mosfet_top1` | higher_is_better | 0.6830 | 0.0220 | 0.7250 | 5 | 0.6500 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `public_lookup_coverage` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_9_judge_aware_portfolio` | `public_lookup_exact_next_when_covered` | not_applicable |  |  |  |  |  |  | not_reported_for_this_solution |
| `solution_9_judge_aware_portfolio` | `task1_canonical_top1` | higher_is_better | 0.9747 | 0.0041 | 0.9817 | 0 | 0.9683 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_canonical_top2` | higher_is_better | 0.9993 | 0.0008 | 1.0000 | 0 | 0.9983 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_mrr` | higher_is_better | 0.8462 | 0.0078 | 0.8556 | 9 | 0.8347 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_same_canonical_miss_rate` | higher_is_better | 0.9160 | 0.0151 | 0.9433 | 0 | 0.8927 | 1 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_top1` | higher_is_better | 0.6970 | 0.0145 | 0.7150 | 4 | 0.6767 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_top2` | higher_is_better | 0.9877 | 0.0038 | 0.9950 | 5 | 0.9817 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_top3` | higher_is_better | 0.9968 | 0.0024 | 1.0000 | 5 | 0.9917 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_9_judge_aware_portfolio` | `task2_block_accuracy` | higher_is_better | 0.7347 | 0.0052 | 0.7393 | 5 | 0.7231 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task2_exact_match` | higher_is_better | 0.0042 | 0.0015 | 0.0067 | 2 | 0.0017 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task2_normalized_edit_distance` | lower_is_better | 0.2334 | 0.0028 | 0.2272 | 2 | 0.2377 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task2_token_accuracy` | higher_is_better | 0.4710 | 0.0074 | 0.4800 | 7 | 0.4530 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_9_judge_aware_portfolio` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_9_judge_aware_portfolio` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_9_judge_aware_portfolio` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_9_judge_aware_portfolio` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |

## Raw Per-Seed Results

See `per_seed_metrics.csv` and `per_seed_metrics.json` in this folder.

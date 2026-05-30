# 10-Seed Solution Evaluation

This report evaluates the current solutions across seeds `0` through `9`.

Important interpretation: these solutions do not train stochastic neural weights. For a fixed local split, each one is deterministic. Here, the seed changes the local train/held-out split, anomaly shuffle, and OOD sample. Metrics marked `constant_across_split_seeds` did not change at all across the 10 runs.

## Main Judging Metrics

| Solution | Task 1 Top-1 mean | best | worst | Task 1 MRR mean | Task 2 block mean | Task 2 edit mean | Task 3 acc mean | OOD avg Top-1 mean | Determinism note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `solution_0_rule_mock` | 0.6723 | 0.6983 (seed 1) | 0.6550 (seed 5) | 0.8310 | 0.3723 | 0.6045 | 1.0000 | 0.6743 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_1_hybrid_retrieval` | 0.6918 | 0.7150 (seed 9) | 0.6650 (seed 6) | 0.8433 | 0.7150 | 0.2387 | 1.0000 | 0.6627 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_2_eval_aware_retrieval` | 0.6962 | 0.7200 (seed 9) | 0.6667 (seed 6) | 0.8455 | 0.7150 | 0.2387 | 1.0000 | 0.6787 | varies_by_local_split_seed_model_itself_deterministic |

## Alias / Canonical Diagnostic

This is not an official judging metric, but it explains why exact Top-1 is much lower than process understanding.

| Solution | Canonical Top-1 mean | best | worst | Canonical Top-2 mean | Same-canonical miss-rate mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `solution_0_rule_mock` | 0.9450 | 0.9517 (seed 8) | 0.9400 (seed 4) | 0.9918 | 0.8319 |
| `solution_1_hybrid_retrieval` | 0.9687 | 0.9733 (seed 0) | 0.9600 (seed 1) | 0.9992 | 0.8981 |
| `solution_2_eval_aware_retrieval` | 0.9730 | 0.9783 (seed 9) | 0.9683 (seed 1) | 0.9992 | 0.9110 |

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
| `solution_0_rule_mock` | `task1_mrr` | higher_is_better | 0.8310 | 0.0065 | 0.8441 | 1 | 0.8214 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_same_canonical_miss_rate` | higher_is_better | 0.8319 | 0.0124 | 0.8535 | 8 | 0.8075 | 4 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top1` | higher_is_better | 0.6723 | 0.0133 | 0.6983 | 1 | 0.6550 | 5 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top2` | higher_is_better | 0.9760 | 0.0058 | 0.9850 | 8 | 0.9667 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top3` | higher_is_better | 0.9895 | 0.0043 | 0.9967 | 8 | 0.9833 | 3 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task1_top5` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task2_block_accuracy` | higher_is_better | 0.3723 | 0.0020 | 0.3763 | 9 | 0.3697 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_exact_match` | higher_is_better | 0.0002 | 0.0005 | 0.0017 | 1 | 0.0000 | 0 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_normalized_edit_distance` | lower_is_better | 0.6045 | 0.0010 | 0.6031 | 9 | 0.6061 | 8 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task2_token_accuracy` | higher_is_better | 0.2052 | 0.0031 | 0.2114 | 9 | 0.2003 | 2 | varies_by_local_split_seed_model_itself_deterministic |
| `solution_0_rule_mock` | `task3_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_f1_valid` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_roc_auc_valid_probability` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
| `solution_0_rule_mock` | `task3_rule_attribution_accuracy` | higher_is_better | 1.0000 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 | constant_across_split_seeds |
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

## Raw Per-Seed Results

See `per_seed_metrics.csv` and `per_seed_metrics.json` in this folder.

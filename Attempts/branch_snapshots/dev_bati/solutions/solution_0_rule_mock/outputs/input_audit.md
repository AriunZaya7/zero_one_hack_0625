# Solution 0 Input Audit

This file records the repo assets considered by `solution_0_rule_mock`.

## Docs Reviewed

- `README.md`
- `CLAUDE.md`
- `PLAN.md`
- `submission/SUBMISSION.md`
- `submission/REPORT_TEMPLATE.md`
- `industrial-infineon/README.md`
- `industrial-infineon/Track_industrial_en.md`
- `industrial-infineon/Track_industrial.md`
- `training_data/README.md`
- `training_data/generation_rules.md`

## Scripts Reviewed

- `data.py`
- `tokenizer.py`
- `ngram.py`
- `run_baseline.py`
- `ablation_n.py`
- `training_data/generate_sequences.py`
- `small_transformer_model/data_generation/generate_more_sequence_data.py`
- `small_transformer_model/data_generation/prepare_dataset.py`
- `small_transformer_model/train/train_model.py`
- `small_transformer_model/train/setup_leonardo.sh`
- `small_transformer_model/train/job.slurm`
- `test_baseline_1/baseline_random.py`
- `test_baseline_1/baseline_ngram.py`
- `test_baseline_1/baseline_llm_zeroshot.py`
- `test_baseline_1/run_all_baselines.py`
- `test_baseline_1/visualize_baselines.py`
- `test_baseline_2/ngram_context_comparison.py`

## Sequence Sources Used By The Baseline

| File | Family | Kind | Sequences | Valid | Invalid | Length min/mean/max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `training_data/IC_Longdescr.csv` | ic | canonical_single_sequence | 1 | 1 | 0 | 109/109.0/109 |
| `training_data/IC_generated_extra.csv` | ic | long_format_sequence | 2000 | 2000 | 0 | 105/115.1/123 |
| `training_data/IC_longdescription_parameters.csv` | ic | canonical_single_sequence | 1 | 1 | 0 | 109/109.0/109 |
| `training_data/IC_variants.csv` | ic | long_format_sequence | 1000 | 1000 | 0 | 107/115.1/122 |
| `training_data/IGBT_Longdescr.csv` | igbt | canonical_single_sequence | 1 | 1 | 0 | 137/137.0/137 |
| `training_data/IGBT_generated_extra.csv` | igbt | long_format_sequence | 2000 | 2000 | 0 | 138/148.0/157 |
| `training_data/IGBT_longdescription_parameters.csv` | igbt | canonical_single_sequence | 1 | 1 | 0 | 139/139.0/139 |
| `training_data/IGBT_variants.csv` | igbt | long_format_sequence | 1000 | 1000 | 0 | 139/148.0/155 |
| `training_data/MOSFET_Longdescr.csv` | mosfet | canonical_single_sequence | 1 | 1 | 0 | 126/126.0/126 |
| `training_data/MOSFET_longdescription_parameters.csv` | mosfet | canonical_single_sequence | 1 | 1 | 0 | 126/126.0/126 |
| `training_data/MOSFET_variants.csv` | mosfet | long_format_sequence | 1000 | 1000 | 0 | 117/125.3/134 |
| `training_data/syntheticIC.csv` | ic | canonical_single_sequence | 1 | 1 | 0 | 107/107.0/107 |
| `training_data/syntheticIGBT.csv` | igbt | canonical_single_sequence | 1 | 1 | 0 | 151/151.0/151 |
| `training_data/synthetic_mosfet.csv` | mosfet | canonical_single_sequence | 1 | 1 | 0 | 126/126.0/126 |

## Metadata-Only CSVs Reviewed But Not Modeled

| File | Rows | Columns |
| --- | ---: | --- |

## Notes

- Official eval inputs and eval_metrics.py are not present in reachable repo history.
- Solution 0 uses all available sequence sources for the baseline model.
- For enriched description/parameter CSVs, Solution 0 uses only the ordered STEP column; natural-language descriptions and fab parameters are not used by this baseline.
- CLAUDE.md and PLAN.md recommend the final winning model use step-level tokens and avoid family embeddings for OOD generalization.

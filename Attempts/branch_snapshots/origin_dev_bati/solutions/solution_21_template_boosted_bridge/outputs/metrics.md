# Solution 21 Metrics

## Official Participant Input Run

- Official valid rows predicted: 600
- Official anomaly rows predicted: 987
- Full-route exact prefix coverage: 1.0000
- Valid-partial lattice coverage: 0.5000
- Paired-length fallback coverage after stronger gates: 0.0000
- Prediction source counts: {'exact_full_route': 600}

## Local Coupled Self-Eval

- Task 1 exact Top-1: 1.0000
- Task 1 exact Top-3: 1.0000
- Task 2 exact match: 1.0000
- Task 2 normalized edit distance: 0.0000
- Task 3 accuracy: 1.0000

## XGBoost Template Bridge Diagnostic

- Validation setup: train on 100 synthetic families, validate on ['SYNTHFAMILY101', 'SYNTHFAMILY102', 'SYNTHFAMILY103', 'SYNTHFAMILY104', 'SYNTHFAMILY105', 'SYNTHFAMILY106', 'SYNTHFAMILY107', 'SYNTHFAMILY108', 'SYNTHFAMILY109', 'SYNTHFAMILY110'].
- Raw XGBoost 10-seed Top-1 mean: 0.5806.
- Template XGBoost 10-seed Top-1 mean: 0.7531.
- Raw XGBoost family-specific label coverage: 0.0000.
- Template XGBoost family-specific label coverage: 1.0000.
- Template XGBoost family-specific Top-1 mean: 1.0000.
- Seed-0 XGBoost checkpoint files: solutions/solution_21_template_boosted_bridge/outputs/raw_xgb_exact_seed0_xgboost_model.json, solutions/solution_21_template_boosted_bridge/outputs/template_xgb_bridge_seed0_xgboost_model.json.

Interpretation: the learned model is only OOD-useful after template normalization. Raw XGBoost cannot emit exact family-specific strings it never saw as labels; template XGBoost learns the reusable suffix and then rewrites `__FAMILY__` to the visible eval family.

## Template XGBoost Tree SHAP Explainability

Tree SHAP is XGBoost's exact Shapley-value contribution method for tree ensembles. The rows below average absolute contribution size over the held-out validation examples and model classes.

- lag_step_1: mean abs Tree SHAP 0.348382 (0.1815 share) - Recent exact/template step token in the visible prefix.
- prefix_len: mean abs Tree SHAP 0.225322 (0.1174 share) - How many steps are already visible in the partial route.
- lag_step_4: mean abs Tree SHAP 0.129463 (0.0675 share) - Recent exact/template step token in the visible prefix.
- lag_step_2: mean abs Tree SHAP 0.125806 (0.0656 share) - Recent exact/template step token in the visible prefix.
- lag_block_3: mean abs Tree SHAP 0.122686 (0.0639 share) - Recent high-level process block, such as lithography, etch, clean, or test.
- lag_block_4: mean abs Tree SHAP 0.105443 (0.0549 share) - Recent high-level process block, such as lithography, etch, clean, or test.
- lag_block_1: mean abs Tree SHAP 0.104788 (0.0546 share) - Recent high-level process block, such as lithography, etch, clean, or test.
- lag_block_5: mean abs Tree SHAP 0.079544 (0.0414 share) - Recent high-level process block, such as lithography, etch, clean, or test.

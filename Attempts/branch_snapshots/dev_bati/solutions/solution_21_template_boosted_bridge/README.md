# Solution 21: Template-Boosted Evidence Bridge

This is the final hybrid candidate for the Industrial AI submission.

It keeps the submit-ready Solution 20 prediction cascade:

- exact validator-valid full-route memory,
- valid-partial lattice,
- paired 60%/80% completion-length guard,
- normalized `__FAMILY__` template fallback,
- validator-based Task 3 labels and rule attribution.

It also adds a real XGBoost OOD diagnostic:

- train raw XGBoost on exact step strings,
- train template XGBoost after converting family-prefixed steps to
  `__FAMILY__ ...`,
- validate both on the same ten unseen synthetic families,
- report a 10-seed summary for the 100-training-family setting.

Run from the repository root:

```bash
python -B solutions/solution_21_template_boosted_bridge/solution.py
```

Important outputs:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/official_submission/nextstep.csv`
- `outputs/official_submission/completion.csv`
- `outputs/official_submission/anomaly.csv`
- `outputs/template_boosting_curve.csv`
- `outputs/template_boosting_10_seed.csv`
- `outputs/template_boosting_examples.csv`
- `outputs/template_boosting_tree_shap.csv`
- `outputs/template_boosting_xgb_training_logloss.csv`
- `outputs/template_boosting_bridge.json`
- `outputs/raw_xgb_exact_seed0_xgboost_model.json`
- `outputs/template_xgb_bridge_seed0_xgboost_model.json`
- `outputs/template_xgb_feature_names.json`
- `outputs/metrics.json`
- `outputs/metrics.md`
- `explanation.html`
- `how_to_submit_this_solution.html`

The XGBoost bridge is a diagnostic and future-model direction. It is not allowed
to override stronger direct evidence in the current submit CSVs.

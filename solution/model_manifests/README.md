# Selected XGBoost Recipes

The selected XGBoost models are task-routed hybrids. Each manifest identifies
the locally trained component checkpoint directories for Tasks 1, 2, and 3.
New checkpoints contain `xgboost_model.json` from XGBoost `save_model()`,
`class_map.json`, and `boosting_metadata.json`. A `model.pkl` compatibility
file is also written. Checkpoint directories are intentionally ignored by Git
because the files are large.

Run the selected all-family recipe:

```powershell
python -m solution.run_eval `
  --model xgboost_hybrid `
  --hybrid_manifest solution/model_manifests/xgb_hybrid_all_family.json `
  --families IC IGBT MOSFET `
  --max_eval_seqs 100 `
  --beam_width 1 `
  --out solution/results/xgb_hybrid_taskbest_eval100.json
```

For leave-one-family-out evaluation, use the corresponding holdout manifest and
set `--families` to `IC`, `IGBT`, or `MOSFET`.

The manifests are reproducibility recipes, not model bundles. Copy the
referenced checkpoint directories separately when deploying to another machine.
The native JSON artifacts are sufficient for loading; `model.pkl` is optional.

## Rebuild the selected all-family recipe

Train the legacy-feature component used for Tasks 1 and 2:

```powershell
python -m solution.train.train_boosting `
  --model xgboost `
  --data_dir training_data `
  --train_families IC IGBT MOSFET `
  --run_name xgboost `
  --out solution/checkpoints/xgboost `
  --max_train_examples 250000 `
  --max_val_examples 50000 `
  --iterations 1200 `
  --feature_version 1 `
  --max_depth 8 `
  --eta 0.08 `
  --subsample 0.9 `
  --colsample_bytree 0.9
```

Train the regularized v2 component used for Task 3:

```powershell
python -m solution.train.train_boosting `
  --model xgboost `
  --data_dir training_data `
  --train_families IC IGBT MOSFET `
  --run_name xgb_v2_reg_100k_300 `
  --out solution/checkpoints/xgb_v2_reg_100k_300 `
  --max_train_examples 100000 `
  --max_val_examples 20000 `
  --iterations 300 `
  --feature_version 2 `
  --max_depth 6 `
  --eta 0.06 `
  --min_child_weight 3 `
  --reg_lambda 2 `
  --subsample 0.85 `
  --colsample_bytree 0.85
```

Both commands save `xgboost_model.json`, `class_map.json`,
`boosting_metadata.json`, and `model.pkl` under their `--out` directories.

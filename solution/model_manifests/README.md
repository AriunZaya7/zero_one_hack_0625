# Selected XGBoost Recipes

The selected XGBoost models are task-routed hybrids. Each manifest identifies
the locally trained component checkpoint directories for Tasks 1, 2, and 3.
Checkpoint directories contain `model.pkl` files serialized with Python
`pickle`; they are intentionally ignored by Git because the files are large.

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

The manifests are reproducibility recipes, not portable model bundles. Copy the
referenced checkpoint directories separately when deploying to another machine.

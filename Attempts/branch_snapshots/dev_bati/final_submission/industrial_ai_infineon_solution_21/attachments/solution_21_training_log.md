# Training / Fitting Log

This solution includes a deterministic evidence cascade and a small XGBoost raw-vs-template diagnostic. The official CSV path is rebuilt from lookup tables and validator evidence; the bridge diagnostic trains XGBoost classifiers from the synthetic 110-family probe.

## Command

```bash
python -B solutions/solution_21_template_boosted_bridge/solution.py
```

## Data Summary

- Families: `mosfet, igbt, ic`
- Valid self-eval rows: `600`
- Anomaly self-eval rows: `600`
- Training sequences: `6709`
- Local split seed: `42`

## Result Snapshot

- Task 1 Top-1: `1.0000`
- Task 1 MRR: `1.0000`
- Task 2 normalized edit distance: `0.0000`
- Task 2 block accuracy: `1.0000`
- Task 3 accuracy: `1.0000`

## XGBoost Bridge Snapshot

- Raw XGBoost 10-seed Top-1: `0.5806`
- Template XGBoost 10-seed Top-1: `0.7531`
- Raw family-specific label coverage: `0.0000`
- Template family-specific label coverage: `1.0000`
- Top Tree SHAP feature: `lag_step_1` with mean absolute contribution `0.348382`.
- Seed-0 model checkpoints:
- `solutions/solution_21_template_boosted_bridge/outputs/raw_xgb_exact_seed0_xgboost_model.json`
- `solutions/solution_21_template_boosted_bridge/outputs/template_xgb_bridge_seed0_xgboost_model.json`

## Checkpoint Status

The official CSV path is deterministic and rebuildable. For the trained bridge diagnostic, seed-0 raw and template XGBoost model JSON checkpoints are committed under `outputs/` and copied into the submission package. For a real neural submission, this folder should be
extended with `.pt`/`.safetensors` checkpoints and cluster logs.

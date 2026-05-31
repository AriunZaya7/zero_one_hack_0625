# Checkpoint Manifest

Binary/checkpoint artifacts: **included for the trained XGBoost diagnostic**.

The official prediction cascade is deterministic and does not need a
neural checkpoint. The trained raw-vs-template XGBoost bridge does
produce model artifacts, and the seed-0 checkpoints are included:

- `solutions/solution_21_template_boosted_bridge/outputs/raw_xgb_exact_seed0_xgboost_model.json`
- `solutions/solution_21_template_boosted_bridge/outputs/template_xgb_bridge_seed0_xgboost_model.json`

Rebuild command:

```bash
python -B solutions/solution_21_template_boosted_bridge/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

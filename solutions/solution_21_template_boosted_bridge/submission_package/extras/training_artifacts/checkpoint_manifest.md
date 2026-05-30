# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: The XGBoost bridge is retrained deterministically from source for each diagnostic run; no binary checkpoint is committed because the model is small and rebuildable.

Rebuild command:

```bash
python -B solutions/solution_21_template_boosted_bridge/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

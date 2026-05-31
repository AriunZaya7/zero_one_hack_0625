# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; pseudo labels and metric logs are regenerated deterministically from the released participant files and validator.

Rebuild command:

```bash
python -B solutions/solution_16_pseudolabel_metric_audit/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

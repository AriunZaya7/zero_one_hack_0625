# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; calibration tables and guard audit files are regenerated deterministically from public data and released participant files.

Rebuild command:

```bash
python -B solutions/solution_17_conformal_route_guard/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

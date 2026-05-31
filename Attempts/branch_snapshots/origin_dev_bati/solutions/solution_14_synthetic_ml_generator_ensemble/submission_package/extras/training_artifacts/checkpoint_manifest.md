# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; generated training data and count tables are rebuilt deterministically from source.

Rebuild command:

```bash
python -B solutions/solution_14_synthetic_ml_generator_ensemble/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

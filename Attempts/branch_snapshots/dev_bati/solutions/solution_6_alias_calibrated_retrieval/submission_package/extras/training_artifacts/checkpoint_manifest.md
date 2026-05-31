# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; alias counts and retrieval tables are rebuilt deterministically.

Rebuild command:

```bash
python -B solutions/solution_6_alias_calibrated_retrieval/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; generated suffix library and retrieval tables are rebuilt deterministically.

Rebuild command:

```bash
python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

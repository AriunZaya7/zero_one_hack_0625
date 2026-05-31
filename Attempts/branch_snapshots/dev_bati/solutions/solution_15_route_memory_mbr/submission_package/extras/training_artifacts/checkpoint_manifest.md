# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; route memory and generated candidates are rebuilt deterministically from source.

Rebuild command:

```bash
python -B solutions/solution_15_route_memory_mbr/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; the retrieval index is rebuilt deterministically.

Rebuild command:

```bash
python -B solutions/solution_1_hybrid_retrieval/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

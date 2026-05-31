# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; all selected specialists and the consensus gate rebuild deterministically.

Rebuild command:

```bash
python -B solutions/solution_10_confidence_gated_consensus/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

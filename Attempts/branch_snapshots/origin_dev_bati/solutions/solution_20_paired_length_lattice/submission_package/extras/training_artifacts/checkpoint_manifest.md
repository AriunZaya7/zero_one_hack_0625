# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; full-route memory, valid-partial lattice, paired-length statistics, template routes, and retrieval indexes are regenerated deterministically from source.

Rebuild command:

```bash
python -B solutions/solution_20_paired_length_lattice/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

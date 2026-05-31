# Checkpoint Manifest

Binary checkpoint: **not produced**.

Reason: No binary checkpoint is needed; the model is rebuilt from the participant inputs, validator, and deterministic fallback source.

Rebuild command:

```bash
python -B solutions/solution_13_transductive_generator_validator/solution.py
```

Source-controlled state needed to rebuild:

- `training_data/`
- solution source file
- shared helper solutions imported by this solution
- `training_data/generate_sequences.py` for validator/generator behavior

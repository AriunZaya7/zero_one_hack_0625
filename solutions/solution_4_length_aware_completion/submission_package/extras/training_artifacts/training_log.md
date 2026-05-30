# Training / Fitting Log

This solution is deterministic. "Training" means fitting counts,
lookup tables, retrieval indexes, generated-data indexes, or fixed
statistics from the public CSV files. It does not run gradient descent.

## Command

```bash
python -B solutions/solution_4_length_aware_completion/solution.py
```

## Data Summary

- Families: `mosfet, igbt, ic`
- Valid self-eval rows: `600`
- Anomaly self-eval rows: `600`
- Training sequences: `6709`
- Local split seed: `42`

## Result Snapshot

- Task 1 Top-1: `0.7317`
- Task 1 MRR: `0.8650`
- Task 2 normalized edit distance: `0.2468`
- Task 2 block accuracy: `0.7188`
- Task 3 accuracy: `1.0000`

## Checkpoint Status

No binary checkpoint is needed; length statistics are deterministic. For a real neural submission, this folder should be
extended with `.pt`/`.safetensors` checkpoints and cluster logs.

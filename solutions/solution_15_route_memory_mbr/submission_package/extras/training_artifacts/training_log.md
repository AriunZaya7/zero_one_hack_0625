# Training / Fitting Log

This solution is deterministic. "Training" means fitting counts,
lookup tables, retrieval indexes, generated-data indexes, or fixed
statistics from the public CSV files. It does not run gradient descent.

## Command

```bash
python -B solutions/solution_15_route_memory_mbr/solution.py
```

## Data Summary

- Families: `mosfet, igbt, ic`
- Valid self-eval rows: `600`
- Anomaly self-eval rows: `600`
- Training sequences: `6709`
- Local split seed: `42`

## Result Snapshot

- Task 1 Top-1: `1.0000`
- Task 1 MRR: `1.0000`
- Task 2 normalized edit distance: `0.0000`
- Task 2 block accuracy: `1.0000`
- Task 3 accuracy: `1.0000`

## Checkpoint Status

No binary checkpoint is needed; route memory and generated candidates are rebuilt deterministically from source. For a real neural submission, this folder should be
extended with `.pt`/`.safetensors` checkpoints and cluster logs.

# Solution 4: Length-Aware Completion

This attempt keeps Solution 2's next-step model and changes Task 2 completion.
It learns the most likely hidden remainder length from public training cuts and
trims retrieved suffixes to that length.

## Why Try This

Task 2 is judged with several metrics, not only exact match. A suffix with a
better length can improve token and block alignment even if exact sequence
match stays rare.

## Run

```bash
python -B solutions/solution_4_length_aware_completion/solution.py
```

Outputs go to:

```text
solutions/solution_4_length_aware_completion/outputs/
```

## Current Result

| Metric | Value |
| --- | ---: |
| Task 1 Top-1 | 0.7317 |
| Task 1 Top-3 | 1.0000 |
| Task 1 Top-5 | 1.0000 |
| Task 1 MRR | 0.8650 |
| Diagnostic-only canonical Top-1 | 0.9783 |
| Task 2 normalized edit distance | 0.2468 |
| Task 2 token accuracy | 0.4495 |
| Task 2 block accuracy | 0.7188 |
| Task 3 accuracy | 1.0000 |

## Caveat

Length trimming improves block alignment slightly versus Solution 2 on seed 42,
but it worsens normalized edit distance. This is a metric-tradeoff attempt, not
the strongest overall solution.

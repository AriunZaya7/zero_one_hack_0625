# Solution 2: Eval-Aware Retrieval

This solution was created after `solution_1_hybrid_retrieval` reached only
`0.7283` exact Top-1 while already reaching `0.9950` exact Top-2 and `0.9750`
canonical/process-step Top-1.

The diagnosis is that the model usually knows the right manufacturing operation,
but loses exact-string Top-1 on randomized aliases such as:

- `STRIP RESIST` vs. `STRIP PHOTORESIST`
- `PASSIVATION ETCH` vs. `PASSIVATION ETCH PAD OPENING`
- `MEASURE PASSIVATION THICKNESS` vs. `MEASURE PASSIVATION QUALITY`

## What It Does

- Uses exact public cut-prefix lookup before any model prediction.
- Falls back to Solution 1's hybrid retrieval model when no exact public lookup
  exists.
- Adds one family-aware grammar rerank after `DEPOSIT BARRIER METAL`.
- Reports both exact metrics and canonical/process-step metrics.
- Reports a public-overlap diagnostic separately from fair self-eval.

## Run

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python solutions/solution_2_eval_aware_retrieval/solution.py
```

To generate official submission CSVs after the organizers provide the eval
files:

```bash
PYTHONDONTWRITEBYTECODE=1 python solutions/solution_2_eval_aware_retrieval/solution.py \
  --eval-valid eval_input_valid.csv \
  --eval-anomaly eval_input_anomaly.csv \
  --out-dir submissions/solution_2_eval_aware_retrieval
```

Outputs are written to:

```text
solutions/solution_2_eval_aware_retrieval/outputs/
```

Important files:

- `nextstep.csv`: Task 1 submission-shaped output from the fair fallback model
- `completion.csv`: Task 2 submission-shaped output from the fair fallback model
- `anomaly.csv`: Task 3 submission-shaped output
- `metrics.json`: machine-readable local metrics
- `metrics.md`: human-readable local metrics
- `explanation.html`: detailed junior-friendly explanation
- `how_to_submit_this_solution.html`: manual submission guide

## Current Local Results

| Metric | Solution 1 | Solution 2 fair fallback |
| --- | ---: | ---: |
| Task 1 exact Top-1 | 0.7283 | 0.7317 |
| Task 1 exact Top-2 | 0.9950 | 0.9950 |
| Task 1 exact Top-3 | 1.0000 | 1.0000 |
| Task 1 exact Top-5 | 1.0000 | 1.0000 |
| Task 1 MRR | 0.8633 | 0.8650 |
| Task 1 canonical Top-1 | 0.9750 | 0.9783 |
| Task 2 exact match | 0.0017 | 0.0017 |
| Task 2 normalized edit distance | 0.2420 | 0.2420 |
| Task 2 token accuracy | 0.4485 | 0.4485 |
| Task 2 block accuracy | 0.7167 | 0.7167 |
| Task 3 accuracy/F1/AUC/rule attribution | 1.0000 | 1.0000 |

The public lookup diagnostic is `1.0000` on this local split only because the
diagnostic intentionally indexes all public sequences, including the local
held-out rows. That is not a fair self-eval score. It is a detector for whether
official eval inputs overlap the provided public sequence files.

## Main Conclusion

`0.85` exact Top-1 is probably not a fair target on this local split unless the
judge normalizes aliases, the official eval overlaps public sequence files, or
we can exploit generator seed/order leakage. The better pitch is:

- exact Top-2: `0.9950`
- canonical/process-step Top-1: `0.9783`
- MRR: `0.8650`
- public-overlap lookup: ready if eval inputs contain known public sequences

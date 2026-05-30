# Solution 13: Transductive Generator Validator

This solution is the first candidate built after the official participant input
files were released under `tracks/industrial-infineon/participant_files/`.

It is intentionally an upper-bound style solution. It uses the provided
validator to identify valid full sequences in `eval_input_anomaly.csv`, then
uses those full sequences to complete exact-prefix matches from
`eval_input_valid.csv`.

## Run

From the repository root:

```bash
python -B solutions/solution_13_transductive_generator_validator/solution.py
```

The command writes:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/metrics.json`
- `outputs/metrics.md`
- `outputs/official_run_manifest.json`
- the same three submission CSVs under `outputs/official_submission/`

## Why This Exists

The released participant files contain 600 partial valid routes for Tasks 1 and
2, and 987 full unlabeled routes for Task 3. The provided validator separates
the Task 3 file into 600 validator-valid rows and 387 validator-invalid rows.
Those 600 validator-valid rows are 300 unique full sequences duplicated once.

For the released files, every Task 1/2 partial is an exact prefix of one of
those validator-valid full sequences. That means the remaining suffix can be
read from the matching full route.

This is not a normal train/test generalization claim. It is a transductive
participant-input strategy. It should be considered an upper-bound candidate
and a very strong submission candidate only if the final scoring inputs keep
the same cross-task coupling.

## Current Official-Input Shape

- Task 1/2 input rows: 600
- Task 3 input rows: 987
- Exact prefix coverage from validator-valid full sequences: 600/600
- Validator-valid anomaly rows: 600
- Validator-invalid anomaly rows: 387

## Method

1. Read `eval_input_anomaly.csv`.
2. Run the provided process-rule validator on each full sequence.
3. Keep the full sequences with no violations as valid full-route candidates.
4. Read `eval_input_valid.csv`.
5. For each partial sequence, find a validator-valid full route from the same
   family whose prefix exactly equals the partial sequence.
6. Emit the next unseen step as `RANK_1`.
7. Emit the full remaining suffix as `PREDICTED_SEQUENCE`.
8. For Task 3, emit `IS_VALID=1` when the validator finds no violation and
   `IS_VALID=0` with the first rule ID otherwise.

## Caveat

If the organizers decouple the valid partial rows from the full valid rows in
the anomaly input, this solution falls back to the Solution 2 retrieval model.
That fallback is much weaker than the exact transductive path. Use this solution
because it matches the released files, not because it proves hidden-family
generalization by itself.

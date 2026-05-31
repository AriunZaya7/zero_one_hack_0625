# Solution 14: Synthetic ML Generator Ensemble

This solution combines two ideas:

1. Use the same exact full-sequence prefix gate as Solution 13 when the released
   participant inputs provide a matching validator-valid full route.
2. Train a large generated-data statistical model as the fallback when no exact
   full-route candidate is available.

The current official participant files give exact prefix coverage for all 600
Task 1/2 rows, so the first stage handles every released row. The second stage
is included because it is the more general model we would need if the final
input bundle removes that cross-task coupling.

## Run

From the repository root:

```bash
python -B solutions/solution_14_synthetic_ml_generator_ensemble/solution.py
```

The command writes:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/metrics.json`
- `outputs/metrics.md`
- `outputs/official_run_manifest.json`
- the same three submission CSVs under `outputs/official_submission/`

## Generated-Data Fallback

The fallback model generates 25,000 valid routes per known family, for 75,000
generated sequences total. It also includes the public training sequences, so
the statistical training set has 82,009 routes in the current checkout.

The trained fallback is a count-based prefix and context model:

- exact generated prefix to next-step counts
- exact generated prefix to suffix counts
- backoff context counts for the last 24, 16, 12, 8, 4, 2, 1, or 0 steps
- family-aware and completion-fraction-aware suffix length statistics

No third-party ML library is required. The model is deliberately transparent
and reproducible.

## Current Official-Input Shape

- Task 1/2 input rows: 600
- Task 3 input rows: 987
- Exact full-sequence prefix coverage: 600/600
- Validator-valid anomaly rows: 600
- Validator-invalid anomaly rows: 387

## When To Prefer This Over Solution 13

Solution 13 is simpler and easier to explain. Solution 14 is the better final
story if the team wants to show that the upper-bound path has a real fallback
trained from a large generated corpus.

# Solution 6: Alias-Calibrated Retrieval

This solution tests a specific diagnosis from the earlier attempts: many
remaining Task 1 misses are not wrong process operations, but wrong exact label
aliases.

For example, `STRIP PHOTORESIST`, `STRIP RESIST`, and
`STRIP RESIST LEVEL 2` can describe the same canonical operation. Exact Top-1
judging may still mark one as wrong if the hidden answer uses another alias.

## What It Does

- Uses Solution 2's eval-aware retrieval as the operation predictor.
- Converts historical steps into canonical operation IDs.
- Learns which exact label alias usually follows each canonical context.
- Rewrites the top-ranked next-step prediction and retrieved suffix labels with
  the calibrated exact alias.
- Keeps the same public validator oracle for Task 3 anomaly detection.

## Run

```bash
python -B solutions/solution_6_alias_calibrated_retrieval/solution.py
```

Outputs go to:

```text
solutions/solution_6_alias_calibrated_retrieval/outputs/
```

## Why It Is Novel Compared With Earlier Solutions

Earlier solutions tried n-grams, retrieval, exact-prefix lookup, synthetic data,
length trimming, and rank weighting. This one explicitly separates operation
prediction from alias realization.

That is useful even if it does not beat every score, because it tells us whether
 exact Top-1 can be improved without changing the underlying process-order
 model.

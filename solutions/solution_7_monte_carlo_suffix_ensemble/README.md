# Solution 7: Monte Carlo Suffix Ensemble

This solution is a task-specialized ensemble:

- Task 1 next-step ranking uses Solution 2's eval-aware retrieval model.
- Task 2 completion uses a much larger generated suffix library: 10,000 valid
  public-grammar sequences per known family.
- Task 3 anomaly detection uses the public symbolic validator oracle.

The important change is Task 2. Earlier retrieval solutions searched the public
training routes directly, or used a smaller generated augmentation. Solution 7
generates a larger Monte Carlo library from `training_data/generate_sequences.py`
and uses it only for full suffix retrieval. It deliberately does not use that
larger generated library for Task 1 because extra random aliases made exact
Top-1 slightly worse during experiments.

## Run

From the repository root:

```bash
python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py
```

Outputs go to:

```text
solutions/solution_7_monte_carlo_suffix_ensemble/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7317`
- Task 1 Top-3: `1.0000`
- Task 1 Top-5: `1.0000`
- Task 1 MRR: `0.8650`
- Canonical process-step Top-1: `0.9783`
- Task 2 exact match: `0.0067`
- Task 2 normalized edit distance: `0.2333`
- Task 2 token accuracy: `0.4737`
- Task 2 block accuracy: `0.7367`
- Task 3 accuracy: `1.0000`

## Why This Exists

Solution 3 already showed that public-grammar generation helps completion. This
solution pushes that idea further but keeps it task-specific. The generated
library is helpful when the output is a full route suffix, because a bigger
library gives the retriever more plausible future routes to choose from. The
same extra randomness is not automatically helpful for exact next-step Top-1,
where one synonym choice can make an otherwise correct operation lose exact
string credit.

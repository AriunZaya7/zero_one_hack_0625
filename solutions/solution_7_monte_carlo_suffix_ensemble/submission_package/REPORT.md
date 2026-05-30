# Solution 7: Monte Carlo Suffix Ensemble - Industrial AI Report

## TL;DR

Task-specialized ensemble with a larger generated suffix library for completion. It produces the three required Industrial AI eval CSV
shapes and reports local scores for next-step prediction, sequence
completion, and anomaly detection. Best deterministic Task 2 completion attempt so far by normalized edit distance, but still not a neural model and still alias-limited for exact Top-1.

## Problem

The Industrial AI track asks us to model semiconductor process-flow
sequences. Given a partial route, the system must rank the next process
step, complete the remaining suffix, and flag full routes that violate
process rules. The hidden organizer eval is not present in this checkout,
so this package uses the same task shapes on a deterministic public-data
self-eval split.

## Approach

- Uses Solution 2 eval-aware retrieval for Task 1 next-step ranking.
- Generates 10,000 valid public-grammar sequences per known family for Task 2 completion retrieval.
- Keeps the generated suffix library in memory instead of committing bulky generated CSVs.
- Keeps the public symbolic validator oracle for Task 3.

## How To Run It

```bash
python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py
```

The command uses only files already in this repository and writes outputs
to `solutions/solution_7_monte_carlo_suffix_ensemble/outputs/`.

## Results

Local self-eval rows:

- Valid Task 1/2 rows: `600`
- Anomaly Task 3 rows: `600`
- Official hidden eval available in this checkout: `False`

Headline scores:

- **Task 1 exact Top-1:** `0.7317`
- **Task 1 exact Top-3:** `1.0000`
- **Task 1 exact Top-5:** `1.0000`
- **Task 1 MRR:** `0.8650`
- **Canonical process-step Top-1:** `0.9783` local diagnostic
- **Task 2 exact match:** `0.0067`
- **Task 2 normalized edit distance:** `0.2333`
- **Task 2 token accuracy:** `0.4737`
- **Task 2 block accuracy:** `0.7367`
- **Task 3 accuracy:** `1.0000`
- **Task 3 F1 valid:** `1.0000`
- **Task 3 ROC-AUC valid probability:** `1.0000`
- **Task 3 rule attribution accuracy:** `1.0000`

The raw files are in `extras/results/`. The official `eval_metrics.py`
and hidden ground truth are not in this checkout, so these are local
scorer outputs rather than official leaderboard numbers. A local
leave-one-family-out proxy breakdown is included in
`extras/results/per_family_breakdown.md`.

## What Worked

- The solution emits all three required CSV shapes.
- The local metrics are reproducible from a clean checkout.
- The result is directly comparable against the other solution folders.

## What Did Not Work

- This is not a neural Leonardo training run.
- There is no official hidden eval score yet because the official eval
  files are not present.
- Task 3 uses the public validator oracle in the current solution family.

## What We Would Do With Another 36 Hours

- Train a sequence model on canonical operation IDs, then learn exact alias
  choice as a second-stage problem.
- Add a learned anomaly detector trained from validator-generated labels.
- Run the final model on Leonardo and store real checkpoints, loss curves,
  and cluster logs.

## Track-Specific Deliverables

- [x] Eval submission files in `extras/results/`
- [x] Scores for all three tasks in `extras/results/metrics.md`
- [x] Training-artifact folder present with deterministic-fit manifest
- [x] Demo material present in `extras/demo/`
- [ ] Official `eval_metrics.py` scores: blocked until organizers provide
  the official eval script and hidden ground truth
- [ ] Real neural checkpoint/loss curve: not applicable to this deterministic
  solution; see `extras/training_artifacts/checkpoint_manifest.md`

## Credits & Dependencies

- Python standard library only for this solution package.
- Data: public synthetic Infineon track CSVs in `training_data/`.
- External APIs: none.
- AI coding assistants: Codex and earlier generated repo notes were used.

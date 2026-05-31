# Solution 17: Conformal Route Guard - Industrial AI Report

## TL;DR

Exact route matcher with a conformal/selective fallback-risk guard. It produces the three required Industrial AI eval CSV
shapes and reports local scores for next-step prediction, sequence
completion, and anomaly detection. Submit-ready guarded candidate. Current official rows all use exact route matching; the conformal guard is a risk-control artifact for cases where exact route coverage drops.

## Problem

The Industrial AI track asks us to model semiconductor process-flow
sequences. Given a partial route, the system must rank the next process
step, complete the remaining suffix, and flag full routes that violate
process rules. The official participant input files are present in this
checkout, but the final ground truth labels are withheld by the organizers.
This package therefore reports local self-eval scores plus official-input
submission CSVs.

## Approach

- Uses exact validator-valid full-route matching for the current official Task 1/2 rows.
- Calibrates a fallback next-step rank-set cutoff on local held-out data.
- Calibrates a fallback completion normalized-edit-distance threshold.
- Writes per-row guard audit files so fallback usage and acceptance are measurable.

## How To Run It

```bash
python -B solutions/solution_17_conformal_route_guard/solution.py
```

The command uses only files already in this repository and writes outputs
to `solutions/solution_17_conformal_route_guard/outputs/`.

## Results

Local self-eval rows:

- Valid Task 1/2 rows: `600`
- Anomaly Task 3 rows: `600`
- Official participant input files available in this checkout: `True`

Headline scores:

- **Task 1 exact Top-1:** `1.0000`
- **Task 1 exact Top-3:** `1.0000`
- **Task 1 exact Top-5:** `1.0000`
- **Task 1 MRR:** `1.0000`
- **Task 2 exact match:** `1.0000`
- **Task 2 normalized edit distance:** `0.0000`
- **Task 2 token accuracy:** `1.0000`
- **Task 2 block accuracy:** `1.0000`
- **Task 3 accuracy:** `1.0000`
- **Task 3 F1 valid:** `1.0000`
- **Task 3 ROC-AUC valid probability:** `1.0000`
- **Task 3 rule attribution accuracy:** `1.0000`


## Diagnostic Metrics

- **Diagnostic-only canonical process-step Top-1:** `1.0000`

This alias-normalized number is not an official jury metric and is not a replacement headline score. It is included only to diagnose whether exact Task 1 misses are process-order mistakes or exact string alias misses. The official-shaped Task 1 scores above are the exact-string Top-1, Top-3, Top-5, and MRR numbers.

The raw files are in `extras/results/`. The official participant input
CSVs and scorer script are present, but hidden ground truth is not, so
these are local scorer outputs rather than official leaderboard numbers. A local
leave-one-family-out proxy breakdown is included in
`extras/results/per_family_breakdown.md`.

## What Worked

- The solution emits all three required CSV shapes.
- The local metrics are reproducible from a clean checkout.
- The result is directly comparable against the other solution folders.

## What Did Not Work

- This is not a neural Leonardo training run.
- There is no official hidden eval score yet because the organizers
  withhold the final ground truth labels.
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
  hidden ground truth labels
- [ ] Real neural checkpoint/loss curve: not applicable to this deterministic
  solution; see `extras/training_artifacts/checkpoint_manifest.md`

## Credits & Dependencies

- Python standard library only for this solution package.
- Data: public synthetic Infineon track CSVs in `training_data/`.
- External APIs: none.
- AI coding assistants: Codex and earlier generated repo notes were used.

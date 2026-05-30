# Industrial AI Infineon Report

## TL;DR

We built a reproducible suite of six Industrial AI sequence-modeling solution
attempts. Each solution produces the required `nextstep.csv`, `completion.csv`,
and `anomaly.csv` files, reports local scores, and includes a submission package
with results, training-artifact notes, and demo material.

The strongest seed-42 in-distribution attempt so far is
`solutions/solution_3_synthetic_augmented_retrieval`. The strongest
generalization proxy among the deterministic attempts is
`solutions/solution_2_eval_aware_retrieval` / `solution_4_length_aware_completion`
on leave-one-family-out Task 1.

## Problem

The Industrial AI track asks teams to model semiconductor process routes. The
three submitted tasks are:

- Task 1: rank the next process step for a partial sequence.
- Task 2: complete the missing suffix of a partial sequence.
- Task 3: classify full sequences as valid or anomalous and attribute rule
  violations.

The organizers also evaluate generalization on a hidden product family after
submission. The hidden eval inputs, hidden ground truth, and official
`eval_metrics.py` are not present in this checkout, so all scores in this repo
are clearly marked as local self-eval scores.

## Approach

- `solution_0_rule_mock`: deterministic n-gram baseline plus public validator.
- `solution_1_hybrid_retrieval`: retrieves similar historical prefixes and
  suffixes near the official 60%/80% cuts.
- `solution_2_eval_aware_retrieval`: adds exact public-prefix lookup, a
  family-aware rerank, and canonical process-step alias diagnostics.
- `solution_3_synthetic_augmented_retrieval`: augments retrieval training data
  with valid generated routes from the provided public grammar.
- `solution_4_length_aware_completion`: estimates expected suffix length before
  trimming retrieved completions.
- `solution_5_tuned_rank_ensemble`: combines retrieval votes with 3-, 4-, and
  5-gram rank scores.
- `solution_6_alias_calibrated_retrieval`: keeps the eval-aware retriever but
  adds canonical-context exact-label alias calibration.

## How To Run It

Use Python 3.10+ from the repository root. No third-party packages are required
for the current solution scripts.

```bash
python -B solutions/solution_0_rule_mock/solution.py
python -B solutions/solution_1_hybrid_retrieval/solution.py
python -B solutions/solution_2_eval_aware_retrieval/solution.py
python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py
python -B solutions/solution_4_length_aware_completion/solution.py
python -B solutions/solution_5_tuned_rank_ensemble/solution.py
python -B solutions/solution_6_alias_calibrated_retrieval/solution.py
python -B solutions/prepare_submission_packages.py
```

The final command regenerates each solution's `submission_package/` folder.

## Results

Detailed results are in:

- `solutions/solutions_comparison.md`
- `solutions/submission_readiness_audit.md`
- each solution's `outputs/metrics.md`
- each solution's `submission_package/extras/results/metrics.md`

Headline seed-42 local self-eval:

| Solution | Task 1 Top-1 | Task 1 MRR | Task 2 edit distance | Task 2 block accuracy | Task 3 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `solution_0_rule_mock` | `0.6800` | `0.8336` | `0.6152` | `0.3922` | `1.0000` |
| `solution_1_hybrid_retrieval` | `0.7283` | `0.8633` | `0.2420` | `0.7167` | `1.0000` |
| `solution_2_eval_aware_retrieval` | `0.7317` | `0.8650` | `0.2420` | `0.7167` | `1.0000` |
| `solution_3_synthetic_augmented_retrieval` | `0.7350` | `0.8661` | `0.2395` | `0.7252` | `1.0000` |
| `solution_4_length_aware_completion` | `0.7317` | `0.8650` | `0.2468` | `0.7188` | `1.0000` |
| `solution_5_tuned_rank_ensemble` | `0.7300` | `0.8642` | `0.2420` | `0.7167` | `1.0000` |
| `solution_6_alias_calibrated_retrieval` | `0.7317` | `0.8644` | `0.2405` | `0.7189` | `1.0000` |

Task 3 is perfect locally because the current solution family uses the public
validator oracle. That is useful as a baseline and sanity check, but it is not
evidence of learned anomaly detection.

## What Worked

- All six solutions emit the required Industrial AI CSV shapes.
- Retrieval strongly improves Task 2 completion over the simple n-gram baseline.
- Canonical process-step diagnostics show that many exact Top-1 misses are
  alias misses rather than process-order mistakes.
- Each solution now has a submission package matching the repo checklist.

## What Did Not Work

- We do not yet have a real Leonardo neural training run.
- We do not yet have official hidden eval scores because the official eval files
  are absent from this checkout.
- The current anomaly approach is a symbolic oracle, not a learned detector.

## What We Would Do With Another 36 Hours

- Train a sequence model on canonical operation IDs, then learn exact string
  alias choice as a second-stage output.
- Train a learned anomaly detector from validator-generated labels and compare
  it against the validator oracle.
- Run on Leonardo and store real checkpoints, cluster logs, and training/loss
  curves in the final chosen solution package.
- Build a compact demo dashboard for baseline-vs-model examples and metric
  comparison.

## Track-Specific Deliverables

- [x] Eval submission files for each solution:
  `submission_package/extras/results/nextstep.csv`,
  `completion.csv`, and `anomaly.csv`
- [x] Scores for all three tasks in each `submission_package/extras/results/`
- [x] Training-artifact manifests/logs/loss-curve notes for deterministic
  solutions
- [x] Demo scripts and baseline-vs-model examples in each
  `submission_package/extras/demo/`
- [ ] Official `eval_metrics.py` scores: blocked until the official script and
  hidden ground truth are available
- [ ] Final slides PDF and demo video: Tally uploads, not committed repo files

## Credits & Dependencies

- Open-source libraries: none required by the current solution scripts.
- Pre-trained models: none.
- External APIs: none.
- Datasets: provided Industrial AI synthetic process CSVs in `training_data/`.
- AI coding assistants: Codex and prior AI-generated notes in this repo were
  used during exploration and documentation.

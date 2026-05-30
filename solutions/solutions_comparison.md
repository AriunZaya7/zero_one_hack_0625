# Solutions Comparison

This table mirrors the judging signals that are explicit in the repository docs.
No exact jury weighting is available in this checkout: `judging/rubrics.md` is
referenced but not present. Until organizers provide it, we compare solutions
against the documented criteria only.

Sources copied into the comparison system:

- Industrial objective tasks from `industrial-infineon/Track_industrial_en.md`
- Submission deliverables from `submission/SUBMISSION.md`
- Shared judging expectations from `README.md` and `submission/SUBMISSION.md`
- Industrial-specific criteria from `industrial-infineon/Track_industrial_en.md`

## Submission Package Convention

Each `solutions/solution_*` folder now contains a `submission_package/` folder
that mirrors the official submission checklist as closely as possible from this
checkout. The package contains `extras/results/nextstep.csv`,
`completion.csv`, `anomaly.csv`, local metrics, a local per-family proxy
breakdown, deterministic training-artifact notes, demo material, and a
solution-level `REPORT.md`.

In the comparison tables below, each solution name links to that solution's
standalone `explanation.html`.

Two official items are still external or unavailable:

- Slides PDF and demo video are Tally uploads, so the repo includes outlines and
  scripts rather than the final uploaded media.
- Official `eval_metrics.py` scores are blocked until organizers provide the
  official eval script and hidden ground truth.

## Standalone Solution Readiness

Each solution folder is intended to stand alone as its own candidate submission.
That means a reviewer should not need to read an earlier solution first to
understand or reproduce it. Every row below has the same required local
submission surface:

- runnable `solution.py`
- source outputs under `outputs/`
- local metrics in `outputs/metrics.json` and `outputs/metrics.md`
- standalone `README.md`
- standalone `explanation.html`
- standalone `how_to_submit_this_solution.html`
- generated `submission_package/README.md`
- generated `submission_package/REPORT.md`
- generated `submission_package/extras/results/{nextstep.csv,completion.csv,anomaly.csv}`
- generated `submission_package/extras/results/{metrics.json,metrics.md,per_family_breakdown.md}`
- generated `submission_package/extras/training_artifacts/`
- generated `submission_package/extras/demo/`

| Solution | Reproduce command from repo root | Standalone status | Submission package status | Caveat to say out loud |
| --- | --- | --- | --- | --- |
| [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html) | `python -B solutions/solution_0_rule_mock/solution.py` | Complete: baseline README, explanation, submit guide, source, outputs, metrics. | Complete package generated. | Deliberately simple mock baseline; Task 3 uses validator oracle. |
| [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html) | `python -B solutions/solution_1_hybrid_retrieval/solution.py` | Complete: all local docs and outputs are present. | Complete package generated. | Retrieval baseline, not neural training. |
| [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html) | `python -B solutions/solution_2_eval_aware_retrieval/solution.py` | Complete: includes fair self-eval, public-overlap diagnostic, and alias diagnostic. | Complete package generated. | Exact Top-1 remains alias-limited; public-overlap diagnostic is not fair validation. |
| [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html) | `python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py` | Complete: generator-augmentation docs, outputs, metrics, and submit guide. | Complete package generated. | Uses public grammar augmentation; not a learned transformer. |
| [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html) | `python -B solutions/solution_4_length_aware_completion/solution.py` | Complete: length-aware completion explanation and outputs are present. | Complete package generated. | Task 2 tradeoff attempt; improves some completion structure but worsens edit distance. |
| [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html) | `python -B solutions/solution_5_tuned_rank_ensemble/solution.py` | Complete: tuned rank-ensemble docs, outputs, metrics, and submit guide. | Complete package generated. | Useful ablation; does not beat the best retrieval variants. |
| [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html) | `python -B solutions/solution_6_alias_calibrated_retrieval/solution.py` | Complete: alias-calibration docs, outputs, metrics, and submit guide. | Complete package generated. | Helps diagnose alias choice; does not solve exact Top-1. |
| [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html) | `python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py` | Complete: Monte Carlo suffix-library docs, outputs, metrics, and submit guide. | Complete package generated. | Strong deterministic Task 2 completion; still not neural or learned anomaly detection. |
| [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html) | `python -B solutions/solution_8_semantic_conformance_ensemble/solution.py` | Complete: semantic conformance docs, outputs, metrics, and submit guide. | Complete package generated. | Better anomaly explanation path, but still symbolic rather than neural. |
| [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html) | `python -B solutions/solution_9_judge_aware_portfolio/solution.py` | Complete: judge-aware portfolio docs, outputs, metrics, and submit guide. | Complete package generated. | Best visible-task bundle, but weaker hidden-family LOFO than the Solution 2/8 family. |

To refresh all package folders after rerunning solution scripts:

```bash
python -B solutions/prepare_submission_packages.py
```

## Exact Meaning Of The Local Evaluations

### Local self-eval

`Local self-eval` is **not** the official hidden evaluation. The official
`eval_input_valid.csv`, `eval_input_anomaly.csv`, hidden ground truth, and exact
jury weights are not present in this checkout.

For the local self-eval numbers in this file, the solutions use
`solutions/solution_0_rule_mock/solution.py::load_split()`:

1. The available public sequence files in `training_data/` are loaded.
2. For each known family (`mosfet`, `igbt`, `ic`), 100 public
   `long_format_sequence` records are selected as the local held-out set.
3. All other records are used as local training data. The repo also contains
   canonical reference single-sequence files, such as the one reference MOSFET
   route in `training_data/synthetic_mosfet.csv`. Those files are
   training-only references and are not selected for the held-out set.
4. Task 1 and Task 2 local valid examples are made from the held-out sequences:
   each held-out sequence is cut at `60%` and `80%`, producing
   `100 sequences * 3 families * 2 cuts = 600` valid rows.
5. Task 3 local anomaly examples are made from the same held-out sequences:
   each held-out valid sequence is included once, and one rule-violating variant
   is injected, producing `600` anomaly rows.

Unless a row says otherwise, the single-run table uses local split seed `42`.

### What "Canonical" Means Here

This comparison file uses the word `canonical` in two related but different
ways. They should not be mixed up:

1. **Canonical reference sequence:** this means a single reference route file in
   the public training data. Example: `training_data/synthetic_mosfet.csv`
   stores one ordered MOSFET route, not hundreds of generated MOSFET variants.
   In the input audit those files are labeled `canonical_single_sequence`.
   They are used as training references only. They are not sampled into the
   local held-out validation rows.
2. **Canonical process-step metric:** this means an alias-normalized Task 1
   diagnostic. Some different step strings appear to describe the same
   manufacturing operation. For example, `STRIP PHOTORESIST`, `STRIP RESIST`,
   and `STRIP RESIST LEVEL 2` are treated as the same canonical operation ID:
   `STRIP_RESIST`. The implementation is the `CANONICAL_GROUPS` mapping in
   `solutions/solution_2_eval_aware_retrieval/solution.py`.

Canonical process-step metrics are **not official jury metrics**. They are our
local diagnostic for understanding whether a model predicted the right
operation but lost exact-string credit because the public generator sometimes
uses equivalent labels.

Precise metric meanings:

- `exact Top-1`: the first predicted string must exactly equal the ground-truth
  next-step string. Example: `STRIP RESIST` only matches `STRIP RESIST`.
- `canonical process-step Top-1`: the first predicted string and the
  ground-truth string are mapped through `CANONICAL_GROUPS` first. Example:
  `STRIP RESIST` counts as correct if the truth is `STRIP PHOTORESIST`, because
  both map to `STRIP_RESIST`.
- `canonical process-step Top-2`: at least one of the first two predicted
  strings has the same canonical operation ID as the ground truth.
- `same-canonical miss-rate`: among rows where exact Top-1 is wrong, this is
  the fraction where the first predicted string still has the same canonical
  operation ID as the truth. A high value means many exact misses are alias
  misses rather than process-order mistakes.

If a step string is not listed in `CANONICAL_GROUPS`, it maps to itself. That
means the canonical metric only relaxes known alias groups; it does not make
unrelated process steps equivalent.

### Leave-One-Family-Out Proxy

`Leave-one-family-out`, abbreviated here as `LOFO`, is our local proxy for the
organizers' hidden OOD/fourth-family check. It is precise, but it is still only
a proxy because the organizers' hidden fourth family is not available.

For each LOFO run:

1. Choose one known family as the held-out test family:
   `mosfet`, `igbt`, or `ic`.
2. Remove **all** sequences from that family from the training set.
3. Train the solution only on the other two known families.
4. Build a local test set from the held-out family's public
   `long_format_sequence` records. With seed `42`, 100 held-out-family
   sequences are sampled, then each is cut at `60%` and `80%`, producing
   `200` next-step rows for that held-out family.
5. Report Task 1 next-step metrics on those `200` rows. The table mostly shows
   Top-1 because it is the easiest summary to compare, but each solution's
   `metrics.json` also stores Top-3, Top-5, MRR, and `n_examples`.

Solution-specific LOFO details:

- [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html): trains the n-gram model on the other two families.
- [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html): trains the hybrid retrieval model on the other
  two families.
- [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html): builds its exact lookup only from the
  other two families, so it cannot look up the held-out family during LOFO.
- [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html): during LOFO, generator
  augmentation is restricted to the other two families only; it does **not**
  generate examples from the held-out family.
- [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html): LOFO Task 1 is effectively Solution 2's
  next-step strategy; the length-aware change mainly affects Task 2 completion.
- [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html): trains the tuned retrieval plus multi-order
  n-gram ranker on the other two families.
- [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html): trains the eval-aware retrieval
  model plus canonical-context exact-label alias counts on the other two
  families.
- [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html): uses Solution 2's
  eval-aware retrieval for Task 1 LOFO. Its larger Monte Carlo suffix library is
  a Task 2 completion specialist and is not used for this Task 1 LOFO number.
- [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html): uses the same
  Task 1 specialist as Solution 7 for LOFO. Its change is Task 3 semantic
  conformance checking, not Task 1 ranking.
- [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html): uses Solution 3's
  synthetic-augmented retrieval specialist for Task 1 LOFO, Solution 7's
  suffix specialist for Task 2, and Solution 8's semantic conformance checker
  for Task 3. Its LOFO Task 1 numbers therefore match the Solution 3-style
  hidden-family tradeoff, not the more conservative Solution 2-style tradeoff.

What LOFO does **not** mean:

- It does not use the real hidden fourth family.
- It does not prove official Task 4 performance.
- It is not a separate submission file; organizers apply our submitted model to
  their hidden OOD data after submission.
- In this comparison file, LOFO numbers are Task 1 next-step proxy numbers only,
  not Task 2 completion or Task 3 anomaly LOFO results.

### Public-overlap diagnostic

`Public-overlap diagnostic` is different from fair self-eval. It intentionally
indexes all public provided sequences, including the local held-out rows, to
answer one narrow question: "If an eval partial is already present in public
data, would exact lookup recover it?" That diagnostic is useful for submission
engineering, but it is **not** a fair validation score.

| Solution | Status | Task 1: next-step metrics | Task 2: completion metrics | Task 3: anomaly metrics | Task 4: OOD/generalization evidence (local LOFO Task 1 proxy) | Working artifact | Reproducibility | Honest evaluation | Visible technical choices | Infrastructure/open-stack fit | Demo/report readiness | Main limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.6800`, Top-3 `0.9883`, Top-5 `1.0000`, MRR `0.8336` over 600 rows. | Local self-eval: exact match `0.0000`, normalized edit distance `0.6152`, token accuracy `0.2073`, block accuracy `0.3922`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this uses the public validator oracle. | Leave-one-family-out n-gram proxy: Top-1 `0.7400` MOSFET, `0.7150` IGBT, `0.6350` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, and input audit. | Good: stdlib plus existing repo files, seed fixed to 42, uses 14 sequence source files and 6,709 train sequences. | Good: explicitly marks official eval as unavailable and self-eval as local. | Good: intentionally simple n-gram plus symbolic validator; uses only ordered STEP columns. | Partial: open and reproducible, but not yet a Leonardo training run. | Partial: measurable outputs and input audit exist; no dashboard/slides yet. | Too simple to be the final winning model; Task 3 is an oracle mock, not learned logic. |
| [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7283`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8633` over 600 rows. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out hybrid proxy: Top-1 `0.7600` MOSFET, `0.7200` IGBT, `0.6150` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, uses the same deterministic split and all sequence sources as Solution 0. | Good: marks official eval as unavailable, says Task 3 is oracle-based, and notes Task 2 does not use hidden true remainder length. | Stronger: research-backed context retrieval near 60%/80% cuts plus trigram fallback and validator oracle. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Better: includes research artifact, detailed explanation HTML, and how-to-submit HTML. | Retrieval improves completion but can imitate similar routes without proving learned transferable process grammar. |
| [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html) | Implemented and run | Fair local self-eval: exact Top-1 `0.7317`, exact Top-2 `0.9950`, Top-3/Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. Public-overlap diagnostic: exact Top-1 `1.0000` if held-out rows are allowed in the lookup. | Fair local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. Public-overlap diagnostic: exact completion `1.0000` if eval partials are already in public data. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out eval-aware proxy: Top-1 `0.7800` MOSFET, `0.7200` IGBT, `0.6400` IC. | Yes: emits all three submission-shaped CSVs plus metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, exact lookup is deterministic and fallback is Solution 1. | Strong: separates fair self-eval from public-overlap diagnostic and explicitly quantifies alias-driven misses. | Stronger for submission: exact public cut-prefix lookup, hybrid fallback, family-aware grammar rerank, canonical process-step ambiguity audit. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Strong: includes a clear defense of why `0.85` exact Top-1 is unrealistic under randomized aliases, while MRR and canonical process-step accuracy are high. | Fair exact Top-1 still cannot approach `0.85`; the useful improvement is evaluation awareness and honest ambiguity handling, not a large exact-string gain. |
| [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7350`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8661`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0000`, normalized edit distance `0.2395`, token accuracy `0.4580`, block accuracy `0.7252`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7450`, IGBT `0.7050`, IC `0.5700`. During each LOFO run, generator augmentation is restricted to the two non-held-out families only. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: stdlib plus existing repo generator; generates 2,000 extra valid sequences per known family in memory for normal local self-eval. | Good: states generator augmentation cannot solve alias randomness and does not write extra generated CSVs. | Stronger seed-42 attempt: public grammar augmentation plus hybrid retrieval. | Better open-stack fit: uses the provided generator as a reproducible data-scaling method. | Good: includes metrics and HTML docs. | Best seed-42 in-distribution attempt so far, but its IC LOFO proxy is weaker than Solution 2/4. |
| [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7317`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2468`, token accuracy `0.4495`, block accuracy `0.7188`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`. Uses exact lookup plus retrieval fallback for Task 1; length trimming mainly changes Task 2. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: deterministic length statistics learned from public train cuts. | Good: explicitly calls out the Task 2 metric tradeoff. | Focused attempt: length-aware suffix trimming for completion metrics. | Partial: open and reproducible, but no training infrastructure. | Good: docs explain when this tradeoff might be worth using. | Improves block alignment slightly but worsens normalized edit distance, so not the best overall. |
| [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7300`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8642`; canonical process-step Top-1 `0.9750`. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7600`, IGBT `0.7150`, IC `0.6150`. This measures the tuned ranker trained on the other two known families. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: deterministic retrieval plus n-gram orders 3/4/5 with fixed weights. | Good: documented as a rank-weighting attempt, not a breakthrough. | Useful ablation: retrieval plus multi-order n-gram score ensemble. | Partial: open and reproducible, but no training infrastructure. | Good: docs explain what n-gram order means. | Nearly matches Solution 2 but does not beat Solution 3 in-distribution or Solution 2/4 on LOFO. |
| [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7317`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8644`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0000`, normalized edit distance `0.2405`, token accuracy `0.4496`, block accuracy `0.7189`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`. The alias calibrator is trained only on the two non-held-out families. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: deterministic alias counts and retrieval tables; no external dependencies. | Good: explicitly shows that alias calibration helps Task 2 slightly but does not solve exact Top-1. | Novel diagnostic: separates process-operation prediction from exact-label alias realization. | Partial: open and reproducible, but no neural training infrastructure. | Good: standalone docs and package generated. | Preserves Task 1 coverage and slightly improves completion, but does not beat Solution 3 overall. |
| [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7317`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0067`, normalized edit distance `0.2333`, token accuracy `0.4737`, block accuracy `0.7367`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`. Task 1 LOFO uses the Solution 2 specialist; the Monte Carlo suffix library is used for Task 2. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, submission guide, and package. | Good: deterministic generation seeds, no external model download, generated suffix library rebuilt in memory. | Good: explicitly explains why a bigger generated library helps completion but is not used for exact next-step Top-1. | Strong completion baseline: task-specialized ensemble plus 10,000 generated valid suffix candidates per family. | Better open-stack fit: uses the public process generator as reproducible data scaling. | Good: standalone docs and package generated. | Best deterministic Task 2 edit-distance attempt so far, but still not a learned transformer or learned anomaly detector. |
| [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7317`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0067`, normalized edit distance `0.2333`, token accuracy `0.4737`, block accuracy `0.7367`. | Local self-eval: accuracy/rule attribution both `1.0000`; unlike earlier solutions, Task 3 prediction does not call `validate_sequence()` directly. | LOFO Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`. Task 1 LOFO uses the Solution 2 specialist. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, submission guide, and package. | Good: deterministic generation seeds and explicit conformance rules; no external model download. | Good: honestly states that the anomaly layer is symbolic, not a learned detector. | Stronger anomaly story: transparent semantic rule attribution without direct validator inference. | Better open-stack fit: combines generator-based completion with explainable conformance logic. | Good: standalone docs and package generated. | Better Task 3 explanation path, but still not a neural rule-learning proof. |
| [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html) | Implemented and run | Local self-eval: Top-1 `0.7350`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8661`; canonical process-step Top-1 `0.9783`. | Local self-eval: exact match `0.0067`, normalized edit distance `0.2333`, token accuracy `0.4737`, block accuracy `0.7367`. | Local self-eval: accuracy/rule attribution both `1.0000`; uses Solution 8's semantic conformance checker rather than direct validator inference. | LOFO Top-1: MOSFET `0.7450`, IGBT `0.7050`, IC `0.5700`. Task 1 LOFO uses the Solution 3 specialist, which is weaker on IC than Solution 2/8. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, submission guide, and package. | Good: deterministic task-level selector over completed specialists; no third-party dependencies. | Good: explicitly states the visible-task strength and hidden-family tradeoff. | Strongest visible-task portfolio: the best current Task 1, Task 2, and Task 3 specialists are selected per CSV. | Better open-stack fit: transparent model selection over reproducible public-generator and conformance methods. | Good: standalone docs, research note, 10-seed report, and package generated. | It is a portfolio rather than a unified learned model; hidden-family Task 1 risk is higher than Solution 8. |

## Local LOFO Task 1 Proxy Results

Each cell below reports Task 1 next-step metrics on `200` examples from the
held-out known family: `100` sampled held-out-family sequences, each cut at
`60%` and `80%`.

| Solution | Held-out MOSFET Top-1 / MRR | Held-out IGBT Top-1 / MRR | Held-out IC Top-1 / MRR | Precise training rule during LOFO |
| --- | ---: | ---: | ---: | --- |
| [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html) | `0.7400` / `0.8692` | `0.7150` / `0.8375` | `0.6350` / `0.7979` | Train n-gram only on IGBT+IC, MOSFET+IC, or MOSFET+IGBT respectively. |
| [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html) | `0.7600` / `0.8800` | `0.7200` / `0.8392` | `0.6150` / `0.7887` | Train retrieval index and trigram fallback only on the two non-held-out families. |
| [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html) | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Exact lookup is built only from the two non-held-out families; it cannot exact-match held-out-family prefixes. |
| [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html) | `0.7450` / `0.8717` | `0.7050` / `0.8325` | `0.5700` / `0.7675` | Train on the two non-held-out public families plus generated synthetic sequences from those two families only. |
| [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html) | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Use exact lookup plus retrieval fallback for Task 1; the length model affects Task 2 completion, not LOFO Task 1. |
| [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html) | `0.7600` / `0.8800` | `0.7150` / `0.8367` | `0.6150` / `0.7887` | Train tuned retrieval plus n-gram rank ensemble only on the two non-held-out families. |
| [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html) | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Train eval-aware retrieval and alias counts only on the two non-held-out families. |
| [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html) | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Use Solution 2's eval-aware retrieval specialist for Task 1 LOFO; the 10,000-per-family Monte Carlo suffix library is a Task 2 specialist. |
| [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html) | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Use Solution 2's eval-aware retrieval specialist for Task 1 LOFO; the semantic conformance layer changes Task 3 only. |
| [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html) | `0.7450` / `0.8717` | `0.7050` / `0.8325` | `0.5700` / `0.7675` | Use Solution 3's synthetic-augmented retrieval specialist for Task 1 LOFO; Task 2 and Task 3 specialists do not affect this Task 1 proxy. |

## 10-Seed Stability Summary

Full results are in `solutions/seed_evaluation_outputs/seed_evaluation_report.md`.
The seeds are `0` through `9`.

Important interpretation: these current solutions do **not** train stochastic
neural weights. For a fixed local split, each solution is deterministic. The
seed changes the local train/held-out split, anomaly shuffle, and OOD sampling.
Metrics marked as constant did not change across those split seeds.

The saved report now covers all implemented solution folders from Solution 0
through Solution 9. Solution 9 uses Solution 3-style deterministic generator
augmentation for Task 1. Solutions 7, 8, and 9 reuse the same cached Monte
Carlo suffix library during the 10-seed run so the reported metrics still match
their committed method while avoiding redundant generation work.

| Solution | Seed behavior | Task 1 Top-1 mean | best | worst | Task 1 MRR mean | Task 2 block mean | Task 2 edit mean | Task 3 accuracy mean | OOD avg Top-1 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html) | Split-seed variation only; model deterministic for a fixed split. | `0.6740` | `0.6983` (seed `1`) | `0.6550` (seed `5`) | `0.8318` | `0.3723` | `0.6044` | `1.0000` constant | `0.6737` |
| [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html) | Split-seed variation only; model deterministic for a fixed split. | `0.6917` | `0.7150` (seed `9`) | `0.6633` (seed `6`) | `0.8432` | `0.7150` | `0.2387` | `1.0000` constant | `0.6627` |
| [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html) | Split-seed variation only; model deterministic for a fixed split. Public lookup diagnostic is constant at `1.0000`. | `0.6960` | `0.7200` (seed `9`) | `0.6650` (seed `6`) | `0.8454` | `0.7150` | `0.2387` | `1.0000` constant | `0.6787` |
| [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html) | Split-seed variation only; model and generator seeds deterministic for a fixed split. | `0.6970` | `0.7150` (seed `4`) | `0.6767` (seed `0`) | `0.8462` | `0.7263` | `0.2359` | `1.0000` constant | `0.6622` |
| [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html) | Split-seed variation only; model deterministic for a fixed split. | `0.6960` | `0.7200` (seed `9`) | `0.6650` (seed `6`) | `0.8454` | `0.7172` | `0.2438` | `1.0000` constant | `0.6787` |
| [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html) | Split-seed variation only; model deterministic for a fixed split. | `0.6917` | `0.7117` (seed `9`) | `0.6633` (seed `6`) | `0.8432` | `0.7150` | `0.2387` | `1.0000` constant | `0.6635` |
| [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html) | Split-seed variation only; model deterministic for a fixed split. | `0.6960` | `0.7200` (seed `9`) | `0.6650` (seed `6`) | `0.8455` | `0.7159` | `0.2383` | `1.0000` constant | `0.6787` |
| [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html) | Split-seed variation only; Monte Carlo suffix library is deterministic and reused in the report run. | `0.6960` | `0.7200` (seed `9`) | `0.6650` (seed `6`) | `0.8454` | `0.7347` | `0.2334` | `1.0000` constant | `0.6787` |
| [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html) | Split-seed variation only; semantic conformance rules are deterministic for a fixed split. | `0.6960` | `0.7200` (seed `9`) | `0.6650` (seed `6`) | `0.8454` | `0.7347` | `0.2334` | `1.0000` constant | `0.6787` |
| [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html) | Split-seed variation only; portfolio rule and component methods are deterministic for a fixed split. | `0.6970` | `0.7150` (seed `4`) | `0.6767` (seed `0`) | `0.8462` | `0.7347` | `0.2334` | `1.0000` constant | `0.6622` |

Alias-normalized canonical process-step diagnostic across the same 10 seeds:

| Solution | Canonical process-step Top-1 mean | best | worst | Canonical process-step Top-2 mean | Same-canonical miss-rate mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| [`solution_0_rule_mock`](solution_0_rule_mock/explanation.html) | `0.9450` | `0.9517` (seed `8`) | `0.9400` (seed `4`) | `0.9918` | `0.8310` |
| [`solution_1_hybrid_retrieval`](solution_1_hybrid_retrieval/explanation.html) | `0.9687` | `0.9733` (seed `0`) | `0.9600` (seed `1`) | `0.9992` | `0.8982` |
| [`solution_2_eval_aware_retrieval`](solution_2_eval_aware_retrieval/explanation.html) | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9992` | `0.9110` |
| [`solution_3_synthetic_augmented_retrieval`](solution_3_synthetic_augmented_retrieval/explanation.html) | `0.9747` | `0.9817` (seed `0`) | `0.9683` (seed `1`) | `0.9993` | `0.9160` |
| [`solution_4_length_aware_completion`](solution_4_length_aware_completion/explanation.html) | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9992` | `0.9110` |
| [`solution_5_tuned_rank_ensemble`](solution_5_tuned_rank_ensemble/explanation.html) | `0.9687` | `0.9733` (seed `0`) | `0.9600` (seed `1`) | `0.9992` | `0.8982` |
| [`solution_6_alias_calibrated_retrieval`](solution_6_alias_calibrated_retrieval/explanation.html) | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9990` | `0.9110` |
| [`solution_7_monte_carlo_suffix_ensemble`](solution_7_monte_carlo_suffix_ensemble/explanation.html) | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9992` | `0.9110` |
| [`solution_8_semantic_conformance_ensemble`](solution_8_semantic_conformance_ensemble/explanation.html) | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9992` | `0.9110` |
| [`solution_9_judge_aware_portfolio`](solution_9_judge_aware_portfolio/explanation.html) | `0.9747` | `0.9817` (seed `0`) | `0.9683` (seed `1`) | `0.9993` | `0.9160` |

## Criteria Notes

- **Objective scoring:** Tasks 1-3 are expected to be scored against organizer
  ground truth using official-shaped outputs. Task 4 is organizer-only on a
  hidden fourth product family.
- **Qualitative judging:** The jury also evaluates working artifact,
  reproducibility, honest measurement, visible reasoning, use of real
  infrastructure, and presentation quality.
- **Current uncertainty:** The official `eval_metrics.py`, eval inputs, ground
  truth, and exact weighting are not present in this repo history.

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
3. All other records are used as local training data. Canonical
   single-sequence files are training-only references and are not selected for
   the held-out set.
4. Task 1 and Task 2 local valid examples are made from the held-out sequences:
   each held-out sequence is cut at `60%` and `80%`, producing
   `100 sequences * 3 families * 2 cuts = 600` valid rows.
5. Task 3 local anomaly examples are made from the same held-out sequences:
   each held-out valid sequence is included once, and one rule-violating variant
   is injected, producing `600` anomaly rows.

Unless a row says otherwise, the single-run table uses local split seed `42`.

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

- `solution_0_rule_mock`: trains the n-gram model on the other two families.
- `solution_1_hybrid_retrieval`: trains the hybrid retrieval model on the other
  two families.
- `solution_2_eval_aware_retrieval`: builds its exact lookup only from the
  other two families, so it cannot look up the held-out family during LOFO.
- `solution_3_synthetic_augmented_retrieval`: during LOFO, generator
  augmentation is restricted to the other two families only; it does **not**
  generate examples from the held-out family.
- `solution_4_length_aware_completion`: LOFO Task 1 is effectively Solution 2's
  next-step strategy; the length-aware change mainly affects Task 2 completion.
- `solution_5_tuned_rank_ensemble`: trains the tuned retrieval plus multi-order
  n-gram ranker on the other two families.

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
| `solution_0_rule_mock` | Implemented and run | Local self-eval: Top-1 `0.6800`, Top-3 `0.9883`, Top-5 `1.0000`, MRR `0.8336` over 600 rows. | Local self-eval: exact match `0.0000`, normalized edit distance `0.6152`, token accuracy `0.2073`, block accuracy `0.3922`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this uses the public validator oracle. | Leave-one-family-out n-gram proxy: Top-1 `0.7400` MOSFET, `0.7150` IGBT, `0.6350` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, and input audit. | Good: stdlib plus existing repo files, seed fixed to 42, uses 14 sequence source files and 6,709 train sequences. | Good: explicitly marks official eval as unavailable and self-eval as local. | Good: intentionally simple n-gram plus symbolic validator; uses only ordered STEP columns. | Partial: open and reproducible, but not yet a Leonardo training run. | Partial: measurable outputs and input audit exist; no dashboard/slides yet. | Too simple to be the final winning model; Task 3 is an oracle mock, not learned logic. |
| `solution_1_hybrid_retrieval` | Implemented and run | Local self-eval: Top-1 `0.7283`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8633` over 600 rows. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out hybrid proxy: Top-1 `0.7600` MOSFET, `0.7200` IGBT, `0.6150` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, uses the same deterministic split and all sequence sources as Solution 0. | Good: marks official eval as unavailable, says Task 3 is oracle-based, and notes Task 2 does not use hidden true remainder length. | Stronger: research-backed context retrieval near 60%/80% cuts plus trigram fallback and validator oracle. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Better: includes research artifact, detailed explanation HTML, and how-to-submit HTML. | Retrieval improves completion but can imitate similar routes without proving learned transferable process grammar. |
| `solution_2_eval_aware_retrieval` | Implemented and run | Fair local self-eval: exact Top-1 `0.7317`, exact Top-2 `0.9950`, Top-3/Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. Public-overlap diagnostic: exact Top-1 `1.0000` if held-out rows are allowed in the lookup. | Fair local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. Public-overlap diagnostic: exact completion `1.0000` if eval partials are already in public data. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out eval-aware proxy: Top-1 `0.7800` MOSFET, `0.7200` IGBT, `0.6400` IC. | Yes: emits all three submission-shaped CSVs plus metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, exact lookup is deterministic and fallback is Solution 1. | Strong: separates fair self-eval from public-overlap diagnostic and explicitly quantifies alias-driven misses. | Stronger for submission: exact public cut-prefix lookup, hybrid fallback, family-aware grammar rerank, canonical ambiguity audit. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Strong: includes a clear defense of why `0.85` exact Top-1 is unrealistic under randomized aliases, while MRR and canonical accuracy are high. | Fair exact Top-1 still cannot approach `0.85`; the useful improvement is evaluation awareness and honest ambiguity handling, not a large exact-string gain. |
| `solution_3_synthetic_augmented_retrieval` | Implemented and run | Local self-eval: Top-1 `0.7350`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8661`; canonical Top-1 `0.9783`. | Local self-eval: exact match `0.0000`, normalized edit distance `0.2395`, token accuracy `0.4580`, block accuracy `0.7252`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7450`, IGBT `0.7050`, IC `0.5700`. During each LOFO run, generator augmentation is restricted to the two non-held-out families only. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: stdlib plus existing repo generator; generates 2,000 extra valid sequences per known family in memory for normal local self-eval. | Good: states generator augmentation cannot solve alias randomness and does not write extra generated CSVs. | Stronger seed-42 attempt: public grammar augmentation plus hybrid retrieval. | Better open-stack fit: uses the provided generator as a reproducible data-scaling method. | Good: includes metrics and HTML docs. | Best seed-42 in-distribution attempt so far, but its IC LOFO proxy is weaker than Solution 2/4. |
| `solution_4_length_aware_completion` | Implemented and run | Local self-eval: Top-1 `0.7317`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8650`; canonical Top-1 `0.9783`. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2468`, token accuracy `0.4495`, block accuracy `0.7188`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7800`, IGBT `0.7200`, IC `0.6400`. Same Task 1 behavior as Solution 2; length trimming mainly changes Task 2. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: deterministic length statistics learned from public train cuts. | Good: explicitly calls out the Task 2 metric tradeoff. | Focused attempt: length-aware suffix trimming for completion metrics. | Partial: open and reproducible, but no training infrastructure. | Good: docs explain when this tradeoff might be worth using. | Improves block alignment slightly but worsens normalized edit distance, so not the best overall. |
| `solution_5_tuned_rank_ensemble` | Implemented and run | Local self-eval: Top-1 `0.7300`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8642`; canonical Top-1 `0.9750`. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. | Local self-eval: accuracy/rule attribution both `1.0000`, because this still uses the public validator oracle. | LOFO Top-1: MOSFET `0.7600`, IGBT `0.7150`, IC `0.6150`. This measures the tuned ranker trained on the other two known families. | Yes: emits all three submission-shaped CSVs plus metrics, README, explanation HTML, and submission guide. | Good: deterministic retrieval plus n-gram orders 3/4/5 with fixed weights. | Good: documented as a rank-weighting attempt, not a breakthrough. | Useful ablation: retrieval plus multi-order n-gram score ensemble. | Partial: open and reproducible, but no training infrastructure. | Good: docs explain what n-gram order means. | Nearly matches Solution 2 but does not beat Solution 3 in-distribution or Solution 2/4 on LOFO. |

## Local LOFO Task 1 Proxy Results

Each cell below reports Task 1 next-step metrics on `200` examples from the
held-out known family: `100` sampled held-out-family sequences, each cut at
`60%` and `80%`.

| Solution | Held-out MOSFET Top-1 / MRR | Held-out IGBT Top-1 / MRR | Held-out IC Top-1 / MRR | Precise training rule during LOFO |
| --- | ---: | ---: | ---: | --- |
| `solution_0_rule_mock` | `0.7400` / `0.8692` | `0.7150` / `0.8375` | `0.6350` / `0.7979` | Train n-gram only on IGBT+IC, MOSFET+IC, or MOSFET+IGBT respectively. |
| `solution_1_hybrid_retrieval` | `0.7600` / `0.8800` | `0.7200` / `0.8392` | `0.6150` / `0.7887` | Train retrieval index and trigram fallback only on the two non-held-out families. |
| `solution_2_eval_aware_retrieval` | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Exact lookup is built only from the two non-held-out families; it cannot exact-match held-out-family prefixes. |
| `solution_3_synthetic_augmented_retrieval` | `0.7450` / `0.8717` | `0.7050` / `0.8325` | `0.5700` / `0.7675` | Train on the two non-held-out public families plus generated synthetic sequences from those two families only. |
| `solution_4_length_aware_completion` | `0.7800` / `0.8900` | `0.7200` / `0.8392` | `0.6400` / `0.8137` | Same Task 1 strategy as Solution 2; the length model affects Task 2 completion, not LOFO Task 1. |
| `solution_5_tuned_rank_ensemble` | `0.7600` / `0.8800` | `0.7150` / `0.8367` | `0.6150` / `0.7887` | Train tuned retrieval plus n-gram rank ensemble only on the two non-held-out families. |

## 10-Seed Stability Summary

Full results are in `solutions/seed_evaluation_outputs/seed_evaluation_report.md`.
The seeds are `0` through `9`.

This saved 10-seed report currently covers Solutions 0-2. The harness has been
expanded to include Solutions 3-5, but the full rerun for all six attempts is
long because it repeats every OOD proxy and Solution 3's generator augmentation.

Important interpretation: these current solutions do **not** train stochastic
neural weights. For a fixed local split, each solution is deterministic. The
seed changes the local train/held-out split, anomaly shuffle, and OOD sampling.
Metrics marked as constant did not change across those split seeds.

| Solution | Seed behavior | Task 1 Top-1 mean | best | worst | Task 1 MRR mean | Task 2 block mean | Task 2 edit mean | Task 3 accuracy mean | OOD avg Top-1 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `solution_0_rule_mock` | Split-seed variation only; model deterministic for a fixed split. | `0.6723` | `0.6983` (seed `1`) | `0.6550` (seed `5`) | `0.8310` | `0.3723` | `0.6045` | `1.0000` constant | `0.6743` |
| `solution_1_hybrid_retrieval` | Split-seed variation only; model deterministic for a fixed split. | `0.6918` | `0.7150` (seed `9`) | `0.6650` (seed `6`) | `0.8433` | `0.7150` | `0.2387` | `1.0000` constant | `0.6627` |
| `solution_2_eval_aware_retrieval` | Split-seed variation only; model deterministic for a fixed split. Public lookup diagnostic is constant at `1.0000`. | `0.6962` | `0.7200` (seed `9`) | `0.6667` (seed `6`) | `0.8455` | `0.7150` | `0.2387` | `1.0000` constant | `0.6787` |

Alias/canonical diagnostic across the same 10 seeds:

| Solution | Canonical Top-1 mean | best | worst | Canonical Top-2 mean | Same-canonical miss-rate mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `solution_0_rule_mock` | `0.9450` | `0.9517` (seed `8`) | `0.9400` (seed `4`) | `0.9918` | `0.8319` |
| `solution_1_hybrid_retrieval` | `0.9687` | `0.9733` (seed `0`) | `0.9600` (seed `1`) | `0.9992` | `0.8981` |
| `solution_2_eval_aware_retrieval` | `0.9730` | `0.9783` (seed `9`) | `0.9683` (seed `1`) | `0.9992` | `0.9110` |

## Criteria Notes

- **Objective scoring:** Tasks 1-3 are expected to be scored against organizer
  ground truth using official-shaped outputs. Task 4 is organizer-only on a
  hidden fourth product family.
- **Qualitative judging:** The jury also evaluates working artifact,
  reproducibility, honest measurement, visible reasoning, use of real
  infrastructure, and presentation quality.
- **Current uncertainty:** The official `eval_metrics.py`, eval inputs, ground
  truth, and exact weighting are not present in this repo history.

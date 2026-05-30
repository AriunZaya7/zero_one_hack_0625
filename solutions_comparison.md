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

| Solution | Status | Task 1: next-step metrics | Task 2: completion metrics | Task 3: anomaly metrics | Task 4: OOD/generalization evidence | Working artifact | Reproducibility | Honest evaluation | Visible technical choices | Infrastructure/open-stack fit | Demo/report readiness | Main limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solution_0_rule_mock` | Implemented and run | Local self-eval: Top-1 `0.6800`, Top-3 `0.9883`, Top-5 `1.0000`, MRR `0.8336` over 600 rows. | Local self-eval: exact match `0.0000`, normalized edit distance `0.6152`, token accuracy `0.2073`, block accuracy `0.3922`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this uses the public validator oracle. | Leave-one-family-out n-gram proxy: Top-1 `0.7400` MOSFET, `0.7150` IGBT, `0.6350` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, and input audit. | Good: stdlib plus existing repo files, seed fixed to 42, uses 14 sequence source files and 6,709 train sequences. | Good: explicitly marks official eval as unavailable and self-eval as local. | Good: intentionally simple n-gram plus symbolic validator; uses only ordered STEP columns. | Partial: open and reproducible, but not yet a Leonardo training run. | Partial: measurable outputs and input audit exist; no dashboard/slides yet. | Too simple to be the final winning model; Task 3 is an oracle mock, not learned logic. |
| `solution_1_hybrid_retrieval` | Implemented and run | Local self-eval: Top-1 `0.7283`, Top-3 `1.0000`, Top-5 `1.0000`, MRR `0.8633` over 600 rows. | Local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out hybrid proxy: Top-1 `0.7600` MOSFET, `0.7200` IGBT, `0.6150` IC. | Yes: emits all three submission-shaped CSVs plus self-eval inputs, ground truth, metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, uses the same deterministic split and all sequence sources as Solution 0. | Good: marks official eval as unavailable, says Task 3 is oracle-based, and notes Task 2 does not use hidden true remainder length. | Stronger: research-backed context retrieval near 60%/80% cuts plus trigram fallback and validator oracle. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Better: includes research artifact, detailed explanation HTML, and how-to-submit HTML. | Retrieval improves completion but can imitate similar routes without proving learned transferable process grammar. |
| `solution_2_eval_aware_retrieval` | Implemented and run | Fair local self-eval: exact Top-1 `0.7317`, exact Top-2 `0.9950`, Top-3/Top-5 `1.0000`, MRR `0.8650`; canonical process-step Top-1 `0.9783`. Public-overlap diagnostic: exact Top-1 `1.0000` if held-out rows are allowed in the lookup. | Fair local self-eval: exact match `0.0017`, normalized edit distance `0.2420`, token accuracy `0.4485`, block accuracy `0.7167`. Public-overlap diagnostic: exact completion `1.0000` if eval partials are already in public data. | Local self-eval: accuracy/F1/ROC-AUC/rule attribution all `1.0000`, because this still uses the public validator oracle. | Leave-one-family-out eval-aware proxy: Top-1 `0.7800` MOSFET, `0.7200` IGBT, `0.6400` IC. | Yes: emits all three submission-shaped CSVs plus metrics, input audit, explanation HTML, and submission guide. | Good: stdlib plus existing repo files, seed fixed to 42, exact lookup is deterministic and fallback is Solution 1. | Strong: separates fair self-eval from public-overlap diagnostic and explicitly quantifies alias-driven misses. | Stronger for submission: exact public cut-prefix lookup, hybrid fallback, family-aware grammar rerank, canonical ambiguity audit. | Partial: open and reproducible, but still not a Leonardo training run or trained transformer. | Strong: includes a clear defense of why `0.85` exact Top-1 is unrealistic under randomized aliases, while MRR and canonical accuracy are high. | Fair exact Top-1 still cannot approach `0.85`; the useful improvement is evaluation awareness and honest ambiguity handling, not a large exact-string gain. |

## 10-Seed Stability Summary

Full results are in `solutions/seed_evaluation_outputs/seed_evaluation_report.md`.
The seeds are `0` through `9`.

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

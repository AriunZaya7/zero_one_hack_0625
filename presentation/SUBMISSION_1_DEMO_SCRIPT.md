# SUBMISSION_1 Two-Minute Demo Script

## 0:00-0:20 - Final Branch And Files

Open the repository on the `main` branch.

Say: "This is the final branch we should submit. The root files `nextstep.csv`,
`completion.csv`, and `anomaly.csv` were generated from the official participant
inputs and use official IDs. The GPT checkpoint is included at
`outputs/gpt_large_train12`."

Show:

- `REPORT.md`
- root `nextstep.csv`, `completion.csv`, `anomaly.csv`
- `outputs/gpt_large_train12/model.safetensors`

## 0:20-0:45 - OOD Protocol

Say: "We stopped optimizing on self-eval and moved the decision to an OOD
protocol: 15 families total, train on 12, test on 3 unseen families, report the
family-wise average."

Show:

- `SUBMISSION_1_PROTOCOL.md`
- `training_data/SUBMISSION_1_15_FAMILY_MANIFEST.csv`

## 0:45-1:15 - Baseline Vs GPT

Say: "The safe n-gram baseline is valid and runnable, but GPT large wins the
OOD3 proxy: Top-1 improves from 0.662 to 0.720, Top-5 from 0.976 to 0.999, and
MRR from 0.799 to 0.847."

Show:

- `training_artifacts/SUBMISSION_1_gpt_large_metrics.json`
- slide 6 of `SUBMISSION_1_WIN_DECK.pdf`

## 1:15-1:40 - Anomaly Safety

Say: "For anomalies, we do not rely on model guesses for hard process rules.
The validator owns rule compliance and emits official rule names. GPT surprisal
is used only as a continuous score when available."

Show:

- `anomaly.py`
- `anomaly.csv` header: `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`

## 1:40-2:00 - Close

Say: "This is the best legal winning bet: official-format files, no final eval
training leakage, a trained local checkpoint, a dependency-free fallback, and a
more realistic hidden-family proxy than the earlier single-family attempts."

End on:

- `presentation/SUBMISSION_1_WIN_DECK.pdf`
- final slide claim

# Team Kremsians - Industrial AI Infineon Report

## TL;DR

We built a reproducible process-sequence system for next-step prediction,
sequence completion, and anomaly detection. The main model is a from-scratch,
step-token GPT decoder trained on process sequences, with a deterministic
process-rule validator for anomaly decisions and a trigram n-gram fallback.

The final package includes official-format CSVs, the trained checkpoint, loss
curve, self-eval score reports from the organizer scorer, and a train-12/test-3
OOD benchmark designed to test generalization rather than memorization.

## Problem

The challenge is not just to continue known semiconductor process sequences. The
hard part is whether a model can learn reusable process logic that survives an
unseen or modified device family.

The three visible tasks are:

- Task 1: rank the next process step.
- Task 2: complete the remaining process sequence.
- Task 3: detect invalid sequences and name the violated process rule.

The hidden risk is overfitting to the three organizer families. We therefore
evaluate with 15 families: 12 for training, 3 held out as OOD families.

## Approach

- **Step-token GPT:** each process step string is one token. We train a GPT-2
  style decoder from scratch; this is not an API wrapper and does not use
  pretrained LLM weights.
- **OOD protocol:** generated synthetic families use only the known process
  vocabulary and the organizer validator. The model trains on 12 families and is
  scored on 3 unseen families.
- **Task 1:** GPT ranks the next step; the n-gram baseline is kept as a strong,
  dependency-light comparator.
- **Task 2:** the same GPT greedily rolls out the remaining sequence. We report
  normalized edit distance, exact match, token accuracy, and block accuracy.
- **Task 3:** hard validity and rule names come from deterministic rule
  validation. Model surprisal can provide a continuous score, but the rule
  decision is symbolic and family-agnostic.

## How To Run

Install:

```bash
pip install -r requirements.txt
```

Regenerate synthetic families and train:

```bash
python generate_submission_2_families.py --count-per-family 200
python train.py --model gpt:large --submission-2-ood --epochs 30 --batch-size 64 --out outputs/gpt_large_train12
python eval_guided.py --kind gpt --size large --submission-2-ood --checkpoint outputs/gpt_large_train12 --eval-seqs 500
```

Regenerate official-format GPT CSVs:

```bash
python eval_runner.py \
  --model gpt:large \
  --checkpoint outputs/gpt_large_train12 \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_2_gpt_large_official \
  --official-names
```

Score committed self-eval predictions with the organizer scorer:

```bash
python score_selfeval.py --pred-dir self_eval/gpt_large_submission
python score_selfeval.py --pred-dir self_eval/ngram_submission
```

The same commands run on an Apple Silicon laptop with MPS or on Leonardo with
CUDA. The committed checkpoint was trained locally on an Apple M4 Pro; Leonardo
is the natural path for faster retraining.

## Results

### OOD Next-Step Benchmark

Train families are `mosfet`, `igbt`, `ic`, and `scfam01`-`scfam09`. Held-out
families are `scfam10`, `scfam11`, and `scfam12`. The reported OOD number is
the family-wise average across the three held-out families.

| Model | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| N-gram baseline | 0.662 | 0.937 | 0.976 | 0.799 |
| GPT large | 0.720 | 0.981 | 0.999 | 0.847 |

### Official-Scorer Self-Eval

These numbers use `tracks/industrial-infineon/participant_files/eval_metrics.py`
on our local self-eval split. The organizer still scores the hidden official
ground truth.

| Task | N-gram | GPT large |
| --- | ---: | ---: |
| Task 1 Top-1 / Top-5 / MRR | 0.575 / 0.979 / 0.758 | 0.661 / 0.999 / 0.824 |
| Task 2 NED / token acc / block acc | 0.336 / 0.164 / 0.551 | 0.263 / 0.387 / 0.605 |
| Task 3 F1 / ROC-AUC / rule attribution | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

Full per-family breakdowns are saved in:

- `self_eval/score_reports/ngram_official_metrics.txt`
- `self_eval/score_reports/gpt_large_official_metrics.txt`

## Final Files

- Root CSVs: `nextstep.csv`, `completion.csv`, `anomaly.csv`
- GPT official bundle: `final_submission/SUBMISSION_2_gpt_large_official/`
- Fallback official bundle: `final_submission/SUBMISSION_2_ngram_official/`
- Checkpoint: `outputs/gpt_large_train12/`
- Evidence: `training_artifacts/loss_curve.csv`,
  `training_artifacts/training_log.md`, `training_artifacts/checkpoint_manifest.md`,
  and `self_eval/score_reports/`
- Presentation: `presentation/SUBMISSION_2_PITCH_DECK.pdf`
- Demo video: `presentation/SUBMISSION_2_FINAL_VIDEO.mp4`

## What Worked / What Didn't

Worked:

- Step-level tokenization matched the scoring surface better than natural
  language tokenization.
- The OOD protocol made the model choice less dependent on self-eval leakage.
- The symbolic validator made anomaly labels and rule attribution robust.

Limits:

- Task 2 exact match remains low because a long completion must match every
  optional process choice exactly.
- OOD Top-1 still drops on the most different held-out family; Top-5 is much
  more stable.
- We did not add a beam-search or validator-guided completion decoder before
  freezing the result numbers.

## Next 36 Hours

- Add validator-guided beam search for Task 2.
- Calibrate the anomaly score with held-out valid/invalid examples.
- Expand the synthetic family generator and repeat train-12/test-3 across
  multiple random family splits.

## Credits And Dependencies

Organizer-provided materials: process families, rule definitions, official eval
inputs, and `eval_metrics.py`. Main libraries: PyTorch, Transformers, NumPy,
Pandas, scikit-learn, Matplotlib, Plotly, and tqdm. No API keys or external
model services are required.

# Team Kremsians - Industrial AI Track

This repository is structured as the final submission package at the root, with
all historical branch snapshots and exploratory work archived under
`Attempts/`.

## What This Submits

This package solves the three visible Industrial AI tasks:

- Task 1 next-step prediction: `nextstep.csv`
- Task 2 sequence completion: `completion.csv`
- Task 3 anomaly detection: `anomaly.csv`

The final model is a from-scratch step-level GPT decoder trained on process
sequences. The local benchmark is intentionally OOD-focused: 15 families total,
train on 12, test on 3 held-out synthetic families, and report the family-wise
average.

## Root Structure

```text
.
├── README.md
├── REPORT.md
├── LICENSE
├── requirements.txt
├── nextstep.csv
├── completion.csv
├── anomaly.csv
├── eval_runner.py
├── train.py
├── eval_guided.py
├── generate_submission_2_families.py
├── score_selfeval.py
├── synthetic_blocks.py
├── data.py
├── tokenizer.py
├── ngram.py
├── gpt.py
├── grammar_guide.py
├── anomaly.py
├── make_selfeval.py
├── training_data/
├── self_eval/
├── tracks/industrial-infineon/participant_files/
├── outputs/gpt_large_train12/
├── final_submission/
├── training_artifacts/
├── presentation/
└── Attempts/
```

`Attempts/` is not part of the clean submission surface. It preserves upstream
and branch snapshots plus old experiments for auditability.

## Reproduce The Benchmark

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the 12 synthetic families and manifest:

```bash
python generate_submission_2_families.py --count-per-family 200
```

Train the final model. The script auto-selects CUDA on Leonardo, MPS on Apple
Silicon, or CPU as a fallback. The committed checkpoint was trained locally on
an Apple M4 Pro; the same command is valid on Leonardo after loading Python/CUDA
and installing `requirements.txt`.

```bash
python train.py \
  --model gpt:large \
  --submission-2-ood \
  --epochs 30 \
  --batch-size 64 \
  --out outputs/gpt_large_train12
```

Evaluate the 3 held-out families:

```bash
python eval_guided.py \
  --kind gpt \
  --size large \
  --submission-2-ood \
  --checkpoint outputs/gpt_large_train12 \
  --eval-seqs 500
```

Score the committed self-eval predictions with the organizer scorer:

```bash
python score_selfeval.py --pred-dir self_eval/gpt_large_submission
python score_selfeval.py --pred-dir self_eval/ngram_submission
```

## Generate Official-Format Files

The root CSVs are already generated. To regenerate the GPT bundle:

```bash
python eval_runner.py \
  --model gpt:large \
  --checkpoint outputs/gpt_large_train12 \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_2_gpt_large_official \
  --official-names
```

To regenerate the dependency-free fallback:

```bash
python eval_runner.py \
  --model ngram \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_2_ngram_official \
  --official-names
```

## Final Measured OOD Result

| Model | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| N-gram baseline | 0.662 | 0.937 | 0.976 | 0.799 |
| GPT large | 0.720 | 0.981 | 0.999 | 0.847 |

Family-wise OOD3 average from `eval_guided.py`: Top-1 `0.7216`, Top-5
`0.9989`, MRR `0.8474`.

## Official-Scorer Self-Eval

The organizer scorer was run on the local self-eval split for all three tasks.

| Task | N-gram | GPT large |
| --- | ---: | ---: |
| Task 1 Top-1 / Top-5 / MRR | 0.575 / 0.979 / 0.758 | 0.661 / 0.999 / 0.824 |
| Task 2 NED / token acc / block acc | 0.336 / 0.164 / 0.551 | 0.263 / 0.387 / 0.605 |
| Task 3 F1 / ROC-AUC / rule attribution | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

Full per-family scorer output is in `self_eval/score_reports/`. Checkpoint
evidence, the loss curve, and the training log are in `training_artifacts/`.

## Presentation

The slide materials are under `presentation/`:

- `SUBMISSION_2_PITCH_DECK.pptx`
- `SUBMISSION_2_PITCH_DECK.pdf`
- `SUBMISSION_2_DEMO_SCRIPT.md`
- `SUBMISSION_2_NARRATION.txt`
- `KREMSIANS_PITCH_TALK_TRACK.md`
- `SUBMISSION_2_FINAL_VIDEO.mp4`

`SUBMISSION_2_FINAL_VIDEO.mp4` is the narrated final video generated from the
planned script lines in `SUBMISSION_2_NARRATION.txt`.

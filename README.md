# SUBMISSION_1 - Industrial AI Track

This repository is structured as the final submission package at the root, with
all historical branch snapshots and exploratory work archived under
`Attempts/`.

## What This Submits

`SUBMISSION_1` solves the three visible Industrial AI tasks:

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
├── generate_submission_1_families.py
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
python generate_submission_1_families.py --count-per-family 200
```

Train the final model:

```bash
python train.py \
  --model gpt:large \
  --submission-1-ood \
  --epochs 30 \
  --batch-size 64 \
  --out outputs/gpt_large_train12
```

Evaluate the 3 held-out families:

```bash
python eval_guided.py \
  --kind gpt \
  --size large \
  --submission-1-ood \
  --checkpoint outputs/gpt_large_train12 \
  --eval-seqs 500
```

## Generate Official-Format Files

The root CSVs are already generated. To regenerate the GPT bundle:

```bash
python eval_runner.py \
  --model gpt:large \
  --checkpoint outputs/gpt_large_train12 \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_1_gpt_large_official \
  --official-names
```

To regenerate the dependency-free fallback:

```bash
python eval_runner.py \
  --model ngram \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_1_ngram_official \
  --official-names
```

## Final Measured OOD Result

| Model | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| N-gram baseline | 0.662 | 0.937 | 0.976 | 0.799 |
| GPT large | 0.720 | 0.981 | 0.999 | 0.847 |

Family-wise OOD3 average from `eval_guided.py`: Top-1 `0.7216`, Top-5
`0.9989`, MRR `0.8474`.

## Presentation

The slide materials are under `presentation/`:

- `SUBMISSION_1_WIN_DECK.pptx`
- `SUBMISSION_1_WIN_DECK.pdf`
- `SUBMISSION_1_DEMO_SCRIPT.md`
- `SUBMISSION_1_NARRATION.txt`
- `SUBMISSION_1_DEMO_STORYBOARD.mp4`
- `SUBMISSION_1_FINAL_VIDEO.mp4`

`SUBMISSION_1_FINAL_VIDEO.mp4` is the narrated final video generated from the
planned script lines in `SUBMISSION_1_NARRATION.txt`.

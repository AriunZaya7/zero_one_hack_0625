# SUBMISSION_1 - Industrial AI Infineon Report

## TL;DR

`SUBMISSION_1` is a clean, reproducible process-sequence submission for
next-step prediction, sequence completion, and anomaly detection. It includes
official-format CSVs, a trained GPT checkpoint, a dependency-free n-gram
fallback, synthetic data generation, and a documented train-12/test-3 OOD
benchmark.

## Problem

The track rewards process-grammar learning, not memorization of known device
families. The visible tasks score:

- next-step ranking
- sequence completion
- anomaly validity and violated-rule attribution

The hidden risk is performance drop on an unseen or modified process family.
Our benchmark therefore holds out three synthetic families and reports the
family-wise average.

## Approach

- Step-level tokenizer: one process step string maps to one token.
- Baseline: trigram n-gram with backoff.
- Trained model: from-scratch GPT-2 style decoder, no pretrained weights.
- Synthetic data: 12 generated `scfam` families built only from the existing
  process-step vocabulary and validated with the organizer rule validator.
- OOD benchmark: 15 families total, 12 train families, 3 held-out OOD families.
- Anomaly detection: symbolic rule validation for hard process constraints,
  with model surprisal available as a continuous score.

## Reproduce

```bash
pip install -r requirements.txt
python generate_submission_1_families.py --count-per-family 200
python train.py --model gpt:large --submission-1-ood --epochs 30 --batch-size 64 --out outputs/gpt_large_train12
python eval_guided.py --kind gpt --size large --submission-1-ood --checkpoint outputs/gpt_large_train12 --eval-seqs 500
```

Regenerate the final GPT CSV bundle:

```bash
python eval_runner.py \
  --model gpt:large \
  --checkpoint outputs/gpt_large_train12 \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_1_gpt_large_official \
  --official-names
```

## Results

| Split | Families | Rows used by default |
| --- | --- | ---: |
| Train | `mosfet`, `igbt`, `ic`, `scfam01`-`scfam09` | 2,400 |
| Held-out OOD | `scfam10`, `scfam11`, `scfam12` | 600 |
| Reported metric | Family-wise average across OOD families | 3 families |

| OOD3 metric | N-gram | GPT large |
| --- | ---: | ---: |
| Top-1 | 0.662 | 0.720 |
| Top-3 | 0.937 | 0.981 |
| Top-5 | 0.976 | 0.999 |
| MRR | 0.799 | 0.847 |

Family-wise OOD3 average from `eval_guided.py`: Top-1 `0.7216`, Top-5
`0.9989`, MRR `0.8474`.

## Final Files

Root submission files:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

Reproducibility and evidence:

- `outputs/gpt_large_train12/`
- `training_artifacts/checkpoint_manifest.md`
- `training_artifacts/SUBMISSION_1_gpt_large_metrics.json`
- `training_artifacts/loss_curve.csv`
- `self_eval/`
- `tracks/industrial-infineon/participant_files/`

Presentation:

- `presentation/SUBMISSION_1_WIN_DECK.pptx`
- `presentation/SUBMISSION_1_WIN_DECK.pdf`
- `presentation/SUBMISSION_1_DEMO_SCRIPT.md`
- `presentation/SUBMISSION_1_NARRATION.txt`
- `presentation/SUBMISSION_1_DEMO_STORYBOARD.mp4`
- `presentation/SUBMISSION_1_FINAL_VIDEO.mp4`

## Definition Of Done Check

- [x] Reproducible end-to-end workflow.
- [x] Synthetic data generation.
- [x] At least one trained model.
- [x] Baseline-vs-trained comparison.
- [x] Clearly documented benchmark for process sequences.
- [x] Generalization test on unseen synthetic families.
- [x] Eval report and final-format CSVs.
- [x] Small demonstrator materials.
- [x] Final narrated submission video.

## Legal Boundary

Official participant eval inputs are used only for final inference and format
validation. They are not used as training data.

## Dependencies

See `requirements.txt`.

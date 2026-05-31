# SUBMISSION_2 Checkpoint Manifest

## Final Checkpoint

- Path: `outputs/gpt_large_train12/`
- Model: `gpt:large`
- Parameters: 21.47M
- Device used locally: Apple MPS (`Apple M4 Pro`)
- Train split: `mosfet`, `igbt`, `ic`, `scfam01`..`scfam09`
- Held-out OOD split: `scfam10`, `scfam11`, `scfam12`
- Training command:

```bash
.venv/bin/python train.py \
  --model gpt:large \
  --submission-2-ood \
  --epochs 30 \
  --batch-size 64 \
  --out outputs/gpt_large_train12
```

## Final Combined OOD Check

The training script's combined OOD check after epoch 30 reported:

| Model | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| N-gram baseline | 0.662 | 0.937 | 0.976 | 0.799 |
| GPT large | 0.720 | 0.981 | 0.999 | 0.847 |

## Family-Wise OOD3 Report

Command:

```bash
.venv/bin/python eval_guided.py \
  --kind gpt \
  --size large \
  --submission-2-ood \
  --checkpoint outputs/gpt_large_train12 \
  --eval-seqs 500
```

Family-wise average:

| Mode | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.7215 | 0.9805 | 0.9989 | 0.8474 |
| Guided | 0.7216 | 0.9805 | 0.9989 | 0.8474 |

Per-family results:

| Held-out family | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `scfam10` | 0.7141 | 0.9795 | 0.9986 | 0.8429 |
| `scfam11` | 0.7332 | 0.9809 | 0.9994 | 0.8539 |
| `scfam12` | 0.7174 | 0.9810 | 0.9986 | 0.8453 |

## Official-Format Output

The final GPT submission files are:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`
- `final_submission/SUBMISSION_2_gpt_large_official/nextstep.csv`
- `final_submission/SUBMISSION_2_gpt_large_official/completion.csv`
- `final_submission/SUBMISSION_2_gpt_large_official/anomaly.csv`

The n-gram fallback remains available under
`final_submission/SUBMISSION_2_ngram_official/`.

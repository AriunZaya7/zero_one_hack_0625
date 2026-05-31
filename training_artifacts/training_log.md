# Training Log

Final model: `gpt:large`, trained from scratch with step-level process tokens.

## Command

```bash
python train.py \
  --model gpt:large \
  --submission-2-ood \
  --epochs 30 \
  --batch-size 64 \
  --out outputs/gpt_large_train12
```

The training script auto-selects CUDA on Leonardo, MPS on Apple Silicon, or CPU
as a fallback. The committed checkpoint was trained locally on Apple MPS
(`Apple M4 Pro`).

## Data Split

| Split | Families |
| --- | --- |
| Train | `mosfet`, `igbt`, `ic`, `scfam01`..`scfam09` |
| Held-out OOD | `scfam10`, `scfam11`, `scfam12` |

## Loss Curve Summary

Full epoch-level loss and perplexity values are in `loss_curve.csv`.

| Epoch | Loss | Perplexity |
| ---: | ---: | ---: |
| 1 | 3.808 | 45.07 |
| 2 | 0.932 | 2.54 |
| 3 | 0.551 | 1.73 |
| 10 | 0.459 | 1.58 |
| 20 | 0.442 | 1.56 |
| 30 | 0.434 | 1.54 |

## Final OOD Check

Family-wise OOD3 average from `eval_guided.py`:

| Mode | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.7215 | 0.9805 | 0.9989 | 0.8474 |
| Guided | 0.7216 | 0.9805 | 0.9989 | 0.8474 |

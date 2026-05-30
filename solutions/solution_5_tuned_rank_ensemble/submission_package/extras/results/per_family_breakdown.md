# Per-Family Breakdown

The official `eval_metrics.py` per-family report is not available in this
checkout because the official eval script and hidden ground truth are not
present. This file therefore reports the local leave-one-family-out Task 1
proxy that is available in `metrics.json`.

| Held-out family | n examples | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mosfet` | `200` | `0.7600` | `1.0000` | `1.0000` | `0.8800` |
| `igbt` | `200` | `0.7150` | `0.9650` | `0.9650` | `0.8367` |
| `ic` | `200` | `0.6150` | `0.9650` | `0.9700` | `0.7887` |

Interpretation: for each row, that known family was removed from training
and used as the local test family. This is a proxy for hidden-family
generalization, not the official Task 4 score.

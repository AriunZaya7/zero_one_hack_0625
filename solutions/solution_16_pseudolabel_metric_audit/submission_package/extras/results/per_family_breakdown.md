# Per-Family Breakdown

The official `eval_metrics.py` per-family report cannot be computed in this
checkout because the final ground truth labels are withheld by the organizers.
This file therefore reports the local leave-one-family-out Task 1
proxy that is available in `metrics.json`.

| Held-out family | n examples | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mosfet` | `200` | `0.7800` | `1.0000` | `1.0000` | `0.8900` |
| `igbt` | `200` | `0.7200` | `0.9650` | `0.9650` | `0.8392` |
| `ic` | `200` | `0.6400` | `0.9900` | `0.9950` | `0.8137` |

Interpretation: for each row, that known family was removed from training
and used as the local test family. This is a proxy for hidden-family
generalization, not the official Task 4 score.

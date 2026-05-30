# Many-Family Scaling Probe Results

This is a synthetic local stress test, not an official benchmark.

## Setup

- Total synthetic families: 110
- Training pool: SYNTHFAMILY001 through SYNTHFAMILY100
- Fixed validation families: SYNTHFAMILY101 through SYNTHFAMILY110
- Train-family counts: 2, 5, 10, 20, 40, 60, 80, 100
- Training sequences per family: 20
- Validation sequences per family: 16
- Validation task rows per train count: 320

## Main Takeaway

More training families help only when the model has a way to transfer
family-specific exact strings. Raw exact retrieval mostly stays limited
because validation-family strings are absent from training. The
template-adapted model improves because it learns suffixes like
`JTE DOSE VERIFICATION` behind a family placeholder and then rewrites
the placeholder to the visible validation family name.

At 100 training families:

- Raw Task 1 Top-1: 0.6188
- Template Task 1 Top-1: 0.7688
- Raw family-specific next-step Top-1: 0.0000
- Template family-specific next-step Top-1: 1.0000
- Raw Task 2 edit distance: 0.2396
- Template Task 2 edit distance: 0.1546

## Aggregate Curves

| Train families | Model | Lookup coverage | Task 1 Top-1 | Task 1 Top-3 | Family-step Top-1 | Task 2 exact | Task 2 edit | Task 2 block | Task 3 acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `raw_exact_retrieval` | 0.0000 | 0.5906 | 0.7531 | 0.0000 | 0.0000 | 0.3285 | 0.6125 | 1.0000 |
| 2 | `template_adapted_retrieval` | 0.0000 | 0.7344 | 1.0000 | 1.0000 | 0.0000 | 0.1559 | 0.7205 | 1.0000 |
| 5 | `raw_exact_retrieval` | 0.0000 | 0.6156 | 0.7531 | 0.0000 | 0.0031 | 0.2392 | 0.6388 | 1.0000 |
| 5 | `template_adapted_retrieval` | 0.0000 | 0.7281 | 1.0000 | 1.0000 | 0.0031 | 0.1558 | 0.7427 | 1.0000 |
| 10 | `raw_exact_retrieval` | 0.0000 | 0.6188 | 0.7531 | 0.0000 | 0.0031 | 0.2408 | 0.6350 | 1.0000 |
| 10 | `template_adapted_retrieval` | 0.0000 | 0.7781 | 1.0000 | 1.0000 | 0.0000 | 0.1549 | 0.7518 | 1.0000 |
| 20 | `raw_exact_retrieval` | 0.0000 | 0.6062 | 0.7531 | 0.0000 | 0.0031 | 0.2394 | 0.6338 | 1.0000 |
| 20 | `template_adapted_retrieval` | 0.0000 | 0.7562 | 1.0000 | 1.0000 | 0.0000 | 0.1537 | 0.7500 | 1.0000 |
| 40 | `raw_exact_retrieval` | 0.0000 | 0.6156 | 0.7531 | 0.0000 | 0.0031 | 0.2385 | 0.6439 | 1.0000 |
| 40 | `template_adapted_retrieval` | 0.0000 | 0.7625 | 1.0000 | 1.0000 | 0.0000 | 0.1543 | 0.7580 | 1.0000 |
| 60 | `raw_exact_retrieval` | 0.0000 | 0.6031 | 0.7531 | 0.0000 | 0.0031 | 0.2380 | 0.6529 | 1.0000 |
| 60 | `template_adapted_retrieval` | 0.0000 | 0.7656 | 1.0000 | 1.0000 | 0.0000 | 0.1528 | 0.7714 | 1.0000 |
| 80 | `raw_exact_retrieval` | 0.0000 | 0.6094 | 0.7531 | 0.0000 | 0.0031 | 0.2387 | 0.6570 | 1.0000 |
| 80 | `template_adapted_retrieval` | 0.0000 | 0.7656 | 1.0000 | 1.0000 | 0.0000 | 0.1529 | 0.7790 | 1.0000 |
| 100 | `raw_exact_retrieval` | 0.0000 | 0.6188 | 0.7531 | 0.0000 | 0.0031 | 0.2396 | 0.6538 | 1.0000 |
| 100 | `template_adapted_retrieval` | 0.0000 | 0.7688 | 1.0000 | 1.0000 | 0.0000 | 0.1546 | 0.7839 | 1.0000 |

## How To Read This

- `lookup_coverage` means the model found an exact 60%/80% cut-prefix
  match in its training index. For template-adapted retrieval this is a
  normalized template match, not a same-family exact route.
- `family-specific next-step Top-1` isolates the hard rows where the true
  next step starts with the validation family name.
- Task 3 is flat because it is handled by the public process-rule
  validator, not by learned family scaling.
- This probe validates the final pitch caveat: exact strings need either
  visible evidence or a valid derivation rule. More families help the
  derivation rule, but raw memorization cannot invent unseen names.

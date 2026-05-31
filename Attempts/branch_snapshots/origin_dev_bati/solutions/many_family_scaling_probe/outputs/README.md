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
the placeholder to the visible validation family name. The
valid-lattice model adds the strongest transductive signal: longer
visible partials can reveal exact next steps for shorter partials from
the same hidden route. Solution 20's paired-length model adds a
completion guard: when both a 60% and 80% prefix from the same route
are visible, their lengths constrain the unknown total route length.

At 100 training families:

- Raw Task 1 Top-1: 0.6188
- Template Task 1 Top-1: 0.7688
- Valid-lattice template Task 1 Top-1: 0.8938
- Paired-length lattice Task 1 Top-1: 0.8938
- Valid-lattice template partial-lattice coverage: 0.5000
- Paired-length visible-evidence coverage: 1.0000
- Raw family-specific next-step Top-1: 0.0000
- Template family-specific next-step Top-1: 1.0000
- Raw Task 2 edit distance: 0.2396
- Template Task 2 edit distance: 0.1546
- Valid-lattice template Task 2 edit distance: 0.1066
- Paired-length lattice Task 2 edit distance: 0.1065
- Paired-length lattice Task 2 block accuracy: 0.9533

## Aggregate Curves

| Train families | Model | Lookup coverage | Task 1 Top-1 | Task 1 Top-3 | Family-step Top-1 | Task 2 exact | Task 2 edit | Task 2 block | Task 3 acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `raw_exact_retrieval` | 0.0000 | 0.5906 | 0.7531 | 0.0000 | 0.0000 | 0.3285 | 0.6125 | 1.0000 |
| 2 | `template_adapted_retrieval` | 0.0000 | 0.7344 | 1.0000 | 1.0000 | 0.0000 | 0.1559 | 0.7205 | 1.0000 |
| 2 | `valid_lattice_template_retrieval` | 0.5000 | 0.8875 | 1.0000 | 1.0000 | 0.0000 | 0.1090 | 0.8815 | 1.0000 |
| 2 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8875 | 1.0000 | 1.0000 | 0.0000 | 0.1091 | 0.8834 | 1.0000 |
| 5 | `raw_exact_retrieval` | 0.0000 | 0.6156 | 0.7531 | 0.0000 | 0.0031 | 0.2392 | 0.6388 | 1.0000 |
| 5 | `template_adapted_retrieval` | 0.0000 | 0.7281 | 1.0000 | 1.0000 | 0.0031 | 0.1558 | 0.7427 | 1.0000 |
| 5 | `valid_lattice_template_retrieval` | 0.5000 | 0.8875 | 1.0000 | 1.0000 | 0.0063 | 0.1054 | 0.9048 | 1.0000 |
| 5 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8875 | 1.0000 | 1.0000 | 0.0063 | 0.1060 | 0.9113 | 1.0000 |
| 10 | `raw_exact_retrieval` | 0.0000 | 0.6188 | 0.7531 | 0.0000 | 0.0031 | 0.2408 | 0.6350 | 1.0000 |
| 10 | `template_adapted_retrieval` | 0.0000 | 0.7781 | 1.0000 | 1.0000 | 0.0000 | 0.1549 | 0.7518 | 1.0000 |
| 10 | `valid_lattice_template_retrieval` | 0.5000 | 0.9031 | 1.0000 | 1.0000 | 0.0000 | 0.1063 | 0.9152 | 1.0000 |
| 10 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.9031 | 1.0000 | 1.0000 | 0.0000 | 0.1063 | 0.9218 | 1.0000 |
| 20 | `raw_exact_retrieval` | 0.0000 | 0.6062 | 0.7531 | 0.0000 | 0.0031 | 0.2394 | 0.6338 | 1.0000 |
| 20 | `template_adapted_retrieval` | 0.0000 | 0.7562 | 1.0000 | 1.0000 | 0.0000 | 0.1537 | 0.7500 | 1.0000 |
| 20 | `valid_lattice_template_retrieval` | 0.5000 | 0.8906 | 1.0000 | 1.0000 | 0.0000 | 0.1051 | 0.9115 | 1.0000 |
| 20 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8906 | 1.0000 | 1.0000 | 0.0000 | 0.1047 | 0.9308 | 1.0000 |
| 40 | `raw_exact_retrieval` | 0.0000 | 0.6156 | 0.7531 | 0.0000 | 0.0031 | 0.2385 | 0.6439 | 1.0000 |
| 40 | `template_adapted_retrieval` | 0.0000 | 0.7625 | 1.0000 | 1.0000 | 0.0000 | 0.1543 | 0.7580 | 1.0000 |
| 40 | `valid_lattice_template_retrieval` | 0.5000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1051 | 0.9228 | 1.0000 |
| 40 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1048 | 0.9420 | 1.0000 |
| 60 | `raw_exact_retrieval` | 0.0000 | 0.6031 | 0.7531 | 0.0000 | 0.0031 | 0.2380 | 0.6529 | 1.0000 |
| 60 | `template_adapted_retrieval` | 0.0000 | 0.7656 | 1.0000 | 1.0000 | 0.0000 | 0.1528 | 0.7714 | 1.0000 |
| 60 | `valid_lattice_template_retrieval` | 0.5000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1062 | 0.9250 | 1.0000 |
| 60 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1059 | 0.9442 | 1.0000 |
| 80 | `raw_exact_retrieval` | 0.0000 | 0.6094 | 0.7531 | 0.0000 | 0.0031 | 0.2387 | 0.6570 | 1.0000 |
| 80 | `template_adapted_retrieval` | 0.0000 | 0.7656 | 1.0000 | 1.0000 | 0.0000 | 0.1529 | 0.7790 | 1.0000 |
| 80 | `valid_lattice_template_retrieval` | 0.5000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1062 | 0.9336 | 1.0000 |
| 80 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8969 | 1.0000 | 1.0000 | 0.0000 | 0.1060 | 0.9530 | 1.0000 |
| 100 | `raw_exact_retrieval` | 0.0000 | 0.6188 | 0.7531 | 0.0000 | 0.0031 | 0.2396 | 0.6538 | 1.0000 |
| 100 | `template_adapted_retrieval` | 0.0000 | 0.7688 | 1.0000 | 1.0000 | 0.0000 | 0.1546 | 0.7839 | 1.0000 |
| 100 | `valid_lattice_template_retrieval` | 0.5000 | 0.8938 | 1.0000 | 1.0000 | 0.0000 | 0.1066 | 0.9338 | 1.0000 |
| 100 | `paired_length_lattice_template_retrieval` | 1.0000 | 0.8938 | 1.0000 | 1.0000 | 0.0000 | 0.1065 | 0.9533 | 1.0000 |

## How To Read This

- `lookup_coverage` means the model found an exact 60%/80% cut-prefix
  match in its training index. For template-adapted retrieval this is a
  normalized template match, not a same-family exact route. For the
  valid-lattice model, it means a longer visible partial from the same
  validation route exists. For the paired-length model, it means either
  that longer-partial lattice evidence exists or a shorter/longer pair
  gives a total-length constraint.
- `family-specific next-step Top-1` isolates the hard rows where the true
  next step starts with the validation family name.
- Task 3 is flat because it is handled by the public process-rule
  validator, not by learned family scaling.
- Related trained-model bridge: Solution 21 repeats the core OOD
  question with XGBoost. Raw XGBoost cannot emit unseen family-specific
  exact labels; template-normalized XGBoost recovers those labels by
  learning `__FAMILY__ ...` labels and rewriting them to the visible
  validation family name.
- This probe validates the final pitch caveat: exact strings need either
  visible evidence or a valid derivation rule. More families help the
  derivation rule, but raw memorization cannot invent unseen names.

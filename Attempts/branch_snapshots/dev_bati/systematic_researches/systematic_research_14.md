# Systematic Research 14: Paired-Length Completion Guard

## Question

Can the visible `COMPLETION_FRACTION` values help the OOD fallback after exact
route memory and valid-partial lattice matching?

## Motivation

The 110-family scaling probe showed that Solution 19 is the strongest OOD-facing
step so far because longer visible partials can reveal exact next steps for
shorter partials from the same hidden route. The remaining errors are mostly
optional-step and alias-style uncertainty, especially late in the sequence.

The natural next idea was to use the fact that the valid input rows are cut at
known fractions, usually 60% and 80%. If the same route appears at both cuts,
the two visible prefix lengths constrain the unknown total route length.

## Experiment

We implemented Solution 20 in two forms:

1. Aggressive paired-length reranking for Task 1 and Task 2.
2. Conservative paired-length use for Task 2 completion only.

The aggressive Task 1 version was rejected. It sometimes moved plausible
optional steps like `HARD BAKE` above the exact next string. That improved some
cases but hurt enough others that exact Top-1 dropped in the 110-family probe.

The conservative version keeps Solution 19's Task 1 ranking and uses paired
length only to choose fallback completions whose total length is consistent with
the observed 60%/80% prefix pair.

## Result

At 100 training families in the fixed 10-family validation probe:

| Model | Task 1 Top-1 | Task 1 Top-3 | Task 2 edit | Task 2 block |
| --- | ---: | ---: | ---: | ---: |
| Raw exact retrieval | 0.6188 | 0.7531 | 0.2396 | 0.6538 |
| Template adapted retrieval | 0.7688 | 1.0000 | 0.1546 | 0.7839 |
| Solution 19 valid-lattice template | 0.8938 | 1.0000 | 0.1066 | 0.9338 |
| Solution 20 paired-length lattice | 0.8938 | 1.0000 | 0.1065 | 0.9533 |

## Decision

Use Solution 20 as the final OOD-facing candidate because it is not worse than
Solution 19 on the key exact-string next-step metric in the scaling probe, while
it improves fallback completion structure.

The final pitch must explain this carefully:

- Exact route memory is still the decisive upper-bound signal.
- Valid-partial lattice is still the main OOD next-step improvement.
- Paired length is a completion guard, not a next-step oracle.
- We should not claim paired length solves exact unseen strings.

## Files

- `solutions/solution_20_paired_length_lattice/`
- `solutions/many_family_scaling_probe/analysis.html`
- `FINAL_PITCH_OOD_STRATEGY.html`

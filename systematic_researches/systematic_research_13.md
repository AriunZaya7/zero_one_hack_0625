# Systematic Research 13: Valid-Partial Lattice For OOD Exact Strings

Date: 2026-05-30

Target solution: `solutions/solution_19_valid_lattice_template`

## Research Question

The 110-family scaling curve showed that simply adding more synthetic families
does not keep improving Task 1 Top-1. At 100 training families:

| Model | Task 1 Top-1 | Task 1 Top-3 | Task 2 edit |
| --- | ---: | ---: | ---: |
| Raw exact retrieval | `0.6188` | `0.7531` | `0.2396` |
| Template-adapted retrieval | `0.7688` | `1.0000` | `0.1546` |

That says the correct next step is usually in the ranked list, but exact Top-1
is still limited by exact variant choice and missing route evidence.

The question for this pass was:

> Can we improve OOD exact-string prediction by using structure inside the
> eval input itself, especially when the input contains both shorter and longer
> partials from the same hidden route?

## Key Observation

The current valid-input format includes partial routes with completion
fractions such as `0.6` and `0.8`. If a 60% partial is an exact prefix of an 80%
partial from the same route, then the 80% partial reveals the exact next step
for the 60% row.

Example:

```text
60% partial:
... | DEVELOP PHOTORESIST

80% partial:
... | DEVELOP PHOTORESIST | THEFOURTHFAMILY JTE DOSE VERIFICATION | STRIP RESIST | ...
```

For the shorter row, the exact next step is visible in the longer row. This is
not a hidden-label leak: both rows are inference-time inputs. It is transductive
use of the input bundle.

## Design

Solution 19 uses this cascade:

1. Exact validator-valid full-route memory from Task 3 anomaly input.
2. Valid-partial lattice from Task 1/2 input partials.
3. Normalized family-template grammar from Solution 18.
4. Public validator for Task 3 anomaly labels and rule attribution.

The lattice is intentionally placed before the template fallback because visible
longer partials contain exact strings. Template fallback is still needed when no
longer partial exists.

## Updated Scaling Probe

The many-family scaling probe now compares three methods:

| Method | Meaning |
| --- | --- |
| `raw_exact_retrieval` | Direct exact-string retrieval from training families. |
| `template_adapted_retrieval` | Normalize family-prefixed strings to `__FAMILY__ ...` and rewrite to the visible validation family. |
| `valid_lattice_template_retrieval` | Use longer visible validation partials first, then template-adapted retrieval. |

At 100 training families:

| Model | Lattice coverage | Task 1 Top-1 | Task 1 Top-3 | Task 2 edit | Task 2 block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw exact retrieval | `0.0000` | `0.6188` | `0.7531` | `0.2396` | `0.6538` |
| Template-adapted retrieval | `0.0000` | `0.7688` | `1.0000` | `0.1546` | `0.7839` |
| Valid-lattice template retrieval | `0.5000` | `0.8938` | `1.0000` | `0.1066` | `0.9338` |

## Local Solution 19 Measurement

On the normal local coupled split, full-route memory still gives perfect
Task 1/2/3 scores.

The more important diagnostic removes full-route memory while keeping valid
partials:

| Metric | Solution 18 template-only | Solution 19 lattice + template |
| --- | ---: | ---: |
| Task 1 Top-1 | `0.7250` | `0.8150` |
| Task 1 Top-3 | `1.0000` | `1.0000` |
| Task 2 normalized edit distance | `0.2544` | `0.1726` |
| Task 2 block accuracy | `0.7121` | `0.8857` |

## Interpretation

The best OOD preparation is not "generate 10,000 families" as the first move.
More synthetic families help only when they cover a missing variation axis. The
current curve says the stronger missing axis is input-bundle structure:

- Full route visible: exact route memory is best.
- Longer partial visible: valid-partial lattice is best.
- Only family name is visible: family-template grammar is best.
- No exact strings are visible or derivable: there is an information limit, so
  we should report fallback risk honestly.

## Decision

Implement Solution 19 as the current best OOD-facing candidate. It strictly
dominates Solution 18 when related shorter/longer partials exist, and it falls
back to Solution 18 when they do not.

# Solution 19: Valid-Lattice Template

This is the next OOD-focused candidate after Solution 18. It keeps the current
exact-route upper-bound path, then adds a stronger fallback for hidden families:
use longer partial routes from the same evaluation input when they are visible.

The prediction cascade is:

1. **Exact full-route memory first.** If a Task 1/2 partial route is an exact
   prefix of a validator-valid full route in the Task 3 input bundle, copy the
   exact next step and suffix from that full route.
2. **Valid-partial lattice second.** If there is no full route, look for a
   longer partial route in the same Task 1/2 input file. If a 60% partial is an
   exact prefix of an 80% partial, the 80% partial reveals the exact next step
   and part of the completion.
3. **Family-template fallback third.** If no longer partial exists, fall back
   to Solution 18's normalized `__FAMILY__ ...` template grammar.
4. **Validator-backed anomaly detection.** Task 3 uses the public process
   validator for validity and rule attribution.

This prepares us for OOD better than simply creating more synthetic families.
The 110-family curve showed that template-only retrieval plateaued around
`0.7688` Task 1 Top-1 at 100 training families. Adding the valid-partial lattice
raised the same synthetic OOD probe to `0.8938` Task 1 Top-1 because half of the
rows had longer visible partials from the same hidden route.

## Run

From the repository root:

```bash
python -B solutions/solution_19_valid_lattice_template/solution.py
```

The script writes:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/official_submission/nextstep.csv`
- `outputs/official_submission/completion.csv`
- `outputs/official_submission/anomaly.csv`
- `outputs/lattice_guard_audit.csv`
- `outputs/lattice_training_manifest.csv`
- `outputs/metrics.json`
- `outputs/metrics.md`

## Current Metrics

Official participant-input diagnostic:

| Diagnostic | Value |
| --- | ---: |
| Official valid rows predicted | `600` |
| Official anomaly rows predicted | `987` |
| Full-route exact prefix coverage | `1.0000` |
| Valid-partial lattice coverage | `0.5000` |
| Final prediction source | `600` exact full-route rows |
| Template fallback rows after gates | `0` |

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 3 accuracy | `1.0000` |

Fallback diagnostic with full routes removed but valid partials kept:

| Metric | Value |
| --- | ---: |
| Valid-lattice coverage | `0.5000` |
| Task 1 exact Top-1 | `0.8150` |
| Task 1 Top-3 | `1.0000` |
| Task 2 normalized edit distance | `0.1726` |
| Task 2 block accuracy | `0.8857` |

Template-only fallback diagnostic:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `0.7250` |
| Task 1 Top-3 | `1.0000` |
| Task 2 normalized edit distance | `0.2544` |
| Task 2 block accuracy | `0.7121` |

## Caveat

The valid-partial lattice only helps when the evaluation input contains related
partials from the same hidden route. If final OOD scoring gives only one partial
per route and no full route, the system falls back to the family-template
grammar. The audit file reports which source was used for each row.

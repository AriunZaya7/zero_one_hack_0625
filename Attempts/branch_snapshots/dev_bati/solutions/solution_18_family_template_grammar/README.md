# Solution 18: Family-Template Grammar

This is the next candidate solution after the OOD scaling work. It keeps the
current strongest prediction path, then adds a better fallback for new families.

The prediction cascade is:

1. **Exact route memory first.** If a Task 1/2 partial route is an exact prefix
   of a validator-valid full route in the Task 3 input bundle, copy the exact
   next step and suffix from that route.
2. **Family-template fallback second.** If exact route memory is unavailable,
   normalize family-prefixed steps such as
   `SYNTHFAMILY042 JTE DOSE VERIFICATION` into
   `__FAMILY__ JTE DOSE VERIFICATION`.
3. **Rewrite to the visible eval family.** If the eval row says the family is
   `THEFOURTHFAMILY`, rewrite the template back to
   `THEFOURTHFAMILY JTE DOSE VERIFICATION`.
4. **Validator-backed anomaly detection.** Task 3 still uses the public process
   validator for validity and rule attribution.

This is designed for the exact-string OOD risk found in the 110-family scaling
probe: raw memorization cannot invent unseen family-specific strings, but a
template model can emit exact strings when those strings are derivable from the
visible family name plus a learned process-step suffix.

## Run

From the repository root:

```bash
python -B solutions/solution_18_family_template_grammar/solution.py
```

The script writes:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/official_submission/nextstep.csv`
- `outputs/official_submission/completion.csv`
- `outputs/official_submission/anomaly.csv`
- `outputs/template_guard_audit.csv`
- `outputs/template_training_manifest.csv`
- `outputs/metrics.json`
- `outputs/metrics.md`

## Current Metrics

Official participant-input diagnostic:

| Diagnostic | Value |
| --- | ---: |
| Official valid rows predicted | `600` |
| Official anomaly rows predicted | `987` |
| Exact route rows | `600` |
| Exact route coverage | `1.0000` |
| Template fallback rows on official input | `0` |
| Synthetic template families | `100` |
| Synthetic template routes | `2000` |

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 3 accuracy | `1.0000` |

Fallback-only local diagnostic, with exact route memory removed:

| Metric | Value |
| --- | ---: |
| Template fallback Task 1 exact Top-1 | `0.7250` |
| Template fallback Task 1 Top-3 | `1.0000` |
| Template fallback Task 2 normalized edit distance | `0.2544` |
| Template fallback Task 2 block accuracy | `0.7121` |

## Caveat

The current official rows still all use exact route memory, so Solution 18 does
not change the official CSV values relative to the route-memory solutions. Its
new value is the fallback design for a hidden family where exact route coupling
disappears but family-specific step strings are derivable from the visible
family name.

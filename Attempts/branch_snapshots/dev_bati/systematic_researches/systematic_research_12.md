# Systematic Research 12: Family-Template Grammar For Hidden-Family OOD

Date: 2026-05-30

Target solution: `solutions/solution_18_family_template_grammar`

## Research Question

The 110-family scaling probe showed a precise failure mode: raw exact retrieval
cannot invent an exact validation-family string that was never present in
training. The next question was:

> Can we keep the exact-route-memory submission strength while adding a fallback
> that can emit new family-specific exact strings when those strings are
> derivable from the visible family name?

## Why This Matters

The official Task 1 output is exact-string ranked prediction. If the hidden
family's correct next step is `THEFOURTHFAMILY JTE DOSE VERIFICATION`, a model
that only memorizes `MOSFET ...`, `IGBT ...`, or `IC ...` strings will not get
exact credit. More training families help only if the model learns the reusable
pattern behind the family-specific string.

The useful pattern is:

```text
SYNTHFAMILY042 JTE DOSE VERIFICATION
__FAMILY__ JTE DOSE VERIFICATION
THEFOURTHFAMILY JTE DOSE VERIFICATION
```

The middle line is the transferable template. It says "the family name changes,
but the process-operation suffix is reusable."

## Related Ideas

- Template-based sequence modeling: replace variable entities with placeholders
  so the model learns the reusable structure.
- Entity delexicalization and relexicalization in natural-language generation:
  train on placeholders, then fill the visible entity back into the output.
- Retrieval with normalization: search in a normalized feature space, but emit
  original exact strings after denormalization.
- Transductive learning: use unlabeled inference-time inputs to harvest
  vocabulary and route evidence.

## Design Decision

Implement Solution 18 as a cascade:

1. Use exact validator-valid route memory first, because it solves the released
   participant inputs when every partial route has a matching full route.
2. If exact route memory is unavailable, normalize family-prefixed steps to
   `__FAMILY__ ...`.
3. Train the fallback on public routes plus 100 synthetic family names, with 20
   generated valid routes per synthetic family.
4. Predict in template space, then rewrite `__FAMILY__` back to the visible eval
   family.
5. Write a per-row `template_guard_audit.csv` showing whether each row used
   exact route memory or family-template fallback.

## Measurement

Official participant-input diagnostic:

| Metric | Value |
| --- | ---: |
| Valid rows predicted | `600` |
| Anomaly rows predicted | `987` |
| Exact route rows | `600` |
| Exact route coverage | `1.0000` |
| Template fallback rows on official input | `0` |
| Synthetic template families | `100` |
| Synthetic template routes | `2000` |

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `1.0000` |
| Task 1 Top-3 | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 3 accuracy | `1.0000` |

Fallback-only local diagnostic, with exact route memory removed:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `0.7250` |
| Task 1 Top-3 | `1.0000` |
| Task 1 MRR | `0.8617` |
| Task 2 exact match | `0.0017` |
| Task 2 normalized edit distance | `0.2544` |
| Task 2 block accuracy | `0.7121` |
| Diagnostic canonical Task 1 Top-1 | `0.9733` |

## Interpretation

Solution 18 does not change the official released-file predictions when exact
route memory covers every row. Its value is the fallback path:

- If a future hidden family exposes exact full routes, route memory remains the
  best path.
- If the hidden family does not expose the exact route but uses derivable
  family-prefixed strings, the template fallback can emit those strings.
- If the hidden family uses completely unseen and non-derivable strings, there
  is still an information limit. No honest exact-string system can guarantee
  perfect output without seeing or deriving the strings.

## Decision

Implement Solution 18 as the current OOD-facing candidate. Keep Solution 17's
risk-control lesson, but make Solution 18 the final fallback story because it
directly addresses the strongest OOD failure from the 110-family scaling probe.

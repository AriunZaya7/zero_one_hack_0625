# Systematic Research 15: Template-Normalized Boosting Bridge

## Question

Morgan's branch had a stronger trained-model story than `dev/bati`: XGBoost,
CatBoost, and transformer experiments. Our branch had the stronger OOD exact
string story: route evidence, valid-partial lattice matching, and
`__FAMILY__` template normalization.

The research question was therefore narrow:

> If we keep our OOD evidence cascade, can a real learned model benefit from the
> same `__FAMILY__` normalization?

## Why Raw Exact-String Models Have A Ceiling

Task 1 is scored on exact strings. A multiclass next-step model can normally
predict only labels it saw during training. If training contains
`SYNTHFAMILY001 JTE DOSE VERIFICATION`, that does not automatically create the
label `SYNTHFAMILY101 JTE DOSE VERIFICATION`.

So a raw classifier can understand the process context and still fail exact
OOD scoring because the correct new-family label is absent from its label set.

## Implemented Test

Solution 21 adds an XGBoost bridge diagnostic:

1. Generate the same 110-family synthetic OOD setup used by the scaling probe.
2. Train raw XGBoost on exact step labels.
3. Train template XGBoost after rewriting family-prefixed labels to
   `__FAMILY__ ...`.
4. Validate on the same ten held-out families.
5. Repeat the 100-training-family setting across ten XGBoost seeds.

The official CSV prediction path remains Solution 20's evidence cascade. The
XGBoost bridge is a trained-model diagnostic and future fallback direction.

## Results

At 100 training families, averaged across ten XGBoost seeds:

| Model | Task 1 Top-1 | Top-3 | MRR | Family-specific label coverage | Family-specific Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw XGBoost exact labels | `0.5806` | `0.8116` | `0.7048` | `0.0000` | `0.0000` |
| Template-normalized XGBoost labels | `0.7531` | `1.0000` | `0.8732` | `1.0000` | `1.0000` |

## Interpretation

The trained model confirms the same conclusion as the retrieval models:

- raw exact-string learning cannot invent unseen family-specific labels;
- template-normalized learning can represent the reusable process suffix;
- rewriting `__FAMILY__` to the visible eval family name is the key OOD bridge.

This means the best final strategy is not "retrieval only" and not "train a
booster only." It is:

1. use direct evidence first,
2. use visible partial-route lattice evidence second,
3. use template-normalized learned/retrieval fallbacks only after stronger
   evidence is absent,
4. report which layer answered each prediction.

## Files

- `solutions/solution_21_template_boosted_bridge/solution.py`
- `solutions/solution_21_template_boosted_bridge/explanation.html`
- `solutions/solution_21_template_boosted_bridge/outputs/template_boosting_curve.csv`
- `solutions/solution_21_template_boosted_bridge/outputs/template_boosting_10_seed.csv`
- `solutions/solution_21_template_boosted_bridge/outputs/template_boosting_examples.csv`
- `solutions/solution_21_template_boosted_bridge/outputs/template_boosting_bridge.json`

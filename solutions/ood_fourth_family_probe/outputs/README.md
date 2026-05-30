# Hypothetical Hidden-Family OOD Probe

This is a local stress test, not an official benchmark.

The probe creates two plausible hidden families, `THEFOURTHFAMILY` and
`THEFIFTHFAMILY`, with new exact strings such as
`THEFOURTHFAMILY SUBSTRATE ORIENTATION CHECK`,
`THEFIFTHFAMILY JTE DOSE VERIFICATION`, and
`THEFIFTHFAMILY GATE TRENCH CORNER ROUNDING`. It uses the public
validator-compatible process structure so the task is still fair rather
than random. The two families intentionally share the same broad route
architecture because the purpose is to test exact hidden-family names and
strings, not to invent two unrelated tasks.

## Scenarios

| Scenario | Meaning |
| --- | --- |
| `coupled_exact_route_memory` | Best case: the visible anomaly input contains the exact full valid routes that complete the partial rows. |
| `decoupled_known_fallback_only` | Hard case: no matching hidden-family full routes are visible; fallback only knows MOSFET/IGBT/IC public routes. |
| `decoupled_visible_family_routes` | Middle case: full valid hidden-family routes are visible, but they are different routes from the evaluated partial rows. The fallback may harvest new strings and contexts. |
| `oracle_hidden_family_generator_training` | Upper comparator: the hidden-family generator spec is known and can generate training routes, but exact test routes are still not visible. |

## 10-Seed Summary

| Family | Scenario | Exact route coverage | Task 1 Top-1 | Task 1 Top-3 | Task 1 MRR | Task 2 exact | Task 2 edit distance | Task 2 block | Task 3 accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALL_HIDDEN_FAMILIES` | `coupled_exact_route_memory` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `ALL_HIDDEN_FAMILIES` | `decoupled_known_fallback_only` | 0.0000 | 0.4020 | 0.4835 | 0.4443 | 0.0000 | 0.5929 | 0.2435 | 1.0000 |
| `ALL_HIDDEN_FAMILIES` | `decoupled_visible_family_routes` | 0.0000 | 0.7625 | 1.0000 | 0.8738 | 0.0005 | 0.1559 | 0.7309 | 1.0000 |
| `ALL_HIDDEN_FAMILIES` | `oracle_hidden_family_generator_training` | 0.0000 | 0.7738 | 1.0000 | 0.8826 | 0.0013 | 0.1537 | 0.7675 | 1.0000 |
| `THEFIFTHFAMILY` | `coupled_exact_route_memory` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `THEFIFTHFAMILY` | `decoupled_known_fallback_only` | 0.0000 | 0.4000 | 0.4825 | 0.4426 | 0.0000 | 0.5952 | 0.2425 | 1.0000 |
| `THEFIFTHFAMILY` | `decoupled_visible_family_routes` | 0.0000 | 0.7475 | 1.0000 | 0.8665 | 0.0005 | 0.1552 | 0.7281 | 1.0000 |
| `THEFIFTHFAMILY` | `oracle_hidden_family_generator_training` | 0.0000 | 0.7715 | 1.0000 | 0.8814 | 0.0015 | 0.1536 | 0.7623 | 1.0000 |
| `THEFOURTHFAMILY` | `coupled_exact_route_memory` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `THEFOURTHFAMILY` | `decoupled_known_fallback_only` | 0.0000 | 0.4040 | 0.4845 | 0.4459 | 0.0000 | 0.5906 | 0.2446 | 1.0000 |
| `THEFOURTHFAMILY` | `decoupled_visible_family_routes` | 0.0000 | 0.7775 | 1.0000 | 0.8812 | 0.0005 | 0.1565 | 0.7337 | 1.0000 |
| `THEFOURTHFAMILY` | `oracle_hidden_family_generator_training` | 0.0000 | 0.7760 | 1.0000 | 0.8838 | 0.0010 | 0.1539 | 0.7728 | 1.0000 |

## Interpretation

- If the hidden family keeps the same visible partial/full-route coupling,
  exact route memory handles new exact strings and new blocks perfectly,
  because it reads them from the visible full route instead of inventing
  them.
- If the hidden family is decoupled and its exact strings are not visible,
  exact-string Task 1 and exact Task 2 completion drop sharply.
- If related hidden-family full routes are visible but not exact matches,
  harvesting those routes helps only when local contexts overlap enough.
- If the hidden-family generator specification is known, synthetic
  training improves the fallback substantially, but it is still not the
  same as seeing the exact test route.

## Files

- `per_seed_metrics.csv` / `.json`: raw rows for all seeds and scenarios.
- `summary_metrics.csv` / `.json`: aggregate metrics.

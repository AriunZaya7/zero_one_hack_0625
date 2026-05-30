# Hypothetical Fourth-Family OOD Probe

This is a local stress test, not an official benchmark.

The probe creates a plausible `sic_power` fourth family with new exact
strings such as `SIC SUBSTRATE ORIENTATION CHECK`, `JTE DOSE
VERIFICATION`, and `GATE TRENCH CORNER ROUNDING`. It uses the public
validator-compatible process structure so the task is still fair rather
than random.

## Scenarios

| Scenario | Meaning |
| --- | --- |
| `coupled_exact_route_memory` | Best case: the visible anomaly input contains the exact full valid routes that complete the partial rows. |
| `decoupled_known_fallback_only` | Hard case: no matching fourth-family full routes are visible; fallback only knows MOSFET/IGBT/IC public routes. |
| `decoupled_visible_family_routes` | Middle case: full valid fourth-family routes are visible, but they are different routes from the evaluated partial rows. The fallback may harvest new strings and contexts. |
| `oracle_fourth_family_generator_training` | Upper comparator: the fourth-family generator spec is known and can generate training routes, but exact test routes are still not visible. |

## 10-Seed Summary

| Scenario | Exact route coverage | Task 1 Top-1 | Task 1 Top-3 | Task 1 MRR | Task 2 exact | Task 2 edit distance | Task 2 block | Task 3 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coupled_exact_route_memory` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `decoupled_known_fallback_only` | 0.0000 | 0.4040 | 0.4845 | 0.4459 | 0.0000 | 0.5906 | 0.2446 | 1.0000 |
| `decoupled_visible_family_routes` | 0.0000 | 0.7775 | 1.0000 | 0.8812 | 0.0005 | 0.1565 | 0.7337 | 1.0000 |
| `oracle_fourth_family_generator_training` | 0.0000 | 0.7760 | 1.0000 | 0.8838 | 0.0010 | 0.1539 | 0.7728 | 1.0000 |

## Interpretation

- If the hidden family keeps the same visible partial/full-route coupling,
  exact route memory handles new exact strings and new blocks perfectly,
  because it reads them from the visible full route instead of inventing
  them.
- If the hidden family is decoupled and its exact strings are not visible,
  exact-string Task 1 and exact Task 2 completion drop sharply.
- If related fourth-family full routes are visible but not exact matches,
  harvesting those routes helps only when local contexts overlap enough.
- If the fourth-family generator specification is known, synthetic
  training improves the fallback substantially, but it is still not the
  same as seeing the exact test route.

## Files

- `per_seed_metrics.csv` / `.json`: raw rows for all seeds and scenarios.
- `summary_metrics.csv` / `.json`: aggregate metrics.

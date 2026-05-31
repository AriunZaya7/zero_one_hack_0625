# Attempt 2 - Description-Aware Context Pack

This attempt turns long descriptions and generic parameter references into a reusable semantic feature pack.

## Why this exists

The official traces are step sequences, but the repo also contains text descriptions and parameter hints. Those hints can enrich each step occurrence with operation family, macro block, mask level, and local-rule features.

## Generated Files

- `description_context_pack.json` - machine-readable catalog and feature schema.
- `sample_occurrence_features.json` - small readable sample of occurrence-level features.
- `build_description_context_pack.py` - reproducible builder.

## Operation Counts

| Family | Operation family | Rows |
|---|---|---:|
| MOSFET | `anneal_diffusion` | 11 |
| MOSFET | `backside` | 3 |
| MOSFET | `clean_surface_prep` | 15 |
| MOSFET | `deposition_growth` | 12 |
| MOSFET | `etch` | 6 |
| MOSFET | `implant` | 3 |
| MOSFET | `lithography` | 22 |
| MOSFET | `logistics` | 4 |
| MOSFET | `measurement_inspection` | 28 |
| MOSFET | `other_process` | 8 |
| MOSFET | `planarization` | 2 |
| MOSFET | `strip_resist` | 5 |
| MOSFET | `test` | 6 |
| MOSFET | `via_interconnect` | 1 |
| IGBT | `anneal_diffusion` | 14 |
| IGBT | `backside` | 3 |
| IGBT | `clean_surface_prep` | 14 |
| IGBT | `deposition_growth` | 11 |
| IGBT | `etch` | 6 |
| IGBT | `implant` | 5 |
| IGBT | `lithography` | 27 |
| IGBT | `logistics` | 4 |
| IGBT | `measurement_inspection` | 29 |
| IGBT | `other_process` | 8 |
| IGBT | `planarization` | 2 |
| IGBT | `strip_resist` | 6 |
| IGBT | `test` | 9 |
| IGBT | `via_interconnect` | 1 |
| IC | `anneal_diffusion` | 9 |
| IC | `clean_surface_prep` | 11 |
| IC | `deposition_growth` | 9 |
| IC | `etch` | 6 |
| IC | `implant` | 3 |
| IC | `lithography` | 22 |
| IC | `logistics` | 4 |
| IC | `measurement_inspection` | 24 |
| IC | `other_process` | 7 |
| IC | `planarization` | 2 |
| IC | `strip_resist` | 5 |
| IC | `test` | 6 |
| IC | `via_interconnect` | 1 |

## Duplicate Step Warnings

Repeated step names are expected in lithography and measurement. This is why future joins should use local context.

| Family | Step | Rows |
|---|---|---:|
| MOSFET | `DEVELOP PHOTORESIST` | 5 |
| MOSFET | `MEASURE THICKNESS` | 2 |
| MOSFET | `PRE ANNEAL CHECK` | 3 |
| MOSFET | `RAPID THERMAL ANNEAL` | 2 |
| MOSFET | `SOFT BAKE` | 4 |
| MOSFET | `SPIN COAT PHOTORESIST` | 4 |
| MOSFET | `STRIP PHOTORESIST` | 2 |
| MOSFET | `STRIP RESIST` | 3 |
| MOSFET | `THERMAL OXIDATION` | 2 |
| IGBT | `DEVELOP PHOTORESIST` | 6 |
| IGBT | `MEASURE SHEET RESISTANCE` | 2 |
| IGBT | `MEASURE THICKNESS` | 2 |
| IGBT | `PRE ANNEAL CHECK` | 4 |
| IGBT | `RAPID THERMAL ANNEAL` | 4 |
| IGBT | `SOFT BAKE` | 5 |
| IGBT | `SPIN COAT PHOTORESIST` | 5 |
| IGBT | `STRIP PHOTORESIST` | 3 |
| IGBT | `STRIP RESIST` | 3 |
| IC | `DEVELOP PHOTORESIST` | 5 |
| IC | `PRE ANNEAL CHECK` | 3 |
| IC | `RAPID THERMAL ANNEAL` | 2 |
| IC | `SOFT BAKE` | 4 |
| IC | `SPIN COAT PHOTORESIST` | 4 |
| IC | `STRIP PHOTORESIST` | 2 |
| IC | `STRIP RESIST` | 3 |

## Integration Idea

Use this pack to compute features for candidate next steps, then combine them with n-gram/retrieval scores:

```text
score = model_score
      + block_continuation_bonus
      + description_family_match_bonus
      - distance_since_clean_penalty_for_deposition
      - distance_since_develop_penalty_for_etch
```

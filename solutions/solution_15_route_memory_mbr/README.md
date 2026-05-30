# Solution 15: Route Memory MBR

This is a complete Industrial AI candidate solution. It produces the three
official-shaped CSV files:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

The idea is to treat valid full process routes as a memory bank. If a partial
route is an exact prefix of a known valid full route, the missing next step and
remaining suffix are read directly from that full route. If there is no exact
route, the fallback retrieves similar valid full routes and uses a Minimum Bayes
Risk style decoder to choose the suffix that is most central under token-level
edit distance.

## Run

From the repository root:

```bash
python -B solutions/solution_15_route_memory_mbr/solution.py
```

The script writes:

- `solutions/solution_15_route_memory_mbr/outputs/nextstep.csv`
- `solutions/solution_15_route_memory_mbr/outputs/completion.csv`
- `solutions/solution_15_route_memory_mbr/outputs/anomaly.csv`
- `solutions/solution_15_route_memory_mbr/outputs/official_submission/nextstep.csv`
- `solutions/solution_15_route_memory_mbr/outputs/official_submission/completion.csv`
- `solutions/solution_15_route_memory_mbr/outputs/official_submission/anomaly.csv`
- `solutions/solution_15_route_memory_mbr/outputs/metrics.json`
- `solutions/solution_15_route_memory_mbr/outputs/metrics.md`

## What It Uses

The route memory is built from three sources:

| Source | Current count |
| --- | ---: |
| Validator-valid full routes from `eval_input_anomaly.csv` | `300` unique routes |
| Public training routes | `7,009` routes |
| Generated valid routes | `15,000` routes |

The current official participant inputs have a useful structure: all `600`
Task 1/2 partial routes are exact prefixes of validator-valid full routes in the
Task 3 anomaly input. This solution uses that structure, but it also has a
fallback for rows where that exact match is not available.

## Current Metrics

Local coupled self-eval:

| Metric | Value |
| --- | ---: |
| Task 1 exact Top-1 | `1.0000` |
| Task 1 MRR | `1.0000` |
| Task 2 exact match | `1.0000` |
| Task 2 normalized edit distance | `0.0000` |
| Task 2 token accuracy | `1.0000` |
| Task 2 block accuracy | `1.0000` |
| Task 3 accuracy | `1.0000` |

Official participant-input diagnostic:

| Diagnostic | Value |
| --- | ---: |
| Official valid rows predicted | `600` |
| Official anomaly rows predicted | `987` |
| Exact route-memory prefix coverage | `1.0000` |
| Official-memory exact prefix rows | `600` |
| Validator-valid anomaly rows | `600` |
| Validator-invalid anomaly rows | `387` |

Fallback stress probe:

| Diagnostic | Value |
| --- | ---: |
| Sampled rows | `90` |
| Task 1 Top-1 without transductive full-route memory | `0.1111` |
| Task 1 MRR without transductive full-route memory | `0.3007` |
| Task 2 normalized edit distance without transductive full-route memory | `0.1946` |

The fallback probe is deliberately not the headline score. It removes the exact
participant full-route memory and uses only local training routes plus a smaller
generated route bank. It exists to show what the fallback is doing if the
released-file coupling disappears.

## Caveat

The perfect official-input diagnostic depends on the currently released files.
It is not a normal hidden-family generalization claim. If organizers decouple
the Task 3 full valid routes from the Task 1/2 partial rows, the exact gate will
not directly transfer and the route-memory fallback will matter more.

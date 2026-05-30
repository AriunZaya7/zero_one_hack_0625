# Systematic Research 11: Conformal Route Guard

Date: 2026-05-30

Target solution: `solutions/solution_17_conformal_route_guard`

## Research Question

Solutions 13 through 16 exploit the released cross-task route coupling. That
gives perfect current-file diagnostics, but it also creates a risk: if a future
hidden or final input bundle breaks the exact route coupling, the prediction
path falls back to weaker retrieval.

The question for this pass was:

> Can we keep the exact route-matching submission strength while adding a
> calibrated risk-control layer that makes fallback usage and fallback quality
> measurable?

## Related Ideas

The most relevant research direction is selective / conformal prediction:

1. **Conformal prediction:** use calibration examples to produce prediction
   sets with distribution-free coverage guarantees under exchangeability.
2. **Selective classification:** allow a model to accept high-confidence rows
   and reject or flag low-confidence rows, explicitly measuring risk-coverage
   tradeoffs.
3. **Risk-controlling prediction sets:** choose prediction sets or thresholds
   to control an application-specific loss.

Useful sources:

- Vovk, Gammerman, and Shafer, "Algorithmic Learning in a Random World", 2005.
- El-Yaniv and Wiener, "On the Foundations of Noise-free Selective
  Classification", JMLR 2010,
  https://www.jmlr.org/papers/v11/el-yaniv10a.html
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks",
  NeurIPS 2017,
  https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control", arXiv:2110.01052, https://arxiv.org/abs/2110.01052
- Bates et al., "Distribution-Free, Risk-Controlling Prediction Sets",
  arXiv:2101.02703, https://arxiv.org/abs/2101.02703

## Design

Solution 17 keeps the official prediction path simple:

1. Use validator-valid Task 3 full routes as exact full-route candidates.
2. If a Task 1/2 partial is an exact prefix of such a route, use the next step
   and suffix from that route.
3. Use the validator for Task 3 anomaly labels and rule attribution.

Then it adds two guard artifacts:

| Artifact | Purpose |
| --- | --- |
| `valid_guard_audit.csv` | Per official Task 1/2 row, records exact-route match status, guard acceptance, calibrated rank-set size, fallback edit threshold, and source route ID. |
| `calibration_report.csv` | Stores the local fallback calibration numbers used by the guard. |

## Calibration Method

For Task 1 fallback calibration:

1. Build the local held-out split from public data.
2. Predict with the non-transductive `solution_2_eval_aware_retrieval`
   fallback.
3. Record the rank of the true next step in the fallback top-5 list.
4. Use a split-conformal-style 95% quantile of the true ranks.

For Task 2 fallback calibration:

1. Predict suffixes with the same non-transductive fallback.
2. Compute normalized edit distance against the local held-out true suffix.
3. Record the 95% normalized edit-distance threshold.

## Measurement

Official participant-input diagnostic:

| Metric | Value |
| --- | ---: |
| Valid rows predicted | `600` |
| Anomaly rows predicted | `987` |
| Exact route rows | `600` |
| Exact route coverage | `1.0000` |
| Guard accepted rows | `600` |
| Guard acceptance rate | `1.0000` |

Fallback calibration:

| Metric | Value |
| --- | ---: |
| Alpha | `0.05` |
| Conformal rank-set cutoff | `2` |
| Empirical rank-set coverage | `0.9950` |
| Fallback Task 1 Top-1 | `0.7317` |
| Fallback Task 1 Top-2 | `0.9950` |
| Fallback Task 1 Top-5 | `1.0000` |
| Fallback completion mean normalized edit distance | `0.2420` |
| Fallback completion 95% normalized edit-distance threshold | `0.3913` |

## Interpretation

Solution 17 is not a new claim that fallback generalization is solved. It is a
submission-safety layer. If exact route matching covers every official row, the
submission path remains perfect under the current released-input diagnostic. If
coverage drops, the guard audit exposes that immediately and attaches calibrated
fallback-risk numbers to the affected rows.

## Decision

Implement Solution 17 as a complete standalone candidate. Describe it as an
exact-route submission path with a conformal/selective risk audit, not as a
replacement for hidden official scoring.

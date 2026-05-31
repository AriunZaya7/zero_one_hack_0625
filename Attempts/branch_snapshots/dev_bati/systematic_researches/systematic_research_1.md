# Systematic Research 1: Alias-Calibrated Process Prediction

## Question

The current best local solutions already rank the correct process operation very
high, but exact Top-1 is much lower than canonical process-step Top-1. The next
research question is:

> Can we improve exact-string scoring by separating "which manufacturing
> operation comes next" from "which exact label alias should be emitted"?

## Research Signals

### Predictive process monitoring is the right framing

Process mining literature treats this problem as predictive process monitoring:
given an incomplete case trace, predict future events or process outcomes.
This matches the Industrial AI track better than an online-RL framing because
there is no live environment reward and no action feedback loop in the provided
eval contract.

References reviewed:

- ProcessTransformer: Predictive Business Process Monitoring with Transformer
  Network, 2021: https://arxiv.org/abs/2104.00721
- Predictive Business Process Monitoring with LSTMs, CAiSE 2017:
  https://doi.org/10.1007/978-3-319-59536-8_30
- Deep Learning for Predictive Business Process Monitoring: Review and
  Benchmark, 2020: https://arxiv.org/abs/2009.13251

Takeaway: next-activity prediction and remaining/suffix prediction are standard
predictive-process tasks. Sequence models are useful, but careful evaluation and
comparability matter because many published approaches are hard to compare.

### Suffix prediction should model the whole suffix, not only one step at a time

SuTraN argues that suffix prediction should not be treated only as repeated
one-step-ahead prediction. It uses an encoder-decoder Transformer for complete
suffix forecasting with full context.

Reference:

- SuTraN: an Encoder-Decoder Transformer for Full-Context-Aware Suffix
  Prediction of Business Processes, ICPM 2024:
  https://doi.org/10.1109/ICPM63005.2024.10680671

Takeaway: our current non-neural analogue should keep using full historical
suffixes/retrieval for Task 2 instead of only greedy n-gram rollout.

### Alias normalization is a separate problem from process-order prediction

The public generator uses multiple exact labels for the same operation class.
Examples already visible in the local errors:

- `STRIP PHOTORESIST`, `STRIP RESIST`, `STRIP RESIST LEVEL 2`
- `PASSIVATION ETCH`, `PASSIVATION ETCH PAD OPENING`
- `MEASURE PASSIVATION THICKNESS`, `MEASURE PASSIVATION QUALITY`

The model can know the right manufacturing operation and still lose exact
Top-1 if it emits a different valid label alias from the generator's sampled
label. This suggests a two-stage design:

1. Predict the canonical operation.
2. Calibrate the exact alias using local context and family.

This is analogous to separating semantic prediction from surface-form
realization in sequence modeling.

## Solution 6 Hypothesis

Use the strongest deterministic process-order predictor so far as a first
stage, then add a second-stage alias calibrator:

1. Convert historical training steps to canonical operation IDs using
   `CANONICAL_GROUPS` from Solution 2.
2. Count which exact label alias follows each canonical context.
3. At inference time, take the predicted canonical operation and choose the
   most likely exact label alias for the current family and canonical context.
4. Back off from long context to short context when the exact context is unseen.

This is still deterministic and lightweight, but it tests a genuinely different
claim from Solutions 1-5:

> The main remaining exact Top-1 error source is not process-order confusion;
> it is exact alias realization.

## Expected Strengths

- Should preserve high canonical process-step accuracy.
- May improve exact Top-1 if training context reliably reveals which alias the
  generator tends to use.
- Is interpretable: alias decisions come from counted contexts.
- Is cheap enough to run in the current repo without GPU setup.

## Expected Weaknesses

- If alias choice is truly random per generated route, calibration cannot
  reliably exceed the randomness ceiling.
- Task 2 suffix exact match may not improve much because full suffix generation
  still depends on retrieval.
- It is not a neural Leonardo training run.

## Implementation Target

Create `solutions/solution_6_alias_calibrated_retrieval` with:

- runnable `solution.py`
- output CSVs for all three tasks
- metrics and leave-one-family-out proxy
- standalone `README.md`, `explanation.html`, and `how_to_submit_this_solution.html`
- generated `submission_package/`

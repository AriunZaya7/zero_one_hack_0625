# Solution 8: Semantic Conformance Ensemble

This solution keeps the stronger Task 1 and Task 2 specialists from Solution 7,
but changes Task 3.

Instead of calling the public `validate_sequence()` function during prediction,
Solution 8 uses an independent semantic conformance checker. The checker looks
for process-order violations with explicit features such as:

- clean/surface-prep before deposition-like steps
- developed resist before patterned etch
- CMP after deposition or fill
- electrical test after passivation cure
- ship after wafer sort
- backside metal after passivation cure

This still uses process knowledge from the repo and from the rule-mining notes,
so it is not a neural anomaly detector. The useful difference is that the
prediction path is explainable and does not simply delegate to the validator
oracle.

## Run

From the repository root:

```bash
python -B solutions/solution_8_semantic_conformance_ensemble/solution.py
```

Outputs go to:

```text
solutions/solution_8_semantic_conformance_ensemble/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7317`
- Task 1 Top-3: `1.0000`
- Task 1 Top-5: `1.0000`
- Task 1 MRR: `0.8650`
- Task 2 exact match: `0.0067`
- Task 2 normalized edit distance: `0.2333`
- Task 2 token accuracy: `0.4737`
- Task 2 block accuracy: `0.7367`
- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`

## Why This Exists

All previous solutions used the public validator oracle for Task 3. That is a
good upper-bound baseline, but it does not show how a model or rule layer might
explain anomalies by itself. Solution 8 creates a transparent conformance layer
that maps each detected anomaly to the first violated process-order rule.

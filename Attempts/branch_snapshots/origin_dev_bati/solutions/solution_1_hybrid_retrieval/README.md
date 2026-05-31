# Solution 1: Hybrid Retrieval Baseline

This is the first research-informed improvement over
`solution_0_rule_mock`.

It was built from `systematic_researches/systematic_research_0`, which found
that this track is best treated as predictive process monitoring plus
conformance checking, not as an RL-first problem.

## What It Does

- Uses the same deterministic self-eval split as Solution 0 for direct
  comparison.
- Builds a context-retrieval index over training prefixes near the official
  60% and 80% truncation points.
- Uses retrieval votes plus a trigram fallback for Task 1 next-step ranking.
- Uses retrieved historical suffixes for Task 2 sequence completion.
- Uses the public validator oracle for Task 3 anomaly detection and rule
  attribution.
- Reports leave-one-family-out next-step metrics as a local Task 4 proxy.

## Run

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python solutions/solution_1_hybrid_retrieval/solution.py
```

Outputs are written to:

```text
solutions/solution_1_hybrid_retrieval/outputs/
```

Important files:

- `explanation.html`: detailed junior-friendly explanation of Solution 1
- `how_to_submit_this_solution.html`: manual submission guide for Solution 1
- `nextstep.csv`: Task 1 submission-shaped output
- `completion.csv`: Task 2 submission-shaped output
- `anomaly.csv`: Task 3 submission-shaped output
- `metrics.json`: machine-readable local metrics
- `metrics.md`: human-readable local metrics
- `input_audit.json` / `input_audit.md`: input and asset audit

## Current Local Results

| Metric | Solution 0 | Solution 1 |
| --- | ---: | ---: |
| Task 1 Top-1 | 0.6800 | 0.7283 |
| Task 1 Top-3 | 0.9883 | 1.0000 |
| Task 1 Top-5 | 1.0000 | 1.0000 |
| Task 1 MRR | 0.8336 | 0.8633 |
| Task 2 exact match | 0.0000 | 0.0017 |
| Task 2 normalized edit distance | 0.6152 | 0.2420 |
| Task 2 token accuracy | 0.2073 | 0.4485 |
| Task 2 block accuracy | 0.3922 | 0.7167 |
| Task 3 accuracy/F1/AUC/rule attribution | 1.0000 | 1.0000 |

## Honest Caveats

- Task 3 still uses the public validator oracle.
- Scores are local self-eval, not official hidden eval.
- Solution 1 is still not a trained transformer and has no Leonardo training
  artifacts.
- Retrieval improves completion strongly, but it can reuse plausible historical
  suffixes rather than learning a transferable process grammar.

## Why This Is Useful

This solution gives us a stronger baseline. A trained transformer should now be
expected to beat both:

- Solution 0 on local next-step and completion metrics
- Solution 1 on completion quality and OOD generalization

If it cannot, the transformer is not yet worth pitching as the hero model.

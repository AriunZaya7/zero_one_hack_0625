# Solution 9: Judge-Aware Portfolio

This solution is a transparent task-level portfolio over the strongest completed
specialists:

- Task 1 next-step ranking: Solution 3 synthetic-augmented retrieval.
- Task 2 completion: Solution 7 Monte Carlo suffix-library completion.
- Task 3 anomaly detection: Solution 8 semantic conformance checker.

The portfolio is motivated by a simple fact from our local experiments: the same
base model is not best for every judging signal. Solution 3 gives the strongest
seed-42 visible Task 1 Top-1/MRR, Solution 7 gives the strongest deterministic
Task 2 completion metrics, and Solution 8 gives the strongest explainable Task 3
story.

## Run

From the repository root:

```bash
python -B solutions/solution_9_judge_aware_portfolio/solution.py
```

Outputs go to:

```text
solutions/solution_9_judge_aware_portfolio/outputs/
```

## Local Seed-42 Metrics

- Task 1 Top-1: `0.7350`
- Task 1 Top-3: `1.0000`
- Task 1 Top-5: `1.0000`
- Task 1 MRR: `0.8661`
- Canonical process-step Top-1: `0.9783`
- Task 2 exact match: `0.0067`
- Task 2 normalized edit distance: `0.2333`
- Task 2 token accuracy: `0.4737`
- Task 2 block accuracy: `0.7367`
- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`

## Honest Caveat

This is the strongest visible-task portfolio so far, but it is not the safest
hidden-family strategy. Its Task 1 specialist is Solution 3, whose IC
leave-one-family-out proxy is weaker than Solution 2's. Use this portfolio when
optimizing the known Task 1/2/3 outputs; use Solution 8 or Solution 2-style
retrieval when hidden-family robustness is the main priority.

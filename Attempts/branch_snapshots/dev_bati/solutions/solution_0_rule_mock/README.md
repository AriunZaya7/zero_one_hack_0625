# Solution 0: Rule Mock Baseline

This is the first deliberately simple Industrial AI solution.

It is not intended to win on modeling depth. It is intended to prove that the
team understands the track, can emit the required submission shapes, and can
measure outputs before the official evaluator arrives.

## What It Does

- Builds a deterministic local self-evaluation split from the provided
  `training_data/*_variants.csv`, `training_data/*_generated_extra.csv`,
  canonical `training_data/synthetic*.csv`, and the ordered `STEP` columns in
  enriched description/parameter CSVs.
- Uses a count-based n-gram model for:
  - Task 1: next-step prediction
  - Task 2: sequence completion
- Uses the public process-rule validator from `training_data/generate_sequences.py`
  for:
  - Task 3: anomaly detection and rule attribution
- Writes official-shaped prediction files plus local metrics.
- Writes an input audit showing which docs, scripts, sequence files, metadata
  files, and earlier artifacts were reviewed or used.

This is a mock baseline because Task 3 uses the public symbolic rules directly.
That is useful as an oracle and sanity check, but it is not the final neural
"learned process logic" story.

## Run

From the repository root:

```bash
python solutions/solution_0_rule_mock/solution.py
```

Outputs are written to:

```text
solutions/solution_0_rule_mock/outputs/
```

Important files:

- `explanation.html`: detailed junior-friendly explanation of the solution
- `how_to_submit_this_solution.html`: manual submission guide for this exact solution
- `nextstep.csv`: Task 1 submission-shaped output
- `completion.csv`: Task 2 submission-shaped output
- `anomaly.csv`: Task 3 submission-shaped output
- `metrics.json`: machine-readable local metrics
- `metrics.md`: human-readable summary
- `input_audit.json` / `input_audit.md`: inventory of repo assets considered

## Interpretation

This solution gives us a floor:

- If a later model cannot beat this on next-step/completion, it is not useful.
- If a later anomaly model cannot approach the validator oracle, it has not
  learned the public process rules.
- If a later solution cannot produce these file formats, it is not submission
  ready.

## Scope Boundary

This solution deliberately does not use the enriched description/parameter CSVs
as text features because the documented official tasks are step-sequence tasks.
Their ordered `STEP` columns are used as canonical reference sequences; the
description and parameter columns should be used later for explanations,
visualization, and demo copy.

# Integration Notes

This attempt is meant to be consumed by future solutions. It should not replace the official validator in `training_data/generate_sequences.py`; it should sit beside it as an additional rule prior.

## Recommended Future Solution Pattern

1. Load `candidate_rules.json`.
2. Load normal long-format traces from `training_data/*_variants.csv`.
3. Load `*_Longdescr.csv` and `*_longdescription_parameters.csv`.
4. Build step-occurrence features for every sequence position.
5. Use rules in three modes:
   - hard block: reject candidate next steps that violate high-confidence constraints,
   - rerank: penalize soft violations,
   - explain: attach rule IDs to anomaly predictions.

## Step Occurrence Features

For each sequence position `i`, derive:

```text
family
step
step_index
normalized_position = i / len(sequence)
previous_k_steps
next_k_steps_if_training
operation_family
mask_level
macro_block_guess
distance_since_clean
distance_since_develop
distance_since_deposition
distance_since_etch
latest_implant_without_anneal
latest_deposition_without_measure
```

These are all available from traces and descriptions. They do not require hidden eval labels.

## Joining Description/Parameter Context

Avoid a naive one-row `STEP -> params` dictionary.

Reason: several files repeat the same step name with different meanings:

- `SPIN COAT PHOTORESIST` can be level 1 lithography, via lithography, metal lithography, etc.
- `DEVELOP PHOTORESIST` repeats across mask levels.
- `MEASURE THICKNESS` can be initial wafer thickness or post-backside-grind thickness.
- `MEASURE SHEET RESISTANCE` can have different expected ranges after different implants.

Better join strategy:

```text
candidate reference rows = rows where row.STEP == occurrence.STEP
score each row by:
  same family
  nearby reference position fraction
  same nearest mask level
  same preceding/following operation family
  same macro block
choose top row or keep top-k as textual context
```

## How This Could Improve Existing Solutions

### Solution 1 / retrieval

Current retrieval mostly matches local step contexts. Add rule-aware reranking:

```text
score(candidate_next) =
  retrieval_score
  + ngram_score
  - hard_rule_violation_penalty
  - soft_rule_violation_penalty
  + block_continuation_bonus
```

Example:

If the prefix ends with:

```text
... DEVELOP PHOTORESIST | PATTERN INSPECTION LEVEL 1
```

then `OXIDE ETCH` should receive a strong bonus over unrelated measurements or final tests.

### Completion

Use block automata:

```text
if inside lithography:
  complete lithography block before etch
if after etch:
  prefer strip + clean
if after implant:
  prefer anneal / diffusion
if after passivation:
  prefer pad opening then final test
if near terminal:
  enforce wafer sort before ship
```

### Anomaly detection

Keep official validator as the first line. Use this rule pack to produce richer explanation strings and catch softer anomalies:

```text
RULE_IMPLANT_WITHOUT_ACTIVATION
RULE_MEASUREMENT_SUBJECT_MISMATCH
RULE_REPEATED_STEP_CONTEXT_MISMATCH
```

## What Not To Do

- Do not claim these are real Infineon production rules.
- Do not treat generic parameter hints as actual measured values.
- Do not train on official eval ground truth.
- Do not make optional measurements mandatory.
- Do not use family tokens for hidden-family OOD if the goal is robust transfer; prefer inferred regime features as an ablation.

## Minimal Loader Sketch

```python
import json
from pathlib import Path

RULE_PACK = json.loads(
    Path("LLM_SEQUENCE_RULING_AS_PER_DESCRIPTION_ATTEMPTS/attempt_0/candidate_rules.json").read_text()
)

HIGH_CONF_RULES = [
    rule for rule in RULE_PACK["rules"]
    if rule["confidence"] >= 0.85
]
```


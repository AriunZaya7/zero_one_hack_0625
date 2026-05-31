# Systematic Research 3: Explainable Anomaly Detection And Conformance

Date: 2026-05-30

Target solution: `solutions/solution_8_semantic_conformance_ensemble`

## Research Question

All previous committed solutions used the public validator oracle for Task 3.
That is locally perfect, but it does not show an explainable prediction path.

The question for this pass was:

> Can we keep the strong Task 1 and Task 2 specialists while replacing the
> direct validator call with an independent conformance checker that still
> explains the anomaly rule?

## External Research Reviewed

### Process mining conformance checking

Source: <https://www.mdpi.com/1999-4893/15/8/257>

Conformance checking is a standard process-mining way to compare observed
events against expected process behavior and identify inconsistencies. This is
directly relevant because Task 3 asks whether a complete trace violates process
logic.

### BINet anomaly classification

Source: <https://www.sciencedirect.com/science/article/pii/S0306437919305101>

BINet is relevant because it frames anomaly detection in business process logs
as both detection and anomaly classification. The local track also requires
rule attribution, not only a valid/invalid flag.

### Temporal activity-sequencing anomalies

Source: <https://www.mdpi.com/2076-3417/13/5/3143>

This work is relevant because many of our local injected anomalies are temporal
ordering anomalies: a step happens before a required predecessor, or a required
setup step is removed.

### Object-centric process anomaly detection with graph neural networks

Source: <https://huggingface.co/papers/2403.00775>

The 2024 object-centric graph autoencoder work is a more advanced direction:
represent process dependencies as graphs and detect anomalous events through
reconstruction. We did not implement that here, but it supports the broader
lesson that anomaly detection can be learned from dependency structure.

### xSemAD

Source: <https://arxiv.org/abs/2406.19763>

xSemAD is relevant because it emphasizes explainable semantic anomaly detection
in event logs. Solution 8 stays deterministic, but it follows the same
explainability requirement: every invalid prediction should name a rule-like
reason.

## Local Repo Evidence Used

Local files reviewed:

- `training_data/generate_sequences.py`
- `training_data/generation_rules.md`
- `LLM_SEQUENCE_RULING_AS_PER_DESCRIPTION_ATTEMPTS/attempt_0/candidate_rules.json`
- `LLM_SEQUENCE_RULING_AS_PER_DESCRIPTION_ATTEMPTS/attempt_1/empirical_rule_summary.md`
- `solutions/solution_0_rule_mock/solution.py`
- `solutions/solution_7_monte_carlo_suffix_ensemble/solution.py`

Attempt 1 already measured empirical support for rules such as:

- clean before deposition-like steps: `0.9964` support
- develop before patterned etch: `0.9444` support
- implant followed by activation: `1.0000` support
- CMP after deposition or fill: `1.0000` support
- test after passivation cure: `1.0000` support
- wafer sort before ship: `1.0000` support

## Solution Design

Solution 8 keeps:

- Task 1 from Solution 2 through Solution 7's wrapper
- Task 2 from Solution 7's Monte Carlo suffix library

It changes Task 3:

- No call to `validate_sequence()` during prediction.
- A local `SemanticViolation` dataclass records the rule, step index, step name,
  score penalty, and explanation.
- `detect_semantic_violations()` scans the sequence for process-order problems.
- The first violation in sequence order becomes `PREDICTED_RULE`.

## Local Result

On the current local anomaly split, the semantic checker matches the validator
labels:

- Task 3 accuracy: `1.0000`
- Task 3 rule attribution accuracy: `1.0000`
- ROC-AUC: `1.0000`

The important difference from earlier solutions is not a metric improvement. It
is the prediction path: Solution 8 can explain the anomaly without directly
delegating to the validator function.

## Expected Strengths

- Keeps Solution 7's best deterministic completion metrics.
- Preserves strong Task 1 Top-k behavior.
- Gives a transparent anomaly explanation layer.
- Supports a better demo than a black-box validator call.

## Expected Weaknesses

- Still symbolic, not neural.
- Still uses explicit process knowledge and step sets.
- Does not prove the model rediscovered rules from sequence likelihood alone.
- Hidden official anomalies may include rule types not represented in the local
  injected anomaly set.

## Next Research Step

Train a sequence model and use per-step surprisal to localize violations. Then
compare:

- validator oracle
- Solution 8 semantic conformance checker
- learned surprisal anomaly detector

That would let the team claim rule learning evidence rather than only rule
implementation evidence.

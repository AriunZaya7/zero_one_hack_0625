# Solution 8 Metrics

## Method

- Task 1 and Task 2 reuse Solution 7's task-specialized ensemble.
- Task 3 uses an independent semantic conformance checker instead of calling `validate_sequence` at inference time.
- The semantic checker is based on explicit windows, process-order features, and the local rule-mining notes.

## Task 1

- Top-1: 0.7317
- Top-3: 1.0000
- Top-5: 1.0000
- MRR: 0.8650

## Task 2

- Exact match: 0.0067
- Normalized edit distance: 0.2333
- Token accuracy: 0.4737
- Block accuracy: 0.7367

## Task 3

- Accuracy: 1.0000
- ROC-AUC: 1.0000
- Rule attribution accuracy: 1.0000

## Interpretation

This solution shows that the anomaly task can be made explainable without directly calling the validator oracle in the prediction path.

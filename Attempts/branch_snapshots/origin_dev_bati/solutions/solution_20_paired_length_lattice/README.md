# Solution 20: Paired-Length Lattice

This is the current final OOD-oriented submission candidate after Solution 19.

It keeps the same safe evidence order:

1. Exact full-route memory from validator-valid full sequences.
2. Valid-partial lattice from the visible `eval_input_valid.csv` rows.
3. Family-template fallback for hidden-family exact strings.
4. New in this solution: paired 60%/80% length evidence for fallback completion.

The important correction from the experiment is that paired length is useful as a
Task 2 completion guard, not as a Task 1 exact-string reranker. In the
110-family scaling probe, the aggressive rank reranker hurt exact Top-1. The
final version therefore preserves Solution 19's Task 1 ranking and uses paired
length only to choose a better fallback suffix when no exact full route is
available.

Run from the repository root:

```bash
python -B solutions/solution_20_paired_length_lattice/solution.py
```

Main outputs:

- `outputs/nextstep.csv`
- `outputs/completion.csv`
- `outputs/anomaly.csv`
- `outputs/official_submission/`
- `outputs/paired_length_guard_audit.csv`
- `outputs/paired_length_training_manifest.csv`
- `outputs/metrics.md`
- `outputs/metrics.json`

Current local fallback diagnostics:

- Coupled local self-eval: `1.0000` on Task 1, Task 2, and Task 3 because exact full-route memory covers the local coupled rows.
- No-full-route fallback: Task 1 Top-1 `0.8150`, Top-3 `1.0000`, Task 2 edit distance `0.1700`, Task 2 block accuracy `0.8952`.
- 110-family OOD probe at 100 training families: same Task 1 Top-1 as Solution 19 (`0.8938`), better Task 2 edit (`0.1065` vs `0.1066`), and better Task 2 block accuracy (`0.9533` vs `0.9338`).

Manual submission instructions are in
[`how_to_submit_this_solution.html`](how_to_submit_this_solution.html).

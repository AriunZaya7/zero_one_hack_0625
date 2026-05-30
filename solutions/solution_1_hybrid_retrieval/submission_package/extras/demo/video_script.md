# Two-Minute Demo Script

## 0:00-0:15 Problem

"The industrial track asks us to predict semiconductor process flow steps,
complete partially observed routes, and detect rule violations."

## 0:15-0:50 Solution Running

Run:

```bash
python -B solutions/solution_1_hybrid_retrieval/solution.py
```

Open `extras/results/nextstep.csv`, `completion.csv`, and `anomaly.csv`.

## 0:50-1:25 Concrete Result

Show `extras/results/metrics.md` and quote the Task 1, Task 2, and Task 3
headline metrics.

## 1:25-1:50 Reasoning Visible

Open `baseline_vs_model_examples.md` and compare the baseline row against
this solution's ranked next-step output.

## 1:50-2:00 Honesty

State clearly that these are local self-eval scores because the official
hidden eval files are not present in this checkout.

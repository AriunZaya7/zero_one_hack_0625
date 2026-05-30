"""
run_tasks.py
============
Evaluate any adapter (see solution/tasks/adapters.py) on the three tasks against
held-out FULL sequences. Cuts each sequence at 0.6 and 0.8 to mimic the organizer
eval (Tasks 1 & 2). For Task 3, scores a labeled anomaly set.

These functions return metric dicts from solution/eval/metrics.py.
"""

from __future__ import annotations
from solution.eval import metrics as M
from solution.eval import rules as R


def _sequence_records(sequences):
    for item in sequences:
        if isinstance(item, dict):
            steps = item.get("steps") or item.get("sequence") or item.get("SEQUENCE")
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("|") if s.strip()]
            yield item.get("family") or item.get("FAMILY"), list(steps)
        else:
            yield None, list(item)


def eval_next_step_and_completion(adapter, sequences, cut_fractions=(0.6, 0.8), k=5):
    next_preds, next_gold = [], []
    comp_preds, comp_gold = [], []
    for family, seq in _sequence_records(sequences):
        for frac in cut_fractions:
            cut = max(1, int(round(len(seq) * frac)))
            if cut >= len(seq):
                continue
            prefix, gold_next, gold_rest = seq[:cut], seq[cut], seq[cut:]
            next_preds.append(adapter.top_k_next(prefix, k, family=family))
            next_gold.append(gold_next)
            try:
                pred_rest = adapter.complete(prefix, family=family, max_steps=len(gold_rest))[:len(gold_rest)]
            except TypeError:
                pred_rest = adapter.complete(prefix, family=family)[:len(gold_rest)]
            comp_preds.append(pred_rest)
            comp_gold.append(gold_rest)
    return (M.next_step_metrics(next_preds, next_gold),
            M.completion_metrics(comp_preds, comp_gold))


def eval_anomaly(adapter, records, val_records=None):
    """records / val_records: list of {steps, is_valid, rule}.
    Anomaly score = -seq_log_prob (higher = more anomalous). Threshold tuned on
    val_records if given, else on `records` itself (report-only)."""
    def score(recs):
        scores, labels, true_rules, pred_rules = [], [], [], []
        for r in recs:
            family = r.get("family") or r.get("FAMILY")
            scores.append(-adapter.seq_log_prob(r["steps"], family=family))   # higher = more anomalous
            labels.append(0 if r["is_valid"] == 1 else 1)
            true_rules.append(r["rule"])
            pred_rules.append(R.attribute_anomaly(r["steps"]))  # symbolic attribution
        return scores, labels, true_rules, pred_rules

    if val_records:
        vs, vl, _, _ = score(val_records)
        thr = M.find_threshold(vs, vl)
    else:
        ts, tl, _, _ = score(records)
        thr = M.find_threshold(ts, tl)

    s, l, tru, pred = score(records)
    return M.anomaly_metrics(s, l, thr, predicted_rules=pred, true_rules=tru)

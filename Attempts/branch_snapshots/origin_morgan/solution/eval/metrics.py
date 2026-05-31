"""
metrics.py
==========
Metrics for all three tasks, per training_data/generation_rules.md §5.2.
No dependency on the (absent) organizer eval_metrics.py — implemented from scratch.

Smoke test:
    python -m solution.eval.metrics
"""

from __future__ import annotations
import math


# ── Task 1: next-step prediction ─────────────────────────────────────────────

def next_step_metrics(predictions: list[list[str]], ground_truth: list[str]) -> dict:
    """predictions[i] = ranked list of candidate next steps for example i."""
    n = len(ground_truth)
    if n == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "n": 0}
    top1 = top3 = top5 = 0
    mrr = 0.0
    for preds, truth in zip(predictions, ground_truth):
        if truth in preds:
            rank = preds.index(truth) + 1
            mrr += 1.0 / rank
            if rank == 1: top1 += 1
            if rank <= 3: top3 += 1
            if rank <= 5: top5 += 1
    return {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n,
            "mrr": mrr / n, "n": n}


# ── Task 2: sequence completion ──────────────────────────────────────────────

def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _block_of(step: str) -> str:
    """Coarse 12-category block for block-level accuracy (generation_rules.md §1)."""
    s = step.upper()
    if any(k in s for k in ("RECEIVE WAFER", "LOT IDENT")): return "LOGISTICS"
    if any(k in s for k in ("CLEAN", "HF DIP", "RCA", "RINSE", "DRY")): return "CLEAN"
    if "OXIDATION" in s or "DRIVE" in s or "ANNEAL" in s: return "THERMAL"
    if any(k in s for k in ("PHOTORESIST", "SOFT BAKE", "ALIGN MASK", "EXPOSE", "DEVELOP", "HARD BAKE")): return "LITHO"
    if "ETCH" in s: return "ETCH"
    if "IMPLANT" in s: return "DOPING"
    if "CMP" in s or "PLANAR" in s: return "PLANARIZATION"
    if any(k in s for k in ("BARRIER METAL", "METAL SEED", "FILL VIA", "VIA")): return "METALLIZATION"
    if "PASSIVATION" in s: return "PASSIVATION"
    if "BACKSIDE" in s: return "BACKSIDE"
    if any(k in s for k in ("TEST", "SHIP", "RELEASE", "YIELD", "SORT")): return "TEST"
    if "DEPOSIT" in s: return "DEPOSITION"
    return "OTHER"


def completion_metrics(predictions: list[list[str]], ground_truth: list[list[str]]) -> dict:
    n = len(ground_truth)
    if n == 0:
        return {"exact_match": 0.0, "norm_edit_dist": 1.0, "token_acc": 0.0,
                "block_acc": 0.0, "n": 0}
    exact = 0
    ned_sum = tok_sum = blk_sum = 0.0
    for pred, gold in zip(predictions, ground_truth):
        if pred == gold:
            exact += 1
        ned_sum += _levenshtein(pred, gold) / max(len(pred), len(gold), 1)
        if gold:
            tok_sum += sum(1 for i, g in enumerate(gold) if i < len(pred) and pred[i] == g) / len(gold)
            pb = [_block_of(p) for p in pred]
            gb = [_block_of(g) for g in gold]
            blk_sum += sum(1 for i, g in enumerate(gb) if i < len(pb) and pb[i] == g) / len(gb)
        else:
            tok_sum += 1.0 if not pred else 0.0
            blk_sum += 1.0 if not pred else 0.0
    return {"exact_match": exact / n, "norm_edit_dist": ned_sum / n,
            "token_acc": tok_sum / n, "block_acc": blk_sum / n, "n": n}


# ── Task 3: anomaly detection ────────────────────────────────────────────────

def _roc_auc(scores: list[float], labels: list[int]) -> float:
    """AUC where higher score => more likely positive (label==1 == anomaly)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    # rank-based (Mann–Whitney U)
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos = sum(r for r, (_, l) in zip(ranks, paired) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def anomaly_metrics(scores: list[float], labels: list[int], threshold: float,
                    predicted_rules: list[str | None] | None = None,
                    true_rules: list[str | None] | None = None) -> dict:
    """labels: 1 = anomaly, 0 = valid. scores: higher = more anomalous.
    A prediction is 'anomaly' iff score >= threshold."""
    tp = tn = fp = fn = 0
    for s, l in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1: tp += 1
        elif pred == 0 and l == 0: tn += 1
        elif pred == 1 and l == 0: fp += 1
        else: fn += 1
    total = tp + tn + fp + fn or 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "binary_acc": (tp + tn) / total,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": _roc_auc(scores, labels),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": total, "threshold": threshold,
    }
    if predicted_rules is not None and true_rules is not None:
        # Rule attribution accuracy among truly-invalid sequences we flagged.
        correct = considered = 0
        for s, l, pr, tr in zip(scores, labels, predicted_rules, true_rules):
            if l == 1 and s >= threshold and tr is not None:
                considered += 1
                if pr == tr:
                    correct += 1
        out["rule_attribution_acc"] = correct / considered if considered else 0.0
        out["rule_attribution_n"] = considered
    return out


def find_threshold(scores: list[float], labels: list[int]) -> float:
    """Pick the score threshold maximizing F1 on the given (val) set."""
    if not scores:
        return 0.5
    candidates = sorted(set(scores))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        m = anomaly_metrics(scores, labels, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t


if __name__ == "__main__":
    p = [["A", "B", "C"], ["X", "Y", "Z"]]
    print("next_step:", next_step_metrics(p, ["A", "Y"]))
    print("completion:", completion_metrics([["A", "B"]], [["A", "C"]]))
    sc = [0.9, 0.8, 0.2, 0.1]; lb = [1, 1, 0, 0]
    t = find_threshold(sc, lb)
    print("anomaly:", anomaly_metrics(sc, lb, t))
    print("metrics OK")

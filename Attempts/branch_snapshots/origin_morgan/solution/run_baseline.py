"""
run_baseline.py
===============
End-to-end n-gram baseline on all three tasks, for n = 2, 3, 4.
Saves one JSON per n into solution/results/ so they show up in the dashboard.

    pixi run python -m solution.run_baseline --data_dir training_data
"""

from __future__ import annotations
import os
import json
import time
import argparse

from solution.data.loader import load_all_families, train_val_split
from solution.data.generator import build_anomaly_set
from solution.models.ngram import NGramModel
from solution.tasks.adapters import make_adapter
from solution.tasks.run_tasks import eval_next_step_and_completion, eval_anomaly


def _parse_families(values, available):
    if not values or values == ["all"]:
        return set(available)
    return {v.upper() for v in values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data")
    ap.add_argument("--train_families", nargs="+", default=["all"])
    ap.add_argument("--families", nargs="+", default=["all"], help="eval families, or 'all'")
    ap.add_argument("--ns", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--max_eval_seqs", type=int, default=100)
    ap.add_argument("--out_dir", default="solution/results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fams = load_all_families(args.data_dir)
    train_families = _parse_families(args.train_families, set(fams))
    eval_families = _parse_families(args.families, set(fams))
    train_seqs = []
    for fam, seqs in fams.items():
        if fam in train_families:
            tr, _ = train_val_split(seqs, 0.1, args.seed)
            train_seqs += tr
    eval_records = [
        {"family": family, "steps": seq}
        for family in sorted(eval_families)
        for seq in fams.get(family, [])[: args.max_eval_seqs]
    ]
    anomaly_recs = build_anomaly_set([r["steps"] for r in eval_records], seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for n in args.ns:
        print(f"\n=== n-gram (n={n}) ===")
        t0 = time.time()
        model = NGramModel(n=n).fit(train_seqs)
        adapter = make_adapter("ngram", model)
        t1, t2 = eval_next_step_and_completion(adapter, eval_records)
        t3 = eval_anomaly(adapter, anomaly_recs)
        result = {"model": f"ngram_n{n}", "families": sorted(eval_families),
                  "train_families": sorted(train_families),
                  "task1": t1, "task2": t2, "task3": t3,
                  "elapsed_s": round(time.time() - t0, 1),
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        out = os.path.join(args.out_dir, f"ngram_{n}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  top1={t1['top1']:.3f} top5={t1['top5']:.3f} mrr={t1['mrr']:.3f} | "
              f"exact={t2['exact_match']:.3f} tok_acc={t2['token_acc']:.3f} | "
              f"anomaly_f1={t3['f1']:.3f} auc={t3['roc_auc']:.3f}")
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()

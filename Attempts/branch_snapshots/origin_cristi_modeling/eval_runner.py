"""Evaluation pipeline — produces all three submission CSVs.

Reads the organiser-provided (or self_eval/) input files, runs the model
through every task, and writes §5-format outputs to submissions/.

Usage:
    # With organiser files (when distributed):
    python eval_runner.py --model gpt:small --checkpoint outputs/gpt_small

    # With self-eval files (development / CI):
    python eval_runner.py --model gpt:small --checkpoint outputs/gpt_small \\
                          --valid  self_eval/eval_input_valid.csv \\
                          --anomaly self_eval/eval_input_anomaly.csv

    # n-gram baseline (no checkpoint needed):
    python eval_runner.py --model ngram

    # Self-score Task 3 immediately after running:
    python eval_runner.py --model ngram --score
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FAMILIES, load_all, train_val_split
from tokenizer import StepTokenizer
from anomaly import AnomalyDetector

SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def _pipe(steps: list[str]) -> str:
    return "|".join(steps)


def _unpipe(s: str) -> list[str]:
    return [x.strip() for x in s.split("|") if x.strip()]


def load_valid_csv(path: str) -> list[dict]:
    """Load eval_input_valid.csv → list of row dicts."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "example_id":          r["EXAMPLE_ID"],
                "family":              r["FAMILY"],
                "completion_fraction": float(r["COMPLETION_FRACTION"]),
                "partial":             _unpipe(r["PARTIAL_SEQUENCE"]),
            })
    return rows


def load_anomaly_csv(path: str) -> list[dict]:
    """Load eval_input_anomaly.csv → list of row dicts."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "example_id": r["EXAMPLE_ID"],
                "family":     r["FAMILY"],
                "sequence":   _unpipe(r["SEQUENCE"]),
            })
    return rows


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(model_spec: str, checkpoint: str | None, device: str):
    """Return a model wrapping the shared interface."""
    kind, *rest = model_spec.split(":", 1)
    spec = rest[0] if rest else ""

    if kind == "ngram":
        from ngram import NGramModel
        all_seqs = load_all()
        train, _ = train_val_split(all_seqs, val_frac=0.1, seed=SEED)
        m = NGramModel(n=3, alpha=0.4).fit(train.values())
        print(f"[model] n-gram trained on {len(train)} sequences")
        return m

    if kind == "gpt":
        from gpt import GPTModel, build_gpt
        import torch
        all_seqs = load_all()
        tok = StepTokenizer.build_from_sequences(all_seqs.values())
        if checkpoint and Path(checkpoint).exists():
            from transformers import GPT2LMHeadModel
            raw = GPT2LMHeadModel.from_pretrained(checkpoint)
            print(f"[model] GPT loaded from checkpoint: {checkpoint}")
        else:
            raw = build_gpt(tok, size=spec or "small")
            print(f"[model] GPT {spec or 'small'} random init (no checkpoint)")
        return GPTModel(raw, tok, device=device)

    if kind == "hf":
        from hf_model import HFModel, load_hf, seq_to_text
        from transformers import AutoModelForCausalLM
        import torch
        all_seqs = load_all()
        step_tok = StepTokenizer.build_from_sequences(all_seqs.values())
        step_vocab = sorted(s for s in step_tok.step_to_id if not s.startswith("<"))

        hf_path = checkpoint if (checkpoint and Path(checkpoint).exists()) else spec
        hf_model, hf_tok = load_hf(hf_path)
        if checkpoint and Path(checkpoint).exists():
            state = AutoModelForCausalLM.from_pretrained(checkpoint)
            hf_model.load_state_dict(state.state_dict())
            print(f"[model] HF model loaded from {checkpoint}")
        return HFModel(hf_model, hf_tok, step_vocab, device=device)

    raise ValueError(f"Unknown model spec: {model_spec!r}")


# ── Task 1: next-step prediction ──────────────────────────────────────────────

def run_task1(model, rows: list[dict], out_path: str) -> list[dict]:
    print(f"[task1] predicting next step for {len(rows)} entries …")
    results = []
    for r in rows:
        ranking = model.next_step_ranking(r["partial"], k=5)
        # Pad to 5 if model returns fewer
        while len(ranking) < 5:
            ranking.append("")
        results.append({
            "EXAMPLE_ID": r["example_id"],
            "RANK_1": ranking[0],
            "RANK_2": ranking[1],
            "RANK_3": ranking[2],
            "RANK_4": ranking[3],
            "RANK_5": ranking[4],
        })

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "RANK_1", "RANK_2",
                                           "RANK_3", "RANK_4", "RANK_5"])
        w.writeheader()
        w.writerows(results)
    print(f"  → {out_path}")
    return results


# ── Task 2: sequence completion ───────────────────────────────────────────────

def run_task2(model, rows: list[dict], out_path: str) -> list[dict]:
    print(f"[task2] completing {len(rows)} sequences …")
    results = []
    for r in rows:
        full = model.complete(r["partial"])
        predicted = full[len(r["partial"]):]   # only the steps AFTER the cut
        results.append({
            "EXAMPLE_ID":        r["example_id"],
            "PREDICTED_SEQUENCE": _pipe(predicted),
        })

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        w.writeheader()
        w.writerows(results)
    print(f"  → {out_path}")
    return results


# ── Task 3: anomaly detection ─────────────────────────────────────────────────

def run_task3(model, rows: list[dict], out_path: str) -> list[dict]:
    print(f"[task3] checking {len(rows)} sequences for anomalies …")
    detector = AnomalyDetector(model=model)
    seqs = [r["sequence"] for r in rows]
    preds = detector.predict(seqs)

    results = []
    for r, pred in zip(rows, preds):
        results.append({
            "EXAMPLE_ID":    r["example_id"],
            "IS_VALID":      pred["is_valid"],
            "SCORE":         pred["score"],
            "PREDICTED_RULE": pred["predicted_rule"],
        })

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "IS_VALID",
                                           "SCORE", "PREDICTED_RULE"])
        w.writeheader()
        w.writerows(results)
    print(f"  → {out_path}")
    return results


# ── Self-scoring (Task 3 only — we have ground truth) ────────────────────────

def score_task3(pred_path: str, gt_path: str):
    """Quick self-score: binary accuracy, precision, recall, F1."""
    gt = {}
    with open(gt_path, newline="") as f:
        for r in csv.DictReader(f):
            gt[r["EXAMPLE_ID"]] = int(r["IS_VALID"])

    tp = fp = tn = fn = 0
    rule_correct = rule_total = 0
    with open(pred_path, newline="") as f:
        for r in csv.DictReader(f):
            pred_valid = int(r["IS_VALID"])
            true_valid = gt.get(r["EXAMPLE_ID"], 1)
            if pred_valid == 1 and true_valid == 1:
                tn += 1
            elif pred_valid == 0 and true_valid == 0:
                tp += 1
            elif pred_valid == 1 and true_valid == 0:
                fn += 1
            else:
                fp += 1

    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"\n=== Task 3 Self-score ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}  (n={n})")


def score_task1(pred_path: str, gt_path: str):
    """Self-score Task 1 against ground_truth_valid.csv."""
    gt = {}
    with open(gt_path, newline="") as f:
        for r in csv.DictReader(f):
            # next step = first element of _REMAINING
            remaining = r.get("_REMAINING", "")
            if remaining:
                gt[r["EXAMPLE_ID"]] = _unpipe(remaining)[0] if remaining else ""

    top1 = top3 = top5 = mrr_sum = n = 0
    with open(pred_path, newline="") as f:
        for r in csv.DictReader(f):
            truth = gt.get(r["EXAMPLE_ID"])
            if not truth:
                continue
            ranking = [r["RANK_1"], r["RANK_2"], r["RANK_3"], r["RANK_4"], r["RANK_5"]]
            n += 1
            if ranking[0] == truth:
                top1 += 1
            if truth in ranking[:3]:
                top3 += 1
            if truth in ranking[:5]:
                top5 += 1
                mrr_sum += 1.0 / (ranking.index(truth) + 1)

    if n:
        print(f"\n=== Task 1 Self-score ===")
        print(f"  Top-1: {top1/n:.4f}  Top-3: {top3/n:.4f}  "
              f"Top-5: {top5/n:.4f}  MRR: {mrr_sum/n:.4f}  (n={n})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default="gpt:small",
                    help="ngram | gpt:tiny|small|large | hf:<name>")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to saved GPT/HF checkpoint")
    ap.add_argument("--valid",     default=None,
                    help="eval_input_valid.csv (default: self_eval/)")
    ap.add_argument("--anomaly",   default=None,
                    help="eval_input_anomaly.csv (default: self_eval/)")
    ap.add_argument("--out",       default="submissions",
                    help="Output directory for submission CSVs")
    ap.add_argument("--tasks",     nargs="+", default=["1", "2", "3"],
                    help="Which tasks to run (default: 1 2 3)")
    ap.add_argument("--score",     action="store_true",
                    help="Self-score using self_eval/ ground truth after running")
    args = ap.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[eval_runner] model={args.model}  device={device}")

    # Resolve input file paths
    valid_path   = args.valid   or "self_eval/eval_input_valid.csv"
    anomaly_path = args.anomaly or "self_eval/eval_input_anomaly.csv"

    if not Path(valid_path).exists() and ("1" in args.tasks or "2" in args.tasks):
        print(f"[warn] {valid_path} not found — skipping Tasks 1 & 2. "
              "Run make_selfeval.py first, or provide --valid path.")
        args.tasks = [t for t in args.tasks if t == "3"]

    if not Path(anomaly_path).exists() and "3" in args.tasks:
        print(f"[warn] {anomaly_path} not found — skipping Task 3.")
        args.tasks = [t for t in args.tasks if t != "3"]

    if not args.tasks:
        print("No tasks to run. Exiting.")
        return

    # Load model (shared across tasks)
    model = load_model(args.model, args.checkpoint, device)

    # Build run tag for output filenames
    tag = args.model.replace(":", "_").replace("/", "-")
    if args.checkpoint:
        tag += f"_ckpt"

    # Task 1 & 2
    if "1" in args.tasks or "2" in args.tasks:
        valid_rows = load_valid_csv(valid_path)
        if "1" in args.tasks:
            run_task1(model, valid_rows,
                      f"{args.out}/task1_{tag}.csv")
        if "2" in args.tasks:
            run_task2(model, valid_rows,
                      f"{args.out}/task2_{tag}.csv")

    # Task 3
    if "3" in args.tasks:
        anomaly_rows = load_anomaly_csv(anomaly_path)
        run_task3(model, anomaly_rows,
                  f"{args.out}/task3_{tag}.csv")

    # Self-scoring
    if args.score:
        gt_valid   = "self_eval/ground_truth_valid.csv"
        gt_anomaly = "self_eval/ground_truth_anomaly.csv"
        if "1" in args.tasks and Path(gt_valid).exists():
            score_task1(f"{args.out}/task1_{tag}.csv", gt_valid)
        if "3" in args.tasks and Path(gt_anomaly).exists():
            score_task3(f"{args.out}/task3_{tag}.csv", gt_anomaly)

    print(f"\n[done] Submissions written to {args.out}/")


if __name__ == "__main__":
    main()

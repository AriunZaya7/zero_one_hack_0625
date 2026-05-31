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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FAMILIES, load_all, train_val_split
from tokenizer import StepTokenizer
from anomaly import AnomalyDetector

SEED = 42
OFFICIAL_FILENAMES = {
    "1": "nextstep.csv",
    "2": "completion.csv",
    "3": "anomaly.csv",
}


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

def resolve_device(model_spec: str) -> str:
    """Return a device without requiring torch for dependency-free baselines."""
    if model_spec.split(":", 1)[0] == "ngram":
        return "cpu"
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for GPT inference. Use --model ngram for "
            "the dependency-free fallback, or install requirements.txt."
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_spec: str, checkpoint: str | None, device: str, allow_random_init: bool = False):
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
            if not allow_random_init:
                raise SystemExit(
                    "GPT checkpoint missing. Refusing to emit random-init submission CSVs. "
                    "Pass a valid --checkpoint, or use --model ngram for the safe baseline."
                )
            raw = build_gpt(tok, size=spec or "small")
            print(f"[model] GPT {spec or 'small'} random init (no checkpoint)")
        return GPTModel(raw, tok, device=device)

    raise ValueError(f"Unknown model spec: {model_spec!r}")


def output_path(out_dir: str, task: str, tag: str, official_names: bool) -> str:
    if official_names:
        return f"{out_dir}/{OFFICIAL_FILENAMES[task]}"
    return f"{out_dir}/task{task}_{tag}.csv"


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


def _levenshtein(seq1: list[str], seq2: list[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _major_block(step: str) -> str:
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def _block_signature(seq: list[str]) -> list[str]:
    sig: list[str] = []
    prev = None
    for step in seq:
        block = _major_block(step)
        if block != prev:
            sig.append(block)
            prev = block
    return sig


def _token_accuracy(pred: list[str], ref: list[str]) -> float:
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred, ref)) / n


def score_task2(pred_path: str, gt_path: str):
    """Self-score Task 2 against ground_truth_valid.csv."""
    gt = {}
    with open(gt_path, newline="") as f:
        for r in csv.DictReader(f):
            remaining = r.get("_REMAINING", "")
            if not remaining and r.get("FULL_SEQUENCE") and r.get("PARTIAL_SEQUENCE"):
                partial = _unpipe(r["PARTIAL_SEQUENCE"])
                full = _unpipe(r["FULL_SEQUENCE"])
                remaining = _pipe(full[len(partial):])
            gt[r["EXAMPLE_ID"]] = _unpipe(remaining)

    ned = []
    exact = []
    token = []
    block = []
    with open(pred_path, newline="") as f:
        for r in csv.DictReader(f):
            ref = gt.get(r["EXAMPLE_ID"])
            if ref is None:
                continue
            pred = _unpipe(r["PREDICTED_SEQUENCE"])
            denom = max(len(pred), len(ref), 1)
            ned.append(_levenshtein(pred, ref) / denom)
            exact.append(pred == ref)
            token.append(_token_accuracy(pred, ref))
            block.append(_token_accuracy(_block_signature(pred), _block_signature(ref)))

    n = len(ned)
    if n:
        print(f"\n=== Task 2 Self-score ===")
        print(f"  Exact Match Rate          : {sum(exact)/n:.4f}")
        print(f"  Normalized Edit Distance  : {sum(ned)/n:.4f}")
        print(f"  Token Accuracy            : {sum(token)/n:.4f}")
        print(f"  Block-level Accuracy      : {sum(block)/n:.4f}  (n={n})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default="gpt:small",
                    help="ngram | gpt:tiny|small|large")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to saved GPT checkpoint")
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
    ap.add_argument("--official-names", action="store_true",
                    help="Write nextstep.csv, completion.csv, anomaly.csv instead of tagged filenames")
    ap.add_argument("--allow-random-init", action="store_true",
                    help="Allow GPT random initialisation when --checkpoint is missing (never use for final submission)")
    args = ap.parse_args()

    device = resolve_device(args.model)
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
    model = load_model(args.model, args.checkpoint, device, allow_random_init=args.allow_random_init)

    # Build run tag for output filenames
    tag = args.model.replace(":", "_").replace("/", "-")
    if args.checkpoint:
        tag += f"_ckpt"

    # Task 1 & 2
    if "1" in args.tasks or "2" in args.tasks:
        valid_rows = load_valid_csv(valid_path)
        if "1" in args.tasks:
            run_task1(model, valid_rows,
                      output_path(args.out, "1", tag, args.official_names))
        if "2" in args.tasks:
            run_task2(model, valid_rows,
                      output_path(args.out, "2", tag, args.official_names))

    # Task 3
    if "3" in args.tasks:
        anomaly_rows = load_anomaly_csv(anomaly_path)
        run_task3(model, anomaly_rows,
                  output_path(args.out, "3", tag, args.official_names))

    # Self-scoring
    if args.score:
        gt_valid   = "self_eval/ground_truth_valid.csv"
        gt_anomaly = "self_eval/ground_truth_anomaly.csv"
        if "1" in args.tasks and Path(gt_valid).exists():
            score_task1(output_path(args.out, "1", tag, args.official_names), gt_valid)
        if "2" in args.tasks and Path(gt_valid).exists():
            score_task2(output_path(args.out, "2", tag, args.official_names), gt_valid)
        if "3" in args.tasks and Path(gt_anomaly).exists():
            score_task3(output_path(args.out, "3", tag, args.official_names), gt_anomaly)

    print(f"\n[done] Submissions written to {args.out}/")


if __name__ == "__main__":
    main()

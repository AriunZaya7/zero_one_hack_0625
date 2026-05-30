"""
Run any solution model on local self-eval data or organizer eval inputs.

Local metrics JSON:
    python -m solution.run_eval --model ngram --n 3 --out solution/results/ngram_3.json

Official submission files:
    python -m solution.run_eval --model xgboost --checkpoint solution/checkpoints/xgboost \
        --eval_dir path/to/eval --submission_dir solution/submission_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

from solution.data.generator import build_anomaly_set
from solution.data.loader import load_all_families, train_val_split
from solution.eval import metrics as M
from solution.eval import rules as R
from solution.tasks.adapters import make_adapter
from solution.tasks.run_tasks import eval_anomaly, eval_next_step_and_completion


def parse_families(values, available):
    if not values or values == ["all"]:
        return set(available)
    return {value.upper() for value in values}


def family_records(fams, selected, max_per_family=None):
    records = []
    for family in sorted(selected):
        seqs = fams.get(family, [])
        if max_per_family:
            seqs = seqs[:max_per_family]
        records.extend({"family": family, "steps": seq} for seq in seqs)
    return records


def build_adapter(args, fams):
    if args.model == "ngram":
        from solution.models.ngram import NGramModel

        train_families = parse_families(args.train_families, set(fams))
        train_seqs = []
        for family, seqs in fams.items():
            if family in train_families:
                train, _ = train_val_split(seqs, 0.1, args.seed)
                train_seqs.extend(train)
        model = NGramModel(n=args.n).fit(train_seqs)
        return make_adapter("ngram", model), f"ngram_n{args.n}", sorted(train_families)

    if args.model == "transformer":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for transformer")
        import torch

        from solution.data.vocab import Vocab
        from solution.models.transformer import ProcessTransformer

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt["config"]
        model = ProcessTransformer(**cfg)
        model.load_state_dict(ckpt["model_state"])
        vocab = Vocab.load(os.path.join(os.path.dirname(args.checkpoint), "vocab.json"))
        device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        return make_adapter("transformer", model, vocab, device), "transformer", []

    if args.model == "llm":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for llm")
        from solution.models.llm import LLMModel

        model = LLMModel(args.base_model, checkpoint_path=args.checkpoint)
        return make_adapter("llm", model), "llm", []

    if args.model == "xgboost":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for xgboost")
        from solution.models.boosting import XGBoostStepModel

        model = XGBoostStepModel.load(args.checkpoint)
        return make_adapter("xgboost", model), "xgboost", []

    if args.model == "catboost":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for catboost")
        from solution.models.boosting import CatBoostStepModel

        model = CatBoostStepModel.load(args.checkpoint)
        return make_adapter("catboost", model), "catboost", []

    raise ValueError(args.model)


def load_eval_valid(eval_dir: str) -> list[dict]:
    path = os.path.join(eval_dir, "eval_input_valid.csv")
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "example_id": row["EXAMPLE_ID"],
                    "family": row.get("FAMILY") or None,
                    "completion_fraction": row.get("COMPLETION_FRACTION"),
                    "partial": [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()],
                }
            )
    return rows


def load_eval_anomaly(eval_dir: str) -> list[dict]:
    path = os.path.join(eval_dir, "eval_input_anomaly.csv")
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "example_id": row["EXAMPLE_ID"],
                    "family": row.get("FAMILY") or None,
                    "steps": [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()],
                }
            )
    return rows


def calibrate_threshold(adapter, records) -> float:
    scores, labels = [], []
    for record in records:
        scores.append(-adapter.seq_log_prob(record["steps"], family=record.get("family")))
        labels.append(0 if record["is_valid"] == 1 else 1)
    return M.find_threshold(scores, labels)


def valid_probability(score: float, threshold: float) -> float:
    # score is anomaly-like: higher means less valid. At threshold, p(valid)=0.5.
    z = max(min(score - threshold, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(z))


def write_submission_files(adapter, valid_rows, anomaly_rows, submission_dir, threshold):
    os.makedirs(submission_dir, exist_ok=True)

    next_path = os.path.join(submission_dir, "nextstep.csv")
    with open(next_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for row in valid_rows:
            preds = adapter.top_k_next(row["partial"], 5, family=row["family"])
            preds = (preds + ["", "", "", "", ""])[:5]
            writer.writerow([row["example_id"], *preds])

    completion_path = os.path.join(submission_dir, "completion.csv")
    with open(completion_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for row in valid_rows:
            pred = adapter.complete(row["partial"], family=row["family"])
            writer.writerow([row["example_id"], "|".join(pred)])

    anomaly_path = os.path.join(submission_dir, "anomaly.csv")
    with open(anomaly_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for row in anomaly_rows:
            rule = R.attribute_anomaly(row["steps"], family=row["family"])
            score = -adapter.seq_log_prob(row["steps"], family=row["family"])
            is_invalid = rule is not None or score >= threshold
            writer.writerow(
                [
                    row["example_id"],
                    0 if is_invalid else 1,
                    f"{valid_probability(score, threshold):.6f}",
                    rule or "",
                ]
            )

    return {"nextstep": next_path, "completion": completion_path, "anomaly": anomaly_path}


def local_metrics(adapter, fams, args, model_name, train_families):
    eval_families = parse_families(args.families, set(fams))
    eval_records = family_records(fams, eval_families, max_per_family=args.max_eval_seqs)
    print(f"eval families={sorted(eval_families)} | sequences={len(eval_records)}")

    t1, t2 = eval_next_step_and_completion(adapter, eval_records)

    anomaly_base = [r["steps"] for r in eval_records]
    anomaly_records = build_anomaly_set(anomaly_base, seed=args.seed)
    anomaly_val = build_anomaly_set(anomaly_base, seed=args.seed + 1)
    t3 = eval_anomaly(adapter, anomaly_records, val_records=anomaly_val)

    by_family = {}
    for family in sorted(eval_families):
        records = [r for r in eval_records if r["family"] == family]
        if not records:
            continue
        ft1, ft2 = eval_next_step_and_completion(adapter, records)
        by_family[family] = {"task1": ft1, "task2": ft2}

    return {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "families": sorted(eval_families),
        "train_families": train_families,
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "by_family": by_family,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ngram", "transformer", "llm", "xgboost", "catboost"])
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--train_families", nargs="+", default=["all"])
    parser.add_argument("--families", nargs="+", default=["all"], help="local eval families, or 'all'")
    parser.add_argument("--max_eval_seqs", type=int, default=100)
    parser.add_argument("--eval_dir", default=None)
    parser.add_argument("--submission_dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    fams = load_all_families(args.data_dir)
    adapter, model_name, train_families = build_adapter(args, fams)

    t0 = time.time()
    result = None

    if args.eval_dir:
        valid_rows = load_eval_valid(args.eval_dir)
        anomaly_rows = load_eval_anomaly(args.eval_dir)
        calib_base = [r["steps"] for r in family_records(fams, set(fams), max_per_family=args.max_eval_seqs)]
        calib_records = build_anomaly_set(calib_base, seed=args.seed)
        threshold = calibrate_threshold(adapter, calib_records)
        submission_dir = args.submission_dir or os.path.join("solution", "submission_outputs", model_name)
        paths = write_submission_files(adapter, valid_rows, anomaly_rows, submission_dir, threshold)
        print("wrote submission files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        result = {
            "model": model_name,
            "checkpoint": args.checkpoint,
            "eval_dir": args.eval_dir,
            "submission_dir": submission_dir,
            "submission_files": paths,
            "anomaly_threshold": threshold,
        }

    if args.out or not args.eval_dir:
        metrics = local_metrics(adapter, fams, args, model_name, train_families)
        result = {**(result or {}), **metrics}

    if result is not None:
        result["elapsed_s"] = round(time.time() - t0, 1)
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = args.out
    if out_path is None and not args.eval_dir:
        out_path = os.path.join("solution", "results", f"{model_name}.json")
    if out_path and result is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"saved -> {out_path}")

    if result and "task1" in result:
        print(
            json.dumps(
                {
                    "task1": result["task1"],
                    "task2": result["task2"],
                    "task3": {k: result["task3"][k] for k in ("binary_acc", "f1", "roc_auc") if k in result["task3"]},
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

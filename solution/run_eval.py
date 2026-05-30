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
from pathlib import Path
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


def apply_hybrid_manifest(args):
    if not args.hybrid_manifest:
        return
    manifest = json.loads(Path(args.hybrid_manifest).read_text(encoding="utf-8"))
    if args.model != "xgboost_hybrid":
        raise ValueError("--hybrid_manifest requires --model xgboost_hybrid")

    args.next_checkpoints = args.next_checkpoints or manifest.get("next_checkpoints")
    args.completion_checkpoint = args.completion_checkpoint or manifest.get("completion_checkpoint")
    args.anomaly_checkpoint = args.anomaly_checkpoint or manifest.get("anomaly_checkpoint")
    args.ensemble_weights = args.ensemble_weights or manifest.get("ensemble_weights")
    args.run_name = args.run_name or manifest.get("run_name")


def family_records(fams, selected, max_per_family=None):
    records = []
    for family in sorted(selected):
        seqs = fams.get(family, [])
        if max_per_family:
            seqs = seqs[:max_per_family]
        records.extend({"family": family, "steps": seq} for seq in seqs)
    return records


def local_eval_records(fams, eval_families, train_families, args):
    records = []
    trained = set(train_families)
    for family in sorted(eval_families):
        seqs = fams.get(family, [])
        if args.eval_split == "val" and family in trained:
            _, seqs = train_val_split(seqs, args.val_ratio, args.seed)
        if args.max_eval_seqs:
            seqs = seqs[:args.max_eval_seqs]
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
        return make_adapter("ngram", model), args.run_name or f"ngram_n{args.n}", sorted(train_families)

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
        return make_adapter("transformer", model, vocab, device), args.run_name or checkpoint_run_name(args.checkpoint, "transformer"), []

    if args.model == "llm":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for llm")
        from solution.models.llm import LLMModel

        model = LLMModel(args.base_model, checkpoint_path=args.checkpoint)
        return make_adapter("llm", model), args.run_name or checkpoint_run_name(args.checkpoint, "llm"), []

    if args.model == "xgboost":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for xgboost")
        from solution.models.boosting import XGBoostStepModel

        model = XGBoostStepModel.load(args.checkpoint)
        apply_beam_overrides(model, args)
        return (
            make_adapter("xgboost", model),
            args.run_name or checkpoint_run_name(args.checkpoint, "xgboost"),
            checkpoint_train_families(args.checkpoint),
        )

    if args.model == "xgboost_ensemble":
        checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
        if not checkpoints:
            raise ValueError("--checkpoints is required for xgboost_ensemble")
        from solution.models.boosting import BoostingEnsembleModel

        model = BoostingEnsembleModel.load_many(checkpoints, weights=args.ensemble_weights)
        apply_beam_overrides(model, args)
        train_families = sorted(set().union(*(checkpoint_train_families(path) for path in checkpoints)))
        name = args.run_name or "xgboost_ensemble_" + "_".join(Path(path).name for path in checkpoints)
        return make_adapter("xgboost", model), name, train_families

    if args.model == "xgboost_hybrid":
        next_paths = args.next_checkpoints or args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
        if not next_paths or not args.completion_checkpoint or not args.anomaly_checkpoint:
            raise ValueError(
                "xgboost_hybrid requires --next_checkpoints/--checkpoints, "
                "--completion_checkpoint, and --anomaly_checkpoint"
            )
        from solution.models.boosting import BoostingTaskHybridModel

        model = BoostingTaskHybridModel.load(
            next_paths=next_paths,
            completion_path=args.completion_checkpoint,
            anomaly_path=args.anomaly_checkpoint,
            next_weights=args.ensemble_weights,
        )
        apply_beam_overrides(model, args)
        all_paths = list(next_paths) + [args.completion_checkpoint, args.anomaly_checkpoint]
        train_families = sorted(set().union(*(checkpoint_train_families(path) for path in all_paths)))
        name = args.run_name or "xgboost_hybrid_" + "_".join(Path(path).name for path in all_paths)
        return make_adapter("xgboost", model), name, train_families

    if args.model == "catboost":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for catboost")
        from solution.models.boosting import CatBoostStepModel

        model = CatBoostStepModel.load(args.checkpoint)
        apply_beam_overrides(model, args)
        return (
            make_adapter("catboost", model),
            args.run_name or checkpoint_run_name(args.checkpoint, "catboost"),
            checkpoint_train_families(args.checkpoint),
        )

    raise ValueError(args.model)


def apply_beam_overrides(model, args):
    if not hasattr(model, "config"):
        return
    if args.beam_width is not None:
        model.config.beam_width = args.beam_width
    if args.beam_branching is not None:
        model.config.beam_branching = args.beam_branching


def checkpoint_run_name(checkpoint: str | None, fallback: str) -> str:
    meta = checkpoint_metadata(checkpoint)
    if meta:
        return meta.get("run_name") or Path(checkpoint).name or fallback
    return Path(checkpoint).name if checkpoint else fallback


def checkpoint_metadata(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {}
    path = Path(checkpoint)
    meta_path = path / "training_meta.json" if path.is_dir() else path.parent / "training_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def checkpoint_train_families(checkpoint: str | None) -> list[str]:
    meta = checkpoint_metadata(checkpoint)
    families = meta.get("train_families") or meta.get("selected_families") or []
    return sorted(families)


def infer_eval_strategy(train_families: list[str], eval_families: set[str]) -> str:
    train = set(train_families)
    if not train:
        return "unknown"
    if eval_families == train and len(train) >= 3:
        return "all_family"
    if eval_families and eval_families.isdisjoint(train):
        return "holdout_family"
    if eval_families.issubset(train):
        return "in_family_subset"
    if eval_families - train:
        return "mixed_in_out_family"
    return "unknown"


def flatten_metrics(prefix: str, value):
    out = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            out.update(flatten_metrics(f"{prefix}/{key}" if prefix else key, nested))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        out[prefix] = value
    return out


def log_result_to_wandb(args, result):
    if args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")
        return
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"eval_{result.get('model', args.model)}",
            mode=args.wandb_mode,
            config=vars(args),
        )
        run.log(flatten_metrics("eval", {
            "task1": result.get("task1", {}),
            "task2": result.get("task2", {}),
            "task3": result.get("task3", {}),
        }))
        run.finish()
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")


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
    eval_records = local_eval_records(fams, eval_families, train_families, args)
    print(f"eval families={sorted(eval_families)} | split={args.eval_split} | sequences={len(eval_records)}")

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
        "model_type": args.model,
        "checkpoint": args.checkpoint,
        "checkpoints": args.checkpoints,
        "next_checkpoints": args.next_checkpoints,
        "completion_checkpoint": args.completion_checkpoint,
        "anomaly_checkpoint": args.anomaly_checkpoint,
        "families": sorted(eval_families),
        "train_families": train_families,
        "eval_split": args.eval_split,
        "eval_strategy": infer_eval_strategy(train_families, eval_families),
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "by_family": by_family,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ngram", "transformer", "llm", "xgboost", "xgboost_ensemble", "xgboost_hybrid", "catboost"])
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None, help="checkpoint dirs for xgboost_ensemble")
    parser.add_argument("--next_checkpoints", nargs="+", default=None, help="Task 1 checkpoint dirs for xgboost_hybrid")
    parser.add_argument("--completion_checkpoint", default=None, help="Task 2 checkpoint dir for xgboost_hybrid")
    parser.add_argument("--anomaly_checkpoint", default=None, help="Task 3 checkpoint dir for xgboost_hybrid")
    parser.add_argument("--hybrid_manifest", default=None, help="JSON recipe for xgboost_hybrid component checkpoints")
    parser.add_argument("--ensemble_weights", nargs="+", type=float, default=None)
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--train_families", nargs="+", default=["all"])
    parser.add_argument("--families", nargs="+", default=["all"], help="local eval families, or 'all'")
    parser.add_argument("--eval_split", default="val", choices=["val", "all"], help="use deterministic validation records for trained families")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_eval_seqs", type=int, default=100)
    parser.add_argument("--eval_dir", default=None)
    parser.add_argument("--submission_dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--beam_width", type=int, default=None, help="override boosted-tree completion beam width")
    parser.add_argument("--beam_branching", type=int, default=None, help="override boosted-tree per-step branching")
    parser.add_argument("--wandb_project", default="zero-one-hack")
    parser.add_argument("--wandb_run", default=None)
    parser.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main():
    args = get_args()
    apply_hybrid_manifest(args)
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
            "model_type": args.model,
            "checkpoint": args.checkpoint,
            "checkpoints": args.checkpoints,
            "next_checkpoints": args.next_checkpoints,
            "completion_checkpoint": args.completion_checkpoint,
            "anomaly_checkpoint": args.anomaly_checkpoint,
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
        log_result_to_wandb(args, result)

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

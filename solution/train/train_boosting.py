"""
Train XGBoost or CatBoost next-step baselines.

Examples:
    python -m solution.train.train_boosting --model xgboost --data_dir training_data --out solution/checkpoints/xgboost
    python -m solution.train.train_boosting --model catboost --data_dir training_data --out solution/checkpoints/catboost
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np

from solution.data.loader import load_all_families, train_val_split
from solution.models.boosting import (
    BoostingConfig,
    CatBoostStepModel,
    XGBoostStepModel,
)


def parse_families(value: list[str] | None, available: set[str]) -> set[str]:
    if not value or value == ["all"]:
        return set(available)
    return {family.upper() for family in value}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["xgboost", "catboost"])
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--out", default=None)
    parser.add_argument("--train_families", nargs="+", default=["all"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_examples", type=int, default=250_000)
    parser.add_argument("--max_val_examples", type=int, default=50_000)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--context_size", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--feature_version", type=int, default=2, choices=[1, 2])
    parser.add_argument("--sample_strategy", default="uniform", choices=["uniform", "family_balanced"])
    parser.add_argument("--family_dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb_project", default="zero-one-hack")
    parser.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])

    # Backend knobs. Only relevant ones are passed to each backend.
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--beam_branching", type=int, default=8)
    return parser.parse_args()


def default_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    examples = "all" if args.max_train_examples == 0 else str(args.max_train_examples)
    suffix = f"fv{args.feature_version}_{args.sample_strategy}"
    if args.model == "xgboost":
        return f"xgboost_{examples}_d{args.max_depth}_eta{args.eta}_it{args.iterations}_{suffix}"
    return f"catboost_{examples}_d{args.depth}_lr{args.learning_rate}_it{args.iterations}_{suffix}"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def extract_training_history(model) -> list[dict]:
    estimator = model.estimator
    history = []

    try:
        raw = estimator.evals_result()
    except Exception:
        try:
            raw = estimator.get_evals_result()
        except Exception:
            raw = {}

    for dataset, metrics in raw.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                while len(history) <= i:
                    history.append({"iteration": len(history)})
                key = f"{dataset}/{metric}"
                try:
                    history[i][key] = float(value)
                except (TypeError, ValueError):
                    pass

    best_iteration = getattr(estimator, "best_iteration", None)
    if best_iteration is None and hasattr(estimator, "get_best_iteration"):
        try:
            best_iteration = estimator.get_best_iteration()
        except Exception:
            best_iteration = None
    best_score = getattr(estimator, "best_score", None)
    for row in history:
        row["best_iteration"] = best_iteration
        if best_score is not None:
            try:
                row["best_score"] = float(best_score)
            except (TypeError, ValueError):
                pass
    return history


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def maybe_start_wandb(args, run_name: str, selected: set[str]):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")
        return None

    try:
        return wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=args.wandb_mode,
            config={**vars(args), "selected_families": sorted(selected)},
        )
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")
        return None


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.max_train_examples = min(args.max_train_examples, 1000)
        args.max_val_examples = min(args.max_val_examples, 1000)
        args.iterations = min(args.iterations, 5)

    run_name = safe_name(default_run_name(args))
    if args.out is None:
        args.out = f"solution/checkpoints/{run_name}"

    families = load_all_families(args.data_dir)
    selected = parse_families(args.train_families, set(families))
    train_records, val_records = [], []
    for family, seqs in families.items():
        if family not in selected:
            continue
        train_seqs, val_seqs = train_val_split(seqs, args.val_ratio, args.seed)
        train_records.extend({"family": family, "steps": seq} for seq in train_seqs)
        val_records.extend({"family": family, "steps": seq} for seq in val_seqs)

    if not train_records:
        raise ValueError(f"No training sequences found for families={sorted(selected)}")

    config = BoostingConfig(
        context_size=args.context_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        beam_width=args.beam_width,
        beam_branching=args.beam_branching,
        feature_version=args.feature_version,
    )
    model = XGBoostStepModel(config) if args.model == "xgboost" else CatBoostStepModel(config)
    params = {}
    if args.model == "xgboost":
        params = {
            "max_depth": args.max_depth,
            "learning_rate": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "gamma": args.gamma,
            "max_bin": args.max_bin,
            "early_stopping_rounds": args.early_stopping_rounds,
        }
    else:
        params = {
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        }

    print(
        f"training {run_name} | model={args.model} | families={sorted(selected)} | "
        f"train_seqs={len(train_records)} val_seqs={len(val_records)} "
        f"max_examples={args.max_train_examples or 'all'} iterations={args.iterations} "
        f"features=v{args.feature_version} sample={args.sample_strategy} family_dropout={args.family_dropout}"
    )
    start = time.time()
    wb = maybe_start_wandb(args, run_name, selected)
    model.fit(
        train_records,
        val_records=val_records,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        iterations=args.iterations,
        device=args.device,
        verbose=args.verbose,
        sample_strategy=args.sample_strategy,
        family_dropout=args.family_dropout,
        **params,
    )

    model.save(args.out)
    history = extract_training_history(model)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "training_log.jsonl")
    write_jsonl(log_path, history)

    if wb:
        for row in history:
            wb.log({f"train/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

    meta = {
        "model": args.model,
        "run_name": run_name,
        "train_families": sorted(selected),
        "train_sequences": len(train_records),
        "val_sequences": len(val_records),
        "max_train_examples": args.max_train_examples,
        "max_val_examples": args.max_val_examples,
        "iterations": args.iterations,
        "feature_version": args.feature_version,
        "sample_strategy": args.sample_strategy,
        "family_dropout": args.family_dropout,
        "elapsed_s": round(time.time() - start, 1),
        "args": vars(args),
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if wb:
        wb.log({
            "summary/elapsed_s": meta["elapsed_s"],
            "summary/train_sequences": len(train_records),
            "summary/val_sequences": len(val_records),
            "summary/max_train_examples": args.max_train_examples,
            "summary/max_val_examples": args.max_val_examples,
        })
        try:
            import wandb

            artifact = wandb.Artifact(run_name, type="model-metadata")
            artifact.add_file(os.path.join(args.out, "training_meta.json"))
            artifact.add_file(log_path)
            wb.log_artifact(artifact)
        except Exception:
            pass
        wb.finish()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

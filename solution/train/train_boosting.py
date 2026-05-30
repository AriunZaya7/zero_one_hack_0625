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
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--smoke", action="store_true")

    # Backend knobs. Only relevant ones are passed to each backend.
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.max_train_examples = min(args.max_train_examples, 1000)
        args.max_val_examples = min(args.max_val_examples, 1000)
        args.iterations = min(args.iterations, 5)

    if args.out is None:
        args.out = f"solution/checkpoints/{args.model}"

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
    )
    model = XGBoostStepModel(config) if args.model == "xgboost" else CatBoostStepModel(config)
    params = {}
    if args.model == "xgboost":
        params = {
            "max_depth": args.max_depth,
            "learning_rate": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
    else:
        params = {
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        }

    print(
        f"training {args.model} | families={sorted(selected)} | "
        f"train_seqs={len(train_records)} val_seqs={len(val_records)} "
        f"max_examples={args.max_train_examples or 'all'} iterations={args.iterations}"
    )
    start = time.time()
    model.fit(
        train_records,
        val_records=val_records,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        iterations=args.iterations,
        device=args.device,
        verbose=args.verbose,
        **params,
    )

    model.save(args.out)
    meta = {
        "model": args.model,
        "train_families": sorted(selected),
        "train_sequences": len(train_records),
        "val_sequences": len(val_records),
        "max_train_examples": args.max_train_examples,
        "max_val_examples": args.max_val_examples,
        "iterations": args.iterations,
        "elapsed_s": round(time.time() - start, 1),
        "args": vars(args),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

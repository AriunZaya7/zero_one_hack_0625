"""
Run reproducible XGBoost comparison sweeps.

The script trains XGBoost variants, evaluates each variant with the normal
run_eval path, and writes both per-run result JSONs and a compact summary.

Examples:
    python -m solution.train.sweep_xgboost --preset fast --strategy both --max_eval_seqs 50
    python -m solution.train.sweep_xgboost --preset full --strategy holdout --max_train_examples 0 --max_eval_seqs 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


FAMILIES = ["IC", "IGBT", "MOSFET"]

TRIAL_PRESETS = {
    "focused": [
        {
            "trial": "legacy_features",
            "args": ["--feature_version", "1"],
        },
        {
            "trial": "regularized_v2",
            "args": [
                "--feature_version", "2",
                "--max_depth", "6",
                "--eta", "0.06",
                "--min_child_weight", "3.0",
                "--reg_lambda", "2.0",
                "--subsample", "0.85",
                "--colsample_bytree", "0.85",
            ],
        },
        {
            "trial": "regularized_v2_family_dropout",
            "args": [
                "--feature_version", "2",
                "--family_dropout", "0.35",
                "--max_depth", "6",
                "--eta", "0.06",
                "--min_child_weight", "3.0",
                "--reg_lambda", "2.0",
                "--subsample", "0.85",
                "--colsample_bytree", "0.85",
            ],
        },
    ],
    "fast": [
        {
            "trial": "default_v2",
            "args": ["--feature_version", "2"],
        },
        {
            "trial": "balanced_v2",
            "args": ["--feature_version", "2", "--sample_strategy", "family_balanced"],
        },
        {
            "trial": "regularized_v2",
            "args": [
                "--feature_version", "2",
                "--max_depth", "6",
                "--eta", "0.06",
                "--min_child_weight", "3.0",
                "--reg_lambda", "2.0",
                "--subsample", "0.85",
                "--colsample_bytree", "0.85",
            ],
        },
    ],
    "full": [
        {
            "trial": "default_v2",
            "args": ["--feature_version", "2"],
        },
        {
            "trial": "balanced_v2",
            "args": ["--feature_version", "2", "--sample_strategy", "family_balanced"],
        },
        {
            "trial": "shallow_regularized_v2",
            "args": [
                "--feature_version", "2",
                "--max_depth", "6",
                "--eta", "0.05",
                "--min_child_weight", "4.0",
                "--reg_lambda", "3.0",
                "--reg_alpha", "0.05",
                "--subsample", "0.85",
                "--colsample_bytree", "0.85",
                "--early_stopping_rounds", "80",
            ],
        },
        {
            "trial": "deep_slow_v2",
            "args": [
                "--feature_version", "2",
                "--max_depth", "10",
                "--eta", "0.04",
                "--min_child_weight", "2.0",
                "--reg_lambda", "2.0",
                "--subsample", "0.9",
                "--colsample_bytree", "0.9",
                "--early_stopping_rounds", "100",
            ],
        },
        {
            "trial": "legacy_features",
            "args": ["--feature_version", "1"],
        },
    ],
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="fast", choices=sorted(TRIAL_PRESETS))
    parser.add_argument("--strategy", default="both", choices=["all", "holdout", "both"])
    parser.add_argument("--holdout_families", nargs="+", default=FAMILIES)
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--checkpoint_root", default="solution/checkpoints/xgboost_sweeps")
    parser.add_argument("--results_dir", default="solution/results")
    parser.add_argument("--sweep_name", default=None)
    parser.add_argument("--max_train_examples", type=int, default=100_000)
    parser.add_argument("--max_val_examples", type=int, default=20_000)
    parser.add_argument("--max_eval_seqs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--beam_branching", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="zero-one-hack")
    parser.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value).strip("_")


def run_command(command: list[str], dry_run: bool):
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def metric_row(result_path: Path, run_name: str, trial: str, strategy: str, train_families: list[str], eval_families: list[str]):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    t1 = result.get("task1", {})
    t2 = result.get("task2", {})
    t3 = result.get("task3", {})
    return {
        "run_name": run_name,
        "trial": trial,
        "strategy": strategy,
        "train_families": ",".join(train_families),
        "eval_families": ",".join(eval_families),
        "top1": t1.get("top1"),
        "top5": t1.get("top5"),
        "mrr": t1.get("mrr"),
        "completion_exact": t2.get("exact_match"),
        "completion_token_acc": t2.get("token_acc"),
        "completion_edit_dist": t2.get("norm_edit_dist"),
        "anomaly_f1": t3.get("f1"),
        "anomaly_auc": t3.get("roc_auc"),
        "elapsed_s": result.get("elapsed_s"),
        "result_path": str(result_path),
    }


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_eval_run(args, trial: dict, run_name: str, train_families: list[str], eval_families: list[str], strategy_label: str):
    checkpoint = Path(args.checkpoint_root) / run_name
    result_path = Path(args.results_dir) / f"{run_name}.json"
    if args.skip_existing and checkpoint.exists() and result_path.exists():
        print(f"[skip existing] {run_name}")
        return result_path

    common_train = [
        sys.executable, "-m", "solution.train.train_boosting",
        "--model", "xgboost",
        "--data_dir", args.data_dir,
        "--out", str(checkpoint),
        "--run_name", run_name,
        "--train_families", *train_families,
        "--max_train_examples", str(args.max_train_examples),
        "--max_val_examples", str(args.max_val_examples),
        "--iterations", str(args.iterations),
        "--beam_width", str(args.beam_width),
        "--beam_branching", str(args.beam_branching),
        "--seed", str(args.seed),
        "--wandb_project", args.wandb_project,
        "--wandb_mode", args.wandb_mode,
    ]
    run_command(common_train + trial["args"], args.dry_run)

    eval_cmd = [
        sys.executable, "-m", "solution.run_eval",
        "--model", "xgboost",
        "--checkpoint", str(checkpoint),
        "--run_name", run_name,
        "--families", *eval_families,
        "--max_eval_seqs", str(args.max_eval_seqs),
        "--out", str(result_path),
        "--beam_width", str(args.beam_width),
        "--beam_branching", str(args.beam_branching),
        "--seed", str(args.seed),
        "--wandb_project", args.wandb_project,
        "--wandb_run", f"eval_{run_name}",
        "--wandb_mode", args.wandb_mode,
    ]
    run_command(eval_cmd, args.dry_run)
    return result_path


def main():
    args = get_args()
    sweep_name = safe_name(args.sweep_name or time.strftime("xgb_sweep_%Y%m%d_%H%M%S"))
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    summary_dir = Path(args.results_dir) / "sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = summary_dir / f"{sweep_name}.jsonl"
    summary_csv = summary_dir / f"{sweep_name}.csv"

    rows = []
    strategies = []
    if args.strategy in {"all", "both"}:
        strategies.append(("all_family", FAMILIES, FAMILIES))
    if args.strategy in {"holdout", "both"}:
        for holdout in [family.upper() for family in args.holdout_families]:
            train = [family for family in FAMILIES if family != holdout]
            strategies.append((f"holdout_{holdout.lower()}", train, [holdout]))

    for trial in TRIAL_PRESETS[args.preset]:
        for strategy_label, train_families, eval_families in strategies:
            run_name = safe_name(f"{sweep_name}_{trial['trial']}_{strategy_label}")
            result_path = train_eval_run(args, trial, run_name, train_families, eval_families, strategy_label)
            if args.dry_run:
                continue
            row = metric_row(result_path, run_name, trial["trial"], strategy_label, train_families, eval_families)
            rows.append(row)
            append_jsonl(summary_jsonl, row)
            write_csv(summary_csv, rows)

    if args.dry_run:
        print(f"dry run only; summary would be written under {summary_dir}")
    else:
        print(f"summary jsonl -> {summary_jsonl}")
        print(f"summary csv   -> {summary_csv}")


if __name__ == "__main__":
    main()

"""
run_all_baselines.py
=====================
Runs all 3 baselines. All results tracked in WandB under the same project.

Usage:
    python run_all_baselines.py --train_dir ./training_data
    python run_all_baselines.py --train_dir ./training_data --skip_llm
"""

import subprocess
import sys
import argparse
import os


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARNING] {label} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../tracks/industrial-infineon/training_data")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--llm_model", default="gpt2",
                        help="HF model name for LLM baseline")
    parser.add_argument("--max_eval_examples", type=int, default=100)
    args = parser.parse_args()

    py = sys.executable
    common = ["--train_dir", args.train_dir, "--wandb_project", args.wandb_project, "--seed", str(args.seed)]

    run([py, "baseline_random.py"] + common, "Baseline 1: Random")
    run([py, "baseline_ngram.py", "--n", "3"] + common, "Baseline 2: N-gram (trigram)")

    if not args.skip_llm:
        run([py, "baseline_llm_zeroshot.py",
             "--model", args.llm_model,
             "--max_eval_examples", str(args.max_eval_examples)] + common,
            f"Baseline 3: LLM Zero-Shot ({args.llm_model})")
    else:
        print("\n[Skipped] Baseline 3: LLM Zero-Shot")

    print(f"""
{'='*60}
  ALL BASELINES COMPLETE
  Check results at: https://wandb.ai  (project: {args.wandb_project})
{'='*60}

All 3 runs are logged under project '{args.wandb_project}'.
Compare them side by side in the WandB dashboard.

Next: once eval files arrive, add --eval_valid and --eval_anomaly flags.
""")


if __name__ == "__main__":
    main()

"""
visualize_baselines.py
=======================
Reads the WandB export CSV and generates baseline comparison plots.

Usage:
    python visualize_baselines.py --csv wandb_export.csv
    python visualize_baselines.py --csv wandb_export.csv --out_dir ./plots
"""

import csv
import argparse
import os

# Use matplotlib - no extra deps needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Load WandB CSV ─────────────────────────────────────────────────────────────

def load_best_runs(csv_path: str) -> dict:
    """
    Load the WandB export CSV and keep only the best (finished) run per baseline.
    Returns a dict: { baseline_name: {metric: value, ...} }
    """
    runs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("State", "").strip() == "finished":
                runs.append(row)

    # Group by baseline name, keep latest finished run per type
    best = {}
    for run in runs:
        name = run.get("baseline", "").strip()
        if name not in best:
            best[name] = run

    print(f"Loaded {len(best)} finished baselines: {list(best.keys())}")
    return best


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Plot helpers ───────────────────────────────────────────────────────────────

COLORS = {
    "random":       "#888780",
    "ngram":        "#3266ad",
    "llm_zeroshot": "#c97b4b",
}

LABELS = {
    "random":       "Random",
    "ngram":        "N-gram (trigram)",
    "llm_zeroshot": "LLM zero-shot (GPT-2)",
}

def model_color(key):
    return COLORS.get(key, "#cccccc")

def model_label(key):
    return LABELS.get(key, key)


# ── Plot 1: Next-Step Prediction ───────────────────────────────────────────────

def plot_next_step(runs: dict, out_dir: str):
    metrics = {
        "Top-1": "next_step/top1_acc",
        "Top-3": "next_step/top3_acc",
        "Top-5": "next_step/top5_acc",
        "MRR":   "next_step/mrr",
    }

    model_keys = list(runs.keys())
    x = np.arange(len(metrics))
    width = 0.25
    n_models = len(model_keys)
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, key in enumerate(model_keys):
        values = [safe_float(runs[key].get(m, 0)) * 100 for m in metrics.values()]
        bars = ax.bar(x + offsets[i], values, width, label=model_label(key),
                      color=model_color(key), alpha=0.85, zorder=3)
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Next-step prediction", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_next_step.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Sequence Completion ────────────────────────────────────────────────

def plot_completion(runs: dict, out_dir: str):
    metrics = {
        "Exact match\n(60%)":  "completion/exact_match_60",
        "Token acc\n(60%)":    "completion/token_acc_60",
        "Exact match\n(80%)":  "completion/exact_match_80",
        "Token acc\n(80%)":    "completion/token_acc_80",
        # fallback keys used by random baseline
        "Exact match":         "completion/exact_match_rate",
        "Token acc":           "completion/token_accuracy",
    }

    # Normalise: some baselines use different key names
    def get_completion_vals(run):
        em = safe_float(run.get("completion/exact_match_60") or run.get("completion/exact_match_rate", 0))
        ta = safe_float(run.get("completion/token_acc_60") or run.get("completion/token_accuracy", 0))
        return [em * 100, ta * 100]

    plot_metrics = ["Exact match", "Token accuracy"]
    model_keys = list(runs.keys())
    x = np.arange(len(plot_metrics))
    width = 0.25
    n_models = len(model_keys)
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, key in enumerate(model_keys):
        values = get_completion_vals(runs[key])
        bars = ax.bar(x + offsets[i], values, width, label=model_label(key),
                      color=model_color(key), alpha=0.85, zorder=3)
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Sequence completion", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(5, ax.get_ylim()[1] * 1.2))
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_completion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: Anomaly Detection ──────────────────────────────────────────────────

def plot_anomaly(runs: dict, out_dir: str):
    model_keys = list(runs.keys())
    values = []
    labels = []
    colors = []

    for key in model_keys:
        val = safe_float(runs[key].get("anomaly/binary_accuracy") or
                         runs[key].get("anomaly/accuracy_on_valid_seqs", 0)) * 100
        values.append(val)
        labels.append(model_label(key))
        colors.append(model_color(key))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, zorder=3, width=0.5)

    for bar, val in zip(bars, values):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # Draw 50% reference line (random chance)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Random chance (50%)")

    ax.set_ylabel("Binary accuracy (%)", fontsize=11)
    ax.set_title("Anomaly detection", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_anomaly.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 4: Summary radar / overview ──────────────────────────────────────────

def plot_summary(runs: dict, out_dir: str):
    """One grouped bar chart with the key metric per task side by side."""
    task_labels = ["Next-step\ntop-1 acc", "Next-step\nMRR", "Completion\ntoken acc", "Anomaly\nacc"]
    model_keys = list(runs.keys())

    def get_summary_vals(run):
        return [
            safe_float(run.get("next_step/top1_acc", 0)) * 100,
            safe_float(run.get("next_step/mrr", 0)) * 100,
            safe_float(run.get("completion/token_acc_60") or run.get("completion/token_accuracy", 0)) * 100,
            safe_float(run.get("anomaly/binary_accuracy") or run.get("anomaly/accuracy_on_valid_seqs", 0)) * 100,
        ]

    x = np.arange(len(task_labels))
    width = 0.25
    n_models = len(model_keys)
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, key in enumerate(model_keys):
        values = get_summary_vals(runs[key])
        bars = ax.bar(x + offsets[i], values, width, label=model_label(key),
                      color=model_color(key), alpha=0.85, zorder=3)
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Baseline comparison — all tasks", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="wandb_export.csv", help="WandB export CSV file")
    parser.add_argument("--out_dir", default="./plots", help="Folder to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_best_runs(args.csv)

    print("\nGenerating plots...")
    plot_next_step(runs, args.out_dir)
    plot_completion(runs, args.out_dir)
    plot_anomaly(runs, args.out_dir)
    plot_summary(runs, args.out_dir)

    print(f"\nAll plots saved to: {args.out_dir}")
    print("Files: plot_next_step.png, plot_completion.png, plot_anomaly.png, plot_summary.png")


if __name__ == "__main__":
    main()
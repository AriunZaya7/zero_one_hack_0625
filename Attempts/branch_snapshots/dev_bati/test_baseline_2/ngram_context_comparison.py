"""
ngram_context_comparison.py
=============================
Trains multiple N-gram models with different context sizes (n=2 to n=50)
and compares their performance. Shows where performance peaks and where
sparsity kills the model.

Put this in test_baseline_2/ and run:
    python ngram_context_comparison.py

All runs tracked in WandB under the same project.
"""

import csv
import argparse
import random
import os
import math
from collections import defaultdict, Counter
import wandb
import matplotlib.pyplot as plt
import numpy as np


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_training_sequences(train_dir: str) -> list[list[str]]:
    sequences = []
    current_id = None
    current_seq = []
    files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    print(f"Found training files: {files}")
    for fname in sorted(files):
        fpath = os.path.join(train_dir, fname)
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_id = row.get("SEQUENCE_ID") or row.get("sequence_id")
                step = (row.get("STEP") or row.get("step", "")).strip()
                if not step:
                    continue
                if seq_id != current_id:
                    if current_seq:
                        sequences.append(current_seq)
                    current_id = seq_id
                    current_seq = [step]
                else:
                    current_seq.append(step)
    if current_seq:
        sequences.append(current_seq)
    print(f"Loaded {len(sequences)} total sequences")
    return sequences


def train_val_split(sequences, val_frac=0.1, seed=42):
    random.seed(seed)
    shuffled = sequences[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


# ── N-gram Model ───────────────────────────────────────────────────────────────

class NGramModel:
    def __init__(self, n: int, smoothing: float = 1e-6):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.vocab = set()
        self.unigram = Counter()  # fallback

    def train(self, sequences: list[list[str]]):
        for seq in sequences:
            padded = ["<START>"] * (self.n - 1) + seq + ["<END>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i: i + self.n - 1])
                next_step = padded[i + self.n - 1]
                self.counts[context][next_step] += 1
                self.vocab.add(next_step)
            # also build unigram for fallback
            for step in seq:
                self.unigram[step] += 1
        self.vocab.discard("<START>")

    def predict_next(self, context_steps: list[str], top_k: int = 5) -> list[str]:
        padded = ["<START>"] * (self.n - 1) + context_steps
        # Try progressively shorter contexts until we find a match (backoff)
        for length in range(self.n - 1, 0, -1):
            context = tuple(padded[-length:])
            counter = self.counts.get(context, Counter())
            if counter:
                preds = [s for s, _ in counter.most_common(top_k)]
                while len(preds) < top_k:
                    preds.append("<UNK>")
                return preds[:top_k]
        # Final fallback: unigram
        preds = [s for s, _ in self.unigram.most_common(top_k)]
        while len(preds) < top_k:
            preds.append("<UNK>")
        return preds[:top_k]

    def sequence_log_prob(self, sequence: list[str]) -> float:
        padded = ["<START>"] * (self.n - 1) + sequence + ["<END>"]
        log_prob = 0.0
        vocab_size = len(self.vocab) + 1
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i: i + self.n - 1])
            next_step = padded[i + self.n - 1]
            counter = self.counts.get(context, Counter())
            total = sum(counter.values())
            count = counter.get(next_step, 0)
            prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            log_prob += math.log(prob)
        return log_prob / max(len(sequence), 1)

    def complete_sequence(self, partial: list[str], max_steps: int = 200) -> list[str]:
        seq = list(partial)
        prev = None
        for _ in range(max_steps):
            next_step = self.predict_next(seq, top_k=1)[0]
            if next_step in ("<END>", "<UNK>") or next_step == prev:
                break
            seq.append(next_step)
            prev = next_step
            if next_step == "SHIP LOT":
                break
        return seq[len(partial):]

    def n_unique_contexts(self) -> int:
        return len(self.counts)

    def coverage(self, val_sequences: list[list[str]]) -> float:
        """Fraction of val transitions where the exact context was seen in training."""
        hits = total = 0
        for seq in val_sequences[:100]:
            padded = ["<START>"] * (self.n - 1) + seq
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i: i + self.n - 1])
                total += 1
                if context in self.counts:
                    hits += 1
        return hits / max(total, 1)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: NGramModel, val_sequences: list[list[str]], seed: int = 42) -> dict:
    random.seed(seed)

    # ── Next-step prediction ──────────────────────────────────────────────────
    examples = []
    for seq in val_sequences:
        for i in range(1, len(seq)):
            examples.append((seq[:i], seq[i]))
    random.shuffle(examples)
    examples = examples[:3000]

    top1 = top3 = top5 = mrr = 0
    for partial, true_next in examples:
        preds = model.predict_next(partial, top_k=5)
        if true_next in preds:
            rank = preds.index(true_next) + 1
            mrr += 1.0 / rank
            if rank <= 5: top5 += 1
            if rank <= 3: top3 += 1
            if rank == 1: top1 += 1
    n = len(examples)

    # ── Sequence completion ───────────────────────────────────────────────────
    samples = random.sample(val_sequences, min(150, len(val_sequences)))
    token_acc_60 = token_acc_80 = 0
    for seq in samples:
        for frac, acc_var in [(0.6, "60"), (0.8, "80")]:
            cut = int(len(seq) * frac)
            partial = seq[:cut]
            true_rest = seq[cut:]
            predicted = model.complete_sequence(partial, max_steps=len(seq))
            matches = sum(a == b for a, b in zip(predicted, true_rest))
            ta = matches / max(len(true_rest), 1)
            if frac == 0.6:
                token_acc_60 += ta
            else:
                token_acc_80 += ta
    token_acc_60 /= len(samples)
    token_acc_80 /= len(samples)

    # ── Anomaly score (log-prob spread) ──────────────────────────────────────
    lp_scores = [model.sequence_log_prob(seq) for seq in val_sequences[:150]]
    mean_lp = sum(lp_scores) / len(lp_scores)
    std_lp = (sum((x - mean_lp) ** 2 for x in lp_scores) / len(lp_scores)) ** 0.5

    return {
        "next_step/top1_acc": top1 / n,
        "next_step/top3_acc": top3 / n,
        "next_step/top5_acc": top5 / n,
        "next_step/mrr": mrr / n,
        "completion/token_acc_60": token_acc_60,
        "completion/token_acc_80": token_acc_80,
        "anomaly/mean_log_prob": mean_lp,
        "anomaly/std_log_prob": std_lp,
        "model/unique_contexts": model.n_unique_contexts(),
        "model/val_coverage": model.coverage(val_sequences),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(all_results: list[dict], n_values: list[int], out_dir: str):
    """Generate comparison plots across all n values."""

    def extract(key):
        return [r.get(key, 0) for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("N-gram context size comparison", fontsize=14, fontweight="bold")

    # ── Plot 1: Next-step accuracy vs n ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(n_values, [x * 100 for x in extract("next_step/top1_acc")],
            "o-", color="#3266ad", label="Top-1", linewidth=2)
    ax.plot(n_values, [x * 100 for x in extract("next_step/top3_acc")],
            "s--", color="#85b7eb", label="Top-3", linewidth=2)
    ax.plot(n_values, [x * 100 for x in extract("next_step/top5_acc")],
            "^:", color="#b5d4f4", label="Top-5", linewidth=2)
    ax.set_xlabel("N (context size)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Next-step prediction accuracy", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)

    # ── Plot 2: Completion token accuracy vs n ────────────────────────────────
    ax = axes[0, 1]
    ax.plot(n_values, [x * 100 for x in extract("completion/token_acc_60")],
            "o-", color="#c97b4b", label="60% cutoff", linewidth=2)
    ax.plot(n_values, [x * 100 for x in extract("completion/token_acc_80")],
            "s--", color="#f0997b", label="80% cutoff", linewidth=2)
    ax.set_xlabel("N (context size)", fontsize=10)
    ax.set_ylabel("Token accuracy (%)", fontsize=10)
    ax.set_title("Sequence completion token accuracy", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Plot 3: Context coverage vs n ─────────────────────────────────────────
    ax = axes[1, 0]
    coverage = extract("model/val_coverage")
    ax.plot(n_values, [x * 100 for x in coverage],
            "o-", color="#639922", linewidth=2)
    ax.set_xlabel("N (context size)", fontsize=10)
    ax.set_ylabel("Coverage (%)", fontsize=10)
    ax.set_title("Val context coverage\n(how often exact context was seen in training)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)

    # Annotate the sparsity cliff
    for i, (n_val, cov) in enumerate(zip(n_values, coverage)):
        if cov < 0.5 and i > 0:
            ax.axvline(n_val, color="red", linestyle="--", alpha=0.5)
            ax.text(n_val + 0.3, 50, f"Sparsity\ncliff ~n={n_val}",
                    color="red", fontsize=8)
            break

    # ── Plot 4: Unique contexts vs n ──────────────────────────────────────────
    ax = axes[1, 1]
    unique_ctx = extract("model/unique_contexts")
    ax.bar(n_values, unique_ctx, color="#7f77dd", alpha=0.75)
    ax.set_xlabel("N (context size)", fontsize=10)
    ax.set_ylabel("Unique contexts seen", fontsize=10)
    ax.set_title("Model size — unique contexts stored", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "ngram_context_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_top1_highlight(all_results: list[dict], n_values: list[int], out_dir: str):
    """Clean single plot highlighting best n for top-1 accuracy."""
    top1_scores = [r.get("next_step/top1_acc", 0) * 100 for r in all_results]
    best_idx = int(np.argmax(top1_scores))
    best_n = n_values[best_idx]
    best_score = top1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#3266ad" if i == best_idx else "#b5d4f4" for i in range(len(n_values))]
    bars = ax.bar([str(n) for n in n_values], top1_scores, color=colors, alpha=0.85)

    for bar, val in zip(bars, top1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("N (context window size)", fontsize=11)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
    ax.set_title(f"N-gram top-1 accuracy by context size  |  Best: n={best_n} ({best_score:.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_ylim(0, 105)

    path = os.path.join(out_dir, "ngram_best_n.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../tracks/industrial-infineon/training_data")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./results_ngram_comparison")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    parser.add_argument(
        "--n_values", type=int, nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50],
        help="List of n values to test e.g. --n_values 2 3 5 10 20 50"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    sequences = load_training_sequences(args.train_dir)
    train_seqs, val_seqs = train_val_split(sequences, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    print(f"\nTesting n values: {args.n_values}\n")

    # ── WandB parent run for the sweep ────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name="ngram-context-sweep",
        config={
            "experiment": "ngram_context_comparison",
            "n_values": args.n_values,
            "train_sequences": len(train_seqs),
            "val_sequences": len(val_seqs),
            "seed": args.seed,
        },
        tags=["ngram", "context-sweep"],
    )

    all_results = []

    for n in args.n_values:
        print(f"\n{'─'*50}")
        print(f"Training N-gram with n={n} (context = last {n-1} steps)")
        print(f"{'─'*50}")

        model = NGramModel(n=n)
        model.train(train_seqs)
        print(f"  Unique contexts: {model.n_unique_contexts():,} | Vocab: {len(model.vocab)}")

        metrics = evaluate(model, val_seqs, seed=args.seed)
        all_results.append(metrics)

        # Log each n as a step in the same WandB run
        log_data = {"n": n, **metrics}
        wandb.log(log_data)

        print(f"  Top-1 acc:      {metrics['next_step/top1_acc']*100:.1f}%")
        print(f"  Top-3 acc:      {metrics['next_step/top3_acc']*100:.1f}%")
        print(f"  MRR:            {metrics['next_step/mrr']*100:.1f}%")
        print(f"  Token acc 60%:  {metrics['completion/token_acc_60']*100:.2f}%")
        print(f"  Val coverage:   {metrics['model/val_coverage']*100:.1f}%")
        print(f"  Unique contexts:{metrics['model/unique_contexts']:,}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_path = plot_results(all_results, args.n_values, args.out_dir)
    highlight_path = plot_top1_highlight(all_results, args.n_values, args.out_dir)

    # Log plots to WandB
    wandb.log({
        "plots/context_comparison": wandb.Image(plot_path),
        "plots/best_n_highlight":   wandb.Image(highlight_path),
    })

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'N':>5} | {'Top-1':>8} | {'Top-3':>8} | {'MRR':>8} | {'Tok60%':>8} | {'Coverage':>9} | {'Contexts':>10}")
    print(f"{'─'*70}")
    for n, r in zip(args.n_values, all_results):
        print(
            f"{n:>5} | "
            f"{r['next_step/top1_acc']*100:>7.1f}% | "
            f"{r['next_step/top3_acc']*100:>7.1f}% | "
            f"{r['next_step/mrr']*100:>7.1f}% | "
            f"{r['completion/token_acc_60']*100:>7.2f}% | "
            f"{r['model/val_coverage']*100:>8.1f}% | "
            f"{r['model/unique_contexts']:>10,}"
        )
    print(f"{'='*70}")

    best_n = args.n_values[int(np.argmax([r["next_step/top1_acc"] for r in all_results]))]
    print(f"\nBest n for top-1 accuracy: n={best_n}")

    wandb.finish()
    print(f"\nDone. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
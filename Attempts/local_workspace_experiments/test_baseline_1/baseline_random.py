"""
Baseline 1: Random Baseline
============================
Predicts random steps from the vocabulary observed in training data.
Holds out 10% of training sequences for internal evaluation.

Usage:
    python baseline_random.py --train_dir ./training_data
"""

import csv
import argparse
import random
import os
import math
from collections import Counter
import wandb


# ── Data Loading ──────────────────────────────────────────────────────────────

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


def train_val_split(sequences: list[list[str]], val_frac: float = 0.1, seed: int = 42):
    random.seed(seed)
    shuffled = sequences[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


def make_next_step_examples(sequences: list[list[str]]) -> list[tuple[list[str], str]]:
    """Generate (partial_sequence, next_step) pairs from sequences."""
    examples = []
    for seq in sequences:
        for i in range(1, len(seq)):
            examples.append((seq[:i], seq[i]))
    return examples


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ── Random Model ──────────────────────────────────────────────────────────────

class RandomBaseline:
    def __init__(self, weighted: bool = True, seed: int = 42):
        self.weighted = weighted
        self.seed = seed
        self.vocab = []
        self.weights = []
        random.seed(seed)

    def train(self, sequences: list[list[str]]):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        self.vocab = list(counter.keys())
        total = sum(counter.values())
        self.weights = [counter[s] / total for s in self.vocab] if self.weighted else [1.0 / len(self.vocab)] * len(self.vocab)
        print(f"Random baseline ready. Vocab size: {len(self.vocab)}")

    def predict_next(self, top_k: int = 5) -> list[str]:
        chosen = random.choices(self.vocab, weights=self.weights, k=top_k * 3)
        seen, unique = set(), []
        for s in chosen:
            if s not in seen:
                seen.add(s)
                unique.append(s)
            if len(unique) == top_k:
                break
        while len(unique) < top_k:
            unique.append(random.choice(self.vocab))
        return unique[:top_k]

    def complete_sequence(self, partial: list[str], avg_total_len: int = 130) -> list[str]:
        steps_remaining = max(10, avg_total_len - len(partial))
        completion = []
        for _ in range(steps_remaining):
            next_step = random.choices(self.vocab, weights=self.weights, k=1)[0]
            completion.append(next_step)
            if next_step == "SHIP LOT":
                break
        return completion


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_next_step(model: RandomBaseline, val_sequences: list[list[str]]) -> dict:
    examples = make_next_step_examples(val_sequences)
    # Sample up to 2000 for speed
    if len(examples) > 2000:
        random.shuffle(examples)
        examples = examples[:2000]

    top1 = top3 = top5 = mrr = 0
    for partial, true_next in examples:
        preds = model.predict_next(top_k=5)
        if true_next in preds:
            rank = preds.index(true_next) + 1
            mrr += 1.0 / rank
            if rank <= 5:
                top5 += 1
            if rank <= 3:
                top3 += 1
            if rank == 1:
                top1 += 1

    n = len(examples)
    return {
        "next_step/top1_acc": top1 / n,
        "next_step/top3_acc": top3 / n,
        "next_step/top5_acc": top5 / n,
        "next_step/mrr": mrr / n,
        "next_step/n_examples": n,
    }


def evaluate_completion(model: RandomBaseline, val_sequences: list[list[str]]) -> dict:
    exact_match = 0
    token_acc_total = 0
    n = min(len(val_sequences), 200)
    samples = random.sample(val_sequences, n)

    for seq in samples:
        cut = int(len(seq) * 0.6)
        partial = seq[:cut]
        true_rest = seq[cut:]
        predicted = model.complete_sequence(partial, avg_total_len=len(seq))

        if predicted == true_rest:
            exact_match += 1

        # Token accuracy: fraction of positions that match
        matches = sum(a == b for a, b in zip(predicted, true_rest))
        token_acc_total += matches / max(len(true_rest), 1)

    return {
        "completion/exact_match_rate": exact_match / n,
        "completion/token_accuracy": token_acc_total / n,
        "completion/n_examples": n,
    }


def evaluate_anomaly(model: RandomBaseline, val_sequences: list[list[str]]) -> dict:
    """Simulate anomaly detection with random scores — should be ~50% accuracy."""
    correct = 0
    n = min(len(val_sequences), 200)
    samples = random.sample(val_sequences, n)
    for _ in samples:
        # All val sequences are valid; model randomly predicts
        pred = random.random() >= 0.5  # True = valid
        if pred:  # correct if it says valid
            correct += 1
    return {
        "anomaly/binary_accuracy": correct / n,
        "anomaly/note": "val sequences are all valid; accuracy = rate of predicting valid",
        "anomaly/n_examples": n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random Baseline")
    parser.add_argument("--train_dir", default="../tracks/industrial-infineon/training_data")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./results_random")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── WandB Init ────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name="baseline-random",
        config={
            "baseline": "random",
            "weighted": True,
            "seed": args.seed,
            "val_frac": args.val_frac,
            "train_dir": args.train_dir,
        },
        tags=["baseline", "random"],
    )

    # ── Load & Split ──────────────────────────────────────────────────────────
    sequences = load_training_sequences(args.train_dir)
    train_seqs, val_seqs = train_val_split(sequences, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    wandb.log({"data/train_sequences": len(train_seqs), "data/val_sequences": len(val_seqs)})

    # ── Train ─────────────────────────────────────────────────────────────────
    model = RandomBaseline(weighted=True, seed=args.seed)
    model.train(train_seqs)
    wandb.log({"model/vocab_size": len(model.vocab)})

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating next-step prediction...")
    metrics_t1 = evaluate_next_step(model, val_seqs)
    print(metrics_t1)
    wandb.log(metrics_t1)

    print("\nEvaluating sequence completion...")
    metrics_t2 = evaluate_completion(model, val_seqs)
    print(metrics_t2)
    wandb.log(metrics_t2)

    print("\nEvaluating anomaly detection...")
    metrics_t3 = evaluate_anomaly(model, val_seqs)
    print(metrics_t3)
    wandb.log({k: v for k, v in metrics_t3.items() if k != "anomaly/note"})

    # ── Sample Predictions Table ──────────────────────────────────────────────
    sample_rows = []
    for seq in val_seqs[:10]:
        partial = seq[:5]
        preds = model.predict_next(top_k=3)
        sample_rows.append({
            "partial": " -> ".join(partial[-3:]),
            "true_next": seq[5] if len(seq) > 5 else "N/A",
            "pred_1": preds[0],
            "pred_2": preds[1],
            "pred_3": preds[2],
        })
    wandb.log({"predictions/sample_table": wandb.Table(
        columns=["partial", "true_next", "pred_1", "pred_2", "pred_3"],
        data=[[r["partial"], r["true_next"], r["pred_1"], r["pred_2"], r["pred_3"]] for r in sample_rows]
    )})

    wandb.finish()
    print(f"\nDone. WandB run: {run.url}")


if __name__ == "__main__":
    main()

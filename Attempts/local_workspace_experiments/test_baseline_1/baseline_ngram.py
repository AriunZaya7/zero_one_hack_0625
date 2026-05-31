"""
Baseline 2: N-gram Statistical Model
======================================
Trigram model over process step sequences.
Holds out 10% of training sequences for internal evaluation.

Usage:
    python baseline_ngram.py --train_dir ./training_data
"""

import csv
import argparse
import random
import os
import math
from collections import defaultdict, Counter
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


def train_val_split(sequences, val_frac=0.1, seed=42):
    random.seed(seed)
    shuffled = sequences[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ── N-gram Model ──────────────────────────────────────────────────────────────

class NGramModel:
    def __init__(self, n: int = 3, smoothing: float = 1e-6):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, sequences: list[list[str]]):
        for seq in sequences:
            padded = ["<START>"] * (self.n - 1) + seq + ["<END>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i: i + self.n - 1])
                next_step = padded[i + self.n - 1]
                self.counts[context][next_step] += 1
                self.vocab.add(next_step)
        self.vocab.discard("<START>")
        print(f"NGram trained | n={self.n} | vocab={len(self.vocab)} | contexts={len(self.counts)}")

    def predict_next(self, context_steps: list[str], top_k: int = 5) -> list[str]:
        padded = ["<START>"] * (self.n - 1) + context_steps
        context = tuple(padded[-(self.n - 1):])
        counter = self.counts.get(context, Counter())

        # Backoff: shorter context
        if not counter and self.n > 2:
            counter = self.counts.get(context[-1:], Counter())

        # Backoff: unigram
        if not counter:
            all_next = Counter()
            for c in self.counts.values():
                all_next.update(c)
            counter = all_next

        preds = [s for s, _ in counter.most_common(top_k)]
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
        for _ in range(max_steps):
            next_step = self.predict_next(seq, top_k=1)[0]
            if next_step in ("<END>", "<UNK>") or next_step == seq[-1]:
                break
            seq.append(next_step)
            if next_step == "SHIP LOT":
                break
        return seq[len(partial):]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_next_step(model: NGramModel, val_sequences: list[list[str]]) -> dict:
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
    return {
        "next_step/top1_acc": top1 / n,
        "next_step/top3_acc": top3 / n,
        "next_step/top5_acc": top5 / n,
        "next_step/mrr": mrr / n,
        "next_step/n_examples": n,
    }


def evaluate_completion(model: NGramModel, val_sequences: list[list[str]]) -> dict:
    results_60, results_80 = [], []
    samples = random.sample(val_sequences, min(200, len(val_sequences)))

    for seq in samples:
        for frac, bucket in [(0.6, results_60), (0.8, results_80)]:
            cut = int(len(seq) * frac)
            partial = seq[:cut]
            true_rest = seq[cut:]
            predicted = model.complete_sequence(partial, max_steps=len(seq))

            matches = sum(a == b for a, b in zip(predicted, true_rest))
            token_acc = matches / max(len(true_rest), 1)
            exact = int(predicted == true_rest)
            bucket.append((exact, token_acc))

    def avg(bucket):
        return {
            "exact_match": sum(e for e, _ in bucket) / len(bucket),
            "token_acc": sum(t for _, t in bucket) / len(bucket),
        }

    r60, r80 = avg(results_60), avg(results_80)
    return {
        "completion/exact_match_60": r60["exact_match"],
        "completion/token_acc_60": r60["token_acc"],
        "completion/exact_match_80": r80["exact_match"],
        "completion/token_acc_80": r80["token_acc"],
        "completion/n_examples": len(samples),
    }


def evaluate_anomaly(model: NGramModel, val_sequences: list[list[str]]) -> dict:
    """
    Use log-probability as anomaly score.
    Val sequences are all valid — a good model should score them HIGH.
    We report the mean log-prob and the fraction scored as 'valid' at various thresholds.
    """
    samples = random.sample(val_sequences, min(300, len(val_sequences)))
    scores = [model.sequence_log_prob(seq) for seq in samples]
    mean_score = sum(scores) / len(scores)
    # What fraction does the model consider 'valid' (above median threshold)?
    median = sorted(scores)[len(scores) // 2]
    frac_valid = sum(1 for s in scores if s >= median) / len(scores)

    return {
        "anomaly/mean_log_prob_valid_seqs": mean_score,
        "anomaly/frac_labeled_valid_at_median": frac_valid,
        "anomaly/n_examples": len(samples),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../tracks/industrial-infineon/training_data")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./results_ngram")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name=f"baseline-ngram-n{args.n}",
        config={
            "baseline": "ngram",
            "n": args.n,
            "smoothing": 1e-6,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "train_dir": args.train_dir,
        },
        tags=["baseline", "ngram"],
    )

    # ── Load & Split ──────────────────────────────────────────────────────────
    sequences = load_training_sequences(args.train_dir)
    train_seqs, val_seqs = train_val_split(sequences, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    wandb.log({"data/train_sequences": len(train_seqs), "data/val_sequences": len(val_seqs)})

    # ── Train ─────────────────────────────────────────────────────────────────
    model = NGramModel(n=args.n)
    model.train(train_seqs)
    wandb.log({"model/vocab_size": len(model.vocab), "model/n_contexts": len(model.counts)})

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating next-step prediction...")
    m1 = evaluate_next_step(model, val_seqs)
    print(m1)
    wandb.log(m1)

    print("\nEvaluating sequence completion...")
    m2 = evaluate_completion(model, val_seqs)
    print(m2)
    wandb.log(m2)

    print("\nEvaluating anomaly scoring...")
    m3 = evaluate_anomaly(model, val_seqs)
    print(m3)
    wandb.log(m3)

    # ── Sample Predictions Table ──────────────────────────────────────────────
    sample_rows = []
    for seq in val_seqs[:10]:
        for cut in [5, 10, 20]:
            if len(seq) > cut:
                partial = seq[:cut]
                preds = model.predict_next(partial, top_k=3)
                true_next = seq[cut]
                sample_rows.append([
                    " -> ".join(partial[-3:]),
                    true_next,
                    preds[0],
                    preds[1],
                    preds[2],
                    "✓" if true_next == preds[0] else "✗",
                ])

    wandb.log({"predictions/sample_table": wandb.Table(
        columns=["context (last 3)", "true_next", "pred_1", "pred_2", "pred_3", "top1_correct"],
        data=sample_rows
    )})

    # ── Log score distribution ─────────────────────────────────────────────────
    val_scores = [model.sequence_log_prob(seq) for seq in val_seqs[:200]]
    wandb.log({"anomaly/log_prob_distribution": wandb.Histogram(val_scores)})

    wandb.finish()
    print(f"\nDone. WandB run: {run.url}")


if __name__ == "__main__":
    main()
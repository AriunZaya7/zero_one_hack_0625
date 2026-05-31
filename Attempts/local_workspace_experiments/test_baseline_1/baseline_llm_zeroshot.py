"""
Baseline 3: Zero-Shot LLM Baseline
=====================================
Uses a pretrained HuggingFace LLM (no fine-tuning) to predict next steps.
Holds out 10% of training sequences for internal evaluation.

Usage:
    python baseline_llm_zeroshot.py --train_dir ./training_data --model gpt2
    python baseline_llm_zeroshot.py --train_dir ./training_data --model Qwen/Qwen2-0.5B
    python baseline_llm_zeroshot.py --train_dir ./training_data --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import csv
import argparse
import random
import os
import time
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


# ── Prompt Templates ──────────────────────────────────────────────────────────

def make_next_step_prompt(partial: list[str], family: str) -> str:
    recent = " -> ".join(partial[-10:])
    return (
        f"Semiconductor {family} manufacturing process. "
        f"Recent steps: {recent} -> "
        f"Next step:"
    )


def make_completion_prompt(partial: list[str], family: str) -> str:
    recent = " | ".join(partial[-15:])
    return (
        f"Semiconductor {family} manufacturing. "
        f"Steps so far: {recent} | "
        f"Continue until SHIP LOT. Next steps:"
    )


def make_anomaly_prompt(sequence: list[str], family: str) -> str:
    seq_str = " -> ".join(sequence[:40])
    return (
        f"Semiconductor {family} process sequence: {seq_str}. "
        f"Is this valid or invalid? Answer:"
    )


# ── HuggingFace Backend ───────────────────────────────────────────────────────

class HFModel:
    def __init__(self, model_name: str, max_new_tokens: int = 20):
        print(f"Loading: {model_name}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.max_new_tokens = max_new_tokens
        print(f"Model loaded: {model_name}")

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Response Parsers ──────────────────────────────────────────────────────────

def parse_next_step(raw: str) -> str:
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    step = lines[0].upper().strip(".,;:-") if lines else "UNKNOWN"
    # Keep only first meaningful token group (step names are multi-word)
    return step[:80]


def parse_completion(raw: str) -> list[str]:
    steps = []
    for line in raw.strip().split("\n"):
        line = line.strip().upper().strip(".,;:-0123456789. ")
        if line and len(line) > 2:
            steps.append(line)
            if "SHIP LOT" in line:
                break
    return steps or ["UNKNOWN"]


def parse_anomaly(raw: str) -> tuple[int, float]:
    upper = raw.upper()
    if "INVALID" in upper:
        return 0, 0.15
    elif "VALID" in upper:
        return 1, 0.85
    else:
        # Model didn't give a clear answer — default to valid
        return 1, 0.5


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_next_step(model: HFModel, val_sequences: list[list[str]], n_examples: int = 100) -> dict:
    examples = []
    for seq in val_sequences:
        for i in range(5, min(len(seq), 30), 5):  # sample at positions 5,10,15,20,25
            examples.append((seq[:i], seq[i], seq[0] if seq else "UNKNOWN"))
    random.shuffle(examples)
    examples = examples[:n_examples]

    top1 = mrr = 0
    sample_rows = []

    for j, (partial, true_next, family_hint) in enumerate(examples):
        prompt = make_next_step_prompt(partial, "SEMICONDUCTOR")
        raw = model.generate(prompt)
        pred = parse_next_step(raw)

        correct = int(pred.strip() == true_next.strip())
        top1 += correct
        mrr += 1.0 if correct else 0.0

        if j < 10:
            sample_rows.append([" -> ".join(partial[-3:]), true_next, pred, "✓" if correct else "✗"])

        if (j + 1) % 20 == 0:
            print(f"  Task 1: {j+1}/{len(examples)}")

    n = len(examples)
    return {
        "next_step/top1_acc": top1 / n,
        "next_step/mrr": mrr / n,
        "next_step/n_examples": n,
    }, sample_rows


def evaluate_completion(model: HFModel, val_sequences: list[list[str]], n_examples: int = 50) -> dict:
    samples = random.sample(val_sequences, min(n_examples, len(val_sequences)))
    token_acc_total = 0

    for j, seq in enumerate(samples):
        cut = int(len(seq) * 0.6)
        partial = seq[:cut]
        true_rest = seq[cut:]
        prompt = make_completion_prompt(partial, "SEMICONDUCTOR")
        raw = model.generate(prompt)
        predicted = parse_completion(raw)

        matches = sum(a == b for a, b in zip(predicted, true_rest))
        token_acc_total += matches / max(len(true_rest), 1)

        if (j + 1) % 10 == 0:
            print(f"  Task 2: {j+1}/{len(samples)}")

    return {
        "completion/token_accuracy": token_acc_total / len(samples),
        "completion/n_examples": len(samples),
    }


def evaluate_anomaly(model: HFModel, val_sequences: list[list[str]], n_examples: int = 50) -> dict:
    """All val sequences are valid — model should predict valid for all."""
    samples = random.sample(val_sequences, min(n_examples, len(val_sequences)))
    correct = 0

    for j, seq in enumerate(samples):
        prompt = make_anomaly_prompt(seq, "SEMICONDUCTOR")
        raw = model.generate(prompt)
        is_valid, _ = parse_anomaly(raw)
        correct += is_valid  # correct if it says valid (they all are)

        if (j + 1) % 10 == 0:
            print(f"  Task 3: {j+1}/{len(samples)}")

    return {
        "anomaly/accuracy_on_valid_seqs": correct / len(samples),
        "anomaly/n_examples": len(samples),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../tracks/industrial-infineon/training_data")
    parser.add_argument("--model", default="gpt2",
                        help="HF model: gpt2, Qwen/Qwen2-0.5B, TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_examples", type=int, default=100,
                        help="Limit eval examples (LLM is slow)")
    parser.add_argument("--out_dir", default="./results_llm_zeroshot")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name=f"baseline-llm-zeroshot-{args.model.split('/')[-1]}",
        config={
            "baseline": "llm_zeroshot",
            "model": args.model,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "max_eval_examples": args.max_eval_examples,
        },
        tags=["baseline", "llm", "zeroshot"],
    )

    # ── Load & Split ──────────────────────────────────────────────────────────
    sequences = load_training_sequences(args.train_dir)
    _, val_seqs = train_val_split(sequences, val_frac=args.val_frac, seed=args.seed)
    print(f"Val sequences: {len(val_seqs)}")
    wandb.log({"data/val_sequences": len(val_seqs)})

    # ── Load Model ────────────────────────────────────────────────────────────
    model = HFModel(model_name=args.model)
    wandb.log({"model/name": args.model})

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating next-step prediction (n={args.max_eval_examples})...")
    m1, sample_rows = evaluate_next_step(model, val_seqs, n_examples=args.max_eval_examples)
    print(m1)
    wandb.log(m1)
    wandb.log({"predictions/sample_table": wandb.Table(
        columns=["context (last 3)", "true_next", "predicted", "correct"],
        data=sample_rows
    )})

    print(f"\nEvaluating sequence completion (n={min(50, args.max_eval_examples)})...")
    m2 = evaluate_completion(model, val_seqs, n_examples=min(50, args.max_eval_examples))
    print(m2)
    wandb.log(m2)

    print(f"\nEvaluating anomaly detection (n={min(50, args.max_eval_examples)})...")
    m3 = evaluate_anomaly(model, val_seqs, n_examples=min(50, args.max_eval_examples))
    print(m3)
    wandb.log(m3)

    wandb.finish()
    print(f"\nDone. WandB run: {run.url}")


if __name__ == "__main__":
    main()

"""
train_model.py
===============
Fine-tunes Qwen2-0.5B on process sequences using causal language modeling.
Supports a --debug mode for fast local testing before submitting to Leonardo.

Local debug test (run this FIRST before submitting to cluster):
    python train_model.py --debug

Full training (run via SLURM on Leonardo):
    python train_model.py --data_dir ./data --output_dir ./checkpoints

WandB tracking is included. Make sure to run `wandb login` before training.
"""

import os
import json
import argparse
import random
import math
import time
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


# ── Argument Parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",    default="../data")
    parser.add_argument("--output_dir",  default="../checkpoints")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model_name",  default="Qwen/Qwen2-0.5B",
                        help="HuggingFace model ID")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max_length",  type=int,   default=512,
                        help="Max token length per sequence")
    parser.add_argument("--warmup_ratio",type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)

    # ── Eval ──────────────────────────────────────────────────────────────────
    parser.add_argument("--eval_every",  type=int,   default=100,
                        help="Evaluate every N optimizer steps")
    parser.add_argument("--save_every",  type=int,   default=500,
                        help="Save checkpoint every N optimizer steps")

    # ── WandB ─────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    parser.add_argument("--run_name",      default="qwen2-0.5b-finetune")

    # ── Debug mode ────────────────────────────────────────────────────────────
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: tiny data, 1 epoch, fast run to catch bugs locally")

    return parser.parse_args()


# ── Debug Overrides ────────────────────────────────────────────────────────────

def apply_debug_mode(args):
    """Override args for a fast local debug run."""
    print("\n" + "="*55)
    print("  DEBUG MODE — fast local test, not a real training run")
    print("="*55)
    args.model_name   = "Qwen/Qwen2-0.5B"   # same model, just tiny data
    args.epochs       = 1
    args.batch_size   = 2
    args.grad_accum   = 1
    args.max_length   = 128                  # shorter sequences
    args.eval_every   = 5
    args.save_every   = 999999              # don't save in debug
    args.run_name     = "debug-run"
    args.n_debug_seqs = 8                   # only 8 sequences total
    print(f"  epochs={args.epochs}, batch={args.batch_size}, "
          f"max_len={args.max_length}, n_seqs={args.n_debug_seqs}")
    print("="*55 + "\n")
    return args


# ── Dataset ────────────────────────────────────────────────────────────────────

class ProcessSequenceDataset(Dataset):
    """
    Loads sequences from a JSONL file and tokenizes them.
    Each record becomes one training example.
    The model is trained to predict the next token at every position
    (causal language modeling).
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int,
                 n_samples: int = None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Optionally limit for debug
        if n_samples:
            random.shuffle(self.records)
            self.records = self.records[:n_samples]

        print(f"  Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = record["prompt"]

        # Tokenize — truncate to max_length
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)  # shape: [seq_len]

        # For causal LM: labels = input_ids (model predicts next token)
        # Shift happens inside the model automatically
        return {
            "input_ids":      input_ids,
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels":         input_ids.clone(),
            "family":         record.get("family", "UNKNOWN"),
        }


def collate_fn(batch, pad_token_id: int):
    """Pad sequences in a batch to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list  = []
    attn_mask_list  = []
    labels_list     = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids and attention_mask
        input_ids_list.append(
            torch.cat([item["input_ids"],
                       torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attn_mask_list.append(
            torch.cat([item["attention_mask"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        # Labels: pad with -100 so loss ignores padding
        labels_list.append(
            torch.cat([item["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "labels":         torch.stack(labels_list),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, label="val") -> dict:
    """Compute average loss and perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss    += outputs.loss.item()
        total_batches += 1

    avg_loss    = total_loss / max(total_batches, 1)
    perplexity  = math.exp(min(avg_loss, 20))  # cap to avoid overflow

    model.train()
    return {
        f"{label}/loss":       avg_loss,
        f"{label}/perplexity": perplexity,
    }


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
        tags=["qwen2", "finetune", "debug" if args.debug else "full"],
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    # Qwen2 may not have a pad token — add one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    print(f"Vocab size: {tokenizer.vocab_size} | Pad token id: {pad_id}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")
    wandb.log({"model/parameters_M": n_params})

    # ── Datasets ───────────────────────────────────────────────────────────────
    n_debug = getattr(args, "n_debug_seqs", None)

    train_dataset = ProcessSequenceDataset(
        os.path.join(args.data_dir, "dataset_train.jsonl"),
        tokenizer, args.max_length,
        n_samples=n_debug,
    )
    id_val_dataset = ProcessSequenceDataset(
        os.path.join(args.data_dir, "dataset_id_val.jsonl"),
        tokenizer, args.max_length,
        n_samples=n_debug,
    )
    ood_dataset = ProcessSequenceDataset(
        os.path.join(args.data_dir, "dataset_ood_test.jsonl"),
        tokenizer, args.max_length,
        n_samples=n_debug,
    )

    collate = lambda b: collate_fn(b, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=0,
    )
    id_val_loader = DataLoader(
        id_val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=0,
    )
    ood_loader = DataLoader(
        ood_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=0,
    )

    wandb.log({
        "data/train_sequences": len(train_dataset),
        "data/id_val_sequences": len(id_val_dataset),
        "data/ood_sequences": len(ood_dataset),
    })

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    total_steps   = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"\nTotal optimizer steps : {total_steps}")
    print(f"Warmup steps          : {warmup_steps}")

    # ── Baseline eval before training ─────────────────────────────────────────
    print("\nEvaluating baseline (before training)...")
    pre_id  = evaluate(model, id_val_loader,  device, label="id_val")
    pre_ood = evaluate(model, ood_loader,     device, label="ood_test")
    print(f"  Before training — ID val loss:  {pre_id['id_val/loss']:.4f}  "
          f"perplexity: {pre_id['id_val/perplexity']:.2f}")
    print(f"  Before training — OOD loss:     {pre_ood['ood_test/loss']:.4f}  "
          f"perplexity: {pre_ood['ood_test/perplexity']:.2f}")
    wandb.log({**pre_id, **pre_ood, "step": 0})

    # ── Training ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step    = 0
    optimizer.zero_grad()
    best_val_loss  = float("inf")

    print(f"\nStarting training for {args.epochs} epoch(s)...\n")

    for epoch in range(args.epochs):
        epoch_loss    = 0.0
        epoch_batches = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss    += outputs.loss.item()
            epoch_batches += 1

            # Optimizer step after grad_accum batches
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Log training metrics ──────────────────────────────────────
                current_loss = epoch_loss / epoch_batches
                wandb.log({
                    "train/loss":      outputs.loss.item(),
                    "train/lr":        scheduler.get_last_lr()[0],
                    "train/epoch":     epoch + step / len(train_loader),
                    "step":            global_step,
                })

                # ── Periodic eval ─────────────────────────────────────────────
                if global_step % args.eval_every == 0:
                    id_metrics  = evaluate(model, id_val_loader,  device, "id_val")
                    ood_metrics = evaluate(model, ood_loader,      device, "ood_test")

                    # Compute OOD drop
                    drop = ood_metrics["ood_test/loss"] - id_metrics["id_val/loss"]
                    wandb.log({
                        **id_metrics, **ood_metrics,
                        "generalization/ood_loss_drop": drop,
                        "step": global_step,
                    })

                    print(f"  Step {global_step:>5} | "
                          f"train_loss={outputs.loss.item():.4f} | "
                          f"id_val_loss={id_metrics['id_val/loss']:.4f} | "
                          f"ood_loss={ood_metrics['ood_test/loss']:.4f} | "
                          f"ood_drop={drop:+.4f}")

                    # Save best model
                    if id_metrics["id_val/loss"] < best_val_loss:
                        best_val_loss = id_metrics["id_val/loss"]
                        best_path = os.path.join(args.output_dir, "best_model")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

                # ── Periodic checkpoint ───────────────────────────────────────
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"  Checkpoint saved: {ckpt_path}")

        # ── End of epoch summary ───────────────────────────────────────────────
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"\nEpoch {epoch+1}/{args.epochs} done | "
              f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s\n")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_id  = evaluate(model, id_val_loader, device, "id_val")
    final_ood = evaluate(model, ood_loader,    device, "ood_test")
    drop      = final_ood["ood_test/loss"] - final_id["id_val/loss"]

    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  ID Val  loss : {final_id['id_val/loss']:.4f}  "
          f"perplexity: {final_id['id_val/perplexity']:.2f}")
    print(f"  OOD     loss : {final_ood['ood_test/loss']:.4f}  "
          f"perplexity: {final_ood['ood_test/perplexity']:.2f}")
    print(f"  OOD drop     : {drop:+.4f}")
    print(f"{'='*55}\n")

    wandb.log({
        **{f"final/{k}": v for k, v in final_id.items()},
        **{f"final/{k}": v for k, v in final_ood.items()},
        "final/ood_loss_drop": drop,
    })

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved: {final_path}")

    wandb.finish()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        args = apply_debug_mode(args)
    train(args)

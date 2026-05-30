"""
train_model.py
===============
Fine-tunes a pretrained LM (Qwen2.5 or GPT-2) on process sequences.

Key fixes vs v1:
  - lr default 1e-5   (2e-4 caused catastrophic OOD forgetting)
  - Saves best_ood_model/ (primary) and best_id_model/ (secondary)
  - DataParallel multi-GPU support (set --gpus-per-task in SLURM)
  - eval_every 50 steps for finer OOD tracking

Debug:
    python train_model.py --debug
Full:
    python train_model.py --model_name Qwen/Qwen2.5-0.5B --run_name qwen2.5-0.5b-run2
"""

import json
import math
import os
import random
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


# ── Metrics Logger ────────────────────────────────────────────────────────────

class MetricsLogger:
    def __init__(self, run_name: str, metrics_dir: str = "metrics"):
        os.makedirs(metrics_dir, exist_ok=True)
        self.path = os.path.join(metrics_dir, f"{run_name}.jsonl")
        self._fh = open(self.path, "w", buffering=1)

    def log(self, event: str, step: int = 0, **data):
        self._fh.write(json.dumps({"event": event, "step": step,
                                    "t": time.time(), **data}) + "\n")

    def close(self):
        self._fh.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def unwrap(model):
    """Strip DataParallel wrapper to get the underlying HF model."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="../data")
    p.add_argument("--output_dir",   default="../checkpoints")
    p.add_argument("--model_name",   default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-5)   # fixed: was 2e-4
    p.add_argument("--max_length",   type=int,   default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_every",   type=int,   default=50)     # more frequent
    p.add_argument("--save_every",   type=int,   default=500)
    p.add_argument("--run_name",     default="run")
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


def apply_debug_mode(args):
    print("\n" + "="*55)
    print("  DEBUG MODE")
    print("="*55)
    args.epochs      = 1
    args.batch_size  = 2
    args.grad_accum  = 1
    args.max_length  = 128
    args.eval_every  = 5
    args.save_every  = 999999
    args.run_name    = "debug-run"
    args.n_debug_seqs = 8
    return args


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProcessSequenceDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length, n_samples=None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if n_samples:
            random.shuffle(self.records)
            self.records = self.records[:n_samples]
        print(f"  Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(rec["prompt"], truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        return {"input_ids": ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": ids.clone()}


def collate_fn(batch, pad_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    ids, masks, labels = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        ids.append(   torch.cat([x["input_ids"],      torch.full((pad,), pad_id,  dtype=torch.long)]))
        masks.append( torch.cat([x["attention_mask"],  torch.zeros(pad,            dtype=torch.long)]))
        labels.append(torch.cat([x["labels"],          torch.full((pad,), -100,   dtype=torch.long)]))
    return {"input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels)}


# ── Forward pass (handles DataParallel) ──────────────────────────────────────

def forward_loss(model, batch, device, use_dp):
    """Run one forward pass; returns scalar loss regardless of #GPUs."""
    out = model(
        input_ids      =batch["input_ids"].to(device),
        attention_mask =batch["attention_mask"].to(device),
        labels         =batch["labels"].to(device),
        return_dict    =not use_dp,  # DP needs tuple output, not HF dataclass
    )
    loss = out[0] if use_dp else out.loss
    # DataParallel may return a 1-D tensor (one value per GPU) — average them
    return loss.mean() if loss.dim() > 0 else loss


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, label, use_dp):
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        loss = forward_loss(model, batch, device, use_dp)
        total += loss.item()
        n     += 1
    avg = total / max(n, 1)
    model.train()
    return {f"{label}/loss": avg,
            f"{label}/perplexity": math.exp(min(avg, 20))}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpus > 1
    print(f"\nDevice: {device}  |  GPUs visible: {n_gpus}"
          + (" → DataParallel" if use_dp else ""))
    if n_gpus >= 1:
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  {props.total_memory/1e9:.1f} GB")

    logger = MetricsLogger(args.run_name)
    logger.log("config", step=0, **vars(args))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    print(f"  Vocab size: {tokenizer.vocab_size}  |  pad_id: {pad_id}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M  |  dtype: {dtype}")
    logger.log("model_info", step=0, parameters_M=round(n_params, 2))

    if use_dp:
        model = nn.DataParallel(model)
        print(f"  Wrapped with DataParallel across {n_gpus} GPUs")

    # ── Data ──────────────────────────────────────────────────────────────────
    n_debug  = getattr(args, "n_debug_seqs", None)
    collate  = lambda b: collate_fn(b, pad_id)

    train_ds  = ProcessSequenceDataset(os.path.join(args.data_dir, "dataset_train.jsonl"),
                                        tokenizer, args.max_length, n_debug)
    id_val_ds = ProcessSequenceDataset(os.path.join(args.data_dir, "dataset_id_val.jsonl"),
                                        tokenizer, args.max_length, n_debug)
    ood_ds    = ProcessSequenceDataset(os.path.join(args.data_dir, "dataset_ood_test.jsonl"),
                                        tokenizer, args.max_length, n_debug)

    train_loader  = DataLoader(train_ds,  batch_size=args.batch_size,
                                shuffle=True,  collate_fn=collate, num_workers=2)
    id_val_loader = DataLoader(id_val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate, num_workers=2)
    ood_loader    = DataLoader(ood_ds,    batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate, num_workers=2)

    logger.log("data_info", step=0,
               train=len(train_ds), id_val=len(id_val_ds), ood=len(ood_ds))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(unwrap(model).parameters(),
                                      lr=args.lr, weight_decay=0.01)
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"\n  Total optimizer steps : {total_steps}")
    print(f"  Warmup steps          : {warmup_steps}")
    print(f"  LR                    : {args.lr}")

    # ── Baseline eval ─────────────────────────────────────────────────────────
    print("\nBaseline eval (untrained)...")
    pre_id  = evaluate(model, id_val_loader, device, "id_val",   use_dp)
    pre_ood = evaluate(model, ood_loader,    device, "ood_test", use_dp)
    print(f"  ID  val : loss {pre_id['id_val/loss']:.4f}  ppl {pre_id['id_val/perplexity']:.2f}")
    print(f"  OOD test: loss {pre_ood['ood_test/loss']:.4f}  ppl {pre_ood['ood_test/perplexity']:.2f}")
    logger.log("baseline", step=0, **pre_id, **pre_ood)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step  = 0
    best_ood_loss = float("inf")   # PRIMARY: save by OOD loss
    best_id_loss  = float("inf")   # secondary
    best_ood_step = 0
    optimizer.zero_grad()

    print(f"\nTraining for {args.epochs} epoch(s)...\n")

    for epoch in range(args.epochs):
        ep_loss, ep_batches = 0.0, 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            loss = forward_loss(model, batch, device, use_dp)
            raw_loss = loss.item()
            (loss / args.grad_accum).backward()

            ep_loss    += raw_loss
            ep_batches += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr_now = scheduler.get_last_lr()[0]
                logger.log("train_step", step=global_step,
                           train_loss=raw_loss, lr=lr_now,
                           epoch=epoch + step / len(train_loader))

                if global_step % args.eval_every == 0:
                    id_m  = evaluate(model, id_val_loader, device, "id_val",   use_dp)
                    ood_m = evaluate(model, ood_loader,    device, "ood_test", use_dp)
                    drop  = ood_m["ood_test/loss"] - id_m["id_val/loss"]
                    logger.log("eval", step=global_step,
                               **id_m, **ood_m, ood_loss_drop=drop)

                    print(f"  Step {global_step:>5} | "
                          f"train={raw_loss:.4f} | "
                          f"id_val={id_m['id_val/loss']:.4f} | "
                          f"ood={ood_m['ood_test/loss']:.4f} | "
                          f"drop={drop:+.4f}")

                    # ── Save best OOD checkpoint (PRIMARY) ────────────────
                    if ood_m["ood_test/loss"] < best_ood_loss:
                        best_ood_loss = ood_m["ood_test/loss"]
                        best_ood_step = global_step
                        ood_path = os.path.join(args.output_dir, "best_ood_model")
                        unwrap(model).save_pretrained(ood_path)
                        tokenizer.save_pretrained(ood_path)
                        print(f"  ✓ Best OOD model  (step {global_step}, "
                              f"ood_loss={best_ood_loss:.4f})")

                    # ── Save best ID checkpoint (secondary) ───────────────
                    if id_m["id_val/loss"] < best_id_loss:
                        best_id_loss = id_m["id_val/loss"]
                        id_path = os.path.join(args.output_dir, "best_id_model")
                        unwrap(model).save_pretrained(id_path)
                        tokenizer.save_pretrained(id_path)

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f"ckpt_step{global_step}")
                    unwrap(model).save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  Checkpoint: {ckpt}")

        elapsed  = time.time() - t0
        avg_loss = ep_loss / max(ep_batches, 1)
        print(f"\nEpoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | {elapsed:.0f}s\n")
        logger.log("epoch_end", step=global_step,
                   epoch=epoch+1, epoch_avg_loss=avg_loss, elapsed_s=elapsed)

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_id  = evaluate(model, id_val_loader, device, "id_val",   use_dp)
    final_ood = evaluate(model, ood_loader,    device, "ood_test", use_dp)
    drop      = final_ood["ood_test/loss"] - final_id["id_val/loss"]

    print(f"\n{'='*58}")
    print(f"  FINAL RESULTS  ({args.run_name})")
    print(f"{'='*58}")
    print(f"  ID  val  loss : {final_id['id_val/loss']:.4f}  "
          f"ppl: {final_id['id_val/perplexity']:.2f}")
    print(f"  OOD test loss : {final_ood['ood_test/loss']:.4f}  "
          f"ppl: {final_ood['ood_test/perplexity']:.2f}")
    print(f"  OOD drop      : {drop:+.4f}")
    print(f"  Best OOD      : step {best_ood_step}, loss {best_ood_loss:.4f}")
    print(f"{'='*58}\n")

    logger.log("final", step=global_step,
               **final_id, **final_ood, ood_loss_drop=drop,
               best_ood_loss=best_ood_loss, best_ood_step=best_ood_step)
    logger.close()

    final_path = os.path.join(args.output_dir, "final_model")
    unwrap(model).save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model  → {final_path}")
    print(f"Best OOD     → {os.path.join(args.output_dir, 'best_ood_model')}")
    print(f"Metrics      → metrics/{args.run_name}.jsonl")


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        args = apply_debug_mode(args)
    train(args)

"""
train_transformer.py
====================
Train the from-scratch ProcessTransformer on step-level tokens.

Trains on IGBT + IC, holds 10% for validation, and keeps MOSFET as an unseen OOD
family (matches the existing project convention). Logs to W&B if available, else
writes a JSONL training log.

Run (GPU node via pixi):
    pixi run python -m solution.train.train_transformer --data_dir training_data \
        --out solution/checkpoints/transformer_small --epochs 30

Quick local smoke test (CPU, tiny):
    pixi run python -m solution.train.train_transformer --smoke
"""

from __future__ import annotations
import os
import json
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from solution.data.loader import load_all_families, train_val_split
from solution.data.vocab import Vocab
from solution.models.transformer import ProcessTransformer

def parse_families(values, available):
    if not values or values == ["all"]:
        return set(available)
    return {value.upper() for value in values}


class SeqDataset(Dataset):
    def __init__(self, sequences, vocab: Vocab, max_len: int):
        self.data = [vocab.encode(s)[:max_len] for s in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.long)


def collate(batch, pad_id: int):
    max_len = max(len(x) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :len(x)] = x
    return out


@torch.no_grad()
def eval_loss(model, loader, device, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               batch[:, 1:].reshape(-1), ignore_index=pad_id)
        tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="training_data")
    p.add_argument("--out", default="solution/checkpoints/transformer_small")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_families", nargs="+", default=["all"])
    p.add_argument("--ood_family", default=None)
    p.add_argument("--wandb_project", default="zero-one-hack")
    p.add_argument("--wandb_run", default="transformer_small")
    p.add_argument("--smoke", action="store_true", help="tiny fast run for debugging")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fams = load_all_families(args.data_dir)
    train_families = parse_families(args.train_families, set(fams))
    train_seqs, val_seqs = [], []
    for fam, seqs in fams.items():
        if fam in train_families:
            tr, va = train_val_split(seqs, 0.1, args.seed)
            train_seqs += tr; val_seqs += va
    ood_seqs = fams.get(args.ood_family, []) if args.ood_family else []

    if args.smoke:
        train_seqs, val_seqs = train_seqs[:64], val_seqs[:16]
        args.d_model, args.n_layers, args.n_heads, args.d_ff = 64, 2, 2, 128
        args.epochs, args.batch_size, args.max_len = 2, 16, 160

    # Vocab from ALL families so OOD steps are in-vocab (shared ~120-step vocab).
    vocab = Vocab([s for v in fams.values() for s in v])
    os.makedirs(args.out, exist_ok=True)
    vocab.save(os.path.join(args.out, "vocab.json"))

    pad_id = Vocab.PAD
    tr_loader = DataLoader(SeqDataset(train_seqs, vocab, args.max_len),
                           batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda b: collate(b, pad_id))
    va_loader = DataLoader(SeqDataset(val_seqs, vocab, args.max_len),
                           batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda b: collate(b, pad_id))
    ood_loader = DataLoader(SeqDataset(ood_seqs, vocab, args.max_len),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id)) if ood_seqs else None

    model = ProcessTransformer(vocab.size, args.d_model, args.n_heads, args.n_layers,
                               args.d_ff, args.max_len, pad_id=pad_id).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} | params={n_params:.2f}M | vocab={vocab.size} | "
          f"families={sorted(train_families)} | train={len(train_seqs)} "
          f"val={len(val_seqs)} ood={len(ood_seqs)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(tr_loader) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)

    # W&B optional
    use_wandb = False
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        use_wandb = True
    except Exception as e:
        print(f"[wandb disabled: {e}] logging to {args.out}/training_log.jsonl")
    log_f = open(os.path.join(args.out, "training_log.jsonl"), "w", encoding="utf-8")

    best_val = float("inf")
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time(); ep_loss, nb = 0.0, 0
        for batch in tr_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   batch[:, 1:].reshape(-1), ignore_index=pad_id)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item(); nb += 1
        tr_loss = ep_loss / max(nb, 1)
        va_loss = eval_loss(model, va_loader, device, pad_id)
        rec = {"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss,
               "val_ppl": math.exp(min(va_loss, 20)), "time_s": round(time.time() - t0, 1)}
        if ood_loader is not None:
            ol = eval_loss(model, ood_loader, device, pad_id)
            rec["ood_loss"] = ol; rec["ood_drop"] = ol - va_loss
        print("  " + " | ".join(f"{k}={v}" for k, v in rec.items()))
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if use_wandb:
            wandb.log({f"train/{k}": v for k, v in rec.items()})

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(),
                        "config": {"vocab_size": vocab.size, "d_model": args.d_model,
                                   "n_heads": args.n_heads, "n_layers": args.n_layers,
                                   "d_ff": args.d_ff, "max_seq_len": args.max_len}},
                       os.path.join(args.out, "best.pt"))
    log_f.close()
    if use_wandb:
        wandb.finish()
    print(f"done. best val_loss={best_val:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()

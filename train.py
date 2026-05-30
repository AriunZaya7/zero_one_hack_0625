"""Train a model (from-scratch GPT *or* pretrained Qwen2.5) on process sequences
and report next-step accuracy vs the n-gram baseline, in ID or leave-one-family-out
(OOD) mode. Writes one JSON per run to results/ so viz.py can plot across models.

Two model tiers (PLAN.md model strategy):
  --model gpt:tiny|small|large     from-scratch GPT-2, step-level tokenizer (anchor)
  --model hf:Qwen/Qwen2.5-0.5B     pretrained LLM, native BPE tokenizer (transfer tier)

Both are measured through the SAME shared interface (closed-vocab next-step ranking),
so the numbers are directly comparable.

Examples:
    python train.py --model gpt:tiny           --leave-out mosfet --epochs 8
    python train.py --model hf:Qwen/Qwen2.5-0.5B --leave-out mosfet --epochs 3 --lr 1e-5
    python train.py --model hf:Qwen/Qwen2.5-1.5B --train-families mosfet igbt ic --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import FAMILIES, leave_one_family_out, load_all, train_val_split
from ngram import NGramModel
from tokenizer import StepTokenizer

SEED = 42
EVAL_SEQS = 150          # full GPT eval is fast; HF eval is capped (see EVAL_SEQS_HF)
EVAL_SEQS_HF = 60        # closed-vocab scoring is heavier -> smaller eval set
RESULTS_DIR = "results"  # one JSON per run -> safe for parallel SLURM jobs


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ======================= datasets =======================
class IdDataset(Dataset):
    """Pre-encoded list of token-id lists (used by both tiers)."""
    def __init__(self, encoded: list[list[int]]):
        self.data = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_collate(pad_id: int):
    def collate(batch):
        m = max(len(x) for x in batch)
        input_ids = [x + [pad_id] * (m - len(x)) for x in batch]
        labels = [x + [-100] * (m - len(x)) for x in batch]
        return torch.tensor(input_ids), torch.tensor(labels)
    return collate


# ======================= eval =======================
@torch.no_grad()
def eval_gpt_fast(gpt, seqs: dict, max_seqs: int = EVAL_SEQS) -> dict:
    """Fast teacher-forced next-step metrics for the step-level GPT: 1 fwd / seq."""
    keys = list(seqs)
    random.Random(SEED).shuffle(keys)
    keys = keys[:max_seqs]
    tok = gpt.tok
    top1 = top3 = top5 = 0
    mrr = 0.0
    n = 0
    for k in keys:
        ids = tok.encode(seqs[k], add_special=True)[: gpt.max_pos]
        x = torch.tensor([ids], device=gpt.device)
        logits = gpt.model(x).logits[0]
        for b in (tok.pad_id, tok.bos_id, tok.unk_id):
            logits[:, b] = float("-inf")
        for t in range(len(ids) - 1):
            target = ids[t + 1]
            if target == tok.eos_id:
                continue
            rank = int((logits[t] > logits[t][target]).sum())
            n += 1
            top1 += rank == 0
            top3 += rank < 3
            top5 += rank < 5
            mrr += 1.0 / (rank + 1)
    return {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n,
            "mrr": mrr / n, "n_predictions": n}


def eval_via_interface(model, seqs: dict, max_seqs: int) -> dict:
    """Generic next-step metrics through next_step_ranking (n-gram and HF)."""
    from run_baseline import next_step_metrics
    return next_step_metrics(model, seqs, max_seqs=max_seqs)


# ======================= training loop =======================
def run_training(model, loader, epochs, lr, device, pad_id):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, epochs * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                pct_start=0.05)
    model.train()
    for ep in range(epochs):
        run_loss = 0.0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            attn = (input_ids != pad_id).long()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += out.loss.item()
        avg = run_loss / len(loader)
        print(f"  epoch {ep+1}/{epochs}  loss={avg:.3f}  ppl={np.exp(avg):.2f}", flush=True)


# ======================= results logging =======================
def log_result(row: dict) -> str:
    """One JSON file per run -> no concurrent-append corruption (parallel jobs).
    viz.py merges results/*.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"{row['model_type']}_{row['model_name'].replace('/', '-')}_" \
           f"{row['mode']}_{row['holdout'] or 'all'}_{int(time.time())}.json"
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(row, f, indent=2)
    return name


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt:tiny",
                    help="gpt:tiny|small|large  OR  hf:<hf-model-name>")
    ap.add_argument("--leave-out", choices=FAMILIES, default=None)
    ap.add_argument("--train-families", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="default: gpt 3e-4, hf 1e-5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind, spec = args.model.split(":", 1)
    assert kind in ("gpt", "hf"), "model must be gpt:<size> or hf:<name>"
    lr = args.lr if args.lr is not None else (3e-4 if kind == "gpt" else 1e-5)

    # ---- data split (shared between baseline and the model) ----------------
    if args.leave_out:
        train_seqs, test_seqs = leave_one_family_out(args.leave_out, seed=SEED)
        mode, holdout = "ood", args.leave_out
    else:
        all_seqs = load_all(args.train_families)
        train_seqs, test_seqs = train_val_split(all_seqs, val_frac=0.1, seed=SEED)
        mode, holdout = "id", ""
    print(f"[mode] {mode} holdout={holdout or '-'}  model={args.model}  device={device}")
    print(f"[data] train={len(train_seqs)} eval={len(test_seqs)}", flush=True)

    # ---- n-gram baseline on the SAME split --------------------------------
    ngram = NGramModel(n=3, alpha=0.4).fit(train_seqs.values())
    ng = eval_via_interface(ngram, test_seqs, EVAL_SEQS)
    print(f"[n-gram]  top1={ng['top1']:.3f} top3={ng['top3']:.3f} "
          f"top5={ng['top5']:.3f} mrr={ng['mrr']:.3f}", flush=True)

    # ---- build + train the model ------------------------------------------
    if kind == "gpt":
        from gpt import GPTModel, build_gpt
        tok = StepTokenizer.build_from_sequences(load_all().values())
        model = build_gpt(tok, size=spec)
        enc = [tok.encode(s, add_special=True) for s in train_seqs.values()]
        pad_id = tok.pad_id
    else:  # hf
        from hf_model import HFModel, load_hf, seq_to_text
        step_tok = StepTokenizer.build_from_sequences(load_all().values())
        step_vocab = sorted(s for s in step_tok.step_to_id if not s.startswith("<"))
        hf_model, hf_tok = load_hf(spec)
        model = hf_model
        enc = [hf_tok(seq_to_text(s)).input_ids for s in train_seqs.values()]
        pad_id = hf_tok.pad_token_id

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{kind}] params={n_params:.1f}M  training {args.epochs} epochs lr={lr}...",
          flush=True)
    model.to(device)
    loader = DataLoader(IdDataset(enc), batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(pad_id))
    run_training(model, loader, args.epochs, lr, device, pad_id)

    # ---- eval -------------------------------------------------------------
    if kind == "gpt":
        from gpt import GPTModel
        wrapped = GPTModel(model, tok, device=device)
        gm = eval_gpt_fast(wrapped, test_seqs)
    else:
        wrapped = HFModel(model, hf_tok, step_vocab, device=device)
        gm = eval_via_interface(wrapped, test_seqs, EVAL_SEQS_HF)

    print(f"[{kind}]     top1={gm['top1']:.3f} top3={gm['top3']:.3f} "
          f"top5={gm['top5']:.3f} mrr={gm['mrr']:.3f}", flush=True)
    tag = "OOD" if mode == "ood" else "ID"
    print(f"\n=== {tag} next-step top1:  n-gram={ng['top1']:.3f}  "
          f"{kind}={gm['top1']:.3f}  (delta={gm['top1']-ng['top1']:+.3f}) ===", flush=True)

    # ---- log + save -------------------------------------------------------
    fname = log_result({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": kind, "model_name": spec, "mode": mode, "holdout": holdout,
        "n_params_M": round(n_params, 2), "epochs": args.epochs,
        "n_eval_seqs": gm["n_predictions"], "ngram_top1": round(ng["top1"], 4),
        "top1": round(gm["top1"], 4), "top3": round(gm["top3"], 4),
        "top5": round(gm["top5"], 4), "mrr": round(gm["mrr"], 4),
    })
    print(f"[results] wrote {RESULTS_DIR}/{fname}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

"""Evaluate GPT checkpoints WITH and WITHOUT grammar-guided decoding.

For every experimental split (ID + 3 LOO) this script:
  1. Loads the appropriate model checkpoint
  2. Builds a GrammarGuide from the SAME training split
  3. Runs next-step eval WITHOUT guidance   (baseline)
  4. Runs next-step eval WITH guidance      (constrained)
  5. Prints a side-by-side comparison table and writes results/guided_*.json

Usage:
    python eval_guided.py --kind gpt --size small   # uses outputs/gpt_small_*
    python eval_guided.py --kind gpt --size large   # uses outputs/gpt_large_*
    python eval_guided.py --kind gpt --size large --leave-out mosfet  # one split only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from data import FAMILIES, load_all, load_family, leave_one_family_out, train_val_split
from grammar_guide import GrammarGuide
from tokenizer import StepTokenizer

SEED = 42
EVAL_SEQS = 200   # sequences to eval per split (more = slower but more accurate)


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


# ── fast teacher-forced eval ──────────────────────────────────────────────────

@torch.no_grad()
def eval_gpt(gpt_model, seqs: dict, guide=None, max_seqs=EVAL_SEQS) -> dict:
    """Single forward pass per sequence — very fast.
    Optionally applies grammar guide at each prediction position."""
    keys = list(seqs)
    random.Random(SEED).shuffle(keys)
    keys = keys[:max_seqs]
    tok  = gpt_model.tok

    top1 = top3 = top5 = 0
    mrr  = 0.0
    n    = 0

    for k in keys:
        ids = tok.encode(seqs[k], add_special=True)[:gpt_model.max_pos]
        x   = torch.tensor([ids], device=gpt_model.device)
        logits = gpt_model.model(x).logits[0].clone()   # (L, V)

        # mask special tokens globally
        for b in (tok.pad_id, tok.bos_id, tok.unk_id):
            logits[:, b] = float("-inf")

        for t in range(len(ids) - 1):
            target = ids[t + 1]
            if target == tok.eos_id:
                continue

            step_logits = logits[t].clone()

            if guide is not None:
                # prefix = real steps before position t (skip <bos> at ids[0])
                prefix_steps = [tok.id_to_step.get(ids[j])
                                for j in range(1, t + 1)
                                if ids[j] in tok.id_to_step]
                prefix_steps = [s for s in prefix_steps if s and not s.startswith("<")]
                guide.mask_logits(step_logits, prefix_steps, tok.id_to_step)

            # rank = how many valid steps score HIGHER than the target
            rank = int((step_logits > step_logits[target]).sum())
            n    += 1
            top1 += (rank == 0)
            top3 += (rank < 3)
            top5 += (rank < 5)
            mrr  += 1.0 / (rank + 1)

    return {"top1": round(top1/n, 4), "top3": round(top3/n, 4),
            "top5": round(top5/n, 4), "mrr":  round(mrr/n,  4),
            "n_preds": n}


# ── model loader ──────────────────────────────────────────────────────────────

def load_gpt_checkpoint(ckpt_path: str, tok: StepTokenizer, device: str):
    from transformers import GPT2LMHeadModel
    from gpt import GPTModel
    model_raw = GPT2LMHeadModel.from_pretrained(ckpt_path)
    return GPTModel(model_raw, tok, device=device)


# ── one split ─────────────────────────────────────────────────────────────────

def run_split(ckpt_path: str | None, split_name: str,
              train_seqs: dict, test_seqs: dict,
              tok: StepTokenizer, device: str, size: str) -> dict | None:
    """Evaluate one (train/test) split with & without guide.  Returns result dict."""

    if ckpt_path is None or not os.path.isdir(ckpt_path):
        print(f"  [{split_name}] checkpoint not found: {ckpt_path} — skipping")
        return None

    print(f"\n  [{split_name}]  checkpoint={ckpt_path}")
    gpt = load_gpt_checkpoint(ckpt_path, tok, device)

    # Build grammar guide from the TRAINING split (not test split)
    vocab = [s for s in tok.step_to_id if not s.startswith("<")]
    guide = GrammarGuide.build(train_seqs.values(), vocab)
    gs    = guide.stats()
    print(f"    Guide:  {gs['bigram_contexts']} bigram ctx, "
          f"avg {gs['avg_trigram_successors']} trigram successors "
          f"(vocab={gs['vocab_size']})")

    # Without guide
    t0 = time.time()
    base = eval_gpt(gpt, test_seqs, guide=None)
    t1 = time.time()
    # With guide
    guided = eval_gpt(gpt, test_seqs, guide=guide)
    t2 = time.time()

    delta_top1 = guided["top1"] - base["top1"]
    delta_mrr  = guided["mrr"]  - base["mrr"]

    color_top1 = "▲" if delta_top1 >= 0 else "▼"
    color_mrr  = "▲" if delta_mrr  >= 0 else "▼"

    print(f"    Without guide: top1={base['top1']:.4f}  top3={base['top3']:.4f}  "
          f"top5={base['top5']:.4f}  mrr={base['mrr']:.4f}  ({t1-t0:.1f}s)")
    print(f"    WITH guide:    top1={guided['top1']:.4f}  top3={guided['top3']:.4f}  "
          f"top5={guided['top5']:.4f}  mrr={guided['mrr']:.4f}  ({t2-t1:.1f}s)")
    print(f"    Delta:         top1={color_top1}{abs(delta_top1):.4f}  "
          f"mrr={color_mrr}{abs(delta_mrr):.4f}")

    return {
        "split":       split_name,
        "checkpoint":  ckpt_path,
        "gpt_size":    size,
        "n_train":     len(train_seqs),
        "n_test":      len(test_seqs),
        "base":        base,
        "guided":      guided,
        "delta_top1":  round(delta_top1, 4),
        "delta_mrr":   round(delta_mrr,  4),
        "guide_stats": gs,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global EVAL_SEQS   # declared first so argparse default ref below is valid
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind",      default="gpt")
    ap.add_argument("--size",      default="large", choices=["tiny", "small", "large"])
    ap.add_argument("--leave-out", default=None, choices=FAMILIES)
    ap.add_argument("--eval-seqs", type=int, default=EVAL_SEQS)
    args = ap.parse_args()
    set_seed()
    EVAL_SEQS = args.eval_seqs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval_guided]  kind={args.kind}  size={args.size}  device={device}")

    # Build shared tokenizer from FULL vocab (including held-out family steps)
    all_seqs = load_all()
    tok      = StepTokenizer.build_from_sequences(all_seqs.values())
    print(f"  Tokenizer: {tok.vocab_size} tokens "
          f"({tok.vocab_size-4} process steps + 4 special)")

    splits_to_run = [args.leave_out] if args.leave_out else [None] + list(FAMILIES)
    results = []

    print(f"\n{'='*65}")
    print(f"  GPT:{args.size} — Grammar-guided decoding comparison")
    print(f"{'='*65}")

    for holdout in splits_to_run:
        if holdout is None:
            # ID run: trained on all families
            train_seqs, test_seqs = train_val_split(all_seqs, val_frac=0.1, seed=SEED)
            split_name = "ID — all families"
            ckpt_path  = f"outputs/gpt_{args.size}_allfam"
        else:
            train_seqs, test_seqs = leave_one_family_out(holdout, seed=SEED)
            split_name = f"OOD — hold-out {holdout.upper()}"
            ckpt_path  = f"outputs/gpt_{args.size}_loo_{holdout}"
            # For kremsians (or any new family), the allfam checkpoint is the right
            # model — it was trained on the same mosfet+igbt+ic split.
            if not os.path.isdir(ckpt_path):
                fallback = f"outputs/gpt_{args.size}_allfam"
                print(f"  [{split_name}] LOO checkpoint not found, "
                      f"using allfam: {fallback}")
                ckpt_path = fallback

        r = run_split(ckpt_path, split_name, train_seqs, test_seqs, tok, device, args.size)
        if r:
            results.append(r)

    # ── summary table ─────────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*65}")
        print(f"  SUMMARY — GPT:{args.size} with grammar-guided decoding")
        print(f"{'='*65}")
        header = f"  {'Split':<28} {'Base':>7} {'Guided':>8} {'Δ':>7} {'Target':>8}"
        print(header)
        print(f"  {'-'*60}")
        for r in results:
            delta_str = f"+{r['delta_top1']:.4f}" if r['delta_top1'] >= 0 else f"{r['delta_top1']:.4f}"
            hit_80 = "✅" if r['guided']['top1'] >= 0.80 else f"   ({r['guided']['top1']*100:.1f}%/80%)"
            print(f"  {r['split']:<28} {r['base']['top1']:>7.4f} "
                  f"{r['guided']['top1']:>8.4f} {delta_str:>7}  {hit_80}")
        print(f"{'='*65}\n")

    # ── save results ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = f"results/guided_{args.size}_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()

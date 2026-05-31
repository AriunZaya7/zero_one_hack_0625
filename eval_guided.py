"""Evaluate GPT checkpoints WITH and WITHOUT grammar-guided decoding.

For every experimental split (ID + 3 LOO) this script:
  1. Loads the appropriate model checkpoint
  2. Builds a GrammarGuide from the SAME training split
  3. Runs next-step eval WITHOUT guidance   (baseline)
  4. Runs next-step eval WITH guidance      (constrained)
  5. Prints a side-by-side comparison table and writes JSON results

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

from data import (
    FAMILIES,
    OOD_3_FAMILIES,
    TRAIN_12_FAMILIES,
    leave_one_family_out,
    load_all,
    train12_test3_by_family,
    train_val_split,
)
from grammar_guide import GrammarGuide
from tokenizer import StepTokenizer

SEED = 42
EVAL_SEQS = 200   # sequences to eval per split (more = slower but more accurate)


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


def best_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _mean_metric(results: list[dict], section: str, metric: str) -> float:
    return round(sum(r[section][metric] for r in results) / max(len(results), 1), 4)


def aggregate_ood3(results: list[dict]) -> dict:
    """Average the three SUBMISSION_1 OOD family results family-wise."""
    return {
        "split": "OOD3 average — train12/test3",
        "families": OOD_3_FAMILIES,
        "base": {
            "top1": _mean_metric(results, "base", "top1"),
            "top3": _mean_metric(results, "base", "top3"),
            "top5": _mean_metric(results, "base", "top5"),
            "mrr": _mean_metric(results, "base", "mrr"),
        },
        "guided": {
            "top1": _mean_metric(results, "guided", "top1"),
            "top3": _mean_metric(results, "guided", "top3"),
            "top5": _mean_metric(results, "guided", "top5"),
            "mrr": _mean_metric(results, "guided", "mrr"),
        },
        "delta_top1": round(
            _mean_metric(results, "guided", "top1") - _mean_metric(results, "base", "top1"),
            4,
        ),
        "delta_mrr": round(
            _mean_metric(results, "guided", "mrr") - _mean_metric(results, "base", "mrr"),
            4,
        ),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global EVAL_SEQS   # declared first so argparse default ref below is valid
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind",      default="gpt")
    ap.add_argument("--size",      default="large", choices=["tiny", "small", "large"])
    ap.add_argument("--leave-out", default=None, choices=FAMILIES)
    ap.add_argument(
        "--submission-1-ood",
        action="store_true",
        help="Evaluate the fixed 15-family protocol: train on 12, report average over 3 held-out OOD families.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to evaluate. Defaults to outputs/gpt_<size>_train12 for --submission-1-ood.",
    )
    ap.add_argument("--eval-seqs", type=int, default=EVAL_SEQS)
    args = ap.parse_args()
    set_seed()
    EVAL_SEQS = args.eval_seqs

    device = best_torch_device()
    print(f"[eval_guided]  kind={args.kind}  size={args.size}  device={device}")

    if args.submission_1_ood:
        train_seqs, tests_by_family = train12_test3_by_family(seed=SEED)
        tok = StepTokenizer.build_from_sequences(train_seqs.values())
        print(f"  Tokenizer: {tok.vocab_size} tokens "
              f"({tok.vocab_size-4} process steps + 4 special)")
        ckpt_path = args.checkpoint or f"outputs/gpt_{args.size}_train12"
        results = []
        print("\nSUBMISSION_1 protocol: 15 families total")
        print(f"  train families ({len(TRAIN_12_FAMILIES)}): {', '.join(TRAIN_12_FAMILIES)}")
        print(f"  held-out families ({len(OOD_3_FAMILIES)}): {', '.join(OOD_3_FAMILIES)}")
        for family, test_seqs in tests_by_family.items():
            split_name = f"OOD3 — hold-out {family.upper()}"
            r = run_split(ckpt_path, split_name, train_seqs, test_seqs, tok, device, args.size)
            if r:
                results.append(r)
        if not results:
            print("No SUBMISSION_1 OOD results produced; checkpoint is missing or unreadable.")
            return
        if len(results) != len(OOD_3_FAMILIES):
            print(f"WARNING: expected {len(OOD_3_FAMILIES)} OOD family results, got {len(results)}.")
        avg = aggregate_ood3(results)
        print(f"\n{'='*65}")
        print("  SUBMISSION_1 OOD3 FAMILY-WISE AVERAGE")
        print(f"{'='*65}")
        print(f"  Base top1={avg['base']['top1']:.4f} top3={avg['base']['top3']:.4f} "
              f"top5={avg['base']['top5']:.4f} mrr={avg['base']['mrr']:.4f}")
        print(f"  Guided   top1={avg['guided']['top1']:.4f} top3={avg['guided']['top3']:.4f} "
              f"top5={avg['guided']['top5']:.4f} mrr={avg['guided']['mrr']:.4f}")
        results.append(avg)
        os.makedirs("results", exist_ok=True)
        out = f"results/submission_1_ood3_{args.size}_{int(time.time())}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved -> {out}")
        return

    all_seqs = load_all(FAMILIES)
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
            # For legacy one-off holdouts, the allfam checkpoint is the fallback
            # model when a dedicated leave-one-out checkpoint is unavailable.
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

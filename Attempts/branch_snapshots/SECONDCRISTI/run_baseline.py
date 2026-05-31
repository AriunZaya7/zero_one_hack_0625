"""Quick baseline experiment — sanity check that the pieces fit + the honest bar.

This is intentionally SMALL. It does not train a GPT, touch W&B, or build a
submission. It just exercises the already-DONE modules (tokenizer, data, ngram)
on the one experiment that matters most in PLAN.md: the leave-one-family-out
(LOO) OOD proxy for the hidden Task 4.

Two things get printed:
  1. In-distribution next-step accuracy of the n-gram (train+eval on all 3 families).
  2. The ID -> OOD drop per held-out family (train on 2, test on the 3rd).

The OOD column is the number to beat with the transformer. If this runs and the
numbers look sane, the data path / tokenizer / n-gram interface all work.

Run:  python run_baseline.py
"""
from __future__ import annotations

import random

from data import FAMILIES, load_all, leave_one_family_out, train_val_split
from ngram import NGramModel
from tokenizer import StepTokenizer

SEED = 42
EVAL_SEQS = 150   # cap eval sequences per split so this stays fast (seconds)


def next_step_metrics(model: NGramModel, seqs: dict[str, list[str]],
                      max_seqs: int = EVAL_SEQS) -> dict[str, float]:
    """Top-1/3/5 accuracy + MRR over every next-step prediction in `seqs`.

    For each position i in a sequence we feed the prefix [:i] and ask the model
    to rank the next step; we score where the true step lands in that ranking.
    """
    keys = list(seqs)
    random.Random(SEED).shuffle(keys)
    keys = keys[:max_seqs]

    top1 = top3 = top5 = 0
    mrr = 0.0
    n = 0
    for k in keys:
        seq = seqs[k]
        for i in range(1, len(seq)):          # predict step i from steps [:i]
            prefix, truth = seq[:i], seq[i]
            ranking = model.next_step_ranking(prefix, k=5)
            n += 1
            if truth in ranking:
                rank = ranking.index(truth)    # 0-based
                if rank == 0:
                    top1 += 1
                if rank < 3:
                    top3 += 1
                top5 += 1
                mrr += 1.0 / (rank + 1)
    return {
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "mrr": mrr / n,
        "n_predictions": n,
    }


def main() -> None:
    # ---- 0. sanity: load everything + build the tokenizer -------------------
    all_seqs = load_all()
    tok = StepTokenizer.build_from_sequences(all_seqs.values())
    lengths = [len(s) for s in all_seqs.values()]
    print(f"[data] loaded {len(all_seqs)} sequences across {FAMILIES}")
    print(f"[data] step length: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths) / len(lengths):.1f}")
    print(f"[tokenizer] vocab_size={tok.vocab_size} "
          f"({tok.vocab_size - 4} process steps + 4 special tokens)")

    # ---- 1. in-distribution baseline ---------------------------------------
    train, val = train_val_split(all_seqs, val_frac=0.1, seed=SEED)
    ngram = NGramModel(n=3, alpha=0.4).fit(train.values())
    id_metrics = next_step_metrics(ngram, val)
    print("\n=== In-distribution next-step (n-gram, all families) ===")
    print(f"  top1={id_metrics['top1']:.3f}  top3={id_metrics['top3']:.3f}  "
          f"top5={id_metrics['top5']:.3f}  mrr={id_metrics['mrr']:.3f}  "
          f"(n={id_metrics['n_predictions']})")

    # ---- 2. leave-one-family-out OOD proxy (the headline) ------------------
    print("\n=== Leave-one-family-out (train on 2, test on held-out) ===")
    print(f"  {'held-out':<10} {'top1':>7} {'top3':>7} {'top5':>7} {'mrr':>7}")
    id_top1 = id_metrics["top1"]
    for fam in FAMILIES:
        tr, te = leave_one_family_out(fam, seed=SEED)
        m = NGramModel(n=10, alpha=0.4).fit(tr.values())
        ood = next_step_metrics(m, te)
        print(f"  {fam:<10} {ood['top1']:>7.3f} {ood['top3']:>7.3f} "
              f"{ood['top5']:>7.3f} {ood['mrr']:>7.3f}")

    print("\n[note] The OOD top1 is the bar the transformer must beat without "
          "shrinking too far below the in-distribution top1 above.")


if __name__ == "__main__":
    main()

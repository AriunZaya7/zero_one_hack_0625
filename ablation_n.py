"""Throwaway ablation: does a bigger n-gram order help, and where?

Sweeps n over a range and reports in-distribution top-1 vs leave-one-family-out
(OOD) top-1, plus the gap. The gap is the point: it tests the project's whole
thesis (memorization vs generalization) on the cheap baseline.
"""
from __future__ import annotations

from data import FAMILIES, leave_one_family_out, load_all, train_val_split
from ngram import NGramModel
from run_baseline import next_step_metrics

SEED = 42

all_seqs = load_all()
train, val = train_val_split(all_seqs, val_frac=0.1, seed=SEED)

print(f"{'n':>3} {'ID top1':>9} {'OOD top1':>9} {'gap':>7}")
for n in [2, 3, 4, 5, 7, 10, 15]:
    id_m = NGramModel(n=n, alpha=0.4).fit(train.values())
    id_top1 = next_step_metrics(id_m, val)["top1"]

    # average OOD over the three held-out families
    ood_vals = []
    for fam in FAMILIES:
        tr, te = leave_one_family_out(fam, seed=SEED)
        m = NGramModel(n=n, alpha=0.4).fit(tr.values())
        ood_vals.append(next_step_metrics(m, te)["top1"])
    ood_top1 = sum(ood_vals) / len(ood_vals)

    print(f"{n:>3} {id_top1:>9.3f} {ood_top1:>9.3f} {id_top1 - ood_top1:>7.3f}")

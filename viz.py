"""Plots from results/*.json (one file per train.py run).

Three figures (PLAN.md §10/§11):
  1. ood_vs_ngram.png   - OOD next-step top-1 per model vs the n-gram bar
  2. id_vs_ood_drop.png - ID vs OOD top-1 per model + the generalization drop
  3. scaling.png        - OOD top-1 vs #params (from-scratch curve vs pretrained curve)

Usage:  python viz.py            # reads results/ -> writes plots/*.png
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = "plots"


def load_rows(results_dir: str = RESULTS_DIR) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def _label(r: dict) -> str:
    return f"{r['model_type']}:{r['model_name']}"


def plot_ood_vs_ngram(rows, out):
    ood = [r for r in rows if r["mode"] == "ood"]
    if not ood:
        return
    labels = [f"{_label(r)}\n(holdout {r['holdout']})" for r in ood]
    x = range(len(ood))
    plt.figure(figsize=(max(6, len(ood) * 1.6), 4))
    plt.bar([i - 0.2 for i in x], [r["ngram_top1"] for r in ood], width=0.4,
            label="n-gram", color="#bbb")
    plt.bar([i + 0.2 for i in x], [r["top1"] for r in ood], width=0.4,
            label="model", color="#3b7dd8")
    plt.xticks(list(x), labels, fontsize=8)
    plt.ylabel("OOD next-step top-1")
    plt.title("Held-out family: model vs n-gram")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_id_vs_ood(rows, out):
    by_model = defaultdict(dict)
    for r in rows:
        by_model[_label(r)][r["mode"]] = r["top1"]
    models = [m for m in by_model if "id" in by_model[m] and "ood" in by_model[m]]
    if not models:
        return
    x = range(len(models))
    plt.figure(figsize=(max(6, len(models) * 1.6), 4))
    plt.bar([i - 0.2 for i in x], [by_model[m]["id"] for m in models], width=0.4,
            label="ID", color="#5fa85f")
    plt.bar([i + 0.2 for i in x], [by_model[m]["ood"] for m in models], width=0.4,
            label="OOD", color="#d8743b")
    for i, m in enumerate(models):
        drop = by_model[m]["id"] - by_model[m]["ood"]
        plt.text(i, max(by_model[m]["id"], by_model[m]["ood"]) + 0.01,
                 f"-{drop:.2f}", ha="center", fontsize=8)
    plt.xticks(list(x), models, fontsize=8)
    plt.ylabel("next-step top-1")
    plt.title("ID vs OOD (smaller drop = better generalization)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_scaling(rows, out):
    ood = [r for r in rows if r["mode"] == "ood"]
    if not ood:
        return
    curves = defaultdict(list)
    for r in ood:
        curves[r["model_type"]].append((r["n_params_M"], r["top1"]))
    plt.figure(figsize=(6, 4))
    for mtype, pts in curves.items():
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, "o-", label=mtype)
    plt.xscale("log")
    plt.xlabel("params (M, log scale)")
    plt.ylabel("OOD next-step top-1")
    plt.title("Scaling: OOD performance vs model size")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    plot_ood_vs_ngram(rows, os.path.join(OUT_DIR, "ood_vs_ngram.png"))
    plot_id_vs_ood(rows, os.path.join(OUT_DIR, "id_vs_ood_drop.png"))
    plot_scaling(rows, os.path.join(OUT_DIR, "scaling.png"))
    print(f"wrote plots to {OUT_DIR}/ from {len(rows)} runs")


if __name__ == "__main__":
    main()

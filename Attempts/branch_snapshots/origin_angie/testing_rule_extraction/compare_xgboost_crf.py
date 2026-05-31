"""
compare_crf_xgboost.py
========================
Branch: angie
Folder: rule_extraction/

Trains and compares two statistical models side by side:
  1. CRF  (Conditional Random Field)   — sequence-aware, category-level learning
  2. XGBoost                           — gradient boosting on engineered features

Both are:
  - Fully deterministic
  - Trained on IGBT + IC (dataset_train.jsonl)
  - Validated on IGBT + IC held-out (dataset_id_val.jsonl)
  - OOD tested on MOSFET (dataset_ood_test.jsonl)
  - Logged side by side to WandB

Usage:
    pip install sklearn-crfsuite xgboost scikit-learn wandb
    python compare_crf_xgboost.py

All results logged to WandB project: industrial-ai-hackathon
Run names: crf-baseline, xgboost-baseline
"""

import json
import os
import random
import math
import argparse
from collections import Counter, defaultdict

import numpy as np
import wandb

# ── Optional imports with helpful error messages ───────────────────────────────
try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
except ImportError:
    raise ImportError("Run: pip install sklearn-crfsuite")

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Run: pip install xgboost")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    classification_report,
)


# ── Step Category Map ──────────────────────────────────────────────────────────
# This is the key to generalization — models learn category-level patterns
# which transfer to unseen families (MOSFET) even if exact step names differ

STEP_CATEGORIES = {
    # RECEIVE / LOGISTICS
    "RECEIVE WAFER LOT":              "RECEIVE",
    "LOT IDENTIFICATION":             "RECEIVE",
    "SHIP LOT":                       "SHIP",
    "YIELD ANALYSIS":                 "ANALYSIS",

    # INSPECTION / MEASUREMENT
    "PRE CLEAN INSPECTION":           "INSPECT",
    "INITIAL WAFER INSPECTION":       "INSPECT",
    "FINAL INSPECTION":               "INSPECT",
    "MEASURE INITIAL THICKNESS":      "MEASURE",
    "MEASURE THICKNESS":              "MEASURE",
    "MEASURE OXIDE THICKNESS":        "MEASURE",
    "MEASURE SURFACE PLANARITY":      "MEASURE",
    "MEASURE GATE OXIDE":             "MEASURE",
    "MEASURE JUNCTION DEPTH":         "MEASURE",
    "MEASURE SHEET RESISTANCE":       "MEASURE",
    "MEASURE ILD THICKNESS":          "MEASURE",
    "MEASURE POST CMP":               "MEASURE",
    "MEASURE METAL THICKNESS":        "MEASURE",

    # CLEAN
    "HF DIP":                         "CLEAN",
    "RCA CLEAN 1":                    "CLEAN",
    "RCA CLEAN 2":                    "CLEAN",
    "DRY WAFER":                      "CLEAN",
    "CLEAN AFTER ETCH":               "CLEAN",
    "CLEAN AFTER CMP":                "CLEAN",
    "CLEAN WAFER SURFACE":            "CLEAN",
    "POST IMPLANT CLEAN":             "CLEAN",
    "PRE DIFFUSION CLEAN":            "CLEAN",
    "SOLVENT CLEAN":                  "CLEAN",

    # DEPOSIT
    "THERMAL OXIDATION":              "DEPOSIT",
    "GATE OXIDE GROWTH":              "DEPOSIT",
    "DEPOSIT PAD OXIDE":              "DEPOSIT",
    "FIELD OXIDE GROWTH":             "DEPOSIT",
    "EPITAXIAL DEPOSITION":           "DEPOSIT",
    "DEPOSIT EPITAXIAL LAYER":        "DEPOSIT",
    "DEPOSIT POLYSILICON":            "DEPOSIT",
    "DEPOSIT GATE OXIDE OR DIELECTRIC":"DEPOSIT",
    "DEPOSIT SPACER DIELECTRIC":      "DEPOSIT",
    "DEPOSIT FIELD OXIDE":            "DEPOSIT",
    "DEPOSIT INTERLAYER DIELECTRIC":  "DEPOSIT",
    "DEPOSIT ILD OXIDE":              "DEPOSIT",
    "DEPOSIT BARRIER METAL":          "DEPOSIT",
    "DEPOSIT METAL SEED":             "DEPOSIT",
    "DEPOSIT METAL 1":                "DEPOSIT",
    "DEPOSIT TOP METAL":              "DEPOSIT",
    "DEPOSIT BACKSIDE METAL":         "DEPOSIT_BACKSIDE",
    "DEPOSIT TUNGSTEN SEED":          "DEPOSIT",
    "DEPOSIT PASSIVATION":            "DEPOSIT_PASSIVATION",
    "DEPOSIT PASSIVATION LAYER":      "DEPOSIT_PASSIVATION",
    "DEPOSIT BACKSIDE PROTECTION":    "DEPOSIT_BACKSIDE",

    # ETCH
    "OXIDE ETCH":                     "ETCH",
    "OXIDE ETCH DRY":                 "ETCH",
    "POLYSILICON ETCH":               "ETCH",
    "POLYSILICON ETCH DRY":           "ETCH",
    "FIELD OXIDE ETCH":               "ETCH",
    "ETCH SILICON OR OXIDE WINDOW":   "ETCH",
    "VIA ETCH":                       "ETCH",
    "VIA ETCH THROUGH DIELECTRIC":    "ETCH",
    "DIELECTRIC ETCH VIA":            "ETCH",
    "METAL ETCH":                     "ETCH_METAL",
    "METAL ETCH DRY":                 "ETCH_METAL",
    "PASSIVATION ETCH PAD OPENING":   "ETCH_PAD",
    "PASSIVATION ETCH":               "ETCH_PAD",
    "ANISOTROPIC ETCH SPACER":        "ETCH_SPACER",

    # LITHO
    "COAT PHOTORESIST":               "LITHO_COAT",
    "SOFT BAKE":                      "LITHO_BAKE",
    "HARD BAKE":                      "LITHO_BAKE",
    "POST EXPOSE BAKE":               "LITHO_BAKE",
    "ALIGN MASK LEVEL 1":             "LITHO_ALIGN",
    "ALIGN MASK LEVEL 2":             "LITHO_ALIGN",
    "ALIGN MASK LEVEL 3":             "LITHO_ALIGN",
    "ALIGN MASK LEVEL 4":             "LITHO_ALIGN",
    "EXPOSE LITHO LEVEL 1":           "LITHO_EXPOSE",
    "EXPOSE LITHO LEVEL 2":           "LITHO_EXPOSE",
    "EXPOSE LITHO LEVEL 3":           "LITHO_EXPOSE",
    "EXPOSE LITHO LEVEL 4":           "LITHO_EXPOSE",
    "DEVELOP PHOTORESIST":            "LITHO_DEVELOP",
    "STRIP RESIST":                   "LITHO_STRIP",
    "STRIP PHOTORESIST":              "LITHO_STRIP",
    "ASH RESIST":                     "LITHO_STRIP",

    # IMPLANT
    "IMPLANT WELL":                   "IMPLANT",
    "IMPLANT SOURCE DRAIN":           "IMPLANT",
    "IMPLANT SOURCE REGION":          "IMPLANT",
    "IMPLANT LDD":                    "IMPLANT",
    "IMPLANT P BODY":                 "IMPLANT",
    "IMPLANT N BUFFER":               "IMPLANT",
    "IMPLANT CHANNEL STOP":           "IMPLANT",
    "IMPLANT DRAIN / CATHODE REGION": "IMPLANT",
    "IMPLANT N-TYPE":                 "IMPLANT",
    "ANNEAL":                         "ANNEAL",
    "RAPID THERMAL ANNEAL":           "ANNEAL",
    "DRIVE IN DIFFUSION":             "ANNEAL",

    # CMP / FILL
    "CMP DIELECTRIC":                 "CMP",
    "CMP INTERLAYER DIELECTRIC":      "CMP",
    "CMP METAL":                      "CMP",
    "CMP VIA FILL":                   "CMP",
    "FILL VIA METAL":                 "FILL",
    "FILL VIA TUNGSTEN":              "FILL",

    # PASSIVATION
    "CURE PASSIVATION":               "CURE_PASSIVATION",
    "OPEN PAD WINDOW":                "PAD_OPEN",
    "OPEN BOND PAD WINDOW":           "PAD_OPEN",
    "PAD WINDOW LITHO":               "PAD_OPEN",

    # BACKSIDE
    "BACKSIDE GRIND":                 "BACKSIDE",
    "BACKSIDE CLEAN":                 "BACKSIDE",
    "BACKSIDE ANNEAL":                "BACKSIDE",

    # TEST
    "PARAMETRIC TEST":                "TEST_ELECTRICAL",
    "ELECTRICAL PARAMETRIC TEST":     "TEST_ELECTRICAL",
    "THRESHOLD VOLTAGE TEST":         "TEST_ELECTRICAL",
    "BREAKDOWN VOLTAGE TEST":         "TEST_ELECTRICAL",
    "LEAKAGE TEST":                   "TEST_ELECTRICAL",
    "SWITCHING TEST":                 "TEST_ELECTRICAL",
    "WAFER SORT TEST":                "TEST_SORT",
}

def categorize(step: str) -> str:
    """Map a step name to its semantic category. Unknown steps get UNKNOWN."""
    return STEP_CATEGORIES.get(step.strip().upper(), "UNKNOWN")


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):>5} records from {os.path.basename(path)}")
    return records


def records_to_sequences(records: list[dict]) -> list[dict]:
    """
    Convert JSONL records to sequences with both raw steps and categories.
    Returns list of { family, steps, categories }
    """
    sequences = []
    for r in records:
        steps = r["steps"]
        categories = [categorize(s) for s in steps]
        sequences.append({
            "family":     r["family"],
            "steps":      steps,
            "categories": categories,
        })
    return sequences


# ── Feature Engineering ────────────────────────────────────────────────────────

# State flags that capture long-range rule dependencies
def compute_state_flags(steps: list[str], position: int) -> dict:
    """
    Compute binary state flags up to current position.
    These capture global sequence state — critical for rules like
    RULE_SHIP_BEFORE_TEST and RULE_TEST_BEFORE_PASSIVATION.
    """
    history = steps[:position]
    cats    = [categorize(s) for s in history]

    return {
        "clean_in_last_5":        any(c == "CLEAN" for c in cats[-5:]),
        "clean_in_last_12":       any(c == "CLEAN" for c in cats[-12:]),
        "litho_develop_in_last_12": any(c == "LITHO_DEVELOP" for c in cats[-12:]),
        "litho_expose_in_last_15":  any(c == "LITHO_EXPOSE" for c in cats[-15:]),
        "deposit_in_last_6":      any(c == "DEPOSIT" for c in cats[-6:]),
        "passivation_deposited":  "DEPOSIT_PASSIVATION" in cats,
        "passivation_cured":      "CURE_PASSIVATION" in cats,
        "wafer_sort_done":        "TEST_SORT" in cats,
        "litho_levels_done":      sum(1 for c in cats if c == "LITHO_EXPOSE"),
        "implant_window_open":    any(c in ("ETCH", "LITHO_DEVELOP") for c in cats[-15:]),
    }


# ── CRF Features ──────────────────────────────────────────────────────────────

def word2features(seq_cats: list[str], seq_steps: list[str],
                  family: str, i: int) -> dict:
    """
    Feature function for CRF.
    Uses categories (not raw step names) for generalization.
    """
    cat   = seq_cats[i]
    step  = seq_steps[i]
    n     = len(seq_cats)
    flags = compute_state_flags(seq_steps, i)

    features = {
        # Current step
        "cat":              cat,
        "step":             step,
        "family":           family,

        # Position features
        "position_bin":     str(min(int(i / n * 10), 9)),  # 0-9 bucket
        "is_early":         str(i < n * 0.2),
        "is_late":          str(i > n * 0.8),

        # Context window (categories only for generalization)
        "cat_minus1":       seq_cats[i-1]  if i > 0 else "BOS",
        "cat_minus2":       seq_cats[i-2]  if i > 1 else "BOS",
        "cat_minus3":       seq_cats[i-3]  if i > 2 else "BOS",
        "cat_plus1":        seq_cats[i+1]  if i < n-1 else "EOS",

        # Bigram / trigram context
        "bigram_m2_m1":     f"{seq_cats[i-2]}_{seq_cats[i-1]}" if i > 1 else "BOS",
        "trigram":          f"{seq_cats[i-2]}_{seq_cats[i-1]}_{cat}" if i > 1 else "BOS",

        # State flags (long-range rule encoding)
        **{f"flag_{k}": str(v) for k, v in flags.items()},
    }
    return features


def seq2features(seq: dict) -> list[dict]:
    cats   = seq["categories"]
    steps  = seq["steps"]
    family = seq["family"]
    return [word2features(cats, steps, family, i) for i in range(len(cats))]


def seq2labels(seq: dict) -> list[str]:
    """Labels = categories of each step (what we predict)."""
    return seq["categories"]


def build_crf_dataset(sequences: list[dict]):
    X = [seq2features(s) for s in sequences]
    y = [seq2labels(s)   for s in sequences]
    return X, y


# ── XGBoost Features ──────────────────────────────────────────────────────────

CATEGORY_LIST = sorted(set(STEP_CATEGORIES.values())) + ["UNKNOWN", "BOS"]
CAT_INDEX     = {c: i for i, c in enumerate(CATEGORY_LIST)}
FAMILY_INDEX  = {"IGBT": 0, "IC": 1, "MOSFET": 2, "UNKNOWN": 3}

def cat_onehot(cat: str) -> list[int]:
    vec = [0] * len(CATEGORY_LIST)
    idx = CAT_INDEX.get(cat, CAT_INDEX["UNKNOWN"])
    vec[idx] = 1
    return vec


def build_xgb_features(seq: dict) -> tuple[list[list[float]], list[int]]:
    """
    Build (features, labels) for every prediction position in one sequence.
    Each position = one row in the XGBoost training matrix.
    """
    cats   = seq["categories"]
    steps  = seq["steps"]
    family = seq["family"]
    n      = len(cats)

    X_rows = []
    y_rows = []

    for i in range(1, n):  # predict from position 1 onward
        flags = compute_state_flags(steps, i)

        row = []

        # Last 5 categories one-hot encoded
        for offset in range(1, 6):
            c = cats[i - offset] if i >= offset else "BOS"
            row.extend(cat_onehot(c))

        # Position features
        row.append(i / n)                    # normalized position
        row.append(float(i < n * 0.2))       # is early
        row.append(float(i > n * 0.8))       # is late
        row.append(float(n))                 # sequence length

        # Family one-hot
        fam_vec = [0] * len(FAMILY_INDEX)
        fam_vec[FAMILY_INDEX.get(family, 3)] = 1
        row.extend(fam_vec)

        # State flags (long-range rules)
        row.extend([
            float(flags["clean_in_last_5"]),
            float(flags["clean_in_last_12"]),
            float(flags["litho_develop_in_last_12"]),
            float(flags["litho_expose_in_last_15"]),
            float(flags["deposit_in_last_6"]),
            float(flags["passivation_deposited"]),
            float(flags["passivation_cured"]),
            float(flags["wafer_sort_done"]),
            float(flags["litho_levels_done"]),
            float(flags["implant_window_open"]),
        ])

        X_rows.append(row)
        # Store the raw category STRING, not the index
        # Remapping to int happens in build_xgb_dataset after we know all classes
        y_rows.append(cats[i])

    return X_rows, y_rows


def build_xgb_dataset(sequences: list[dict], class_map=None):
    """
    Build XGBoost dataset.
    class_map: dict mapping category string -> int index.
    Must be built from training data first, then reused for all splits.
    Unknown categories (in OOD but not training) map to index 0.
    """
    X_all, y_all = [], []
    for seq in sequences:
        X_rows, y_rows = build_xgb_features(seq)
        X_all.extend(X_rows)
        y_all.extend(y_rows)

    X = np.array(X_all, dtype=np.float32)

    if class_map is not None:
        # Map string category -> int, unknown -> 0
        y = np.array([class_map.get(c, 0) for c in y_all], dtype=np.int32)
    else:
        # First pass: return raw strings as-is for building the class_map
        return X, y_all

    return X, y


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_crf(model, sequences: list[dict], label: str) -> dict:
    """Evaluate CRF on next-step category prediction."""
    X, y_true = build_crf_dataset(sequences)
    y_pred = model.predict(X)

    # Flatten for metrics
    y_true_flat = [label for seq in y_true for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]

    acc = accuracy_score(y_true_flat, y_pred_flat)

    # Top-1 per transition (exclude BOS)
    correct = sum(t == p for t, p in zip(y_true_flat, y_pred_flat))
    total   = len(y_true_flat)

    # Category coverage — how many unique categories correctly predicted
    correct_cats = set(t for t, p in zip(y_true_flat, y_pred_flat) if t == p)

    return {
        f"{label}/top1_accuracy":   acc,
        f"{label}/correct":         correct,
        f"{label}/total":           total,
        f"{label}/correct_categories": len(correct_cats),
    }


def evaluate_xgb(model, sequences: list[dict],
                 class_map: dict, label: str) -> dict:
    """Evaluate XGBoost on next-step category prediction."""
    X, y_true = build_xgb_dataset(sequences, class_map)

    y_pred      = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)

    # Top-3 and Top-5 accuracy
    n_classes = y_pred_proba.shape[1]
    all_classes = list(range(n_classes))
    top3 = top_k_accuracy_score(y_true, y_pred_proba, k=min(3, n_classes), labels=all_classes)
    top5 = top_k_accuracy_score(y_true, y_pred_proba, k=min(5, n_classes), labels=all_classes)

    # MRR
    mrr = 0.0
    for true_label, proba in zip(y_true, y_pred_proba):
        ranked = np.argsort(proba)[::-1]
        rank   = np.where(ranked == true_label)[0]
        if len(rank) > 0:
            mrr += 1.0 / (rank[0] + 1)
    mrr /= max(len(y_true), 1)

    return {
        f"{label}/top1_accuracy": acc,
        f"{label}/top3_accuracy": top3,
        f"{label}/top5_accuracy": top5,
        f"{label}/mrr":           mrr,
        f"{label}/total":         len(y_true),
    }


# ── Anomaly Detection ──────────────────────────────────────────────────────────

def anomaly_score_crf(model, sequence: dict) -> float:
    """
    Use CRF marginal probabilities as anomaly score.
    Lower average probability = more anomalous.
    """
    X = [seq2features(sequence)]
    # sklearn-crfsuite marginals
    marginals = model.predict_marginals(X)[0]
    true_cats  = sequence["categories"]

    log_prob = 0.0
    for marginal, true_cat in zip(marginals, true_cats):
        prob = marginal.get(true_cat, 1e-10)
        log_prob += math.log(prob)

    return log_prob / max(len(true_cats), 1)


def anomaly_score_xgb(model, sequence: dict, class_map: dict = None) -> float:
    """
    Use XGBoost prediction confidence as anomaly score.
    Low average max_proba = model is uncertain = potentially anomalous.
    """
    X_rows, _ = build_xgb_features(sequence)
    if not X_rows:
        return 0.0
    X = np.array(X_rows, dtype=np.float32)
    probas = model.predict_proba(X)
    avg_confidence = np.mean(np.max(probas, axis=1))
    return float(avg_confidence)


# ── WandB Comparison Table ─────────────────────────────────────────────────────

def log_comparison_table(crf_metrics: dict, xgb_metrics: dict,
                          split: str):
    """Log a side-by-side comparison table to WandB."""
    crf_top1 = crf_metrics.get(f"{split}/top1_accuracy", 0) * 100
    xgb_top1 = xgb_metrics.get(f"{split}/top1_accuracy", 0) * 100
    xgb_top3 = xgb_metrics.get(f"{split}/top3_accuracy", 0) * 100
    xgb_top5 = xgb_metrics.get(f"{split}/top5_accuracy", 0) * 100
    xgb_mrr  = xgb_metrics.get(f"{split}/mrr", 0) * 100

    table = wandb.Table(
        columns=["Model", "Split", "Top-1 %", "Top-3 %", "Top-5 %", "MRR %"],
        data=[
            ["CRF",     split, round(crf_top1, 2), 0.0, 0.0, 0.0],
            ["XGBoost", split, round(xgb_top1, 2), round(xgb_top3, 2),
             round(xgb_top5, 2), round(xgb_mrr, 2)],
        ]
    )
    wandb.log({f"comparison_table_{split}": table})


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="../small_transformer_model/data",
                        help="Path to folder with dataset_*.jsonl files")
    parser.add_argument("--wandb_project", default="industrial-ai-hackathon")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--xgb_trees",     type=int, default=300,
                        help="Number of XGBoost trees")
    parser.add_argument("--xgb_depth",     type=int, default=6,
                        help="XGBoost max tree depth")
    parser.add_argument("--crf_max_iter",  type=int, default=100,
                        help="CRF max iterations")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── WandB ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name="crf-vs-xgboost",
        config={
            "branch":       "angie",
            "experiment":   "crf_vs_xgboost_comparison",
            "xgb_trees":    args.xgb_trees,
            "xgb_depth":    args.xgb_depth,
            "crf_max_iter": args.crf_max_iter,
            "seed":         args.seed,
            "data_dir":     args.data_dir,
        },
        tags=["angie", "crf", "xgboost", "comparison", "statistical"],
    )

    # ── Load Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_records   = load_jsonl(os.path.join(args.data_dir, "dataset_train.jsonl"))
    id_val_records  = load_jsonl(os.path.join(args.data_dir, "dataset_id_val.jsonl"))
    ood_records     = load_jsonl(os.path.join(args.data_dir, "dataset_ood_test.jsonl"))

    train_seqs   = records_to_sequences(train_records)
    id_val_seqs  = records_to_sequences(id_val_records)
    ood_seqs     = records_to_sequences(ood_records)

    wandb.log({
        "data/train_sequences":   len(train_seqs),
        "data/id_val_sequences":  len(id_val_seqs),
        "data/ood_sequences":     len(ood_seqs),
        "data/unique_categories": len(CATEGORY_LIST),
    })

    print(f"\nTrain: {len(train_seqs)} | ID Val: {len(id_val_seqs)} | OOD: {len(ood_seqs)}")

    # ── Verify category coverage ───────────────────────────────────────────────
    all_steps = set()
    for seq in train_seqs + id_val_seqs + ood_seqs:
        all_steps.update(seq["steps"])
    unknown_steps = [s for s in all_steps if categorize(s) == "UNKNOWN"]
    if unknown_steps:
        print(f"\n[WARNING] {len(unknown_steps)} steps not in category map:")
        for s in sorted(unknown_steps)[:10]:
            print(f"  '{s}'")
        print("  Consider adding these to STEP_CATEGORIES for better accuracy")
    wandb.log({"data/uncategorized_steps": len(unknown_steps)})


    # ══════════════════════════════════════════════════════════════════════
    # MODEL 1 — CRF
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Training CRF (max_iter={args.crf_max_iter})")
    print(f"{'='*55}")

    X_train_crf, y_train_crf = build_crf_dataset(train_seqs)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=args.crf_max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train_crf, y_train_crf)
    print(f"  CRF trained | classes: {len(crf.classes_)}")
    wandb.log({"crf/n_classes": len(crf.classes_)})

    # Evaluate CRF
    print("\n  Evaluating CRF...")
    crf_id_metrics  = evaluate_crf(crf, id_val_seqs, "crf_id_val")
    crf_ood_metrics = evaluate_crf(crf, ood_seqs,    "crf_ood_test")

    crf_id_top1  = crf_id_metrics["crf_id_val/top1_accuracy"]  * 100
    crf_ood_top1 = crf_ood_metrics["crf_ood_test/top1_accuracy"] * 100
    crf_drop     = crf_id_top1 - crf_ood_top1

    print(f"  CRF ID Val  Top-1: {crf_id_top1:.2f}%")
    print(f"  CRF OOD     Top-1: {crf_ood_top1:.2f}%")
    print(f"  CRF OOD drop:      {crf_drop:.2f}%")

    wandb.log({
        **crf_id_metrics,
        **crf_ood_metrics,
        "crf/ood_drop_pct": crf_drop,
    })


    # ══════════════════════════════════════════════════════════════════════
    # MODEL 2 — XGBoost
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Building XGBoost features...")
    print(f"{'='*55}")

    # Build class_map from training data ONLY
    # This maps category string -> consecutive int 0..N-1
    # All splits reuse this map — unknown OOD categories map to 0 safely
    print("  Building class map from training data...")
    _, y_train_strings = build_xgb_dataset(train_seqs, class_map=None)
    unique_cats = sorted(set(y_train_strings))
    class_map   = {cat: idx for idx, cat in enumerate(unique_cats)}
    n_classes   = len(class_map)
    print(f"  Class map built: {n_classes} classes from training data")
    wandb.log({"xgb/n_classes_fitted": n_classes})

    # Build all splits using the same class_map
    X_train_xgb, y_train_xgb = build_xgb_dataset(train_seqs,  class_map)
    X_idval_xgb, y_idval_xgb = build_xgb_dataset(id_val_seqs, class_map)
    X_ood_xgb,   y_ood_xgb   = build_xgb_dataset(ood_seqs,    class_map)

    print(f"  Train rows: {len(X_train_xgb):,} | Features: {X_train_xgb.shape[1]}")
    wandb.log({
        "xgb/train_rows":    len(X_train_xgb),
        "xgb/n_features":    X_train_xgb.shape[1],
        "xgb/n_classes":     len(CATEGORY_LIST),
    })

    print(f"\n  Training XGBoost (trees={args.xgb_trees}, depth={args.xgb_depth})...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=args.xgb_trees,
        max_depth=args.xgb_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        verbosity=1,
    )

    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_idval_xgb, y_idval_xgb)],
        verbose=50,
    )
    print("  XGBoost trained")

    # Evaluate XGBoost
    print("\n  Evaluating XGBoost...")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(CAT_INDEX.keys()))

    xgb_id_metrics  = evaluate_xgb(xgb_model, id_val_seqs, class_map, "xgb_id_val")
    xgb_ood_metrics = evaluate_xgb(xgb_model, ood_seqs,    class_map, "xgb_ood_test")

    xgb_id_top1  = xgb_id_metrics["xgb_id_val/top1_accuracy"]   * 100
    xgb_ood_top1 = xgb_ood_metrics["xgb_ood_test/top1_accuracy"] * 100
    xgb_drop     = xgb_id_top1 - xgb_ood_top1

    print(f"  XGBoost ID Val  Top-1: {xgb_id_top1:.2f}%")
    print(f"  XGBoost ID Val  Top-3: {xgb_id_metrics['xgb_id_val/top3_accuracy']*100:.2f}%")
    print(f"  XGBoost ID Val  MRR:   {xgb_id_metrics['xgb_id_val/mrr']*100:.2f}%")
    print(f"  XGBoost OOD     Top-1: {xgb_ood_top1:.2f}%")
    print(f"  XGBoost OOD drop:      {xgb_drop:.2f}%")

    wandb.log({
        **xgb_id_metrics,
        **xgb_ood_metrics,
        "xgb/ood_drop_pct": xgb_drop,
    })

    # ── Feature importance (XGBoost only) ─────────────────────────────────────
    importances = xgb_model.feature_importances_
    top10_idx   = np.argsort(importances)[-10:][::-1]
    fi_table    = wandb.Table(
        columns=["feature_index", "importance"],
        data=[[int(i), float(importances[i])] for i in top10_idx]
    )
    wandb.log({"xgb/top10_feature_importance": fi_table})


    # ══════════════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'CRF':>10} {'XGBoost':>10} {'Winner':>10}")
    print(f"{'─'*60}")

    def compare_row(name, crf_val, xgb_val, higher_is_better=True):
        winner = "CRF" if (crf_val > xgb_val) == higher_is_better else "XGBoost"
        if abs(crf_val - xgb_val) < 0.5:
            winner = "TIE"
        print(f"{name:<35} {crf_val:>9.2f}% {xgb_val:>9.2f}% {winner:>10}")
        return winner

    compare_row("ID Val Top-1 Accuracy",  crf_id_top1,  xgb_id_top1)
    compare_row("OOD Top-1 Accuracy",     crf_ood_top1, xgb_ood_top1)
    compare_row("OOD Drop (lower=better)",crf_drop,     xgb_drop, higher_is_better=False)
    print(f"{'─'*60}")
    print(f"  N-gram (n=10) baseline:   78.4% Top-1  (your previous best)")
    print(f"{'='*60}")

    # Log comparison table to WandB
    log_comparison_table(
        {"crf_id_val/top1_accuracy": crf_id_top1/100},
        {**{k.replace("xgb_", ""): v for k, v in xgb_id_metrics.items()}},
        "ID_Val"
    )
    log_comparison_table(
        {"crf_ood_test/top1_accuracy": crf_ood_top1/100},
        {**{k.replace("xgb_", ""): v for k, v in xgb_ood_metrics.items()}},
        "OOD_Test"
    )

    # Final summary metrics
    wandb.log({
        "summary/crf_id_top1":    crf_id_top1,
        "summary/crf_ood_top1":   crf_ood_top1,
        "summary/crf_ood_drop":   crf_drop,
        "summary/xgb_id_top1":    xgb_id_top1,
        "summary/xgb_ood_top1":   xgb_ood_top1,
        "summary/xgb_ood_drop":   xgb_drop,
        "summary/ngram_baseline": 78.4,
    })

    wandb.finish()
    print(f"\nDone. View results at: {run.url}")


if __name__ == "__main__":
    main()
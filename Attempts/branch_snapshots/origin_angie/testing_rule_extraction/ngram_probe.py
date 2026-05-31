"""
ngram_probe.py
===============
Branch: angie
Folder: rule_extraction/

Trains an n-gram model (n=10) on raw step names (no categorization)
then probes it with valid vs rule-violated sequences to measure
which of the 10 process rules the model has learned.

For each rule:
  - Feed a VALID sequence → record perplexity
  - Feed an INVALID sequence (rule violated) → record perplexity
  - Compute ratio: invalid_ppl / valid_ppl
  - Ratio >> 1 means the model learned the rule
  - Ratio ≈ 1 means the model did NOT learn the rule

Results saved to: ./probe_results/ngram_probe_results.json
Ready for: python extract_rules_llm.py

Usage:
    python ngram_probe.py --train_dir ../tracks/industrial-infineon/training_data
"""

import csv
import json
import math
import os
import random
import argparse
from collections import defaultdict, Counter


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_training_sequences(train_dir: str) -> list[list[str]]:
    sequences = []
    current_id = None
    current_seq = []
    files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} CSV files")
    for fname in sorted(files):
        fpath = os.path.join(train_dir, fname)
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.strip().upper() for c in (reader.fieldnames or [])]
            if "SEQUENCE_ID" not in cols or "STEP" not in cols:
                continue
            for row in reader:
                seq_id = (row.get("SEQUENCE_ID") or row.get("sequence_id", "")).strip()
                step   = (row.get("STEP")        or row.get("step", "")).strip()
                if not step or not seq_id:
                    continue
                if seq_id != current_id:
                    if current_seq:
                        sequences.append(current_seq)
                    current_id = seq_id
                    current_seq = [step]
                else:
                    current_seq.append(step)
    if current_seq:
        sequences.append(current_seq)
    print(f"Loaded {len(sequences)} sequences")
    return sequences


def train_val_split(sequences, val_frac=0.1, seed=42):
    random.seed(seed)
    shuffled = sequences[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


# ── N-gram Model ───────────────────────────────────────────────────────────────

class NGramModel:
    def __init__(self, n: int = 10, smoothing: float = 1e-6):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.vocab = set()
        self.unigram = Counter()

    def train(self, sequences: list[list[str]]):
        for seq in sequences:
            padded = ["<START>"] * (self.n - 1) + seq + ["<END>"]
            for i in range(len(padded) - self.n + 1):
                context   = tuple(padded[i: i + self.n - 1])
                next_step = padded[i + self.n - 1]
                self.counts[context][next_step] += 1
                self.vocab.add(next_step)
            for step in seq:
                self.unigram[step] += 1
        self.vocab.discard("<START>")
        print(f"N-gram trained | n={self.n} | vocab={len(self.vocab)} | "
              f"contexts={len(self.counts)}")

    def sequence_log_prob(self, sequence: list[str]) -> float:
        """
        Compute normalized log-probability of a sequence.
        Lower = model is more surprised = more anomalous.
        """
        padded     = ["<START>"] * (self.n - 1) + sequence + ["<END>"]
        log_prob   = 0.0
        vocab_size = len(self.vocab) + 1

        for i in range(len(padded) - self.n + 1):
            context   = tuple(padded[i: i + self.n - 1])
            next_step = padded[i + self.n - 1]

            # Backoff: try shorter contexts if exact not found
            counter = Counter()
            for length in range(self.n - 1, 0, -1):
                ctx = tuple(padded[i + (self.n - 1 - length): i + self.n - 1])
                counter = self.counts.get(ctx, Counter())
                if counter:
                    break

            # Final fallback: unigram
            if not counter:
                counter = self.unigram

            total = sum(counter.values())
            count = counter.get(next_step, 0)
            prob  = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            log_prob += math.log(prob)

        return log_prob / max(len(sequence), 1)

    def perplexity(self, sequence: list[str]) -> float:
        log_prob = self.sequence_log_prob(sequence)
        return math.exp(-log_prob)

    def predict_next(self, context_steps: list[str], top_k: int = 5) -> list[str]:
        padded = ["<START>"] * (self.n - 1) + context_steps
        for length in range(self.n - 1, 0, -1):
            context = tuple(padded[-length:])
            counter = self.counts.get(context, Counter())
            if counter:
                preds = [s for s, _ in counter.most_common(top_k)]
                while len(preds) < top_k:
                    preds.append("<UNK>")
                return preds[:top_k]
        preds = [s for s, _ in self.unigram.most_common(top_k)]
        while len(preds) < top_k:
            preds.append("<UNK>")
        return preds[:top_k]


# ── Rule Probes ────────────────────────────────────────────────────────────────
# Each probe tests one rule using a minimal valid vs invalid sequence pair.
# These use real step names from the training data (no categories).

RULE_PROBES = [
    {
        "rule": "RULE_DEP_NO_CLEAN",
        "description": "Deposition must always be preceded by a cleaning step",
        "valid_seq": [
            "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN INSPECTION",
            "MEASURE INITIAL THICKNESS", "HF DIP", "DRY WAFER",
            "THERMAL OXIDATION", "MEASURE OXIDE THICKNESS",
        ],
        "invalid_seq": [
            "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN INSPECTION",
            "MEASURE INITIAL THICKNESS", "MEASURE SURFACE PLANARITY",
            "THERMAL OXIDATION", "MEASURE OXIDE THICKNESS",
        ],
        "violation": "THERMAL OXIDATION (deposition) without prior clean step",
    },
    {
        "rule": "RULE_METAL_ETCH_NO_LITHO",
        "description": "Lithography must precede metal etching",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT METAL 1", "COAT PHOTORESIST",
            "SOFT BAKE", "EXPOSE LITHO LEVEL 3", "DEVELOP PHOTORESIST",
            "METAL ETCH", "STRIP RESIST",
        ],
        "invalid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT METAL 1", "COAT PHOTORESIST",
            "SOFT BAKE", "METAL ETCH", "STRIP RESIST",
        ],
        "violation": "METAL ETCH without EXPOSE + DEVELOP preceding it",
    },
    {
        "rule": "RULE_ETCH_NO_MASK",
        "description": "A developed mask must exist before any etch",
        "valid_seq": [
            "COAT PHOTORESIST", "SOFT BAKE", "EXPOSE LITHO LEVEL 1",
            "DEVELOP PHOTORESIST", "OXIDE ETCH", "STRIP RESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "SOFT BAKE", "EXPOSE LITHO LEVEL 1",
            "OXIDE ETCH", "STRIP RESIST",
        ],
        "violation": "OXIDE ETCH without DEVELOP PHOTORESIST first",
    },
    {
        "rule": "RULE_LITHO_LEVEL_SKIP",
        "description": "Lithography levels must increment sequentially",
        "valid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "STRIP RESIST", "CLEAN AFTER ETCH",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 2", "DEVELOP PHOTORESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "STRIP RESIST", "CLEAN AFTER ETCH",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 3", "DEVELOP PHOTORESIST",
        ],
        "violation": "Jumped from LITHO LEVEL 1 directly to LITHO LEVEL 3",
    },
    {
        "rule": "RULE_IMPLANT_NO_MASK",
        "description": "An implant window must be opened before ion implantation",
        "valid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "IMPLANT P BODY", "STRIP RESIST", "ANNEAL",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "IMPLANT P BODY", "STRIP RESIST", "ANNEAL",
        ],
        "violation": "IMPLANT P BODY without prior OXIDE ETCH to open window",
    },
    {
        "rule": "RULE_CMP_NO_DEP",
        "description": "CMP planarization requires a prior deposition step",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT INTERLAYER DIELECTRIC",
            "MEASURE ILD THICKNESS", "CMP DIELECTRIC", "MEASURE POST CMP",
        ],
        "invalid_seq": [
            "MEASURE SURFACE PLANARITY", "MEASURE INITIAL THICKNESS",
            "CMP DIELECTRIC", "MEASURE POST CMP",
        ],
        "violation": "CMP DIELECTRIC with no prior deposition step",
    },
    {
        "rule": "RULE_PAD_OPEN_BEFORE_DEP",
        "description": "Bond pad window only after passivation is deposited and cured",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 4", "DEVELOP PHOTORESIST",
            "OPEN PAD WINDOW", "STRIP RESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 4", "DEVELOP PHOTORESIST",
            "OPEN PAD WINDOW", "STRIP RESIST",
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
        ],
        "violation": "OPEN PAD WINDOW before DEPOSIT PASSIVATION and CURE PASSIVATION",
    },
    {
        "rule": "RULE_TEST_BEFORE_PASSIVATION",
        "description": "Electrical tests must come after passivation is cured",
        "valid_seq": [
            "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "OPEN PAD WINDOW", "PARAMETRIC TEST", "WAFER SORT TEST",
        ],
        "invalid_seq": [
            "DEPOSIT PASSIVATION", "PARAMETRIC TEST",
            "CURE PASSIVATION", "OPEN PAD WINDOW", "WAFER SORT TEST",
        ],
        "violation": "PARAMETRIC TEST before CURE PASSIVATION",
    },
    {
        "rule": "RULE_SHIP_BEFORE_TEST",
        "description": "Wafer sort test must complete before shipping",
        "valid_seq": [
            "CURE PASSIVATION", "OPEN PAD WINDOW",
            "WAFER SORT TEST", "FINAL INSPECTION", "SHIP LOT",
        ],
        "invalid_seq": [
            "CURE PASSIVATION", "OPEN PAD WINDOW",
            "FINAL INSPECTION", "SHIP LOT", "WAFER SORT TEST",
        ],
        "violation": "SHIP LOT appears before WAFER SORT TEST",
    },
    {
        "rule": "RULE_BACKSIDE_BEFORE_PASSIVATION",
        "description": "Backside metal only after frontside passivation is cured",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "BACKSIDE GRIND", "BACKSIDE CLEAN", "DEPOSIT BACKSIDE METAL",
        ],
        "invalid_seq": [
            "BACKSIDE GRIND", "BACKSIDE CLEAN", "DEPOSIT BACKSIDE METAL",
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
        ],
        "violation": "DEPOSIT BACKSIDE METAL before DEPOSIT PASSIVATION",
    },
]


# ── Run Probes ─────────────────────────────────────────────────────────────────

def run_probes(model: NGramModel) -> list[dict]:
    results = []

    print(f"\n{'─'*80}")
    print(f"{'Rule':<38} {'Valid PPL':>10} {'Invalid PPL':>12} "
          f"{'Ratio':>8} {'Learned?':>12}")
    print(f"{'─'*80}")

    for probe in RULE_PROBES:
        valid_ppl   = model.perplexity(probe["valid_seq"])
        invalid_ppl = model.perplexity(probe["invalid_seq"])
        ratio       = invalid_ppl / max(valid_ppl, 1e-6)

        if ratio >= 2.0:
            learned = "STRONG"
        elif ratio >= 1.3:
            learned = "PARTIAL"
        elif ratio >= 1.05:
            learned = "WEAK"
        else:
            learned = "NOT LEARNED"

        print(f"{probe['rule']:<38} {valid_ppl:>10.2f} {invalid_ppl:>12.2f} "
              f"{ratio:>8.2f}x  {learned:>12}")

        results.append({
            "rule":         probe["rule"],
            "description":  probe["description"],
            "violation":    probe["violation"],
            "valid_seq":    probe["valid_seq"],
            "invalid_seq":  probe["invalid_seq"],
            "valid_ppl":    round(valid_ppl,   4),
            "invalid_ppl":  round(invalid_ppl, 4),
            "valid_log_prob":   round(model.sequence_log_prob(probe["valid_seq"]), 4),
            "invalid_log_prob": round(model.sequence_log_prob(probe["invalid_seq"]), 4),
            "ppl_ratio":    round(ratio, 4),
            "learned":      learned,
        })

    return results


def print_summary(results: list[dict], n: int, n_train: int):
    strong  = sum(1 for r in results if r["learned"] == "STRONG")
    partial = sum(1 for r in results if r["learned"] == "PARTIAL")
    weak    = sum(1 for r in results if r["learned"] == "WEAK")
    none_   = sum(1 for r in results if r["learned"] == "NOT LEARNED")
    total   = len(results)
    score   = (strong + 0.5 * partial + 0.2 * weak) / total * 100

    print(f"\n{'='*55}")
    print(f"  N-GRAM (n={n}) RULE LEARNING SUMMARY")
    print(f"  Training sequences: {n_train}")
    print(f"{'='*55}")
    print(f"  STRONG      : {strong}/{total}")
    print(f"  PARTIAL     : {partial}/{total}")
    print(f"  WEAK        : {weak}/{total}")
    print(f"  NOT LEARNED : {none_}/{total}")
    print(f"  Overall score: {score:.1f}%")
    print(f"{'='*55}")

    print("\nRules NOT learned or weak:")
    for r in results:
        if r["learned"] in ("NOT LEARNED", "WEAK"):
            print(f"  ✗ {r['rule']}")
            print(f"    reason: {r['violation']}")
            print(f"    ratio: {r['ppl_ratio']:.2f}x (need ≥1.3 for partial, ≥2.0 for strong)")

    print("\nRules STRONGLY learned:")
    for r in results:
        if r["learned"] == "STRONG":
            print(f"  ✓ {r['rule']} (ratio={r['ppl_ratio']:.2f}x)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",
                        default="../training_data")
    parser.add_argument("--n",        type=int,   default=10)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out_dir",  default="./probe_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load and train ─────────────────────────────────────────────────────────
    sequences = load_training_sequences(args.train_dir)
    train_seqs, val_seqs = train_val_split(
        sequences, val_frac=args.val_frac, seed=args.seed
    )
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")

    model = NGramModel(n=args.n)
    model.train(train_seqs)

    # ── Run probes ─────────────────────────────────────────────────────────────
    print(f"\nRunning {len(RULE_PROBES)} rule probes on n-gram (n={args.n})...")
    results = run_probes(model)
    print_summary(results, args.n, len(train_seqs))

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "model_type":        "ngram",
        "n":                 args.n,
        "train_sequences":   len(train_seqs),
        "val_sequences":     len(val_seqs),
        "vocab_size":        len(model.vocab),
        "unique_contexts":   len(model.counts),
        "probe_results":     results,
        "summary": {
            "strong":      sum(1 for r in results if r["learned"] == "STRONG"),
            "partial":     sum(1 for r in results if r["learned"] == "PARTIAL"),
            "weak":        sum(1 for r in results if r["learned"] == "WEAK"),
            "not_learned": sum(1 for r in results if r["learned"] == "NOT LEARNED"),
            "total":       len(results),
            "score_pct":   round(
                (sum(1 for r in results if r["learned"] == "STRONG") +
                 0.5 * sum(1 for r in results if r["learned"] == "PARTIAL") +
                 0.2 * sum(1 for r in results if r["learned"] == "WEAK"))
                / len(results) * 100, 1
            ),
        }
    }

    out_path = os.path.join(args.out_dir, "ngram_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print(f"Next: python extract_rules_llm.py "
          f"--probe_results {out_path}")


if __name__ == "__main__":
    main()
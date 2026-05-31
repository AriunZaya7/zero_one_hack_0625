"""
prepare_dataset.py
===================
Reads all training CSVs, builds structured prompts, and saves:

  dataset_train.jsonl      — IGBT + IC sequences for training
  dataset_id_val.jsonl     — IGBT + IC held-out sequences for ID validation
  dataset_ood_test.jsonl   — MOSFET sequences for OOD testing (never seen in training)

Each line in the JSONL files is one sequence formatted as:
  {
    "sequence_id": "seq_001",
    "family": "IGBT",
    "prompt": "<FAMILY>IGBT</FAMILY> RECEIVE WAFER LOT | LOT IDENTIFICATION | ...",
    "steps": ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", ...],
    "n_steps": 142
  }

The prompt format is what gets fed to the model during training and inference.

Put this in test_baseline_2/ and run:
    python prepare_dataset.py
"""

import csv
import json
import os
import random
import argparse
from collections import defaultdict


# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_DIR = "../../tracks/industrial-infineon/training_data"

# Which families go where
TRAIN_FAMILIES  = {"igbt", "ic"}       # used for training
OOD_FAMILIES    = {"mosfet"}           # never seen during training

# Map filename keywords to family names
FAMILY_MAP = {
    "igbt":   "IGBT",
    "ic":     "IC",
    "mosfet": "MOSFET",
}

def detect_family(filename: str) -> str | None:
    """Detect family from filename."""
    fname = filename.lower()
    for key, name in FAMILY_MAP.items():
        if key in fname:
            return name
    return None


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_all_sequences(train_dir: str) -> dict[str, list[list[str]]]:
    """
    Load all sequences grouped by family.
    Returns: { "IGBT": [[step, step, ...], ...], "IC": [...], "MOSFET": [...] }
    Only loads files that contain SEQUENCE_ID + STEP columns (long format).
    Skips description/parameter files.
    """
    family_sequences = defaultdict(list)
    files = sorted(f for f in os.listdir(train_dir) if f.endswith(".csv"))

    for fname in files:
        family = detect_family(fname)
        if family is None:
            print(f"  [skip] {fname} — could not detect family")
            continue

        fpath = os.path.join(train_dir, fname)
        seqs = load_long_format_csv(fpath)

        if not seqs:
            print(f"  [skip] {fname} — no SEQUENCE_ID/STEP columns or empty")
            continue

        family_sequences[family].extend(seqs)
        print(f"  [load] {fname:<45} {len(seqs):>5} sequences  (family={family})")

    return dict(family_sequences)


def load_long_format_csv(path: str) -> list[list[str]]:
    """
    Load a long-format CSV (SEQUENCE_ID, STEP) into a list of sequences.
    Skips files that don't have the right columns.
    """
    sequences = []
    current_id = None
    current_seq = []

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.strip().upper() for c in (reader.fieldnames or [])]

            # Only process long-format files
            if "SEQUENCE_ID" not in cols and "STEP" not in cols:
                return []

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

    except Exception as e:
        print(f"  [error] {path}: {e}")
        return []

    return sequences


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(steps: list[str], family: str) -> str:
    """
    Build a structured prompt from a sequence.

    Format:
        <FAMILY>IGBT</FAMILY> RECEIVE WAFER LOT | LOT IDENTIFICATION | ...

    The family tag tells the model which product family this sequence belongs to.
    Steps are joined with ' | ' as separator.
    """
    steps_str = " | ".join(steps)
    return f"<FAMILY>{family}</FAMILY> {steps_str}"


def build_record(seq_id: str, steps: list[str], family: str) -> dict:
    """Build a single JSONL record."""
    return {
        "sequence_id": seq_id,
        "family": family,
        "prompt": build_prompt(steps, family),
        "steps": steps,
        "n_steps": len(steps),
    }


# ── Split Logic ────────────────────────────────────────────────────────────────

def split_sequences(
    sequences: list[list[str]],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]]]:
    """Split into train and val."""
    random.seed(seed)
    shuffled = sequences[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


# ── JSONL Writer ───────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>5} records → {path}")


def preview_record(record: dict, label: str = ""):
    """Print a preview of one record."""
    print(f"\n  {'─'*60}")
    if label:
        print(f"  {label}")
    print(f"  sequence_id : {record['sequence_id']}")
    print(f"  family      : {record['family']}")
    print(f"  n_steps     : {record['n_steps']}")
    print(f"  prompt[:120]: {record['prompt'][:120]}...")
    print(f"  {'─'*60}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=TRAIN_DIR)
    parser.add_argument("--out_dir",   default="../data",
                        help="Where to save the JSONL files")
    parser.add_argument("--val_frac",  type=float, default=0.1,
                        help="Fraction of train families to hold out for ID validation")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"\nLoading sequences from: {args.train_dir}")
    print(f"Train families : {sorted(TRAIN_FAMILIES)}")
    print(f"OOD families   : {sorted(OOD_FAMILIES)}")
    print()

    # ── Load all sequences ─────────────────────────────────────────────────────
    all_seqs = load_all_sequences(args.train_dir)

    print(f"\nLoaded families: {list(all_seqs.keys())}")
    for fam, seqs in all_seqs.items():
        lengths = [len(s) for s in seqs]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"  {fam:<8}: {len(seqs):>5} sequences  avg_len={avg_len:.0f} steps")

    # ── Build train/val splits for training families ───────────────────────────
    train_records = []
    id_val_records = []

    for family_name, seqs in all_seqs.items():
        if family_name.lower() not in TRAIN_FAMILIES:
            continue

        train_seqs, val_seqs = split_sequences(seqs, val_frac=args.val_frac, seed=args.seed)

        for i, steps in enumerate(train_seqs):
            train_records.append(build_record(
                seq_id=f"{family_name}_train_{i:05d}",
                steps=steps,
                family=family_name,
            ))

        for i, steps in enumerate(val_seqs):
            id_val_records.append(build_record(
                seq_id=f"{family_name}_val_{i:05d}",
                steps=steps,
                family=family_name,
            ))

    # ── Build OOD test set ─────────────────────────────────────────────────────
    ood_records = []
    for family_name, seqs in all_seqs.items():
        if family_name.lower() not in OOD_FAMILIES:
            continue

        for i, steps in enumerate(seqs):
            ood_records.append(build_record(
                seq_id=f"{family_name}_ood_{i:05d}",
                steps=steps,
                family=family_name,
            ))

    # ── Shuffle ────────────────────────────────────────────────────────────────
    random.shuffle(train_records)
    random.shuffle(id_val_records)
    random.shuffle(ood_records)

    # ── Write JSONL files ──────────────────────────────────────────────────────
    print(f"\nWriting dataset files to: {args.out_dir}")
    train_path   = os.path.join(args.out_dir, "dataset_train.jsonl")
    id_val_path  = os.path.join(args.out_dir, "dataset_id_val.jsonl")
    ood_path     = os.path.join(args.out_dir, "dataset_ood_test.jsonl")

    write_jsonl(train_records,  train_path)
    write_jsonl(id_val_records, id_val_path)
    write_jsonl(ood_records,    ood_path)

    # ── Preview one record from each split ─────────────────────────────────────
    if train_records:
        preview_record(train_records[0],  "Sample TRAIN record")
    if id_val_records:
        preview_record(id_val_records[0], "Sample ID VAL record")
    if ood_records:
        preview_record(ood_records[0],    "Sample OOD TEST record")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"""
{'='*60}
  DATASET READY
{'='*60}
  Train (IGBT + IC):     {len(train_records):>5} sequences → dataset_train.jsonl
  ID Val (IGBT + IC):    {len(id_val_records):>5} sequences → dataset_id_val.jsonl
  OOD Test (MOSFET):     {len(ood_records):>5} sequences → dataset_ood_test.jsonl
{'='*60}

  Prompt format example:
  <FAMILY>IGBT</FAMILY> RECEIVE WAFER LOT | LOT IDENTIFICATION | ...

  Next step:
      python train_model.py --data_dir ./data
""")

    # ── Save a vocab file (useful for training) ────────────────────────────────
    all_steps = set()
    for record in train_records + id_val_records + ood_records:
        all_steps.update(record["steps"])

    # Add special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>",
                      "<FAMILY>IGBT</FAMILY>",
                      "<FAMILY>IC</FAMILY>",
                      "<FAMILY>MOSFET</FAMILY>",
                      "<FAMILY>UNKNOWN</FAMILY>"]

    vocab = special_tokens + sorted(all_steps)
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "size": len(vocab)}, f, indent=2)

    print(f"  Vocab: {len(vocab)} tokens → {vocab_path}")
    print(f"  (includes {len(special_tokens)} special tokens + {len(all_steps)} step names)\n")


if __name__ == "__main__":
    main()
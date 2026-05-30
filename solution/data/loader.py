"""
loader.py
=========
Load long-format process-sequence CSVs into lists of step-name sequences.

The training CSVs use long format (one step per row):

    SEQUENCE_ID,STEP
    MOSFET_0001,RECEIVE WAFER LOT
    MOSFET_0001,LOT IDENTIFICATION
    ...

In this repo the per-family files live in ``training_data/`` and are named
``<FAMILY>_variants.csv`` (plus ``<FAMILY>_generated_extra.csv`` for the train
families). See PLAN.md "Path mapping for this repo".

Smoke test:
    python -m solution.data.loader --data_dir training_data
"""

import csv
import os
import random
import argparse
from collections import defaultdict


# Map a filename to a family. We only read the long-format *variants* / *extra*
# files; description / parameter / synthetic files are skipped.
FAMILY_KEYS = {"mosfet": "MOSFET", "igbt": "IGBT", "ic": "IC"}
LONG_FORMAT_SUFFIXES = ("_variants.csv", "_generated_extra.csv")


def detect_family(filename: str) -> str | None:
    fname = filename.lower()
    # Check IGBT before IC, because "ic" is a substring of nothing here but be safe.
    for key in ("mosfet", "igbt", "ic"):
        if fname.startswith(key):
            return FAMILY_KEYS[key]
    return None


def load_sequences(csv_path: str) -> list[list[str]]:
    """Load one long-format CSV into a list of sequences (each a list of steps)."""
    sequences: list[list[str]] = []
    current_id = None
    current_seq: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip().upper() for c in (reader.fieldnames or [])]
        if "SEQUENCE_ID" not in cols or "STEP" not in cols:
            return []  # not a long-format file
        for row in reader:
            seq_id = (row.get("SEQUENCE_ID") or row.get("sequence_id") or "").strip()
            step = (row.get("STEP") or row.get("step") or "").strip()
            if not seq_id or not step:
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
    return sequences


def load_all_families(data_dir: str) -> dict[str, list[list[str]]]:
    """Return {"IC": [...], "IGBT": [...], "MOSFET": [...]} from all long-format CSVs."""
    out: dict[str, list[list[str]]] = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(LONG_FORMAT_SUFFIXES):
            continue
        family = detect_family(fname)
        if family is None:
            continue
        seqs = load_sequences(os.path.join(data_dir, fname))
        if seqs:
            out[family].extend(seqs)
            print(f"  [load] {fname:<35} {len(seqs):>5} sequences  (family={family})")
    return dict(out)


def train_val_split(sequences: list, val_ratio: float = 0.1, seed: int = 42):
    """Deterministic shuffle + split. Same seed everywhere."""
    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:n_train], shuffled[n_train:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data")
    args = ap.parse_args()
    fams = load_all_families(args.data_dir)
    for fam, seqs in fams.items():
        lengths = [len(s) for s in seqs]
        avg = sum(lengths) / len(lengths) if lengths else 0
        print(f"{fam:<8}: {len(seqs):>5} seqs | avg_len={avg:.0f} | "
              f"min={min(lengths, default=0)} max={max(lengths, default=0)}")
    print("loader OK")

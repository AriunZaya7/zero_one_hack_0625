"""Data loading + splits for the Industrial AI track.

Wraps the organizer-provided read_csv_sequences() when importable, with a
fallback long-format CSV parser so nobody is blocked on import paths.

Key function: leave_one_family_out() -- our LOCAL proxy for the hidden
4th-family OOD test (Task 4). Train on two families, evaluate on the third.
"""
from __future__ import annotations

import csv
import random
import warnings
from collections import defaultdict
from pathlib import Path

FAMILIES = ["mosfet", "igbt", "ic", "kremsians"]

# adjust paths if your checkout differs
VARIANTS = {
    "mosfet": [
        "training_data/MOSFET_variants.csv",
        "training_data/MOSFET_extra.csv",         # 5 000 extra
        "training_data/MOSFET_extra2.csv",        # 4 000 extra  → ~10 K total
    ],
    "igbt": [
        "training_data/IGBT_variants.csv",
        "training_data/IGBT_generated_extra.csv", # ~2 000 extra
        "training_data/IGBT_extra2.csv",          # 7 000 extra  → ~10 K total
    ],
    "ic": [
        "training_data/IC_variants.csv",
        "training_data/IC_generated_extra.csv",   # ~2 000 extra
        "training_data/IC_extra2.csv",            # 7 000 extra  → ~10 K total
    ],
    "kremsians": [
        "training_data/KREMSIANS_variants.csv",   # 2 000 synthetic 4th-family sequences
    ],
}


def _fallback_read(path: str | Path) -> dict[str, list[str]]:
    seqs: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seqs[row["SEQUENCE_ID"]].append(row["STEP"])
    return dict(seqs)


def read_sequences(path: str | Path) -> dict[str, list[str]]:
    """Return {sequence_id: [step, step, ...]}."""
    try:
        from training_data.generate_sequences import read_csv_sequences  # type: ignore
        return read_csv_sequences(Path(path))
    except Exception:
        return _fallback_read(path)


def load_family(family: str) -> dict[str, list[str]]:
    paths = VARIANTS[family]
    if isinstance(paths, str):
        paths = [paths]
    out: dict[str, list[str]] = {}
    for file_idx, p in enumerate(paths):
        if not Path(p).exists():
            warnings.warn(f"[data] {p} not found, skipping.")
            continue
        for sid, seq in read_sequences(p).items():
            # prefix with file index so IDs are unique across files
            out[f"f{file_idx}_{sid}"] = seq
    return out


def load_all(families: list[str] | None = None) -> dict[str, list[str]]:
    """Family-prefixed ids so they never collide across files."""
    families = families or FAMILIES
    out: dict[str, list[str]] = {}
    for fam in families:
        for sid, seq in load_family(fam).items():
            out[f"{fam}:{sid}"] = seq
    return out


def train_val_split(seqs: dict[str, list[str]], val_frac: float = 0.1, seed: int = 42):
    keys = list(seqs)
    random.Random(seed).shuffle(keys)
    n_val = int(len(keys) * val_frac)
    val = {k: seqs[k] for k in keys[:n_val]}
    train = {k: seqs[k] for k in keys[n_val:]}
    return train, val


def leave_one_family_out(holdout: str, seed: int = 42):
    """Train on the other two families, test on `holdout`.

    This is our offline stand-in for the organizers' hidden 4th family.
    Same vocabulary, different block structure / cycle counts == exactly
    the kind of shift Task 4 measures.
    """
    assert holdout in FAMILIES
    train: dict[str, list[str]] = {}
    for fam in FAMILIES:
        if fam == holdout:
            continue
        for sid, seq in load_family(fam).items():
            train[f"{fam}:{sid}"] = seq
    test = {f"{holdout}:{sid}": seq for sid, seq in load_family(holdout).items()}
    return train, test

"""Data loading + splits for the Industrial AI track.

Wraps the organizer-provided read_csv_sequences() when importable, with a
fallback long-format CSV parser so nobody is blocked on import paths.

Key functions:
- leave_one_family_out() -- legacy one-family OOD proxy.
- train12_test3_split() -- SUBMISSION_2 protocol: 15 families total,
  train on 12 and report the average across 3 held-out OOD families.
"""
from __future__ import annotations

import csv
import random
import warnings
from collections import defaultdict
from pathlib import Path

OFFICIAL_FAMILIES = ["mosfet", "igbt", "ic"]
SYNTHETIC_TRAIN_FAMILIES = [
    "scfam01", "scfam02", "scfam03", "scfam04", "scfam05",
    "scfam06", "scfam07", "scfam08", "scfam09",
]
SYNTHETIC_OOD_FAMILIES = ["scfam10", "scfam11", "scfam12"]

TRAIN_12_FAMILIES = OFFICIAL_FAMILIES + SYNTHETIC_TRAIN_FAMILIES
OOD_3_FAMILIES = SYNTHETIC_OOD_FAMILIES
FAMILIES = TRAIN_12_FAMILIES + OOD_3_FAMILIES

FAMILY_GROUPS = {
    "official": OFFICIAL_FAMILIES,
    "synthetic_train": SYNTHETIC_TRAIN_FAMILIES,
    "synthetic_ood": SYNTHETIC_OOD_FAMILIES,
    "train12": TRAIN_12_FAMILIES,
    "ood3": OOD_3_FAMILIES,
    "all15": FAMILIES,
}

# adjust paths if your checkout differs
VARIANTS = {
    "mosfet": [
        "training_data/MOSFET_variants.csv",
    ],
    "igbt": [
        "training_data/IGBT_variants.csv",
        "training_data/IGBT_generated_extra.csv", # ~2 000 extra
    ],
    "ic": [
        "training_data/IC_variants.csv",
        "training_data/IC_generated_extra.csv",   # ~2 000 extra
    ],
    "scfam01": ["training_data/SCFAM01_variants.csv"],
    "scfam02": ["training_data/SCFAM02_variants.csv"],
    "scfam03": ["training_data/SCFAM03_variants.csv"],
    "scfam04": ["training_data/SCFAM04_variants.csv"],
    "scfam05": ["training_data/SCFAM05_variants.csv"],
    "scfam06": ["training_data/SCFAM06_variants.csv"],
    "scfam07": ["training_data/SCFAM07_variants.csv"],
    "scfam08": ["training_data/SCFAM08_variants.csv"],
    "scfam09": ["training_data/SCFAM09_variants.csv"],
    "scfam10": ["training_data/SCFAM10_variants.csv"],
    "scfam11": ["training_data/SCFAM11_variants.csv"],
    "scfam12": ["training_data/SCFAM12_variants.csv"],
}


def expand_family_groups(families: list[str] | None, default_group: str = "train12") -> list[str]:
    """Expand family names and group aliases while preserving order."""
    if not families:
        return list(FAMILY_GROUPS[default_group])
    expanded: list[str] = []
    for item in families:
        key = item.lower()
        values = FAMILY_GROUPS.get(key, [key])
        for family in values:
            if family not in VARIANTS:
                raise ValueError(f"Unknown family or family group: {item!r}")
            if family not in expanded:
                expanded.append(family)
    return expanded


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
    family = family.lower()
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
    families = expand_family_groups(families, default_group="train12")
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


def _load_limited_family(family: str, limit: int | None, seed: int) -> dict[str, list[str]]:
    seqs = load_family(family)
    if limit is None or len(seqs) <= limit:
        return seqs
    keys = list(seqs)
    random.Random(seed + sum(ord(c) for c in family)).shuffle(keys)
    keys = sorted(keys[:limit])
    return {key: seqs[key] for key in keys}


def leave_one_family_out(holdout: str, seed: int = 42):
    """Legacy one-family holdout split.

    SUBMISSION_2 uses train12_test3_split() instead. This helper is kept for
    older leave-one-family-out experiments.
    """
    holdout = holdout.lower()
    assert holdout in FAMILIES
    train: dict[str, list[str]] = {}
    for fam in FAMILIES:
        if fam == holdout:
            continue
        for sid, seq in load_family(fam).items():
            train[f"{fam}:{sid}"] = seq
    test = {f"{holdout}:{sid}": seq for sid, seq in load_family(holdout).items()}
    return train, test


def train12_test3_split(seed: int = 42, per_family_limit: int | None = 200):
    """Return the fixed SUBMISSION_2 train/test split.

    Training families are the 3 official families plus 9 synthetic families.
    Test families are 3 held-out synthetic OOD families. The OOD report should
    average over those 3 family-level scores.

    By default, each family is capped at 200 sequences so the 12-family training
    mix is family-balanced and the 3-family OOD test is exactly 600 sequences.
    """
    train: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    for family in TRAIN_12_FAMILIES:
        for sid, seq in _load_limited_family(family, per_family_limit, seed).items():
            train[f"{family}:{sid}"] = seq
    for family in OOD_3_FAMILIES:
        for sid, seq in _load_limited_family(family, per_family_limit, seed).items():
            test[f"{family}:{sid}"] = seq
    return train, test


def train12_test3_by_family(seed: int = 42, per_family_limit: int | None = 200):
    """Return train sequences and a mapping of each held-out OOD family to test sequences."""
    train, _ = train12_test3_split(seed=seed, per_family_limit=per_family_limit)
    tests = {
        family: {
            f"{family}:{sid}": seq
            for sid, seq in _load_limited_family(family, per_family_limit, seed).items()
        }
        for family in OOD_3_FAMILIES
    }
    return train, tests

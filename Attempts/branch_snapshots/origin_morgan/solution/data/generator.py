"""
Helpers for valid sequence generation and local anomaly-set construction.

The organizer eval files are not present in this repo, so build_anomaly_set()
creates labeled development data by injecting each of the 10 official forbidden
patterns into otherwise-valid sequences. Labels are assigned by the official
validator through solution.eval.rules.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
from collections.abc import Callable

from solution.eval import rules as R


def generate_extra(
    family: str,
    count: int,
    output: str,
    seed: int = 123,
    script: str = "training_data/generate_sequences.py",
) -> bool:
    """Call the official generator CLI for extra valid data."""
    cmd = [
        "python",
        script,
        "--family",
        family,
        "--count",
        str(count),
        "--output",
        output,
        "--seed",
        str(seed),
    ]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def _remove_positions(seq: list[str], positions: set[int]) -> list[str]:
    return [step for i, step in enumerate(seq) if i not in positions]


def _move_step(seq: list[str], src: int, dst: int) -> list[str]:
    out = list(seq)
    step = out.pop(src)
    if src < dst:
        dst -= 1
    out.insert(max(0, min(dst, len(out))), step)
    return out


def _first_index(seq: list[str], pred: Callable[[str], bool]) -> int | None:
    return next((i for i, step in enumerate(seq) if pred(step)), None)


def _last_indices_before(seq: list[str], i: int, window: int, pred: Callable[[str], bool]) -> set[int]:
    return {j for j in range(max(0, i - window), i) if pred(seq[j])}


def _inject_dep_no_clean(seq: list[str], rng: random.Random) -> list[str] | None:
    candidates = [i for i, step in enumerate(seq) if R.is_deposit(step)]
    rng.shuffle(candidates)
    for i in candidates:
        pos = _last_indices_before(seq, i, 12, R.is_clean)
        if pos:
            return _remove_positions(seq, pos)
    return None


def _inject_metal_etch_no_litho(seq: list[str], rng: random.Random) -> list[str] | None:
    candidates = [i for i, step in enumerate(seq) if R.is_metal_etch(step)]
    rng.shuffle(candidates)
    for i in candidates:
        pos = {
            j
            for j in range(max(0, i - 15), i)
            if R.is_develop(seq[j]) or seq[j].strip().upper().startswith("EXPOSE LITHO LEVEL")
        }
        if pos:
            return _remove_positions(seq, pos)
    return None


def _inject_etch_no_mask(seq: list[str], rng: random.Random) -> list[str] | None:
    candidates = [i for i, step in enumerate(seq) if R.is_etch(step)]
    rng.shuffle(candidates)
    for i in candidates:
        pos = _last_indices_before(seq, i, 12, R.is_develop)
        if pos:
            return _remove_positions(seq, pos)
    return None


def _inject_litho_level_skip(seq: list[str], rng: random.Random) -> list[str] | None:
    aligns = [(i, R.align_level(step)) for i, step in enumerate(seq)]
    aligns = [(i, lvl) for i, lvl in aligns if lvl is not None and lvl > 1]
    rng.shuffle(aligns)
    for _, lvl in aligns:
        prev = next((i for i, step in enumerate(seq) if R.align_level(step) == lvl - 1), None)
        if prev is not None:
            return seq[:prev] + seq[prev + 1 :]
    return None


def _inject_implant_no_mask(seq: list[str], rng: random.Random) -> list[str] | None:
    candidates = [i for i, step in enumerate(seq) if R.is_implant(step)]
    rng.shuffle(candidates)
    for i in candidates:
        pos = _last_indices_before(seq, i, 15, R.is_implant_opener)
        if pos:
            return _remove_positions(seq, pos)
    return None


def _inject_cmp_no_dep(seq: list[str], rng: random.Random) -> list[str] | None:
    candidates = [i for i, step in enumerate(seq) if R.is_cmp(step)]
    rng.shuffle(candidates)
    for i in candidates:
        pos = _last_indices_before(seq, i, 6, R.is_fill)
        if pos:
            return _remove_positions(seq, pos)
    return None


def _inject_pad_open_before_dep(seq: list[str], rng: random.Random) -> list[str] | None:
    pad_idx = _first_index(seq, R.is_pad_window)
    dep_idx = _first_index(seq, R.is_passivation)
    if pad_idx is not None and dep_idx is not None and pad_idx > dep_idx:
        return _move_step(seq, pad_idx, dep_idx)
    return None


def _inject_test_before_passivation(seq: list[str], rng: random.Random) -> list[str] | None:
    test_idx = _first_index(seq, R.is_electrical_test)
    cure_idx = _first_index(seq, R.is_cure)
    if test_idx is not None and cure_idx is not None and test_idx > cure_idx:
        return _move_step(seq, test_idx, cure_idx)
    return None


def _inject_ship_before_test(seq: list[str], rng: random.Random) -> list[str] | None:
    ship_idx = _first_index(seq, R.is_ship)
    sort_idx = _first_index(seq, R.is_wafer_sort)
    if ship_idx is not None and sort_idx is not None and ship_idx > sort_idx:
        return _move_step(seq, ship_idx, sort_idx)
    return None


def _inject_backside_before_passivation(seq: list[str], rng: random.Random) -> list[str] | None:
    backside_idx = _first_index(seq, R.is_backside_metal)
    cure_idx = _first_index(seq, R.is_cure)
    if backside_idx is not None and cure_idx is not None and backside_idx > cure_idx:
        return _move_step(seq, backside_idx, cure_idx)
    return None


INJECTORS: dict[str, Callable[[list[str], random.Random], list[str] | None]] = {
    "RULE_DEP_NO_CLEAN": _inject_dep_no_clean,
    "RULE_METAL_ETCH_NO_LITHO": _inject_metal_etch_no_litho,
    "RULE_ETCH_NO_MASK": _inject_etch_no_mask,
    "RULE_LITHO_LEVEL_SKIP": _inject_litho_level_skip,
    "RULE_IMPLANT_NO_MASK": _inject_implant_no_mask,
    "RULE_CMP_NO_DEP": _inject_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP": _inject_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": _inject_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": _inject_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": _inject_backside_before_passivation,
}


def _try_inject(seq: list[str], rng: random.Random, rule_order: list[str]) -> tuple[list[str], str] | None:
    for rule_id in rule_order:
        mutated = INJECTORS[rule_id](list(seq), rng)
        if mutated is None:
            continue
        detected = R.attribute_anomaly(mutated)
        if detected is not None:
            return mutated, detected
    return None


def build_anomaly_set(
    valid_sequences: list[list[str]],
    seed: int = 42,
    invalid_frac: float = 0.5,
) -> list[dict]:
    """Return records shaped as {steps, is_valid, rule}.

    Injectors are cycled by rule id for coverage, then shuffled per sequence as
    fallback. The final rule label always comes from the official validator.
    """
    rng = random.Random(seed)
    out = []
    rule_cycle = list(R.RULE_IDS)

    for idx, seq in enumerate(valid_sequences):
        if not R.is_valid(seq):
            continue

        if rng.random() >= invalid_frac:
            out.append({"steps": list(seq), "is_valid": 1, "rule": None})
            continue

        preferred = rule_cycle[idx % len(rule_cycle)]
        rule_order = [preferred] + [rid for rid in rule_cycle if rid != preferred]
        if rng.random() < 0.25:
            rng.shuffle(rule_order)

        injected = _try_inject(list(seq), rng, rule_order)
        if injected is None:
            out.append({"steps": list(seq), "is_valid": 1, "rule": None})
            continue

        mutated, rule = injected
        out.append({"steps": mutated, "is_valid": 0, "rule": rule})

    rng.shuffle(out)
    return out


def write_anomaly_csv(records, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "SEQUENCE", "IS_VALID", "RULE"])
        for i, record in enumerate(records):
            writer.writerow(
                [
                    f"ex_{i:05d}",
                    "|".join(record["steps"]),
                    record["is_valid"],
                    record["rule"] or "",
                ]
            )
    print(f"wrote {len(records)} rows -> {path}")


if __name__ == "__main__":
    from collections import Counter

    from solution.data.loader import load_all_families

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="solution/results/anomaly_dev.csv")
    args = parser.parse_args()

    families = load_all_families(args.data_dir)
    seqs = [seq for family_seqs in families.values() for seq in family_seqs][: args.n]
    records = build_anomaly_set(seqs, seed=42)
    counts = Counter(record["rule"] for record in records if record["rule"])
    print(f"built {len(records)} records, {sum(counts.values())} injected-invalid")
    print("rules:", dict(sorted(counts.items())))
    write_anomaly_csv(records, args.out)
    print("generator OK")

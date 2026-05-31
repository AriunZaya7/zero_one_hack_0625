"""Generate the SUBMISSION_2 15-family OOD benchmark.

The active protocol is:

- 3 organizer families: mosfet, igbt, ic
- 9 synthetic training families: scfam01..scfam09
- 3 synthetic held-out OOD families: scfam10..scfam12

That gives exactly 15 families total. Model training uses the first 12 families
and the OOD report averages over the last 3 families.

The synthetic families reuse only the existing process-step vocabulary and are
validated with the organizer validator. They are intentionally generated from
multiple profile families rather than one single-template family so the OOD
estimate is less brittle.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from synthetic_blocks import (
    _backside_block,
    _choice,
    _cycle1_pwell,
    _cycle2_pbody,
    _cycle3_poly_dual_implant,
    _cycle4_field_via,
    _cycle5_metal,
    _final_inspection,
    _first_oxidation,
    _initial_measurements,
    _template_prep,
    _litho,
    _maybe,
    _passivation_block,
    _pre_process_clean,
    _prefix,
    _test_suite,
)
from training_data.generate_sequences import validate_sequence


@dataclass(frozen=True)
class SyntheticFamily:
    name: str
    split: str
    litho_levels: int
    second_field_via: bool
    spacer_bias: float
    seed_offset: int


SYNTHETIC_FAMILIES: tuple[SyntheticFamily, ...] = (
    SyntheticFamily("scfam01", "train", 4, False, 0.25, 101),
    SyntheticFamily("scfam02", "train", 4, True, 0.35, 102),
    SyntheticFamily("scfam03", "train", 5, False, 0.45, 103),
    SyntheticFamily("scfam04", "train", 5, True, 0.55, 104),
    SyntheticFamily("scfam05", "train", 5, False, 0.65, 105),
    SyntheticFamily("scfam06", "train", 6, False, 0.40, 106),
    SyntheticFamily("scfam07", "train", 6, True, 0.50, 107),
    SyntheticFamily("scfam08", "train", 6, False, 0.70, 108),
    SyntheticFamily("scfam09", "train", 5, True, 0.80, 109),
    SyntheticFamily("scfam10", "ood", 4, True, 0.75, 201),
    SyntheticFamily("scfam11", "ood", 5, False, 0.20, 202),
    SyntheticFamily("scfam12", "ood", 6, True, 0.90, 203),
)


def _cycle6_top_metal(rng: random.Random) -> list[str]:
    """Additional top-metal layer for 6-litho synthetic families."""
    steps = [
        _choice(rng, "WET CLEAN RCA1", "RCA CLEAN 1"),
        _choice(rng, "WET CLEAN RCA2", "RCA CLEAN 2"),
        "HF DIP",
        _choice(rng, "DEPOSIT TOP METAL", "DEPOSIT METAL 1"),
    ]
    steps += _maybe(rng, 0.55, "MEASURE METAL THICKNESS")
    steps += _litho(rng, 6)
    steps[-1] = "METAL PATTERN INSPECTION"
    steps.append(_choice(rng, "METAL ETCH", "METAL ETCH DRY"))
    steps += [
        _choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
        _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER METAL ETCH"),
    ]
    steps += _maybe(rng, 0.55, "MEASURE LINE WIDTH")
    return steps


def _maybe_extra_spacer(rng: random.Random, profile: SyntheticFamily) -> list[str]:
    if rng.random() >= profile.spacer_bias:
        return []
    return [
        _choice(rng, "WET CLEAN RCA1", "RCA CLEAN 1"),
        _choice(rng, "WET CLEAN RCA2", "RCA CLEAN 2"),
        "HF DIP",
        "DEPOSIT SPACER DIELECTRIC",
        "ANISOTROPIC ETCH SPACER",
    ]


def generate_sequence(profile: SyntheticFamily, rng: random.Random) -> list[str]:
    seq: list[str] = []
    seq += _prefix(rng)
    seq += _initial_measurements(rng)
    seq += _pre_process_clean(rng)
    seq += _template_prep(rng)
    seq += _first_oxidation(rng)
    seq += _cycle1_pwell(rng)
    seq += _cycle2_pbody(rng)
    seq += _cycle3_poly_dual_implant(rng)
    seq += _maybe_extra_spacer(rng, profile)
    seq += _cycle4_field_via(rng)
    if profile.second_field_via:
        seq += _cycle4_field_via(rng)
    if profile.litho_levels >= 5:
        seq += _cycle5_metal(rng)
    if profile.litho_levels >= 6:
        seq += _cycle6_top_metal(rng)
    seq += _passivation_block(rng)
    seq += _backside_block(rng)
    seq += _final_inspection(rng)
    seq += _test_suite(rng)
    seq += [_choice(rng, "LOT RELEASE", "FINAL LOT RELEASE"), "SHIP LOT"]
    return seq


def generate_family(profile: SyntheticFamily, count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed + profile.seed_offset)
    sequences: list[list[str]] = []
    attempts = 0
    invalid = 0
    while len(sequences) < count:
        attempts += 1
        seq = generate_sequence(profile, rng)
        errors = validate_sequence(seq)
        if errors:
            invalid += 1
            if invalid <= 3:
                print(f"[warn] {profile.name} invalid attempt {attempts}: {errors[0]}")
            continue
        sequences.append(seq)
    print(
        f"{profile.name}: generated {len(sequences)} valid sequences "
        f"({attempts} attempts, {invalid} invalid discarded)"
    )
    return sequences


def write_family_csv(profile: SyntheticFamily, sequences: list[list[str]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{profile.name.upper()}_variants.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SEQUENCE_ID", "STEP"])
        for i, seq in enumerate(sequences, 1):
            sid = f"{profile.name.upper()}_{i:05d}"
            for step in seq:
                writer.writerow([sid, step])
    return path


def write_manifest(out_dir: Path, count_per_family: int) -> Path:
    path = out_dir / "SUBMISSION_2_15_FAMILY_MANIFEST.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "split", "source_file", "litho_levels", "second_field_via"])
        writer.writerow(["mosfet", "train", "MOSFET_variants.csv", 4, ""])
        writer.writerow(["igbt", "train", "IGBT_variants.csv", 6, ""])
        writer.writerow(["ic", "train", "IC_variants.csv", 4, ""])
        for profile in SYNTHETIC_FAMILIES:
            writer.writerow([
                profile.name,
                profile.split,
                f"{profile.name.upper()}_variants.csv",
                profile.litho_levels,
                int(profile.second_field_via),
            ])
    print(f"Wrote manifest for {len(SYNTHETIC_FAMILIES) + 3} total families "
          f"({count_per_family} sequences each) -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("training_data"))
    args = parser.parse_args()

    for profile in SYNTHETIC_FAMILIES:
        sequences = generate_family(profile, args.count_per_family, args.seed)
        path = write_family_csv(profile, sequences, args.out_dir)
        lengths = [len(seq) for seq in sequences]
        print(
            f"  wrote {path} rows={sum(lengths)} "
            f"len=min/avg/max {min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}"
        )
    write_manifest(args.out_dir, args.count_per_family)


if __name__ == "__main__":
    main()

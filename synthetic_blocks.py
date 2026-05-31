"""Reusable synthetic process blocks for SUBMISSION_2 family generation.

The blocks combine existing MOSFET, IGBT, and IC vocabulary in validated
process orders. They are used by `generate_submission_2_families.py` to create
multiple train and held-out OOD families without introducing new step tokens.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training_data.generate_sequences import validate_sequence


# ── helpers ───────────────────────────────────────────────────────────────────

def _maybe(rng, prob, *steps):
    return list(steps) if rng.random() < prob else []

def _choice(rng, *options):
    return rng.choice(options)


# ── grammar blocks ────────────────────────────────────────────────────────────

def _prefix(rng):
    return ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
            _choice(rng, "INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION")]


def _initial_measurements(rng):
    s = []
    s += _maybe(rng, 0.7, _choice(rng, "MEASURE THICKNESS", "MEASURE INITIAL THICKNESS",
                                        "MEASURE INITIAL GEOMETRY"))
    s += _maybe(rng, 0.6, _choice(rng, "MEASURE SURFACE PARTICLES", "MEASURE SURFACE DEFECTS"))
    return s


def _pre_process_clean(rng):
    s = [_choice(rng, "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS")]
    s += _maybe(rng, 0.55, "BACKSIDE CLEAN")
    s += _maybe(rng, 0.45, "FRONTSIDE CLEAN")
    s += [_choice(rng, "WET CLEAN RCA1", "RCA CLEAN 1"),
          _choice(rng, "WET CLEAN RCA2", "RCA CLEAN 2"),
          "HF DIP"]
    s += _maybe(rng, 0.5, _choice(rng, "DRY WAFER", "DRY WAFER BACKSIDE"))
    return s


def _template_prep(rng):
    """IGBT epitaxial check → MOSFET substrate+epitaxy.  No training family has this."""
    s = ["EPITAXIAL WAFER CHECK"]                      # IGBT step first
    s += _maybe(rng, 0.65, "MEASURE EPITAXY THICKNESS")
    s += _maybe(rng, 0.55, "MEASURE RESISTIVITY")
    s += _maybe(rng, 0.25, "EPITAXIAL REWORK CHECK")   # IGBT optional
    s += ["SUBSTRATE CHECK",                           # then MOSFET substrate check
          "EPITAXY PREP",
          "EPITAXIAL DEPOSITION",
          "MEASURE EPITAXY THICKNESS",
          "MEASURE RESISTIVITY",
          "EPITAXY ANNEAL",
          "WAFER SURFACE CLEAN"]
    return s


def _first_oxidation(rng):
    s = ["THERMAL OXIDATION", "GATE OXIDE PREP", "GATE OXIDE GROWTH"]
    s += _maybe(rng, 0.6, "ANNEAL OXIDE")
    s.append(_choice(rng, "MEASURE OXIDE THICKNESS", "MEASURE GATE OXIDE THICKNESS"))
    return s


_PATTERN_INSPECTION = {
    # Only strings that exist in the MOSFET/IGBT/IC training vocabulary
    1: ("INSPECT PATTERN LEVEL 1", "PATTERN INSPECTION LEVEL 1"),
    2: ("P BODY WINDOW INSPECTION",),
    3: ("POLY PATTERN INSPECTION",),
    4: ("FIELD PATTERN INSPECTION",),
    5: ("VIA INSPECTION",           "VIA OPENING INSPECTION"),
    6: ("METAL PATTERN INSPECTION",),
}


def _litho(rng, level):
    """Standard litho block for a given level — uses only training-vocab inspections."""
    s = ["SPIN COAT PHOTORESIST", "SOFT BAKE",
         f"ALIGN MASK LEVEL {level}", f"EXPOSE LITHO LEVEL {level}"]
    s += _maybe(rng, 0.35, "POST EXPOSE BAKE")
    s.append("DEVELOP PHOTORESIST")
    opts = _PATTERN_INSPECTION.get(level, ("INSPECT PATTERN LEVEL 1",))
    s.append(_choice(rng, *opts))
    s += _maybe(rng, 0.4, "HARD BAKE")
    return s


def _cycle1_pwell(rng):
    """P-well implant — MOSFET-like (litho level 1)."""
    s = _litho(rng, 1)
    s.append(_choice(rng, "OXIDE ETCH", "OXIDE ETCH DRY"))
    s += [_choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
          _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH")]
    s += _maybe(rng, 0.5, _choice(rng, "MEASURE OPENING CD", "MEASURE WINDOW CD"))
    s.append("IMPLANT WELL")
    s += _maybe(rng, 0.25, "PRE ANNEAL CHECK")
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    s += _maybe(rng, 0.55, _choice(rng, "MEASURE JUNCTION DEPTH", "MEASURE SHEET RESISTANCE"))
    return s


def _cycle2_pbody(rng):
    """P-body implant — IGBT-like window etch (litho level 2)."""
    s = _litho(rng, 2)
    s.append("ETCH SILICON OR OXIDE WINDOW")
    s += [_choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
          _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER WINDOW ETCH")]
    s += _maybe(rng, 0.45, "MEASURE WINDOW CD")
    s.append("IMPLANT P BODY")
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    s += _maybe(rng, 0.5, _choice(rng, "MEASURE JUNCTION DEPTH", "MEASURE JUNCTION PROFILE"))
    return s


def _cycle3_poly_dual_implant(rng):
    """Poly gate + SOURCE DRAIN + N BUFFER in same cycle — template-specific (litho level 3).

    MOSFET: poly etch -> SOURCE DRAIN only
    IGBT: poly etch -> CHANNEL STOP only; N BUFFER is a separate litho cycle
    Template: poly etch -> SOURCE DRAIN -> N BUFFER.
    """
    # optional pre-clean before poly deposition
    s = []
    if rng.random() < 0.4:
        s += [_choice(rng, "WET CLEAN RCA1", "RCA CLEAN 1"),
              _choice(rng, "WET CLEAN RCA2", "RCA CLEAN 2"), "HF DIP"]
    s += ["DEPOSIT POLYSILICON"]
    s += _maybe(rng, 0.5, _choice(rng, "POLYSILICON ANNEAL", "ANNEAL POLYSILICON"))
    s += _maybe(rng, 0.6, "MEASURE POLY THICKNESS")
    s += _litho(rng, 3)
    s.append(_choice(rng, "POLYSILICON ETCH", "POLYSILICON ETCH DRY"))
    s += [_choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
          _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER POLY ETCH")]
    s += _maybe(rng, 0.5, _choice(rng, "MEASURE GATE CD", "MEASURE GATE OXIDE THICKNESS"))
    # DUAL IMPLANT — the template-specific transition
    s.append(_choice(rng, "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION"))
    s.append("IMPLANT N BUFFER")      # ← novel: N BUFFER after SOURCE DRAIN in same cycle
    s += _maybe(rng, 0.2, "PRE ANNEAL CHECK")
    s.append(_choice(rng, "RAPID THERMAL ANNEAL", "LIGHT ANNEAL"))
    s += _maybe(rng, 0.4, "MEASURE JUNCTION DEPTH")
    # optional MOSFET-style spacer sub-block
    if rng.random() < 0.55:
        s += ["DEPOSIT SPACER DIELECTRIC", "ANISOTROPIC ETCH SPACER",
              "IMPLANT LDD", "RAPID THERMAL ANNEAL"]
        s += _maybe(rng, 0.45, "MEASURE SPACER WIDTH")
    return s


def _cycle4_field_via(rng):
    """Field oxide + via prep — DEPOSIT FIELD OXIDE (IGBT-only step) in a MOSFET-style
    sequence, then via open (litho level 4).  Novel use of field oxide position."""
    s = []
    # optional intermediate clean
    if rng.random() < 0.4:
        s += [_choice(rng, "WET CLEAN RCA1", "RCA CLEAN 1"),
              _choice(rng, "WET CLEAN RCA2", "RCA CLEAN 2"), "HF DIP"]
    s.append("DEPOSIT FIELD OXIDE")           # IGBT step in the field-oxide position
    s += _maybe(rng, 0.5, "MEASURE FILM THICKNESS")
    # ILD (interlayer dielectric between last implant and via)
    s.append(_choice(rng, "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"))
    s.append(_choice(rng, "DENSIFY DIELECTRIC", "DENSIFY OXIDE"))
    s += _maybe(rng, 0.65, _choice(rng, "MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"))
    s.append(_choice(rng, "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"))
    s += _maybe(rng, 0.65, _choice(rng, "MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"))
    # Via litho (level 4)
    s += _litho(rng, 4)
    s.append(_choice(rng, "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC"))
    s += [_choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
          _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER VIA ETCH")]
    s += _maybe(rng, 0.5, _choice(rng, "MEASURE VIA CD", "MEASURE VIA RESISTANCE"))
    # Via fill block
    s += ["DEPOSIT BARRIER METAL",
          _choice(rng, "DEPOSIT METAL SEED", "DEPOSIT TUNGSTEN SEED"),
          _choice(rng, "FILL VIA METAL", "FILL VIA TUNGSTEN"),
          _choice(rng, "CMP METAL", "CMP VIA FILL")]
    s += _maybe(rng, 0.6, _choice(rng, "MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"))
    return s


def _cycle5_metal(rng):
    """Metal deposition + litho + etch (litho level 5)."""
    s = ["DEPOSIT METAL 1"]
    s += _maybe(rng, 0.55, "MEASURE METAL THICKNESS")
    # Metal litho (level 5)
    s += _litho(rng, 5)
    # override to use metal pattern inspection (only training-vocab option)
    s[-1] = "METAL PATTERN INSPECTION"
    s.append(_choice(rng, "METAL ETCH", "METAL ETCH DRY"))
    s += [_choice(rng, "STRIP PHOTORESIST", "STRIP RESIST"),
          _choice(rng, "CLEAN AFTER ETCH", "CLEAN AFTER METAL ETCH")]
    s += _maybe(rng, 0.5, "MEASURE CONTACT RESISTANCE")
    return s


def _passivation_block(rng):
    s = [_choice(rng, "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"),
         "CURE PASSIVATION",
         _choice(rng, "MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"),
         _choice(rng, "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"),
         _choice(rng, "PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"),
         _choice(rng, "DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"),
         _choice(rng, "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"),
         "STRIP RESIST",
         "CLEAN PAD OPENING",
         "MEASURE PAD OPENING"]
    return s


def _backside_block(rng):
    s = [_choice(rng, "BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL")]
    s += ["BACKSIDE GRIND",
          _choice(rng, "MEASURE THICKNESS", "MEASURE WAFER THICKNESS"),
          "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE", "BACKSIDE DRY",
          "BACKSIDE METALLIZATION PREP", "DEPOSIT BACKSIDE METAL",
          "BACKSIDE ANNEAL", "MEASURE BACKSIDE CONTACT"]
    return s


def _final_inspection(rng):
    s = ["FINAL CLEAN", "FINAL THICKNESS MEASURE", "FINAL GEOMETRY CHECK"]
    s += _maybe(rng, 0.5, "FINAL OXIDE CHECK")
    s += _maybe(rng, 0.4, "FINAL CD INSPECTION")
    s += ["FINAL PARTICLE INSPECTION"]
    s += _maybe(rng, 0.35, "FRONTSIDE CLEAN FINAL")
    s += _maybe(rng, 0.4, "FINAL ELECTRICAL TEST PREP")
    return s


def _test_suite(rng):
    s = [_choice(rng, "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"),
         "LEAKAGE TEST",
         "THRESHOLD VOLTAGE TEST",   # gate-based like MOSFET
         "SWITCHING TEST",
         "WAFER SORT TEST",
         "YIELD ANALYSIS"]
    return s


# ── full sequence ─────────────────────────────────────────────────────────────

def generate_template_sequence(rng) -> list[str]:
    seq = []
    seq += _prefix(rng)
    seq += _initial_measurements(rng)
    seq += _pre_process_clean(rng)
    seq += _template_prep(rng)
    seq += _first_oxidation(rng)
    seq += _cycle1_pwell(rng)
    seq += _cycle2_pbody(rng)
    seq += _cycle3_poly_dual_implant(rng)
    seq += _cycle4_field_via(rng)
    seq += _cycle5_metal(rng)
    seq += _passivation_block(rng)
    seq += _backside_block(rng)
    seq += _final_inspection(rng)
    seq += _test_suite(rng)
    seq += [rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"]), "SHIP LOT"]
    return seq


# ── generation loop ───────────────────────────────────────────────────────────

def generate(count: int, seed: int = 42, max_retries: int = 20) -> list[list[str]]:
    rng = random.Random(seed)
    sequences, attempts, violations = [], 0, 0
    while len(sequences) < count:
        attempts += 1
        seq = generate_template_sequence(rng)
        errs = validate_sequence(seq)
        if not errs:
            sequences.append(seq)
        else:
            violations += 1
            if violations <= 3:
                print(f"  [warn] violation in attempt {attempts}: {errs[0]}", flush=True)
    print(f"Generated {len(sequences)} valid sequences "
          f"({attempts} attempts, {violations} invalid discarded)", flush=True)
    return sequences


def write_csv(sequences: list[list[str]], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "STEP"])
        for i, seq in enumerate(sequences):
            sid = f"SCFAM_TEMPLATE_{i+1:05d}"
            for step in seq:
                w.writerow([sid, step])
    print(f"Wrote {len(sequences)} sequences ({sum(len(s) for s in sequences)} rows) → {path}")


def validate_file(path: str):
    from training_data.generate_sequences import read_csv_sequences
    seqs = read_csv_sequences(Path(path))
    ok, fail = 0, 0
    for sid, steps in seqs.items():
        errs = validate_sequence(steps)
        if errs:
            fail += 1
            print(f"  FAIL {sid}: {errs[0]}")
        else:
            ok += 1
    print(f"Validation: {ok} valid, {fail} invalid (out of {ok+fail})")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count",    type=int, default=2000)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="training_data/SCFAM_TEMPLATE_variants.csv")
    ap.add_argument("--validate", default=None, metavar="CSV",
                    help="Validate an existing file instead of generating")
    args = ap.parse_args()

    if args.validate:
        validate_file(args.validate)
    else:
        print(f"Generating {args.count} synthetic template sequences (seed={args.seed})...")
        seqs = generate(args.count, seed=args.seed)
        write_csv(seqs, args.out)
        # Quick stats
        lengths = [len(s) for s in seqs]
        print(f"Sequence lengths: min={min(lengths)} max={max(lengths)} "
              f"avg={sum(lengths)/len(lengths):.1f}")

"""
generate_data.py
=================
Generates additional training sequences for IGBT and IC families
using the provided generate_sequences.py from the track organizers.

MOSFET is held out as OOD test family — we do NOT generate more MOSFET.

Run from test_baseline_2/:
    python generate_data.py
"""

import subprocess
import sys
import os
import argparse
import csv


TRAIN_DIR       = "../../tracks/industrial-infineon/training_data"
GENERATE_SCRIPT = "../../tracks/industrial-infineon/training_data/generate_sequences.py"


def count_sequences(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("SEQUENCE_ID") or row.get("sequence_id", "")
            if sid:
                ids.add(sid)
    return len(ids)


def generate(family: str, count: int, output_path: str, seed: int) -> bool:
    print(f"\n{'─'*55}")
    print(f"  Generating {count} {family.upper()} sequences")
    print(f"  Output: {output_path}")
    print(f"{'─'*55}")

    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--family", family,
        "--count",  str(count),
        "--output", output_path,
        "--seed",   str(seed),
    ]
    print(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Generation failed for {family}")
        return False

    n = count_sequences(output_path)
    print(f"\n  Done — {n} sequences written")
    return True


def validate(csv_path: str, family: str):
    """Run the built-in validator on generated file."""
    print(f"\n  Validating {csv_path}...")
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--validate", csv_path,
        "--family", family,
    ]
    subprocess.run(cmd)


def main():
    global GENERATE_SCRIPT

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_script", default=GENERATE_SCRIPT,
                        help="Path to generate_sequences.py")
    parser.add_argument("--out_dir", default=TRAIN_DIR,
                        help="Where to save generated CSVs")
    parser.add_argument("--igbt_count", type=int, default=2000)
    parser.add_argument("--ic_count",   type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=123)
    parser.add_argument("--validate",   action="store_true",
                        help="Run validation on generated files after generation")
    args = parser.parse_args()

    GENERATE_SCRIPT = args.generate_script

    if not os.path.exists(GENERATE_SCRIPT):
        print(f"[ERROR] generate_sequences.py not found at: {GENERATE_SCRIPT}")
        print("Check --generate_script path")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Estimate combinatoric space first ─────────────────────────────────────
    print("\nEstimating combinatoric space...")
    for family in ["igbt", "ic"]:
        subprocess.run([sys.executable, GENERATE_SCRIPT,
                        "--family", family, "--estimate-only"])

    # ── Generate IGBT ─────────────────────────────────────────────────────────
    igbt_out = os.path.join(args.out_dir, "IGBT_generated_extra.csv")
    ok_igbt = generate("igbt", args.igbt_count, igbt_out, args.seed)
    if ok_igbt and args.validate:
        validate(igbt_out, "igbt")

    # ── Generate IC ───────────────────────────────────────────────────────────
    ic_out = os.path.join(args.out_dir, "IC_generated_extra.csv")
    ok_ic = generate("ic", args.ic_count, ic_out, args.seed + 1)
    if ok_ic and args.validate:
        validate(ic_out, "ic")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  GENERATION COMPLETE")
    print(f"{'='*55}")
    if ok_igbt:
        print(f"  IGBT : {count_sequences(igbt_out):>5} sequences → {igbt_out}")
    if ok_ic:
        print(f"  IC   : {count_sequences(ic_out):>5} sequences → {ic_out}")
    print(f"  MOSFET: skipped (held out as OOD test family)")
    print(f"""
  Next step:
      python prepare_dataset.py
""")


if __name__ == "__main__":
    main()
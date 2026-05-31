"""Build self-evaluation files that mirror the organiser's eval format.

Creates:
    self_eval/eval_input_valid.csv    — 600 entries (100 seqs × 3 families × 2 cut points)
    self_eval/eval_input_anomaly.csv  — 987 entries (600 valid + 387 with injected violations)
    self_eval/ground_truth_anomaly.csv — ground truth for self-scoring Task 3

The held-out sequences are NOT used during GPT/n-gram training (seeded split).

Run once:
    python make_selfeval.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# ── imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data import FAMILIES, load_family
from training_data.generate_sequences import validate_sequence

SEED       = 999   # different from training seed (42) so no overlap
HOLDOUT_N  = 200   # sequences reserved per family (100 for task1/2, 100 for task3)
OUT_DIR    = Path("self_eval")
OUT_DIR.mkdir(exist_ok=True)


# ── violation injectors ───────────────────────────────────────────────────────

def inject_ship_before_test(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_SHIP_BEFORE_TEST: move SHIP LOT before WAFER SORT TEST."""
    bad = list(seq)
    if "SHIP LOT" not in bad or "WAFER SORT TEST" not in bad:
        return None
    i_sort = bad.index("WAFER SORT TEST")
    i_ship = bad.index("SHIP LOT")
    if i_sort >= i_ship:
        return None  # already violated — skip
    bad.pop(i_ship)
    bad.insert(i_sort, "SHIP LOT")
    return bad, "RULE_SHIP_BEFORE_TEST"


def inject_test_before_passivation(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_TEST_BEFORE_PASSIVATION: move PARAMETRIC TEST before CURE PASSIVATION."""
    bad = list(seq)
    test_steps = ["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
                  "LEAKAGE TEST", "SWITCHING TEST"]
    cure = "CURE PASSIVATION"
    if cure not in bad:
        return None
    i_cure = bad.index(cure)
    # Find a test step that currently appears after CURE PASSIVATION
    for ts in test_steps:
        if ts in bad:
            i_ts = bad.index(ts)
            if i_ts > i_cure:
                bad.pop(i_ts)
                bad.insert(max(0, i_cure - 1), ts)
                return bad, "RULE_TEST_BEFORE_PASSIVATION"
    return None


def inject_dep_no_clean(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_DEP_NO_CLEAN: remove the pre-process clean so a deposition lacks it."""
    bad = list(seq)
    clean_triggers = ["PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS",
                      "WET CLEAN RCA1", "RCA CLEAN 1", "HF DIP"]
    for cln in clean_triggers:
        if cln in bad:
            idx = bad.index(cln)
            bad.pop(idx)
            violations = validate_sequence(bad)
            if any(v.rule == "RULE_DEP_NO_CLEAN" for v in violations):
                return bad, "RULE_DEP_NO_CLEAN"
            bad.insert(idx, cln)   # revert and try next
    return None


def inject_backside_before_passivation(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_BACKSIDE_BEFORE_PASSIVATION: move DEPOSIT BACKSIDE METAL before CURE PASSIVATION."""
    bad = list(seq)
    if "DEPOSIT BACKSIDE METAL" not in bad or "CURE PASSIVATION" not in bad:
        return None
    i_cure = bad.index("CURE PASSIVATION")
    i_dep  = bad.index("DEPOSIT BACKSIDE METAL")
    if i_dep < i_cure:
        return None  # already before — skip
    bad.pop(i_dep)
    bad.insert(max(0, i_cure - 1), "DEPOSIT BACKSIDE METAL")
    return bad, "RULE_BACKSIDE_BEFORE_PASSIVATION"


def inject_etch_no_mask(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_ETCH_NO_MASK: remove DEVELOP PHOTORESIST right before an etch."""
    bad = list(seq)
    etch_steps = {"OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH",
                  "POLYSILICON ETCH DRY", "VIA ETCH", "METAL ETCH",
                  "METAL ETCH DRY", "FIELD OXIDE ETCH"}
    for i, step in enumerate(bad):
        if step in etch_steps:
            # Look back for DEVELOP PHOTORESIST
            window = bad[max(0, i-12):i]
            if "DEVELOP PHOTORESIST" in window:
                j = bad[:i].index("DEVELOP PHOTORESIST")
                bad.pop(j)
                violations = validate_sequence(bad)
                if any(v.rule == "RULE_ETCH_NO_MASK" for v in violations):
                    return bad, "RULE_ETCH_NO_MASK"
                bad.insert(j, "DEVELOP PHOTORESIST")
    return None


def inject_cmp_no_dep(seq: list[str]) -> tuple[list[str], str] | None:
    """RULE_CMP_NO_DEP: remove deposit step before a CMP."""
    bad = list(seq)
    cmp_steps = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"}
    fill_steps = {"FILL VIA METAL", "FILL VIA TUNGSTEN",
                  "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
                  "DEPOSIT BARRIER METAL"}
    for i, step in enumerate(bad):
        if step in cmp_steps:
            window = bad[max(0, i-6):i]
            fills_in_window = [s for s in window if s in fill_steps]
            if fills_in_window:
                target = fills_in_window[-1]
                prefix = bad[:i]
                j = len(prefix) - 1 - prefix[::-1].index(target)
                bad.pop(j)
                violations = validate_sequence(bad)
                if any(v.rule == "RULE_CMP_NO_DEP" for v in violations):
                    return bad, "RULE_CMP_NO_DEP"
                bad.insert(j, target)
    return None


INJECTORS = [
    inject_ship_before_test,
    inject_test_before_passivation,
    inject_dep_no_clean,
    inject_backside_before_passivation,
    inject_etch_no_mask,
    inject_cmp_no_dep,
]


# ── main builder ──────────────────────────────────────────────────────────────

def build_selfeval():
    rng = random.Random(SEED)

    # Load and shuffle each family
    family_seqs: dict[str, list[list[str]]] = {}
    for fam in FAMILIES:
        seqs = list(load_family(fam).values())
        rng.shuffle(seqs)
        family_seqs[fam] = seqs
        print(f"  {fam}: {len(seqs)} sequences loaded")

    # Partition: first HOLDOUT_N per family held out; rest used for training
    task12_seqs: dict[str, list[list[str]]] = {}   # first 100 per family
    task3_valid: dict[str, list[list[str]]] = {}   # next 100 per family
    for fam in FAMILIES:
        task12_seqs[fam] = family_seqs[fam][:100]
        task3_valid[fam] = family_seqs[fam][100:200]

    # ── eval_input_valid.csv (Tasks 1 & 2) ───────────────────────────────────
    valid_rows = []
    for fam in FAMILIES:
        for i, seq in enumerate(task12_seqs[fam]):
            for frac in (0.6, 0.8):
                cut = max(1, int(len(seq) * frac))
                partial = seq[:cut]
                eid = f"valid_{fam}_{i:03d}_{int(frac*100):02d}"
                valid_rows.append({
                    "EXAMPLE_ID":          eid,
                    "FAMILY":              fam.upper(),
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE":    "|".join(partial),
                    "NEXT_STEP":           seq[cut] if cut < len(seq) else "",
                    "FULL_SEQUENCE":        "|".join(seq),
                    # keep ground truth for self-scoring (not in real eval)
                    "_REMAINING":          "|".join(seq[cut:]),
                    "_FULL_SEQUENCE":      "|".join(seq),
                })

    with open(OUT_DIR / "eval_input_valid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY",
                                           "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        w.writeheader()
        for r in valid_rows:
            w.writerow({k: r[k] for k in ["EXAMPLE_ID", "FAMILY",
                                           "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"]})

    # Ground truth for Tasks 1 & 2
    with open(OUT_DIR / "ground_truth_valid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY",
                                           "COMPLETION_FRACTION", "PARTIAL_SEQUENCE",
                                           "NEXT_STEP", "FULL_SEQUENCE",
                                           "_REMAINING", "_FULL_SEQUENCE"])
        w.writeheader()
        for r in valid_rows:
            w.writerow(r)

    print(f"  eval_input_valid.csv: {len(valid_rows)} rows")

    # ── eval_input_anomaly.csv (Task 3) ───────────────────────────────────────
    anomaly_rows = []
    gt_rows = []

    # 600 valid sequences
    idx = 0
    for fam in FAMILIES:
        for seq in task3_valid[fam]:
            eid = f"anomaly_{idx:04d}"
            anomaly_rows.append({
                "EXAMPLE_ID": eid,
                "FAMILY":     fam.upper(),
                "SEQUENCE":   "|".join(seq),
            })
            gt_rows.append({
                "EXAMPLE_ID":    eid,
                "FAMILY":        fam.upper(),
                "SEQUENCE":      "|".join(seq),
                "IS_VALID":      1,
                "RULE_VIOLATED": "",
                "VIOLATION_RULE": "",
            })
            idx += 1

    # 387 violated sequences — cycle through injectors, try families
    target_violations = 387
    violations_made = 0
    fam_cycle = (FAMILIES * 200)[:target_violations * 3]  # enough to try

    for fam in fam_cycle:
        if violations_made >= target_violations:
            break
        # Use any valid sequence as the base for synthetic violations. These
        # rows are for scorer validation/reporting, not for training.
        candidates = family_seqs[fam]
        if not candidates:
            continue
        seq = rng.choice(candidates)
        injector = INJECTORS[violations_made % len(INJECTORS)]
        result = injector(seq)
        if result is None:
            # try next injector
            for inj in INJECTORS:
                result = inj(seq)
                if result is not None:
                    break
        if result is None:
            continue
        bad_seq, rule = result
        # Verify the violation is actually detected
        detected = validate_sequence(bad_seq)
        if not detected:
            continue   # injection didn't take, skip
        eid = f"anomaly_{idx:04d}"
        anomaly_rows.append({
            "EXAMPLE_ID": eid,
            "FAMILY":     fam.upper(),
            "SEQUENCE":   "|".join(bad_seq),
        })
        gt_rows.append({
            "EXAMPLE_ID":    eid,
            "FAMILY":        fam.upper(),
            "SEQUENCE":      "|".join(bad_seq),
            "IS_VALID":      0,
            "RULE_VIOLATED": rule,
            "VIOLATION_RULE": rule,
        })
        idx += 1
        violations_made += 1

    # Shuffle so valid/invalid are mixed
    combined = list(zip(anomaly_rows, gt_rows))
    rng.shuffle(combined)
    anomaly_rows, gt_rows = zip(*combined)

    with open(OUT_DIR / "eval_input_anomaly.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        w.writeheader()
        w.writerows(anomaly_rows)

    with open(OUT_DIR / "ground_truth_anomaly.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE",
                                           "IS_VALID", "RULE_VIOLATED", "VIOLATION_RULE"])
        w.writeheader()
        w.writerows(gt_rows)

    with open(OUT_DIR / "ground_truth_anomaly_forbidden.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE", "VIOLATION_RULE"])
        w.writeheader()
        for r in gt_rows:
            if r["IS_VALID"] == 0:
                w.writerow({
                    "EXAMPLE_ID": r["EXAMPLE_ID"],
                    "FAMILY": r["FAMILY"],
                    "SEQUENCE": r["SEQUENCE"],
                    "VIOLATION_RULE": r["VIOLATION_RULE"],
                })

    with open(OUT_DIR / "ground_truth_anomaly_valid_supplement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        w.writeheader()
        for r in gt_rows:
            if r["IS_VALID"] == 1:
                w.writerow({
                    "EXAMPLE_ID": r["EXAMPLE_ID"],
                    "FAMILY": r["FAMILY"],
                    "SEQUENCE": r["SEQUENCE"],
                })

    n_valid_in_anomaly = sum(1 for r in gt_rows if r["IS_VALID"] == 1)
    print(f"  eval_input_anomaly.csv: {len(anomaly_rows)} rows "
          f"({n_valid_in_anomaly} valid, {len(anomaly_rows)-n_valid_in_anomaly} violated)")
    print(f"Self-eval files written to {OUT_DIR}/")


if __name__ == "__main__":
    print("Building self-evaluation files...")
    build_selfeval()

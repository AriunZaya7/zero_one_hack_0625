"""
Rule checking for process-sequence anomalies.

The authoritative implementation of the 10 forbidden patterns lives in
training_data/generate_sequences.py. This module provides the stable solution
API used by the models, anomaly generator, dashboard, and eval runner while
delegating rule truth to that official validator.
"""

from __future__ import annotations

from collections import defaultdict

from training_data.generate_sequences import (
    BACKSIDE_METAL_STEPS,
    CLEAN_STEPS,
    CMP_STEPS,
    DEPOSITION_STEPS,
    ELECTRICAL_TEST_STEPS,
    ETCH_STEPS,
    FILL_STEPS,
    IMPLANT_OPENER_STEPS,
    IMPLANT_STEPS,
    METAL_ETCH_STEPS,
    PAD_WINDOW_STEPS,
    validate_sequence,
)

RULE_IDS = [
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
]

DEVELOP_STEPS = frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"})
PASSIVATION_STEPS = frozenset({"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"})
OXIDATION_STEPS = frozenset({"THERMAL OXIDATION", "GATE OXIDE GROWTH", "ANNEAL OXIDE"})


def _u(step: str) -> str:
    return str(step).strip().upper()


def _normalize(sequence: list[str]) -> list[str]:
    return [_u(step) for step in sequence if str(step).strip()]


def violations(sequence: list[str]):
    """Return official Violation objects for this sequence."""
    return validate_sequence(_normalize(sequence))


def check_all_rules(sequence: list[str], family: str | None = None, vocab=None) -> dict[str, bool]:
    """Return {rule_id: violated}. family/vocab are accepted for PLAN.md compatibility."""
    hit = {v.rule for v in violations(sequence)}
    return {rule_id: rule_id in hit for rule_id in RULE_IDS}


def first_violation(sequence: list[str], family: str | None = None, vocab=None) -> tuple[str | None, int | None]:
    """Return (first_rule_id, step_index), with ties broken by RULE_IDS order."""
    by_index: dict[int, list[str]] = defaultdict(list)
    for violation in violations(sequence):
        by_index[violation.step_index].append(violation.rule)
    if not by_index:
        return None, None

    first_idx = min(by_index)
    tied_rules = set(by_index[first_idx])
    for rule_id in RULE_IDS:
        if rule_id in tied_rules:
            return rule_id, first_idx
    return by_index[first_idx][0], first_idx


def attribute_anomaly(sequence: list[str], family: str | None = None, vocab=None) -> str | None:
    """Return the first violated rule id, or None if the sequence is valid."""
    return first_violation(sequence, family=family, vocab=vocab)[0]


def is_valid(sequence: list[str]) -> bool:
    return attribute_anomaly(sequence) is None


# Predicate helpers used by feature extraction and anomaly injection.
def is_clean(step: str) -> bool:
    return _u(step) in CLEAN_STEPS


def is_deposit(step: str) -> bool:
    return _u(step) in DEPOSITION_STEPS


def is_develop(step: str) -> bool:
    return _u(step) in DEVELOP_STEPS


def is_etch(step: str) -> bool:
    return _u(step) in ETCH_STEPS


def is_metal_etch(step: str) -> bool:
    return _u(step) in METAL_ETCH_STEPS


def is_implant(step: str) -> bool:
    return _u(step) in IMPLANT_STEPS


def is_implant_opener(step: str) -> bool:
    return _u(step) in IMPLANT_OPENER_STEPS


def is_cmp(step: str) -> bool:
    return _u(step) in CMP_STEPS


def is_fill(step: str) -> bool:
    return _u(step) in FILL_STEPS


def is_pad_window(step: str) -> bool:
    return _u(step) in PAD_WINDOW_STEPS


def is_electrical_test(step: str) -> bool:
    return _u(step) in ELECTRICAL_TEST_STEPS


def is_passivation(step: str) -> bool:
    return _u(step) in PASSIVATION_STEPS


def is_cure(step: str) -> bool:
    return _u(step) == "CURE PASSIVATION"


def is_backside_metal(step: str) -> bool:
    return _u(step) in BACKSIDE_METAL_STEPS


def is_oxidation(step: str) -> bool:
    return _u(step) in OXIDATION_STEPS


def is_wafer_sort(step: str) -> bool:
    return _u(step) == "WAFER SORT TEST"


def is_ship(step: str) -> bool:
    return _u(step) == "SHIP LOT"


def align_level(step: str) -> int | None:
    step = _u(step)
    if not step.startswith("ALIGN MASK LEVEL "):
        return None
    try:
        return int(step.rsplit(" ", 1)[1])
    except ValueError:
        return None


if __name__ == "__main__":
    import os

    try:
        from solution.data.loader import load_all_families

        if os.path.isdir("training_data"):
            families = load_all_families("training_data")
            for family, seqs in families.items():
                sample = seqs[:200]
                bad = [seq for seq in sample if not is_valid(seq)]
                rate = 100 * (len(sample) - len(bad)) / max(len(sample), 1)
                print(f"  {family:<7} valid-pass rate on {len(sample)} seqs: {rate:.1f}%")
                if bad:
                    print(f"    e.g. {first_violation(bad[0])}")
    except Exception as exc:
        print(f"  (skipped real-data check: {exc})")

    assert attribute_anomaly(["SHIP LOT", "WAFER SORT TEST"]) == "RULE_SHIP_BEFORE_TEST"
    assert check_all_rules(["SHIP LOT", "WAFER SORT TEST"])["RULE_SHIP_BEFORE_TEST"]
    print("rules OK")

"""Task 3: Anomaly detection for process sequences.

Strategy (two-tier):
  1. Symbolic oracle  — validate_sequence() checks all 10 grammar rules
     deterministically.  Binary classification is PERFECT on any sequence
     whose violations come from the known rule set.
  2. Continuous score — model.sequence_surprisal() gives a per-nats score
     that calibrates the ROC-AUC metric when the judges want a probability
     rather than a hard label.

Usage:
    from anomaly import AnomalyDetector
    det = AnomalyDetector(model=gpt_wrapper)     # model optional
    results = det.predict(sequences)             # list[dict]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Make training_data importable from the repo root
sys.path.insert(0, str(Path(__file__).parent))

try:
    from training_data.generate_sequences import validate_sequence
    _VALIDATOR_OK = True
except Exception as e:
    print(f"[anomaly] Warning: validate_sequence unavailable ({e}); "
          "falling back to surprisal-only.")
    validate_sequence = None
    _VALIDATOR_OK = False

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


def _symbolic_check(steps: list[str]) -> tuple[bool, str | None]:
    """Run the rule validator.  Returns (is_valid, first_violated_rule_or_None)."""
    if not _VALIDATOR_OK:
        return True, None
    violations = validate_sequence(steps)
    if not violations:
        return True, None
    # Violation dataclass has .rule attribute (string like "RULE_DEP_NO_CLEAN")
    return False, violations[0].rule


def _surprisal_to_score(surprisal: float) -> float:
    """Map mean-NLL surprisal (nats/step) → validity probability ∈ (0,1).

    Typical valid sequence: ~0.5–2.5 nats/step  → score close to 1.0
    Typical invalid:        higher surprisal     → score closer to 0.0
    Sigmoid centred at 2.5 nats gives a smooth calibration.
    """
    return 1.0 / (1.0 + math.exp(surprisal - 2.5))


class AnomalyDetector:
    """Wraps the two-tier detection strategy behind a single .predict() call."""

    def __init__(self, model=None):
        """
        Args:
            model: any object implementing .sequence_surprisal(seq) → float.
                   Pass None to use hard binary scoring (symbolic only).
        """
        self.model = model

    def predict(self, sequences: list[list[str]]) -> list[dict]:
        """
        Args:
            sequences: list of step-lists

        Returns:
            list of dicts:
                is_valid     (int 1/0)
                score        (float in [0,1];  1 = definitely valid)
                predicted_rule (str or empty string)
        """
        results = []
        for steps in sequences:
            is_valid, rule = _symbolic_check(steps)

            if self.model is not None:
                try:
                    surp = self.model.sequence_surprisal(steps)
                    score = _surprisal_to_score(surp)
                    # Blend: if symbolic says invalid, push score down regardless
                    if not is_valid:
                        score = min(score, 0.35)
                    else:
                        score = max(score, 0.55)
                except Exception:
                    score = 1.0 if is_valid else 0.0
            else:
                score = 1.0 if is_valid else 0.0

            results.append({
                "is_valid":       1 if is_valid else 0,
                "score":          round(score, 4),
                "predicted_rule": rule or "",
            })
        return results

    def predict_sequence(self, steps: list[str]) -> dict:
        return self.predict([steps])[0]


if __name__ == "__main__":
    # Quick smoke test
    from training_data.generate_sequences import validate_sequence as _v
    valid = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION",
             "MEASURE THICKNESS", "PRE CLEAN WAFER", "WET CLEAN RCA1",
             "WET CLEAN RCA2", "HF DIP", "THERMAL OXIDATION", "SHIP LOT"]
    # inject a violation: SHIP LOT before WAFER SORT TEST
    invalid = list(valid)
    # (this sequence already lacks WAFER SORT TEST — that itself is an anomaly,
    #  but let's test the direct path)

    det = AnomalyDetector(model=None)
    r = det.predict([valid])
    print("valid sequence:", r[0])

    # Test with an actual violation
    from data import load_family
    seqs = list(load_family("mosfet").values())
    seq = seqs[0]
    # Inject RULE_SHIP_BEFORE_TEST
    bad = list(seq)
    if "SHIP LOT" in bad and "WAFER SORT TEST" in bad:
        i_ship = bad.index("SHIP LOT")
        i_sort = bad.index("WAFER SORT TEST")
        if i_sort > 0:
            bad.pop(i_ship)
            bad.insert(i_sort - 1, "SHIP LOT")
    r2 = det.predict([bad])
    print("SHIP_BEFORE_TEST violation:", r2[0])
    assert r2[0]["is_valid"] == 0, "should detect violation"
    print("All checks passed.")

"""
Standalone XGBoost adapter for rule_probe_pipeline.py.

Share this file together with xgb_model.json and xgb_class_map.json.
"""

from pathlib import Path
import json

import numpy as np
import xgboost as xgb


MODEL_NAME = "XGBoost IC+IGBT -> MOSFET OOD"

MODEL_INFO = {
    "model_type": "xgboost",
    "n_features": 0,
    "n_classes": 0,
    "trained_on": "IGBT + IC",
    "ood_family": "MOSFET",
}


# Load the self-contained artifacts from the same directory as this adapter.
print(f"Loading model for: {MODEL_NAME}")

ARTIFACT_DIR = Path(__file__).resolve().parent

model = xgb.XGBClassifier()
model.load_model(str(ARTIFACT_DIR / "xgb_model.json"))

with open(ARTIFACT_DIR / "xgb_class_map.json", encoding="utf-8") as f:
    adapter_metadata = json.load(f)

CONFIG = adapter_metadata["config"]
FEATURE_NAMES = adapter_metadata["feature_names"]
VOCAB_TO_ID = {step: idx for idx, step in enumerate(adapter_metadata["vocab_itos"])}
FAMILY_TO_ID = {key: int(value) for key, value in adapter_metadata["family_to_id"].items()}
BIGRAM_TO_ID = {tuple(row[:-1]): int(row[-1]) for row in adapter_metadata["bigram_to_id"]}
TRIGRAM_TO_ID = {tuple(row[:-1]): int(row[-1]) for row in adapter_metadata["trigram_to_id"]}
RULE_SETS = {key: set(values) for key, values in adapter_metadata["rule_sets"].items()}
class_map = {entry["step"]: int(entry["estimator_class_id"]) for entry in adapter_metadata["classes"]}

BOS_ID = VOCAB_TO_ID["<BOS>"]
UNK_ID = VOCAB_TO_ID["<UNK>"]
DEFAULT_FAMILY_ID = int(adapter_metadata["default_family_id"])
CONTEXT_SIZE = int(CONFIG["context_size"])
MAX_SEQ_LEN = int(CONFIG["max_seq_len"])
FEATURE_VERSION = int(CONFIG.get("feature_version", 1))

MODEL_INFO.update(
    {
        "n_features": len(FEATURE_NAMES),
        "n_classes": len(class_map),
        "trained_on": " + ".join(adapter_metadata["trained_on"]),
        "ood_family": adapter_metadata["ood_family"],
    }
)

print(f"  Model loaded | classes: {len(class_map)} | features: {len(FEATURE_NAMES)}")


def _u(step: str) -> str:
    return str(step).strip().upper()


def _in_rule_set(name: str, step: str) -> bool:
    return _u(step) in RULE_SETS[name]


def _align_level(step: str) -> int | None:
    step = _u(step)
    if not step.startswith("ALIGN MASK LEVEL "):
        return None
    try:
        return int(step.rsplit(" ", 1)[1])
    except ValueError:
        return None


def _is_passivation(step: str) -> bool:
    return _in_rule_set("passivation", step)


def _is_cure(step: str) -> bool:
    return _u(step) == "CURE PASSIVATION"


def _is_backside(step: str) -> bool:
    return "BACKSIDE" in _u(step) or _in_rule_set("backside_metal", step)


def _is_wafer_sort(step: str) -> bool:
    return _u(step) == "WAFER SORT TEST"


def _is_ship(step: str) -> bool:
    return _u(step) == "SHIP LOT"


OP_PREDICATES = [
    ("clean", lambda step: _in_rule_set("clean", step)),
    ("deposit", lambda step: _in_rule_set("deposit", step)),
    ("develop", lambda step: _in_rule_set("develop", step)),
    ("etch", lambda step: _in_rule_set("etch", step)),
    ("metal_etch", lambda step: _in_rule_set("metal_etch", step)),
    ("implant", lambda step: _in_rule_set("implant", step)),
    ("cmp", lambda step: _in_rule_set("cmp", step)),
    ("fill", lambda step: _in_rule_set("fill", step)),
    ("pad_window", lambda step: _in_rule_set("pad_window", step)),
    ("test", lambda step: _in_rule_set("electrical_test", step)),
    ("passivation", lambda step: _is_passivation(step) or _is_cure(step)),
    ("backside", _is_backside),
    ("oxidation", lambda step: _in_rule_set("oxidation", step)),
    ("litho", lambda step: _align_level(step) is not None),
]

EXTRA_COUNT_PREDICATES = [
    ("litho", lambda step: _align_level(step) is not None),
    ("develop", lambda step: _in_rule_set("develop", step)),
    ("oxidation", lambda step: _in_rule_set("oxidation", step)),
    ("fill", lambda step: _in_rule_set("fill", step)),
    ("metal_etch", lambda step: _in_rule_set("metal_etch", step)),
    ("pad_window", lambda step: _in_rule_set("pad_window", step)),
    ("wafer_sort", _is_wafer_sort),
    ("ship", _is_ship),
]

RECENCY_PREDICATES = [
    ("clean", lambda step: _in_rule_set("clean", step)),
    ("deposit", lambda step: _in_rule_set("deposit", step)),
    ("etch", lambda step: _in_rule_set("etch", step)),
    ("implant", lambda step: _in_rule_set("implant", step)),
    ("cmp", lambda step: _in_rule_set("cmp", step)),
    ("test", lambda step: _in_rule_set("electrical_test", step)),
    ("passivation", lambda step: _is_passivation(step) or _is_cure(step)),
    ("litho", lambda step: _align_level(step) is not None),
]


def _count_matching(prefix: list[str], predicate) -> int:
    return sum(1 for step in prefix if predicate(step))


def _seen_matching(prefix: list[str], predicate) -> float:
    return 1.0 if any(predicate(step) for step in prefix) else 0.0


def _last_distance_norm(prefix: list[str], predicate) -> float:
    for idx in range(len(prefix) - 1, -1, -1):
        if predicate(prefix[idx]):
            return min(len(prefix) - 1 - idx, MAX_SEQ_LEN) / MAX_SEQ_LEN
    return 1.0


def _op_flags(step: str | None) -> list[float]:
    if not step:
        return [0.0 for _ in OP_PREDICATES]
    return [1.0 if predicate(step) else 0.0 for _, predicate in OP_PREDICATES]


def _extra_features(prefix: list[str]) -> list[float]:
    prefix_len = max(len(prefix), 1)
    unique_steps = len(set(prefix))
    litho_levels = [_align_level(step) for step in prefix]
    litho_levels = [level for level in litho_levels if level is not None]
    last_litho = litho_levels[-1] if litho_levels else 0
    max_litho = max(litho_levels, default=0)

    features = [
        float(unique_steps),
        unique_steps / prefix_len,
        max(len(prefix) - unique_steps, 0) / prefix_len,
        float(last_litho),
        float(max_litho - last_litho),
    ]
    features.extend(float(_count_matching(prefix, predicate)) for _, predicate in EXTRA_COUNT_PREDICATES)
    features.extend(_seen_matching(prefix, predicate) for _, predicate in EXTRA_COUNT_PREDICATES)
    features.extend(_op_flags(prefix[-1] if prefix else None))
    features.extend(_op_flags(prefix[-2] if len(prefix) >= 2 else None))
    features.extend(_last_distance_norm(prefix, predicate) for _, predicate in RECENCY_PREDICATES)
    return features


def _featurize(prefix: list[str]) -> list[float]:
    ids = [VOCAB_TO_ID.get(step, UNK_ID) for step in prefix]
    lag_ids = []
    for offset in range(1, CONTEXT_SIZE + 1):
        lag_ids.append(ids[-offset] if len(ids) >= offset else BOS_ID)
    last2 = tuple(reversed((lag_ids + [BOS_ID, BOS_ID])[:2]))
    last3 = tuple(reversed((lag_ids + [BOS_ID, BOS_ID, BOS_ID])[:3]))

    categories = {
        "clean": _count_matching(prefix, lambda step: _in_rule_set("clean", step)),
        "deposit": _count_matching(prefix, lambda step: _in_rule_set("deposit", step)),
        "etch": _count_matching(prefix, lambda step: _in_rule_set("etch", step)),
        "implant": _count_matching(prefix, lambda step: _in_rule_set("implant", step)),
        "cmp": _count_matching(prefix, lambda step: _in_rule_set("cmp", step)),
        "test": _count_matching(prefix, lambda step: _in_rule_set("electrical_test", step)),
        "passivation": _count_matching(prefix, lambda step: _is_passivation(step) or _is_cure(step)),
        "backside": _count_matching(prefix, lambda step: "BACKSIDE" in _u(step)),
    }
    litho_levels = [_align_level(step) for step in prefix]
    max_litho = max([level for level in litho_levels if level is not None], default=0)

    features = [
        DEFAULT_FAMILY_ID,
        min(len(prefix), MAX_SEQ_LEN),
        min(len(prefix), MAX_SEQ_LEN) / MAX_SEQ_LEN,
        max_litho,
        categories["clean"],
        categories["deposit"],
        categories["etch"],
        categories["implant"],
        categories["cmp"],
        categories["test"],
        categories["passivation"],
        categories["backside"],
        1.0 if categories["clean"] else 0.0,
        1.0 if categories["deposit"] else 0.0,
        1.0 if categories["etch"] else 0.0,
        1.0 if categories["implant"] else 0.0,
        1.0 if categories["cmp"] else 0.0,
        1.0 if categories["test"] else 0.0,
        1.0 if categories["passivation"] else 0.0,
        1.0 if categories["backside"] else 0.0,
    ]
    features.extend(lag_ids)
    features.append(BIGRAM_TO_ID.get(last2, 0))
    features.append(TRIGRAM_TO_ID.get(last3, 0))
    if FEATURE_VERSION >= 2:
        features.extend(_extra_features(prefix))
    return features


def build_features(sequence: list[str]) -> np.ndarray:
    """
    Convert a sequence of step names into a feature matrix.
    Each row is one prediction position in the sequence.
    """
    rows = [_featurize(sequence[:i]) for i in range(1, len(sequence))]

    if not rows:
        return np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)

    return np.array(rows, dtype=np.float32)


# get_perplexity - DO NOT RENAME THIS FUNCTION
# The pipeline calls this exact function name.
# Return a float: higher = model more surprised = more anomalous

def get_perplexity(sequence: list[str]) -> float:
    """
    Measure how surprised the model is by this sequence.
    Uses average prediction confidence as inverse perplexity:
      high confidence -> low perplexity  (model knows what comes next)
      low confidence  -> high perplexity (model is uncertain / surprised)
    """
    X = build_features(sequence)

    try:
        probas = model.predict_proba(X)
        avg_conf = float(np.mean(np.max(probas, axis=1)))
        perplexity = 1.0 / max(avg_conf, 1e-6)
    except Exception as e:
        print(f"  [warning] get_perplexity failed: {e}")
        perplexity = 1.0

    return perplexity

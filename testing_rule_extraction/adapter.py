"""
my_model_adapter.py
====================
Template for plugging ANY model into rule_probe_pipeline.py.

Your teammate fills this in with their own model loading and
feature engineering code. Nothing else needs to be shared.

Required:
    - get_perplexity(sequence) function  ← must exist, pipeline calls this
    - MODEL_NAME string                  ← shows in the HTML report
    - MODEL_INFO dict                    ← shows in the HTML report

Usage:
    python rule_probe_pipeline.py --model xgboost --adapter_path ./my_model_adapter.py
"""

# ── Fill in your model name and info ──────────────────────────────────────────

MODEL_NAME = "XGBoost (teammate)"  # change to whatever you want

MODEL_INFO = {
    "model_type": "xgboost",
    "n_features": 168,  # update with your actual feature count
    "n_classes": 27,  # update with your actual class count
    "trained_on": "IGBT + IC",
    "ood_family": "MOSFET",
}

# ── Load your model here ───────────────────────────────────────────────────────
# This code runs once when the adapter is imported

import numpy as np
import xgboost as xgb
import json

print(f"Loading model for: {MODEL_NAME}")

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")  # path to your saved model

with open("xgb_class_map.json") as f:
    class_map = json.load(f)  # your category -> int mapping

print(f"  Model loaded | classes: {len(class_map)}")


# ── Your feature engineering ───────────────────────────────────────────────────
# Copy your feature building code here exactly as it was during training
# The pipeline calls get_perplexity(sequence) for each probe sequence

def build_features(sequence: list[str]) -> np.ndarray:
    """
    Convert a sequence of step names into a feature matrix.
    Each row = one prediction position in the sequence.

    IMPORTANT: use the EXACT same feature engineering as during training.
    If you used categories, use the same category map.
    If you used one-hot encoding, use the same vocabulary.
    """
    rows = []
    n = len(sequence)

    for i in range(1, n):
        row = []

        # ── Replace the feature engineering below with your own ───────────────

        # Example: last 5 step names one-hot encoded
        # (replace with whatever features your model was trained on)
        for offset in range(1, 6):
            step = sequence[i - offset] if i >= offset else "<BOS>"
            # encode step however your model expects
            # e.g. category index, one-hot, raw string hash, etc.
            row.append(class_map.get(step, 0))  # placeholder

        # Example: position feature
        row.append(i / max(n, 1))

        # ── End of your feature engineering ──────────────────────────────────

        rows.append(row)

    if not rows:
        return np.zeros((1, len(rows[0]) if rows else 1), dtype=np.float32)

    return np.array(rows, dtype=np.float32)


# ── get_perplexity — DO NOT RENAME THIS FUNCTION ──────────────────────────────
# The pipeline calls this exact function name.
# Return a float: higher = model more surprised = more anomalous

def get_perplexity(sequence: list[str]) -> float:
    """
    Measure how surprised the model is by this sequence.
    Uses average prediction confidence as inverse perplexity:
      high confidence → low perplexity  (model knows what comes next)
      low confidence  → high perplexity (model is uncertain / surprised)
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
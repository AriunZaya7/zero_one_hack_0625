#!/usr/bin/env python3
"""Solution 21: template-boosted evidence bridge.

This solution keeps Solution 20 as the submit-ready prediction cascade, then
adds the trained-model experiment needed for a final hybrid OOD strategy:

1. Train an XGBoost next-step model on raw exact strings.
2. Train the same XGBoost model after replacing family-specific prefixes with
   `__FAMILY__`.
3. Evaluate both on the same ten unseen synthetic families.

The goal is not to replace the evidence cascade with XGBoost. The goal is to
prove that a learned model needs the same template normalization as our
retrieval/grammar fallback if it must emit exact strings for a new family name.

Run from repo root:
    python -B solutions/solution_21_template_boosted_bridge/solution.py
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np
from xgboost import DMatrix, XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
BRIDGE_CONTEXT_SIZE = 12
BRIDGE_BLOCK_CONTEXT_SIZE = 6
BRIDGE_ESTIMATORS = 45
BRIDGE_TRAIN_COUNTS = (2, 5, 10, 20, 40, 60, 80, 100)
BRIDGE_SEEDS = tuple(range(10))
BRIDGE_MODEL_NAMES = ("raw_xgb_exact", "template_xgb_bridge")
LEVEL_RE = re.compile(r"\bLEVEL\s+(\d+)\b")

sys.path.insert(0, str(ROOT))

from solutions.many_family_scaling_probe import family_scaling_probe as scaling  # noqa: E402
from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402
from solutions.solution_18_family_template_grammar import solution as sol18  # noqa: E402
from solutions.solution_20_paired_length_lattice import solution as sol20  # noqa: E402


NEXTSTEP_FIELDS = sol20.NEXTSTEP_FIELDS
COMPLETION_FIELDS = sol20.COMPLETION_FIELDS
ANOMALY_FIELDS = sol20.ANOMALY_FIELDS
BRIDGE_METRIC_FIELDS = [
    "phase",
    "model",
    "seed",
    "train_family_count",
    "train_sequences",
    "train_examples",
    "label_count",
    "validation_rows",
    "truth_label_coverage",
    "family_specific_truth_share",
    "family_specific_truth_label_coverage",
    "task1_top1",
    "task1_top3",
    "task1_top5",
    "task1_mrr",
    "family_specific_task1_top1",
    "family_specific_task1_top3",
    "family_specific_task1_mrr",
]
EXAMPLE_FIELDS = [
    "EXAMPLE_ID",
    "FAMILY",
    "COMPLETION_FRACTION",
    "TRUTH_NEXT",
    "RAW_RANK_1",
    "RAW_RANK_2",
    "RAW_RANK_3",
    "TEMPLATE_RANK_1",
    "TEMPLATE_RANK_2",
    "TEMPLATE_RANK_3",
    "RAW_TRUTH_LABEL_AVAILABLE",
    "TEMPLATE_TRUTH_LABEL_AVAILABLE",
]
TREE_SHAP_FIELDS = [
    "rank",
    "feature",
    "mean_abs_shap",
    "share",
    "plain_english",
]
XGB_LOSS_FIELDS = [
    "model",
    "seed",
    "iteration",
    "mlogloss",
]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cut_position(total_length: int, fraction: float) -> int:
    return max(1, min(total_length - 1, int(total_length * fraction)))


def family_from_key(key: str) -> str:
    return key.split(":", 1)[0]


def max_level_seen(sequence: list[str]) -> int:
    best = 0
    for step in sequence:
        for match in LEVEL_RE.finditer(step):
            best = max(best, int(match.group(1)))
    return best


def last_level_seen(sequence: list[str]) -> int:
    for step in reversed(sequence):
        matches = [int(match.group(1)) for match in LEVEL_RE.finditer(step)]
        if matches:
            return max(matches)
    return 0


def dedupe_keep_order(items: Iterable[str], k: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
        if len(out) >= k:
            break
    return (out + [""] * k)[:k]


def feature_plain_english(feature: str) -> str:
    if feature.startswith("lag_step_"):
        return "Recent exact/template step token in the visible prefix."
    if feature.startswith("lag_block_"):
        return "Recent high-level process block, such as lithography, etch, clean, or test."
    if feature == "family_id":
        return "Raw family identity. Template mode intentionally makes this mostly unneeded."
    if feature == "prefix_len":
        return "How many steps are already visible in the partial route."
    if feature == "completion_fraction_pct":
        return "Whether the partial was cut around 60% or 80% through the route."
    if feature == "position_norm_200":
        return "Route progress normalized to a roughly 200-step route length."
    if feature == "unique_step_count":
        return "How many distinct steps appeared in the visible prefix."
    if feature == "unique_step_frac":
        return "How diverse the visible prefix is relative to its length."
    if feature == "repeat_step_frac":
        return "How much the visible prefix repeats earlier steps."
    if feature == "family_specific_prefix_steps":
        return "How many visible prefix steps already contain the current family name."
    if feature == "template_prefix_steps":
        return "How many visible prefix steps became __FAMILY__ template steps."
    if feature == "max_mask_level_seen":
        return "Highest lithography mask level observed so far."
    if feature == "last_mask_level_seen":
        return "Most recent lithography mask level observed so far."
    return "Engineered context feature used by the XGBoost bridge."


class XGBNextStepBridge:
    """Small XGBoost next-step model for raw-vs-template OOD testing."""

    def __init__(self, *, template_mode: bool, seed: int) -> None:
        self.template_mode = template_mode
        self.seed = seed
        self.model_name = "template_xgb_bridge" if template_mode else "raw_xgb_exact"
        self.step_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.block_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.family_to_id: dict[str, int] = {"<UNK_FAMILY>": 0}
        self.label_steps: list[str] = []
        self.step_to_label: dict[str, int] = {}
        self.feature_names: list[str] = []
        self.estimator: XGBClassifier | None = None
        self.train_examples = 0
        self.training_logloss: list[float] = []

    def transform_step(self, step: str, family: str) -> str:
        if self.template_mode:
            return sol18.normalize_step(step, family)
        return step

    def transform_sequence(self, sequence: list[str], family: str) -> list[str]:
        if self.template_mode:
            return sol18.normalize_sequence(sequence, family)
        return list(sequence)

    def inverse_step(self, step: str, family: str) -> str:
        if self.template_mode:
            return sol18.denormalize_step(step, family)
        return step

    def _build_metadata(self, train_sequences: dict[str, list[str]]) -> None:
        step_values = set()
        block_values = set()
        label_values = set()
        for key, sequence in sorted(train_sequences.items()):
            family = family_from_key(key)
            transformed = self.transform_sequence(sequence, family)
            step_values.update(transformed)
            block_values.update(base.block_label(step) for step in transformed)
            for fraction in base.FRACTIONS:
                cut = cut_position(len(transformed), fraction)
                label_values.add(transformed[cut])
            if not self.template_mode and family not in self.family_to_id:
                self.family_to_id[family] = len(self.family_to_id)

        for step in sorted(step_values):
            self.step_to_id.setdefault(step, len(self.step_to_id))
        for block in sorted(block_values):
            self.block_to_id.setdefault(block, len(self.block_to_id))
        self.label_steps = sorted(label_values)
        self.step_to_label = {step: idx for idx, step in enumerate(self.label_steps)}
        self.feature_names = (
            [
                "family_id",
                "prefix_len",
                "completion_fraction_pct",
                "position_norm_200",
                "unique_step_count",
                "unique_step_frac",
                "repeat_step_frac",
                "family_specific_prefix_steps",
                "template_prefix_steps",
                "max_mask_level_seen",
                "last_mask_level_seen",
            ]
            + [f"lag_step_{i}" for i in range(BRIDGE_CONTEXT_SIZE, 0, -1)]
            + [f"lag_block_{i}" for i in range(BRIDGE_BLOCK_CONTEXT_SIZE, 0, -1)]
        )

    def _step_id(self, step: str) -> int:
        return self.step_to_id.get(step, self.step_to_id["<UNK>"])

    def _block_id(self, step: str) -> int:
        return self.block_to_id.get(base.block_label(step), self.block_to_id["<UNK>"])

    def features(self, prefix: list[str], family: str, completion_fraction: float) -> list[float]:
        transformed = self.transform_sequence(prefix, family)
        prefix_len = len(transformed)
        unique_count = len(set(transformed))
        last_steps = transformed[-BRIDGE_CONTEXT_SIZE:]
        last_blocks = transformed[-BRIDGE_BLOCK_CONTEXT_SIZE:]
        step_ids = [0] * (BRIDGE_CONTEXT_SIZE - len(last_steps)) + [self._step_id(step) for step in last_steps]
        block_ids = [0] * (BRIDGE_BLOCK_CONTEXT_SIZE - len(last_blocks)) + [
            self._block_id(step) for step in last_blocks
        ]
        family_specific_prefix_steps = sum(1 for step in prefix if step.startswith(f"{family} "))
        template_prefix_steps = sum(1 for step in transformed if step.startswith(f"{sol18.TEMPLATE_FAMILY_PLACEHOLDER} "))
        family_id = 0 if self.template_mode else self.family_to_id.get(family, 0)
        return [
            float(family_id),
            float(prefix_len),
            float(round(completion_fraction * 100)),
            prefix_len / 200.0,
            float(unique_count),
            unique_count / max(prefix_len, 1),
            (prefix_len - unique_count) / max(prefix_len, 1),
            float(family_specific_prefix_steps),
            float(template_prefix_steps),
            float(max_level_seen(transformed)),
            float(last_level_seen(transformed)),
            *[float(value) for value in step_ids],
            *[float(value) for value in block_ids],
        ]

    def fit(self, train_sequences: dict[str, list[str]]) -> "XGBNextStepBridge":
        self._build_metadata(train_sequences)
        x_rows: list[list[float]] = []
        y_rows: list[int] = []
        for key, sequence in sorted(train_sequences.items()):
            family = family_from_key(key)
            transformed = self.transform_sequence(sequence, family)
            for fraction in base.FRACTIONS:
                cut = cut_position(len(transformed), fraction)
                target = transformed[cut]
                label = self.step_to_label.get(target)
                if label is None:
                    continue
                x_rows.append(self.features(sequence[:cut], family, fraction))
                y_rows.append(label)

        if len(set(y_rows)) < 2:
            raise RuntimeError(f"{self.model_name} needs at least two target labels")

        self.train_examples = len(y_rows)
        x_train = np.asarray(x_rows, dtype=np.float32)
        y_train = np.asarray(y_rows, dtype=np.int64)
        self.estimator = XGBClassifier(
            n_estimators=BRIDGE_ESTIMATORS,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.95,
            colsample_bytree=0.95,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            verbosity=0,
        )
        self.estimator.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=False)
        results = self.estimator.evals_result()
        self.training_logloss = [
            float(value)
            for value in results.get("validation_0", {}).get("mlogloss", [])
        ]
        return self

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        x = np.asarray([self.features(prefix, family, completion_fraction)], dtype=np.float32)
        probabilities = self.estimator.predict_proba(x)[0]
        ranked_label_indices = np.argsort(-probabilities)[: max(k * 4, k)]
        ranked_steps = [
            self.inverse_step(self.label_steps[int(label_idx)], family)
            for label_idx in ranked_label_indices
            if int(label_idx) < len(self.label_steps)
        ]
        return dedupe_keep_order(ranked_steps, k)

    def truth_label_available(self, truth_step: str, family: str) -> bool:
        return self.transform_step(truth_step, family) in self.step_to_label

    def feature_importance_rows(self, limit: int = 14) -> list[dict[str, object]]:
        if self.estimator is None:
            return []
        importances = self.estimator.feature_importances_
        rows = []
        for idx in np.argsort(-importances)[:limit]:
            rows.append(
                {
                    "feature": self.feature_names[int(idx)],
                    "importance": float(importances[int(idx)]),
                }
            )
        return rows

    def tree_shap_rows(
        self,
        valid_examples: list[base.ValidExample],
        limit: int = 14,
    ) -> list[dict[str, object]]:
        """Return exact Tree SHAP contribution magnitudes for this XGBoost model.

        XGBoost's ``pred_contribs=True`` computes Tree SHAP values for tree
        ensembles. For the multiclass next-step model we average absolute
        contributions over examples and classes, then show the largest features.
        """
        if self.estimator is None or not valid_examples:
            return []

        x_rows = np.asarray(
            [
                self.features(ex.partial, ex.family, ex.completion_fraction)
                for ex in valid_examples
            ],
            dtype=np.float32,
        )
        matrix = DMatrix(x_rows, feature_names=self.feature_names)
        booster = self.estimator.get_booster()
        try:
            contribs = booster.predict(matrix, pred_contribs=True, strict_shape=True)
        except TypeError:
            contribs = booster.predict(matrix, pred_contribs=True)
        contribs_arr = np.asarray(contribs)
        feature_count = len(self.feature_names)

        if contribs_arr.ndim == 3:
            # Expected strict multiclass shape: examples x classes x features+1.
            feature_contribs = contribs_arr[:, :, :feature_count]
            mean_abs = np.abs(feature_contribs).mean(axis=(0, 1))
        elif contribs_arr.ndim == 2 and contribs_arr.shape[1] == feature_count + 1:
            # Binary/single-output shape: examples x features+1.
            mean_abs = np.abs(contribs_arr[:, :feature_count]).mean(axis=0)
        elif contribs_arr.ndim == 2 and contribs_arr.shape[1] % (feature_count + 1) == 0:
            # Some XGBoost versions flatten multiclass as examples x classes*(features+1).
            class_count = contribs_arr.shape[1] // (feature_count + 1)
            reshaped = contribs_arr.reshape(contribs_arr.shape[0], class_count, feature_count + 1)
            mean_abs = np.abs(reshaped[:, :, :feature_count]).mean(axis=(0, 1))
        else:
            raise RuntimeError(f"Unexpected Tree SHAP contribution shape: {contribs_arr.shape}")

        total = float(mean_abs.sum()) or 1.0
        rows: list[dict[str, object]] = []
        for rank, idx in enumerate(np.argsort(-mean_abs)[:limit], start=1):
            feature = self.feature_names[int(idx)]
            value = float(mean_abs[int(idx)])
            rows.append(
                {
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": value,
                    "share": value / total,
                    "plain_english": feature_plain_english(feature),
                }
            )
        return rows


def train_sequences_for_count(
    all_train: dict[str, dict[str, list[str]]],
    train_family_count: int,
) -> dict[str, list[str]]:
    selected = {
        family: all_train[family]
        for family in scaling.TRAIN_FAMILIES[:train_family_count]
    }
    return scaling.flatten(selected)


def predict_bridge_task1(
    model: XGBNextStepBridge,
    examples: list[base.ValidExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        ranks = model.next_step_ranking(ex.partial, ex.family, ex.completion_fraction, k=5)
        rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "RANK_1": ranks[0],
                "RANK_2": ranks[1],
                "RANK_3": ranks[2],
                "RANK_4": ranks[3],
                "RANK_5": ranks[4],
            }
        )
    return rows


def family_specific_subset(examples: list[base.ValidExample]) -> list[base.ValidExample]:
    return [
        ex for ex in examples
        if ex.truth_next.startswith(f"{ex.family} ")
    ]


def evaluate_bridge_model(
    *,
    phase: str,
    model: XGBNextStepBridge,
    train_family_count: int,
    train_sequences: dict[str, list[str]],
    valid_examples: list[base.ValidExample],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    task1_rows = predict_bridge_task1(model, valid_examples)
    task1 = base.evaluate_task1(task1_rows, valid_examples)
    specific_examples = family_specific_subset(valid_examples)
    specific = base.evaluate_task1(task1_rows, specific_examples) if specific_examples else {}
    truth_label_hits = sum(
        model.truth_label_available(ex.truth_next, ex.family)
        for ex in valid_examples
    )
    specific_truth_label_hits = sum(
        model.truth_label_available(ex.truth_next, ex.family)
        for ex in specific_examples
    )
    metric_row = {
        "phase": phase,
        "model": model.model_name,
        "seed": model.seed,
        "train_family_count": train_family_count,
        "train_sequences": len(train_sequences),
        "train_examples": model.train_examples,
        "label_count": len(model.label_steps),
        "validation_rows": len(valid_examples),
        "truth_label_coverage": truth_label_hits / max(len(valid_examples), 1),
        "family_specific_truth_share": len(specific_examples) / max(len(valid_examples), 1),
        "family_specific_truth_label_coverage": specific_truth_label_hits / max(len(specific_examples), 1),
        "task1_top1": task1["top1"],
        "task1_top3": task1["top3"],
        "task1_top5": task1["top5"],
        "task1_mrr": task1["mrr"],
        "family_specific_task1_top1": specific.get("top1", float("nan")),
        "family_specific_task1_top3": specific.get("top3", float("nan")),
        "family_specific_task1_mrr": specific.get("mrr", float("nan")),
    }
    return metric_row, task1_rows


def summarize_bridge_seed_rows(seed_rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    metrics = [
        "truth_label_coverage",
        "family_specific_truth_label_coverage",
        "task1_top1",
        "task1_top3",
        "task1_top5",
        "task1_mrr",
        "family_specific_task1_top1",
        "family_specific_task1_top3",
        "family_specific_task1_mrr",
    ]
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for model_name in BRIDGE_MODEL_NAMES:
        model_rows = [row for row in seed_rows if row["model"] == model_name]
        summary[model_name] = {}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in model_rows
                if not math.isnan(float(row[metric]))
            ]
            if not values:
                continue
            higher_is_better = True
            best_value = max(values) if higher_is_better else min(values)
            worst_value = min(values) if higher_is_better else max(values)
            best_seed = next(row["seed"] for row in model_rows if float(row[metric]) == best_value)
            worst_seed = next(row["seed"] for row in model_rows if float(row[metric]) == worst_value)
            summary[model_name][metric] = {
                "mean": mean(values),
                "std": pstdev(values),
                "best": best_value,
                "best_seed": best_seed,
                "worst": worst_value,
                "worst_seed": worst_seed,
            }
    return summary


def bridge_examples(
    raw_rows: list[dict[str, object]],
    template_rows: list[dict[str, object]],
    raw_model: XGBNextStepBridge,
    template_model: XGBNextStepBridge,
    valid_examples: list[base.ValidExample],
    limit: int = 24,
) -> list[dict[str, object]]:
    raw_by_id = {str(row["EXAMPLE_ID"]): row for row in raw_rows}
    template_by_id = {str(row["EXAMPLE_ID"]): row for row in template_rows}
    examples = [
        ex for ex in valid_examples
        if ex.truth_next.startswith(f"{ex.family} ")
    ][:limit]
    out: list[dict[str, object]] = []
    for ex in examples:
        raw = raw_by_id[ex.example_id]
        template = template_by_id[ex.example_id]
        out.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "FAMILY": ex.family,
                "COMPLETION_FRACTION": ex.completion_fraction,
                "TRUTH_NEXT": ex.truth_next,
                "RAW_RANK_1": raw["RANK_1"],
                "RAW_RANK_2": raw["RANK_2"],
                "RAW_RANK_3": raw["RANK_3"],
                "TEMPLATE_RANK_1": template["RANK_1"],
                "TEMPLATE_RANK_2": template["RANK_2"],
                "TEMPLATE_RANK_3": template["RANK_3"],
                "RAW_TRUTH_LABEL_AVAILABLE": int(raw_model.truth_label_available(ex.truth_next, ex.family)),
                "TEMPLATE_TRUTH_LABEL_AVAILABLE": int(template_model.truth_label_available(ex.truth_next, ex.family)),
            }
        )
    return out


def run_boosting_bridge_probe() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_train, validation = scaling.generate_family_sequences()
    valid_examples = scaling.build_valid_examples(validation)
    curve_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    feature_importance: list[dict[str, object]] = []
    tree_shap: list[dict[str, object]] = []
    model_checkpoints: list[str] = []
    xgb_loss_rows: list[dict[str, object]] = []

    for train_count in BRIDGE_TRAIN_COUNTS:
        train_sequences = train_sequences_for_count(all_train, train_count)
        for template_mode in (False, True):
            model = XGBNextStepBridge(template_mode=template_mode, seed=0).fit(train_sequences)
            metric_row, _task1_rows = evaluate_bridge_model(
                phase="single_seed_scaling_curve",
                model=model,
                train_family_count=train_count,
                train_sequences=train_sequences,
                valid_examples=valid_examples,
            )
            curve_rows.append(metric_row)

    final_train_count = max(BRIDGE_TRAIN_COUNTS)
    final_train_sequences = train_sequences_for_count(all_train, final_train_count)
    raw_seed0_rows: list[dict[str, object]] | None = None
    template_seed0_rows: list[dict[str, object]] | None = None
    raw_seed0_model: XGBNextStepBridge | None = None
    template_seed0_model: XGBNextStepBridge | None = None

    for seed in BRIDGE_SEEDS:
        for template_mode in (False, True):
            model = XGBNextStepBridge(template_mode=template_mode, seed=seed).fit(final_train_sequences)
            metric_row, task1_rows = evaluate_bridge_model(
                phase="ten_seed_final_count",
                model=model,
                train_family_count=final_train_count,
                train_sequences=final_train_sequences,
                valid_examples=valid_examples,
            )
            seed_rows.append(metric_row)
            if seed == 0 and model.estimator is not None:
                checkpoint_path = OUT_DIR / f"{model.model_name}_seed0_xgboost_model.json"
                model.estimator.save_model(checkpoint_path)
                model_checkpoints.append(str(checkpoint_path.relative_to(ROOT)))
                for iteration, loss in enumerate(model.training_logloss):
                    xgb_loss_rows.append(
                        {
                            "model": model.model_name,
                            "seed": seed,
                            "iteration": iteration,
                            "mlogloss": loss,
                        }
                    )
            if seed == 0 and template_mode:
                template_seed0_rows = task1_rows
                template_seed0_model = model
                feature_importance = model.feature_importance_rows()
                tree_shap = model.tree_shap_rows(valid_examples)
                (OUT_DIR / "template_xgb_feature_names.json").write_text(
                    json.dumps(model.feature_names, indent=2) + "\n",
                    encoding="utf-8",
                )
            if seed == 0 and not template_mode:
                raw_seed0_rows = task1_rows
                raw_seed0_model = model

    if raw_seed0_rows is not None and template_seed0_rows is not None and raw_seed0_model and template_seed0_model:
        example_rows = bridge_examples(
            raw_seed0_rows,
            template_seed0_rows,
            raw_seed0_model,
            template_seed0_model,
            valid_examples,
        )

    summary = summarize_bridge_seed_rows(seed_rows)
    write_csv(OUT_DIR / "template_boosting_curve.csv", BRIDGE_METRIC_FIELDS, curve_rows)
    write_csv(OUT_DIR / "template_boosting_10_seed.csv", BRIDGE_METRIC_FIELDS, seed_rows)
    write_csv(OUT_DIR / "template_boosting_examples.csv", EXAMPLE_FIELDS, example_rows)
    write_csv(OUT_DIR / "template_boosting_tree_shap.csv", TREE_SHAP_FIELDS, tree_shap)
    write_csv(OUT_DIR / "template_boosting_xgb_training_logloss.csv", XGB_LOSS_FIELDS, xgb_loss_rows)
    (OUT_DIR / "template_boosting_bridge.json").write_text(
        json.dumps(
            {
                "curve_rows": curve_rows,
                "seed_rows": seed_rows,
                "seed_summary": summary,
                "feature_importance": feature_importance,
                "tree_shap": tree_shap,
                "examples": example_rows,
                "model_checkpoints": model_checkpoints,
                "xgb_training_logloss_rows": xgb_loss_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "train_counts": list(BRIDGE_TRAIN_COUNTS),
        "seeds": list(BRIDGE_SEEDS),
        "validation_families": list(scaling.VALIDATION_FAMILIES),
        "validation_rows": len(valid_examples),
        "validation_sequences": scaling.VALIDATION_FAMILY_COUNT * scaling.VALIDATION_SEQUENCES_PER_FAMILY,
        "curve_rows": curve_rows,
        "seed_rows": seed_rows,
        "seed_summary": summary,
        "feature_importance": feature_importance,
        "tree_shap": tree_shap,
        "examples": example_rows,
        "model_checkpoints": model_checkpoints,
        "xgb_training_logloss_rows": xgb_loss_rows,
        "outputs": {
            "curve_csv": str((OUT_DIR / "template_boosting_curve.csv").relative_to(ROOT)),
            "ten_seed_csv": str((OUT_DIR / "template_boosting_10_seed.csv").relative_to(ROOT)),
            "examples_csv": str((OUT_DIR / "template_boosting_examples.csv").relative_to(ROOT)),
            "tree_shap_csv": str((OUT_DIR / "template_boosting_tree_shap.csv").relative_to(ROOT)),
            "xgb_training_logloss_csv": str((OUT_DIR / "template_boosting_xgb_training_logloss.csv").relative_to(ROOT)),
            "raw_seed0_checkpoint": str((OUT_DIR / "raw_xgb_exact_seed0_xgboost_model.json").relative_to(ROOT)),
            "template_seed0_checkpoint": str((OUT_DIR / "template_xgb_bridge_seed0_xgboost_model.json").relative_to(ROOT)),
            "template_feature_names": str((OUT_DIR / "template_xgb_feature_names.json").relative_to(ROOT)),
            "bridge_json": str((OUT_DIR / "template_boosting_bridge.json").relative_to(ROOT)),
        },
    }


def run_official_prediction(template_training: sol18.TemplateTrainingBundle | None = None) -> dict[str, object]:
    if not sol13.OFFICIAL_VALID.exists() or not sol13.OFFICIAL_ANOMALY.exists():
        raise FileNotFoundError(
            "Official participant files are missing under tracks/industrial-infineon/participant_files."
        )
    if template_training is None:
        template_training = sol18.build_template_training_bundle(
            sol18.public_sequences(),
            sol18.generated_template_sequences(),
        )
    valid_inputs = sol13.read_official_valid_inputs()
    anomaly_inputs = sol13.read_official_anomaly_inputs()
    model = sol20.PairedLengthLatticeModel(valid_inputs, anomaly_inputs, template_training)

    nextstep_rows = sol20.predict_task1(model, valid_inputs)
    completion_rows = sol20.predict_task2(model, valid_inputs)
    anomaly_rows = sol20.predict_task3(anomaly_inputs)
    audit_rows = model.audit_rows(valid_inputs, nextstep_rows)

    write_csv(OUT_DIR / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(OUT_DIR / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(OUT_DIR / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)
    write_csv(OUT_DIR / "paired_length_guard_audit.csv", sol20.PAIR_AUDIT_FIELDS, audit_rows)

    official_dir = OUT_DIR / "official_submission"
    write_csv(official_dir / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(official_dir / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(official_dir / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)

    coverage = model.coverage(valid_inputs)
    write_csv(
        OUT_DIR / "paired_length_training_manifest.csv",
        sol20.PAIR_MANIFEST_FIELDS,
        sol20.manifest_rows(template_training, coverage),
    )
    source_counts = coverage["prediction_source_counts"]
    validator_valid = sum(1 for row in anomaly_rows if int(row["IS_VALID"]) == 1)
    manifest = {
        "solution": "solution_21_template_boosted_bridge",
        "mode": "official_prediction_with_solution20_evidence_cascade_plus_xgb_bridge_diagnostic",
        "eval_valid": str(sol13.OFFICIAL_VALID.relative_to(ROOT)),
        "eval_anomaly": str(sol13.OFFICIAL_ANOMALY.relative_to(ROOT)),
        "valid_rows": len(valid_inputs),
        "anomaly_rows": len(anomaly_inputs),
        "prediction_source_counts": source_counts,
        "full_route_exact_prefix_coverage": coverage["full_route"]["exact_prefix_coverage"],
        "valid_lattice_coverage": coverage["valid_lattice"]["valid_lattice_coverage"],
        "paired_length_coverage": coverage["paired_length"]["paired_length_coverage"],
        "template_fallback_rows": source_counts.get("family_template_fallback", 0),
        "validator_valid_rows": validator_valid,
        "validator_invalid_rows": len(anomaly_rows) - validator_valid,
        "coverage": coverage,
        "template_training": {
            "public_sequences": template_training.public_sequences,
            "synthetic_template_families": template_training.synthetic_template_families,
            "synthetic_template_sequences": template_training.synthetic_template_sequences,
            "total_sequences": template_training.total_sequences,
            "placeholder": sol18.TEMPLATE_FAMILY_PLACEHOLDER,
        },
        "outputs": {
            "nextstep": str((OUT_DIR / "nextstep.csv").relative_to(ROOT)),
            "completion": str((OUT_DIR / "completion.csv").relative_to(ROOT)),
            "anomaly": str((OUT_DIR / "anomaly.csv").relative_to(ROOT)),
            "paired_length_guard_audit": str((OUT_DIR / "paired_length_guard_audit.csv").relative_to(ROOT)),
            "paired_length_training_manifest": str((OUT_DIR / "paired_length_training_manifest.csv").relative_to(ROOT)),
        },
    }
    for manifest_path in (OUT_DIR / "official_run_manifest.json", official_dir / "official_run_manifest.json"):
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def summary_metric(
    bridge: dict[str, object],
    model_name: str,
    metric: str,
    stat: str = "mean",
) -> float:
    return float(bridge["seed_summary"][model_name][metric][stat])  # type: ignore[index]


def curve_at(
    bridge: dict[str, object],
    model_name: str,
    train_count: int,
    metric: str,
) -> float:
    for row in bridge["curve_rows"]:  # type: ignore[index]
        if row["model"] == model_name and int(row["train_family_count"]) == train_count:
            return float(row[metric])
    raise KeyError((model_name, train_count, metric))


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    official = metrics["official_input_prediction"]
    local = metrics["fair_self_eval"]
    bridge = metrics["template_boosting_bridge"]
    task1 = local["task1_next_step"]
    task2 = local["task2_completion"]
    task3 = local["task3_anomaly"]
    lines = [
        "# Solution 21 Metrics",
        "",
        "## Official Participant Input Run",
        "",
        f"- Official valid rows predicted: {official['valid_rows']}",
        f"- Official anomaly rows predicted: {official['anomaly_rows']}",
        f"- Full-route exact prefix coverage: {official['full_route_exact_prefix_coverage']:.4f}",
        f"- Valid-partial lattice coverage: {official['valid_lattice_coverage']:.4f}",
        f"- Paired-length fallback coverage after stronger gates: {official['paired_length_coverage']:.4f}",
        f"- Prediction source counts: {official['prediction_source_counts']}",
        "",
        "## Local Coupled Self-Eval",
        "",
        f"- Task 1 exact Top-1: {task1['top1']:.4f}",
        f"- Task 1 exact Top-3: {task1['top3']:.4f}",
        f"- Task 2 exact match: {task2['exact_match']:.4f}",
        f"- Task 2 normalized edit distance: {task2['normalized_edit_distance']:.4f}",
        f"- Task 3 accuracy: {task3['accuracy']:.4f}",
        "",
        "## XGBoost Template Bridge Diagnostic",
        "",
        f"- Validation setup: train on 100 synthetic families, validate on {bridge['validation_families']}.",
        f"- Raw XGBoost 10-seed Top-1 mean: {summary_metric(bridge, 'raw_xgb_exact', 'task1_top1'):.4f}.",
        f"- Template XGBoost 10-seed Top-1 mean: {summary_metric(bridge, 'template_xgb_bridge', 'task1_top1'):.4f}.",
        f"- Raw XGBoost family-specific label coverage: {summary_metric(bridge, 'raw_xgb_exact', 'family_specific_truth_label_coverage'):.4f}.",
        f"- Template XGBoost family-specific label coverage: {summary_metric(bridge, 'template_xgb_bridge', 'family_specific_truth_label_coverage'):.4f}.",
        f"- Template XGBoost family-specific Top-1 mean: {summary_metric(bridge, 'template_xgb_bridge', 'family_specific_task1_top1'):.4f}.",
        f"- Seed-0 XGBoost checkpoint files: {', '.join(bridge.get('model_checkpoints', []))}.",
        "",
        "Interpretation: the learned model is only OOD-useful after template normalization. Raw XGBoost cannot emit exact family-specific strings it never saw as labels; template XGBoost learns the reusable suffix and then rewrites `__FAMILY__` to the visible eval family.",
        "",
    ]
    tree_shap_rows = bridge.get("tree_shap", [])
    if tree_shap_rows:
        lines.extend(
            [
                "## Template XGBoost Tree SHAP Explainability",
                "",
                "Tree SHAP is XGBoost's exact Shapley-value contribution method for tree ensembles. The rows below average absolute contribution size over the held-out validation examples and model classes.",
                "",
            ]
        )
        for row in tree_shap_rows[:8]:
            lines.append(
                f"- {row['feature']}: mean abs Tree SHAP {float(row['mean_abs_shap']):.6f} "
                f"({float(row['share']):.4f} share) - {row['plain_english']}"
            )
        lines.append("")
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def html_metric_card(label: str, value: object, note: str) -> str:
    return (
        "<div class=\"metric-card\">"
        f"<strong>{html.escape(str(label))}</strong>"
        f"<span>{html.escape(str(value))}</span>"
        f"<p>{html.escape(note)}</p>"
        "</div>"
    )


def write_explanation_html(metrics: dict[str, object]) -> None:
    bridge = metrics["template_boosting_bridge"]
    official = metrics["official_input_prediction"]
    raw_top1 = summary_metric(bridge, "raw_xgb_exact", "task1_top1")
    template_top1 = summary_metric(bridge, "template_xgb_bridge", "task1_top1")
    raw_family_label = summary_metric(bridge, "raw_xgb_exact", "family_specific_truth_label_coverage")
    template_family_label = summary_metric(bridge, "template_xgb_bridge", "family_specific_truth_label_coverage")
    raw_family_top1 = summary_metric(bridge, "raw_xgb_exact", "family_specific_task1_top1")
    template_family_top1 = summary_metric(bridge, "template_xgb_bridge", "family_specific_task1_top1")
    cards = "\n".join(
        [
            html_metric_card("Raw XGBoost Top-1", fmt(raw_top1), "Ten-seed mean on the same ten unseen synthetic families."),
            html_metric_card("Template XGBoost Top-1", fmt(template_top1), "Same model, same data, but family-prefixed steps are normalized first."),
            html_metric_card("Raw family-label coverage", fmt(raw_family_label), "Can the raw model even choose the exact hidden-family label?"),
            html_metric_card("Template family-label coverage", fmt(template_family_label), "The template label exists, then gets rewritten to the eval family."),
        ]
    )
    feature_rows = "\n".join(
        f"<tr><td>{html.escape(str(row['feature']))}</td><td>{fmt(float(row['importance']), 6)}</td></tr>"
        for row in bridge["feature_importance"]
    )
    tree_shap_rows = list(bridge.get("tree_shap", []))
    max_tree_shap = max((float(row["mean_abs_shap"]) for row in tree_shap_rows), default=1.0)
    tree_shap_bars = "\n".join(
        "<div class=\"shap-row\">"
        "<div class=\"shap-label\">"
        f"<strong>{html.escape(str(row['feature']))}</strong>"
        f"<span>{html.escape(str(row['plain_english']))}</span>"
        "</div>"
        "<div class=\"shap-track\">"
        f"<div class=\"shap-fill\" style=\"--bar-width: {max(3.0, float(row['mean_abs_shap']) / max_tree_shap * 100):.1f}%\"></div>"
        "</div>"
        f"<div class=\"shap-value\">{fmt(float(row['mean_abs_shap']), 5)}</div>"
        "</div>"
        for row in tree_shap_rows[:10]
    )
    tree_shap_table_rows = "\n".join(
        "<tr>"
        f"<td>{int(row['rank'])}</td>"
        f"<td>{html.escape(str(row['feature']))}</td>"
        f"<td>{fmt(float(row['mean_abs_shap']), 6)}</td>"
        f"<td>{fmt(float(row['share']))}</td>"
        f"<td>{html.escape(str(row['plain_english']))}</td>"
        "</tr>"
        for row in tree_shap_rows[:10]
    )
    example_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['EXAMPLE_ID']))}</td>"
        f"<td>{html.escape(str(row['TRUTH_NEXT']))}</td>"
        f"<td>{html.escape(str(row['RAW_RANK_1']))}</td>"
        f"<td>{html.escape(str(row['TEMPLATE_RANK_1']))}</td>"
        f"<td>{html.escape(str(row['RAW_TRUTH_LABEL_AVAILABLE']))}</td>"
        f"<td>{html.escape(str(row['TEMPLATE_TRUTH_LABEL_AVAILABLE']))}</td>"
        "</tr>"
        for row in bridge["examples"][:12]
    )
    curve_rows = "\n".join(
        "<tr>"
        f"<td>{count}</td>"
        f"<td>{fmt(curve_at(bridge, 'raw_xgb_exact', count, 'task1_top1'))}</td>"
        f"<td>{fmt(curve_at(bridge, 'template_xgb_bridge', count, 'task1_top1'))}</td>"
        f"<td>{fmt(curve_at(bridge, 'raw_xgb_exact', count, 'family_specific_truth_label_coverage'))}</td>"
        f"<td>{fmt(curve_at(bridge, 'template_xgb_bridge', count, 'family_specific_truth_label_coverage'))}</td>"
        "</tr>"
        for count in BRIDGE_TRAIN_COUNTS
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Solution 21 - Template-Boosted Evidence Bridge</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #52616b;
      --line: #d7dde2;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --blue: #1e5b8a;
      --teal: #08756f;
      --gold: #9b6a12;
      --red: #9a3d31;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 18px 64px;
    }}
    h1, h2, h3 {{
      line-height: 1.18;
      margin: 0 0 12px;
    }}
    h1 {{ font-size: 34px; }}
    h2 {{ margin-top: 34px; font-size: 25px; }}
    h3 {{ margin-top: 22px; font-size: 19px; }}
    p {{ margin: 0 0 14px; }}
    code {{
      background: #eef2f5;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 18px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .metric-card strong {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .metric-card span {{
      display: block;
      font-size: 28px;
      font-weight: 700;
      margin: 6px 0;
    }}
    .metric-card p {{ color: var(--muted); font-size: 14px; margin: 0; }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .step {{
      border: 1px solid var(--line);
      border-top: 5px solid var(--blue);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 112px;
    }}
    .step:nth-child(2) {{ border-top-color: var(--gold); }}
    .step:nth-child(3) {{ border-top-color: var(--teal); }}
    .step:nth-child(4) {{ border-top-color: var(--red); }}
    .step strong {{ display: block; margin-bottom: 6px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 20px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef3f6; }}
    .callout {{
      border-left: 5px solid var(--teal);
      background: #eef8f7;
      padding: 14px 16px;
      margin: 16px 0;
      border-radius: 0 8px 8px 0;
    }}
    .shap-bars {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
      margin: 14px 0 18px;
    }}
    .shap-row {{
      display: grid;
      grid-template-columns: minmax(220px, 1.4fr) minmax(180px, 2fr) 78px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
    }}
    .shap-label strong {{
      display: block;
      font-size: 14px;
    }}
    .shap-label span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .shap-track {{
      height: 13px;
      background: #e9eef2;
      border-radius: 999px;
      overflow: hidden;
    }}
    .shap-fill {{
      width: var(--bar-width);
      height: 100%;
      background: linear-gradient(90deg, var(--teal), var(--blue));
      border-radius: 999px;
      transform-origin: left center;
      animation: shapGrow 850ms ease-out both;
    }}
    .shap-value {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
      font-size: 13px;
    }}
    @keyframes shapGrow {{
      from {{ transform: scaleX(0); }}
      to {{ transform: scaleX(1); }}
    }}
    ul {{ margin-top: 8px; }}
    @media (max-width: 760px) {{
      .shap-row {{ grid-template-columns: 1fr; }}
      .shap-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Solution 21: Template-Boosted Evidence Bridge</h1>
    <p>
      This solution answers the final modeling question: can we combine a real
      trained model, like XGBoost, with the <code>__FAMILY__</code> template
      idea that made our OOD strategy stronger? The answer is yes, and the
      diagnostic below shows why template normalization is the important part.
    </p>
    <p>
      The submit-ready CSVs still come from the strongest evidence cascade from
      Solution 20: exact full-route memory, valid-partial lattice, paired-length
      completion guard, and template fallback. The XGBoost bridge is included as
      trained-model evidence and as a future path for a final hybrid model.
    </p>
  </section>

  <section class="section">
    <h2>TLDR</h2>
    <div class="metric-grid">
      {cards}
    </div>
    <div class="callout">
      <p>
        Raw XGBoost has the same OOD ceiling as any exact-string classifier:
        if the label <code>THEFOURTHFAMILY JTE DOSE VERIFICATION</code> never
        existed during training, it cannot select that exact label. Template
        XGBoost instead learns <code>__FAMILY__ JTE DOSE VERIFICATION</code>,
        then rewrites it to the visible eval family name.
      </p>
    </div>
    <div class="flow">
      <div class="step"><strong>1. Evidence cascade</strong><p>Use exact full routes and visible valid-partial relationships first, because those are direct evidence.</p></div>
      <div class="step"><strong>2. Raw learner</strong><p>Train XGBoost on exact step strings. This is the plain exact-label baseline.</p></div>
      <div class="step"><strong>3. Template learner</strong><p>Train the same model after replacing family names with <code>__FAMILY__</code>.</p></div>
      <div class="step"><strong>4. OOD check</strong><p>Validate on ten unseen families and compare exact-string recoverability.</p></div>
    </div>
  </section>

  <section class="section">
    <h2>Detailed Explanation</h2>
    <h3>What changed from Solution 20?</h3>
    <p>
      The official prediction path is intentionally conservative. It uses the
      same cascade as Solution 20 and writes the same three required submission
      files:
      <code>nextstep.csv</code>, <code>completion.csv</code>, and
      <code>anomaly.csv</code>. On the released participant inputs it predicts
      {official['valid_rows']} valid rows and {official['anomaly_rows']} anomaly
      rows. The diagnostic coverage is still:
      full-route exact prefix coverage {fmt(float(official['full_route_exact_prefix_coverage']))},
      valid-lattice coverage {fmt(float(official['valid_lattice_coverage']))},
      and paired-length fallback coverage {fmt(float(official['paired_length_coverage']))}.
    </p>
    <p>
      The new part is the trained bridge test. It is deliberately isolated from
      the submit CSVs so that a weaker learned model cannot damage the current
      best submission. Its purpose is to prove what kind of learned model would
      make sense next.
    </p>

    <h3>Why raw XGBoost cannot solve new exact strings</h3>
    <p>
      A multiclass next-step model chooses from labels it saw during training.
      If it trained on <code>SYNTHFAMILY001 SOURCE WINDOW INSPECTION</code> and
      <code>SYNTHFAMILY002 SOURCE WINDOW INSPECTION</code>, those are different
      labels. A new family-specific string is a new label. Raw XGBoost therefore
      has family-specific label coverage {fmt(raw_family_label)}, and its
      family-specific Top-1 is {fmt(raw_family_top1)}.
    </p>

    <h3>Why template XGBoost can transfer</h3>
    <p>
      Template mode changes the label space. Every family-specific training
      label becomes a reusable label like
      <code>__FAMILY__ SOURCE WINDOW INSPECTION</code>. At prediction time, if
      the eval row says the family is <code>THEFOURTHFAMILY</code>, the model's
      template prediction is rewritten to
      <code>THEFOURTHFAMILY SOURCE WINDOW INSPECTION</code>. That gives
      family-specific label coverage {fmt(template_family_label)} and
      family-specific Top-1 {fmt(template_family_top1)} across ten XGBoost
      seeds.
    </p>

    <h3>Scaling curve</h3>
    <p>
      The table below uses one XGBoost seed for each training-family count. The
      validation families are always the same ten held-out families from the
      110-family probe. This lets us see whether more families help, and whether
      the improvement is actually available to a raw exact-string model.
    </p>
    <table>
      <thead>
        <tr><th>Training families</th><th>Raw Top-1</th><th>Template Top-1</th><th>Raw family-label coverage</th><th>Template family-label coverage</th></tr>
      </thead>
      <tbody>{curve_rows}</tbody>
    </table>

    <h3>Feature importance</h3>
    <p>
      These are the largest XGBoost feature importances from the template model
      at the 100-family setting. A junior way to read this table: the model is
      mostly learning from the recent process context and the current position,
      not from the literal hidden family name.
    </p>
    <table>
      <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
      <tbody>{feature_rows}</tbody>
    </table>

    <h3>Tree SHAP explainability</h3>
    <p>
      Feature importance tells us which split features the trees used often.
      Tree SHAP answers a more local question: on the held-out validation rows,
      which features changed the model's score the most? XGBoost computes these
      Tree SHAP values directly for its tree ensemble, so this is a real
      Shapley-style explanation without adding another dependency.
    </p>
    <div class="shap-bars" aria-label="Template XGBoost Tree SHAP bar chart">
      {tree_shap_bars}
    </div>
    <table>
      <thead><tr><th>Rank</th><th>Feature</th><th>Mean abs Tree SHAP</th><th>Share</th><th>Plain-English meaning</th></tr></thead>
      <tbody>{tree_shap_table_rows}</tbody>
    </table>

    <h3>Concrete examples</h3>
    <p>
      These rows show the core difference. The raw model often predicts a known
      training-family exact string, while the template model can output the
      correct new-family exact string because the template label exists.
    </p>
    <table>
      <thead>
        <tr><th>Example</th><th>Truth next step</th><th>Raw rank 1</th><th>Template rank 1</th><th>Raw label available</th><th>Template label available</th></tr>
      </thead>
      <tbody>{example_rows}</tbody>
    </table>

    <h3>How this should influence the final hackathon strategy</h3>
    <p>
      The best final strategy is not "just train XGBoost" and it is not "just
      use retrieval". It is an evidence cascade with a template-normalized
      learned fallback. Direct evidence should win when it exists. When direct
      evidence does not exist, the model should operate in a label space that
      can express new-family exact strings.
    </p>
  </section>
</main>
</body>
</html>
"""
    (Path(__file__).resolve().parent / "explanation.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    synthetic_template_sequences = sol18.generated_template_sequences()
    template_training = sol18.build_template_training_bundle(sol18.public_sequences(), synthetic_template_sequences)
    official_manifest = run_official_prediction(template_training)
    local_metrics = sol20.local_self_eval(synthetic_template_sequences)
    bridge_metrics = run_boosting_bridge_probe()
    metrics = {
        "solution": "solution_21_template_boosted_bridge",
        "seed": SEED,
        "method": {
            "name": "Solution 20 evidence cascade plus XGBoost raw-vs-template OOD bridge diagnostic",
            "official_prediction_path": "solution_20_paired_length_lattice",
            "trained_model": "xgboost.XGBClassifier",
            "template_family_placeholder": sol18.TEMPLATE_FAMILY_PLACEHOLDER,
            "bridge_estimators": BRIDGE_ESTIMATORS,
            "bridge_context_size": BRIDGE_CONTEXT_SIZE,
            "official_hidden_ground_truth_used": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "valid_task_rows": local_metrics["valid_task_rows"],
            "anomaly_task_rows": local_metrics["anomaly_task_rows"],
            "official_eval_available": sol13.OFFICIAL_VALID.exists() and sol13.OFFICIAL_ANOMALY.exists(),
            "train_sequences": local_metrics["train_sequences"],
            "synthetic_template_sequences": len(synthetic_template_sequences),
        },
        "official_input_prediction": official_manifest,
        "fair_self_eval": local_metrics,
        "template_boosting_bridge": bridge_metrics,
    }
    write_metrics(metrics)
    write_explanation_html(metrics)
    print(f"Wrote Solution 21 outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

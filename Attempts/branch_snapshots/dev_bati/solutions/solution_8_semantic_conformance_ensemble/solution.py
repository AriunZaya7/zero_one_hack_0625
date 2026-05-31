#!/usr/bin/env python3
"""Solution 8: semantic conformance ensemble.

This solution keeps Solution 7's next-step and completion specialists, but
replaces the direct validator-oracle Task 3 call with an independent semantic
conformance checker. The checker uses explicit process-order features and step
sets, but it does not call `validate_sequence` during inference.

Run from repo root:
    python -B solutions/solution_8_semantic_conformance_ensemble/solution.py
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_7_monte_carlo_suffix_ensemble import solution as sol7  # noqa: E402
from training_data.generate_sequences import (  # noqa: E402
    BACKSIDE_METAL_STEPS,
    CLEAN_STEPS,
    IMPLANT_OPENER_STEPS,
    IMPLANT_STEPS,
    METAL_ETCH_STEPS,
    PAD_WINDOW_STEPS,
)


@dataclass(frozen=True)
class SemanticViolation:
    rule: str
    step_index: int
    step_name: str
    score_penalty: float
    explanation: str


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def window(sequence: list[str], index: int, size: int) -> list[str]:
    return sequence[max(0, index - size):index]


def any_in_window(sequence: list[str], index: int, size: int, targets: frozenset[str]) -> bool:
    return any(step in targets for step in window(sequence, index, size))


def detect_semantic_violations(sequence: list[str]) -> list[SemanticViolation]:
    """Detect process-order violations without calling the public validator."""
    violations: list[SemanticViolation] = []

    for i, step in enumerate(sequence):
        if step in base.DEPOSITION_STEPS:
            if not any_in_window(sequence, i, 12, CLEAN_STEPS):
                violations.append(
                    SemanticViolation(
                        "RULE_DEP_NO_CLEAN",
                        i,
                        step,
                        0.95,
                        "Deposition-like step has no clean/surface-prep step in the previous 12 steps.",
                    )
                )

    for i, step in enumerate(sequence):
        if step in METAL_ETCH_STEPS:
            w = window(sequence, i, 15)
            has_expose = any(s.startswith("EXPOSE LITHO LEVEL") for s in w)
            has_develop = "DEVELOP PHOTORESIST" in w or "DEVELOP PAD WINDOW" in w
            if not (has_expose and has_develop):
                violations.append(
                    SemanticViolation(
                        "RULE_METAL_ETCH_NO_LITHO",
                        i,
                        step,
                        0.98,
                        "Metal etch lacks a recent expose/develop lithography mask context.",
                    )
                )

    for i, step in enumerate(sequence):
        if step in base.ETCH_STEPS:
            w = window(sequence, i, 12)
            if "DEVELOP PHOTORESIST" not in w and "DEVELOP PAD WINDOW" not in w:
                violations.append(
                    SemanticViolation(
                        "RULE_ETCH_NO_MASK",
                        i,
                        step,
                        0.98,
                        "Patterned etch lacks a recently developed resist or pad-window mask.",
                    )
                )

    align_steps: list[tuple[int, int]] = []
    for i, step in enumerate(sequence):
        if step.startswith("ALIGN MASK LEVEL "):
            level_text = step.split("ALIGN MASK LEVEL ", 1)[1]
            if level_text.isdigit():
                align_steps.append((i, int(level_text)))
    for (_, previous_level), (i, current_level) in zip(align_steps, align_steps[1:]):
        if current_level > previous_level + 1 or current_level < previous_level:
            violations.append(
                SemanticViolation(
                    "RULE_LITHO_LEVEL_SKIP",
                    i,
                    sequence[i],
                    0.90,
                    "Lithography mask levels are not monotonically sequential.",
                )
            )

    for i, step in enumerate(sequence):
        if step in IMPLANT_STEPS:
            if not any_in_window(sequence, i, 15, IMPLANT_OPENER_STEPS):
                violations.append(
                    SemanticViolation(
                        "RULE_IMPLANT_NO_MASK",
                        i,
                        step,
                        0.93,
                        "Implant lacks a recent opened implant window or developed resist context.",
                    )
                )

    for i, step in enumerate(sequence):
        if step in base.CMP_STEPS:
            if not any_in_window(sequence, i, 6, base.FILL_STEPS):
                violations.append(
                    SemanticViolation(
                        "RULE_CMP_NO_DEP",
                        i,
                        step,
                        0.94,
                        "CMP appears without a recent deposition or fill step to planarize.",
                    )
                )

    passivation_dep_idx: int | None = None
    cure_passivation_idx: int | None = None
    for i, step in enumerate(sequence):
        if step in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"):
            passivation_dep_idx = i
        if step == "CURE PASSIVATION":
            cure_passivation_idx = i
        if step in PAD_WINDOW_STEPS:
            if (
                passivation_dep_idx is None
                or i < passivation_dep_idx
                or cure_passivation_idx is None
                or i < cure_passivation_idx
            ):
                violations.append(
                    SemanticViolation(
                        "RULE_PAD_OPEN_BEFORE_DEP",
                        i,
                        step,
                        0.94,
                        "Pad window appears before passivation has been deposited and cured.",
                    )
                )

    cure_idx = next((i for i, step in enumerate(sequence) if step == "CURE PASSIVATION"), None)
    for i, step in enumerate(sequence):
        if step in base.ELECTRICAL_TEST_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append(
                    SemanticViolation(
                        "RULE_TEST_BEFORE_PASSIVATION",
                        i,
                        step,
                        0.96,
                        "Electrical testing appears before passivation cure.",
                    )
                )

    ship_idx = next((i for i, step in enumerate(sequence) if step == "SHIP LOT"), None)
    sort_idx = next((i for i, step in enumerate(sequence) if step == "WAFER SORT TEST"), None)
    if ship_idx is not None and (sort_idx is None or ship_idx < sort_idx):
        violations.append(
            SemanticViolation(
                "RULE_SHIP_BEFORE_TEST",
                ship_idx,
                "SHIP LOT",
                0.99,
                "Lot is shipped before wafer sort testing is complete.",
            )
        )

    for i, step in enumerate(sequence):
        if step in BACKSIDE_METAL_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append(
                    SemanticViolation(
                        "RULE_BACKSIDE_BEFORE_PASSIVATION",
                        i,
                        step,
                        0.97,
                        "Backside metal is deposited before frontside passivation cure.",
                    )
                )

    return sorted(violations, key=lambda violation: violation.step_index)


def predict_task3_semantic(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        violations = detect_semantic_violations(ex.sequence)
        if not violations:
            rows.append(
                {
                    "EXAMPLE_ID": ex.example_id,
                    "IS_VALID": 1,
                    "SCORE": 1.0,
                    "PREDICTED_RULE": "",
                }
            )
            continue

        top = violations[0]
        rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "IS_VALID": 0,
                "SCORE": max(0.0, 1.0 - top.score_penalty),
                "PREDICTED_RULE": top.rule,
            }
        )
    return rows


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task1 = metrics["task1_next_step"]
    task2 = metrics["task2_completion"]
    task3 = metrics["task3_anomaly"]
    lines = [
        "# Solution 8 Metrics",
        "",
        "## Method",
        "",
        "- Task 1 and Task 2 reuse Solution 7's task-specialized ensemble.",
        "- Task 3 uses an independent semantic conformance checker instead of calling `validate_sequence` at inference time.",
        "- The semantic checker is based on explicit windows, process-order features, and the local rule-mining notes.",
        "",
        "## Task 1",
        "",
        f"- Top-1: {task1['top1']:.4f}",
        f"- Top-3: {task1['top3']:.4f}",
        f"- Top-5: {task1['top5']:.4f}",
        f"- MRR: {task1['mrr']:.4f}",
        "",
        "## Task 2",
        "",
        f"- Exact match: {task2['exact_match']:.4f}",
        f"- Normalized edit distance: {task2['normalized_edit_distance']:.4f}",
        f"- Token accuracy: {task2['token_accuracy']:.4f}",
        f"- Block accuracy: {task2['block_accuracy']:.4f}",
        "",
        "## Task 3",
        "",
        f"- Accuracy: {task3['accuracy']:.4f}",
        f"- ROC-AUC: {task3['roc_auc_valid_probability']:.4f}",
        f"- Rule attribution accuracy: {task3['rule_attribution_accuracy']:.4f}",
        "",
        "## Interpretation",
        "",
        "This solution shows that the anomaly task can be made explainable without directly calling the validator oracle in the prediction path.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = sol7.TaskSpecializedMonteCarloModel(train)

    task1_rows = sol7.predict_task1(model, valid_examples)
    task2_rows = sol7.predict_task2(model, valid_examples)
    task3_rows = predict_task3_semantic(anomaly_examples)

    write_csv(
        OUT_DIR / "nextstep.csv",
        ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"],
        task1_rows,
    )
    write_csv(OUT_DIR / "completion.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], task2_rows)
    write_csv(
        OUT_DIR / "anomaly.csv",
        ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"],
        task3_rows,
    )

    metrics = {
        "solution": "solution_8_semantic_conformance_ensemble",
        "seed": SEED,
        "method": {
            "name": "Solution 7 task specialists plus independent semantic conformance checker",
            "task1_specialist": "solution_2_eval_aware_retrieval",
            "task2_specialist": "solution_7_monte_carlo_suffix_library",
            "task3_specialist": "semantic_conformance_checker_without_validate_sequence_inference",
            "task2_uses_truth_remainder_length": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences": len(train),
            "completion_library_sequences": model.completion_library_size,
            "valid_task_rows": len(valid_examples),
            "anomaly_task_rows": len(anomaly_examples),
            "official_eval_available": False,
        },
        "data_inventory": sequence_inventory,
        "task1_next_step": base.evaluate_task1(task1_rows, valid_examples),
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_examples),
        "task2_completion": base.evaluate_task2(task2_rows, valid_examples),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_examples),
        "task4_ood_proxy_next_step": sol2.evaluate_ood_proxy(by_family),
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

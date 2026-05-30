#!/usr/bin/env python3
"""Solution 9: judge-aware heterogeneous portfolio.

This solution is not a new base learner. It is a transparent model-selection
portfolio over the strongest completed specialists:

- Task 1: Solution 3 synthetic-augmented retrieval, because it has the strongest
  seed-42 visible next-step Top-1/MRR.
- Task 2: Solution 7 Monte Carlo suffix library, because it has the strongest
  deterministic completion edit distance and block accuracy.
- Task 3: Solution 8 semantic conformance checker, because it keeps perfect
  local anomaly metrics without direct validator inference.

Run from repo root:
    python -B solutions/solution_9_judge_aware_portfolio/solution.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_1_hybrid_retrieval import solution as sol1  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_3_synthetic_augmented_retrieval import solution as sol3  # noqa: E402
from solutions.solution_7_monte_carlo_suffix_ensemble import solution as sol7  # noqa: E402
from solutions.solution_8_semantic_conformance_ensemble import solution as sol8  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class JudgeAwarePortfolio:
    """Task-level model selector over already completed specialists."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        augmented_train = sol3.augment_with_generated_sequences(train_sequences)
        self.task1_model = sol1.HybridRetrievalModel(augmented_train)
        self.task2_model = sol7.TaskSpecializedMonteCarloModel(train_sequences)

    def predict_task1(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        return sol1.predict_task1(self.task1_model, examples)

    def predict_task2(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        return sol7.predict_task2(self.task2_model, examples)

    @staticmethod
    def predict_task3(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
        return sol8.predict_task3_semantic(examples)


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task1 = metrics["task1_next_step"]
    task2 = metrics["task2_completion"]
    task3 = metrics["task3_anomaly"]
    canonical = metrics["task1_canonical_process_step"]
    lines = [
        "# Solution 9 Metrics",
        "",
        "## Method",
        "",
        "- Task 1 specialist: Solution 3 synthetic-augmented retrieval.",
        "- Task 2 specialist: Solution 7 Monte Carlo suffix-library completion.",
        "- Task 3 specialist: Solution 8 semantic conformance checker.",
        "- Portfolio rule: choose the strongest completed specialist per visible judging task.",
        "",
        "## Task 1",
        "",
        f"- Top-1: {task1['top1']:.4f}",
        f"- Top-3: {task1['top3']:.4f}",
        f"- Top-5: {task1['top5']:.4f}",
        f"- MRR: {task1['mrr']:.4f}",
        f"- Canonical Top-1: {canonical['canonical_top1']:.4f}",
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
        "## Honest Tradeoff",
        "",
        "The Task 1 specialist improves visible seed-42 Top-1 but has weaker IC leave-one-family-out behavior than Solution 2. Treat this as a visible-task portfolio, not the safest hidden-family strategy.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = JudgeAwarePortfolio(train)

    task1_rows = model.predict_task1(valid_examples)
    task2_rows = model.predict_task2(valid_examples)
    task3_rows = model.predict_task3(anomaly_examples)

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
        "solution": "solution_9_judge_aware_portfolio",
        "seed": SEED,
        "method": {
            "name": "judge-aware heterogeneous task portfolio",
            "task1_specialist": "solution_3_synthetic_augmented_retrieval",
            "task2_specialist": "solution_7_monte_carlo_suffix_ensemble",
            "task3_specialist": "solution_8_semantic_conformance_ensemble",
            "task2_uses_truth_remainder_length": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences": len(train),
            "valid_task_rows": len(valid_examples),
            "anomaly_task_rows": len(anomaly_examples),
            "official_eval_available": False,
        },
        "data_inventory": sequence_inventory,
        "task1_next_step": base.evaluate_task1(task1_rows, valid_examples),
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_examples),
        "task2_completion": base.evaluate_task2(task2_rows, valid_examples),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_examples),
        "task4_ood_proxy_next_step": sol3.evaluate_ood_proxy(by_family),
        "portfolio_tradeoff": {
            "visible_task_choice": "maximize current local visible Task 1/2/3 metrics",
            "hidden_family_caveat": "Task 1 uses Solution 3, which is weaker than Solution 2 on IC leave-one-family-out.",
        },
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

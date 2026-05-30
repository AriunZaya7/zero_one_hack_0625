#!/usr/bin/env python3
"""Solution 7: task-specialized Monte Carlo suffix ensemble.

This solution keeps the strongest generalizing next-step model from Solution 2,
then uses a larger grammar-generated suffix library for Task 2 completion.

Run from repo root:
    python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py
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
COMPLETION_SYNTHETIC_PER_FAMILY = 10_000
COMPLETION_SYNTHETIC_SEED_BASE = 18_000

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_1_hybrid_retrieval import solution as sol1  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from training_data.generate_sequences import generate_dataset  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def augment_completion_library(
    train_sequences: dict[str, list[str]],
    count_per_family: int = COMPLETION_SYNTHETIC_PER_FAMILY,
    seed_base: int = COMPLETION_SYNTHETIC_SEED_BASE,
) -> dict[str, list[str]]:
    """Add many valid public-grammar samples for suffix retrieval only."""
    augmented = dict(train_sequences)
    for family_index, family in enumerate(base.FAMILIES):
        generated = generate_dataset(
            family,
            count=count_per_family,
            seed=seed_base + family_index,
            validate=True,
        )
        for i, sequence in enumerate(generated):
            augmented[f"{family}:solution7_completion_mc:{i:05d}"] = sequence
    return augmented


class TaskSpecializedMonteCarloModel:
    """Use separate specialists for ranking and full-suffix completion."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.next_step_model = sol2.EvalAwareRetrievalModel(train_sequences, train_sequences)
        completion_library = augment_completion_library(train_sequences)
        self.completion_model = sol1.HybridRetrievalModel(completion_library)
        self.completion_library_size = len(completion_library)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        return self.next_step_model.next_step_ranking(
            prefix,
            family=family,
            completion_fraction=completion_fraction,
            k=k,
        )

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        return self.completion_model.complete(
            prefix,
            family=family,
            completion_fraction=completion_fraction,
        )


def predict_task1(
    model: TaskSpecializedMonteCarloModel,
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


def predict_task2(
    model: TaskSpecializedMonteCarloModel,
    examples: list[base.ValidExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        suffix = model.complete(ex.partial, ex.family, ex.completion_fraction)
        rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
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
    canonical = metrics["task1_canonical_process_step"]
    lines = [
        "# Solution 7 Metrics",
        "",
        "## Method",
        "",
        "- Task 1 model: Solution 2 eval-aware retrieval.",
        "- Task 2 model: Solution 1 hybrid suffix retrieval trained on public train sequences plus a larger Monte Carlo library from the public grammar.",
        f"- Generated completion sequences per family: {COMPLETION_SYNTHETIC_PER_FAMILY}",
        f"- Completion synthetic seed base: {COMPLETION_SYNTHETIC_SEED_BASE}",
        f"- Total completion library sequences: {metrics['self_eval']['completion_library_sequences']}",
        "- Task 3 model: public symbolic validator oracle.",
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
        f"- Rule attribution accuracy: {task3['rule_attribution_accuracy']:.4f}",
        "",
        "## Interpretation",
        "",
        "This is a task-specialized ensemble. It deliberately avoids using the very large synthetic library for Task 1 because the extra random aliases slightly reduce exact Top-1. For completion, the same larger library helps because the metric rewards a full suffix that is structurally close to the target route.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = TaskSpecializedMonteCarloModel(train)

    task1_rows = predict_task1(model, valid_examples)
    task2_rows = predict_task2(model, valid_examples)
    task3_rows = sol2.predict_task3(anomaly_examples)

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
        "solution": "solution_7_monte_carlo_suffix_ensemble",
        "seed": SEED,
        "method": {
            "name": "eval-aware next-step retrieval plus Monte Carlo suffix-library completion",
            "task1_specialist": "solution_2_eval_aware_retrieval",
            "task2_specialist": "solution_1_hybrid_retrieval_on_generated_suffix_library",
            "completion_synthetic_per_family": COMPLETION_SYNTHETIC_PER_FAMILY,
            "completion_synthetic_seed_base": COMPLETION_SYNTHETIC_SEED_BASE,
            "task2_uses_truth_remainder_length": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences_before_completion_augmentation": len(train),
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

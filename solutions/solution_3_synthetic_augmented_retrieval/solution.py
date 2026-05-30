#!/usr/bin/env python3
"""Solution 3: synthetic-augmented hybrid retrieval.

This attempt uses the public sequence generator as a legitimate source of more
training examples. The hypothesis is simple: if the hidden eval data comes from
the same documented grammar, more generated routes should make retrieval less
dependent on the small public variant files.

Run from repo root:
    python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
SYNTHETIC_PER_FAMILY = 2000
SYNTHETIC_SEED_BASE = 9000

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


def augment_with_generated_sequences(
    train: dict[str, list[str]],
    families: Iterable[str] = base.FAMILIES,
    count_per_family: int = SYNTHETIC_PER_FAMILY,
    seed_base: int = SYNTHETIC_SEED_BASE,
) -> dict[str, list[str]]:
    augmented = dict(train)
    for family_index, family in enumerate(families):
        generated = generate_dataset(
            family,
            count=count_per_family,
            seed=seed_base + family_index,
            validate=True,
        )
        for i, sequence in enumerate(generated):
            augmented[f"{family}:solution3_generated:{i:05d}"] = sequence
    return augmented


def evaluate_ood_proxy(
    by_family: dict[str, dict[str, base.SequenceRecord]],
) -> dict[str, dict[str, float]]:
    """Leave-one-family-out proxy without generating the held-out family."""
    results: dict[str, dict[str, float]] = {}
    for family in base.FAMILIES:
        train: dict[str, list[str]] = {}
        test: dict[str, list[str]] = {}
        train_families: list[str] = []
        for fam in base.FAMILIES:
            if fam != family:
                train_families.append(fam)
            for key, record in by_family[fam].items():
                if fam == family:
                    if record.source_kind == "long_format_sequence":
                        test[key] = record.steps
                else:
                    train[key] = record.steps

        augmented_train = augment_with_generated_sequences(
            train,
            families=train_families,
            count_per_family=1000,
            seed_base=SYNTHETIC_SEED_BASE + 100,
        )
        model = sol1.HybridRetrievalModel(augmented_train)
        rng = random.Random(SEED)
        keys = sorted(test)
        rng.shuffle(keys)
        sampled_test = {key: test[key] for key in keys[:base.EVAL_PER_FAMILY]}
        examples = base.build_valid_examples({family: sampled_test})
        rows = sol1.predict_task1(model, examples)
        results[family] = base.evaluate_task1(rows, examples)
    return results


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
        "# Solution 3 Metrics",
        "",
        "## Method",
        "",
        f"- Synthetic generated sequences per family: {SYNTHETIC_PER_FAMILY}",
        f"- Total train sequences after augmentation: {metrics['self_eval']['train_sequences_after_augmentation']}",
        "- Model: Solution 1 hybrid retrieval trained on public plus generated grammar data.",
        "",
        "## Task 1",
        "",
        f"- Top-1: {task1['top1']:.4f}",
        f"- Top-3: {task1['top3']:.4f}",
        f"- Top-5: {task1['top5']:.4f}",
        f"- MRR: {task1['mrr']:.4f}",
        f"- Diagnostic-only canonical Top-1: {canonical['canonical_top1']:.4f}",
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
        "## Caveat",
        "",
        "More synthetic data does not remove randomized exact-label ambiguity, so this is mainly a grammar-coverage attempt.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    augmented_train = augment_with_generated_sequences(train)
    model = sol1.HybridRetrievalModel(augmented_train)

    task1_rows = sol1.predict_task1(model, valid_examples)
    task2_rows = sol1.predict_task2(model, valid_examples)
    task3_rows = sol1.predict_task3(anomaly_examples)

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
        "solution": "solution_3_synthetic_augmented_retrieval",
        "seed": SEED,
        "method": {
            "name": "public generator augmentation + hybrid retrieval",
            "synthetic_per_family": SYNTHETIC_PER_FAMILY,
            "synthetic_seed_base": SYNTHETIC_SEED_BASE,
            "task2_uses_truth_remainder_length": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences_before_augmentation": len(train),
            "train_sequences_after_augmentation": len(augmented_train),
            "valid_task_rows": len(valid_examples),
            "anomaly_task_rows": len(anomaly_examples),
            "official_eval_available": False,
        },
        "data_inventory": sequence_inventory,
        "task1_next_step": base.evaluate_task1(task1_rows, valid_examples),
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_examples),
        "task2_completion": base.evaluate_task2(task2_rows, valid_examples),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_examples),
        "task4_ood_proxy_next_step": evaluate_ood_proxy(by_family),
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

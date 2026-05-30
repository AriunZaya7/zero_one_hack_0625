#!/usr/bin/env python3
"""Solution 4: length-aware completion reranker.

This attempt keeps Solution 2's next-step strategy but changes Task 2. It learns
the likely hidden remainder length from public training cuts and trims retrieved
suffixes to that length. The goal is not exact sequence reconstruction; it is to
improve token/block alignment for the documented completion metrics.

Run from repo root:
    python -B solutions/solution_4_length_aware_completion/solution.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class LengthAwareCompletionModel:
    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.base_model = sol2.EvalAwareRetrievalModel(train_sequences, train_sequences)
        self.length_stats: dict[tuple[object, ...], Counter[int]] = defaultdict(Counter)
        self._fit_lengths(train_sequences)

    def _fit_lengths(self, train_sequences: dict[str, list[str]]) -> None:
        for key, sequence in train_sequences.items():
            family = key.split(":", 1)[0]
            for fraction in base.FRACTIONS:
                cut = max(1, min(len(sequence) - 1, int(len(sequence) * fraction)))
                remainder_length = len(sequence) - cut
                self.length_stats[("family_fraction_cut", family, fraction, cut)][remainder_length] += 1
                self.length_stats[("fraction_cut", fraction, cut)][remainder_length] += 1
                self.length_stats[("family_fraction", family, fraction)][remainder_length] += 1
                self.length_stats[("fraction", fraction)][remainder_length] += 1

    def expected_remainder_length(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> int:
        cut = len(prefix)
        keys = [
            ("family_fraction_cut", family, completion_fraction, cut),
            ("fraction_cut", completion_fraction, cut),
            ("family_fraction", family, completion_fraction),
            ("fraction", completion_fraction),
        ]
        for key in keys:
            counts = self.length_stats.get(key)
            if counts:
                return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        estimated_total = round(cut / max(completion_fraction, 0.01))
        return max(1, estimated_total - cut)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        return self.base_model.next_step_ranking(prefix, family, completion_fraction, k=k)

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        suffix = self.base_model.complete(prefix, family, completion_fraction)
        target_length = self.expected_remainder_length(prefix, family, completion_fraction)
        if len(suffix) > target_length:
            return suffix[:target_length]
        return suffix


def predict_task1(
    model: LengthAwareCompletionModel,
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
    model: LengthAwareCompletionModel,
    examples: list[base.ValidExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        suffix = model.complete(ex.partial, ex.family, ex.completion_fraction)
        rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
    return rows


def evaluate_ood_proxy(
    by_family: dict[str, dict[str, base.SequenceRecord]],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for family in base.FAMILIES:
        train: dict[str, list[str]] = {}
        test: dict[str, list[str]] = {}
        for fam in base.FAMILIES:
            for key, record in by_family[fam].items():
                if fam == family:
                    if record.source_kind == "long_format_sequence":
                        test[key] = record.steps
                else:
                    train[key] = record.steps

        model = LengthAwareCompletionModel(train)
        rng = random.Random(SEED)
        keys = sorted(test)
        rng.shuffle(keys)
        sampled_test = {key: test[key] for key in keys[:base.EVAL_PER_FAMILY]}
        examples = base.build_valid_examples({family: sampled_test})
        rows = predict_task1(model, examples)
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
        "# Solution 4 Metrics",
        "",
        "## Method",
        "",
        "- Task 1: Solution 2 eval-aware retrieval.",
        "- Task 2: retrieved suffix trimmed to the most likely remainder length from public training cuts.",
        "- Task 3: public validator oracle.",
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
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = LengthAwareCompletionModel(train)

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
        "solution": "solution_4_length_aware_completion",
        "seed": SEED,
        "method": {
            "name": "eval-aware retrieval + length-aware completion trimming",
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
        "task4_ood_proxy_next_step": evaluate_ood_proxy(by_family),
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate every current solution across multiple split seeds.

The current solutions are deterministic algorithms, not neural models with
random initialization. In this report, a "seed" changes the deterministic local
train/held-out split, anomaly shuffle, and OOD sample. For a fixed split, each
solution gives the same predictions every time.

Run from repo root:
    python -B solutions/evaluate_across_seeds.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "solutions" / "seed_evaluation_outputs"
SEEDS = tuple(range(10))

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as sol0  # noqa: E402
from solutions.solution_1_hybrid_retrieval import solution as sol1  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_3_synthetic_augmented_retrieval import solution as sol3  # noqa: E402
from solutions.solution_4_length_aware_completion import solution as sol4  # noqa: E402
from solutions.solution_5_tuned_rank_ensemble import solution as sol5  # noqa: E402
from solutions.solution_6_alias_calibrated_retrieval import solution as sol6  # noqa: E402
from solutions.solution_7_monte_carlo_suffix_ensemble import solution as sol7  # noqa: E402
from solutions.solution_8_semantic_conformance_ensemble import solution as sol8  # noqa: E402
from solutions.solution_10_confidence_gated_consensus import solution as sol10  # noqa: E402
from solutions.solution_11_ood_guarded_consensus import solution as sol11  # noqa: E402
from training_data.generate_sequences import generate_dataset  # noqa: E402


MetricDict = dict[str, float | int | str]


def set_solution_seed(seed: int) -> None:
    """Patch module-level seeds used by the local self-eval helpers."""
    sol0.SEED = seed
    sol1.SEED = seed
    sol2.SEED = seed
    sol3.SEED = seed
    sol4.SEED = seed
    sol5.SEED = seed
    sol6.SEED = seed
    sol7.SEED = seed
    sol8.SEED = seed
    sol10.SEED = seed
    sol11.SEED = seed


def add_top2(rows: list[dict[str, object]], examples: list[sol0.ValidExample]) -> float:
    hits = 0
    by_id = {row["EXAMPLE_ID"]: row for row in rows}
    for ex in examples:
        row = by_id[ex.example_id]
        ranks = [str(row[f"RANK_{i}"]) for i in range(1, 3)]
        hits += ex.truth_next in ranks
    return hits / max(len(examples), 1)


def average_ood_top1(ood: dict[str, dict[str, float]]) -> float:
    return mean(family_metrics["top1"] for family_metrics in ood.values())


def flatten_common_metrics(
    *,
    seed: int,
    solution_name: str,
    task1: dict[str, float],
    task2: dict[str, float],
    task3: dict[str, float],
    ood: dict[str, dict[str, float]],
    canonical: dict[str, object] | None = None,
    public_lookup: dict[str, float] | None = None,
) -> MetricDict:
    row: MetricDict = {
        "solution": solution_name,
        "seed": seed,
        "seed_interpretation": "split_seed_only_model_deterministic",
        "task1_top1": task1["top1"],
        "task1_top2": task1["top2"],
        "task1_top3": task1["top3"],
        "task1_top5": task1["top5"],
        "task1_mrr": task1["mrr"],
        "task2_exact_match": task2["exact_match"],
        "task2_normalized_edit_distance": task2["normalized_edit_distance"],
        "task2_token_accuracy": task2["token_accuracy"],
        "task2_block_accuracy": task2["block_accuracy"],
        "task3_accuracy": task3["accuracy"],
        "task3_f1_valid": task3["f1_valid"],
        "task3_roc_auc_valid_probability": task3["roc_auc_valid_probability"],
        "task3_rule_attribution_accuracy": task3["rule_attribution_accuracy"],
        "ood_avg_top1": average_ood_top1(ood),
    }
    for family, family_metrics in sorted(ood.items()):
        row[f"ood_{family}_top1"] = family_metrics["top1"]
        row[f"ood_{family}_mrr"] = family_metrics["mrr"]
    if canonical is not None:
        row["task1_canonical_top1"] = float(canonical["canonical_top1"])
        row["task1_canonical_top2"] = float(canonical["canonical_top2"])
        row["task1_same_canonical_miss_rate"] = float(
            canonical["same_canonical_miss_rate_among_exact_misses"]
        )
    else:
        row["task1_canonical_top1"] = ""
        row["task1_canonical_top2"] = ""
        row["task1_same_canonical_miss_rate"] = ""
    if public_lookup is not None:
        row["public_lookup_coverage"] = public_lookup["coverage"]
        row["public_lookup_exact_next_when_covered"] = public_lookup[
            "exact_next_hit_rate_when_covered"
        ]
    else:
        row["public_lookup_coverage"] = ""
        row["public_lookup_exact_next_when_covered"] = ""
    return row


def load_seed_data(seed: int) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, list[str]]],
    list[sol0.ValidExample],
    list[sol0.AnomalyExample],
    dict[str, dict[str, sol0.SequenceRecord]],
]:
    set_solution_seed(seed)
    train, heldout, _inventory, by_family = sol0.load_split()
    valid_examples = sol0.build_valid_examples(heldout)
    anomaly_examples = sol0.build_anomaly_examples(heldout)
    return train, heldout, valid_examples, anomaly_examples, by_family


def evaluate_solution_0(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol0.NGramModel(n=3, alpha=0.4).fit(train.values())
    task1_rows = sol0.predict_task1(model, valid_examples)
    task2_rows = sol0.predict_task2(model, valid_examples)
    task3_rows = sol0.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    canonical = sol2.evaluate_canonical_task1(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_0_rule_mock",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol0.evaluate_ood_proxy(by_family),
        canonical=canonical,
    )


def evaluate_solution_1(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol1.HybridRetrievalModel(train)
    task1_rows = sol1.predict_task1(model, valid_examples)
    task2_rows = sol1.predict_task2(model, valid_examples)
    task3_rows = sol1.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    canonical = sol2.evaluate_canonical_task1(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_1_hybrid_retrieval",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol1.evaluate_ood_proxy(by_family),
        canonical=canonical,
    )


def evaluate_solution_2(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    fair_model = sol2.EvalAwareRetrievalModel(train, train)
    public_model = sol2.EvalAwareRetrievalModel(train, sol2.all_public_sequences(by_family))
    task1_rows = sol2.predict_task1(fair_model, valid_examples)
    task2_rows = sol2.predict_task2(fair_model, valid_examples)
    task3_rows = sol2.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_2_eval_aware_retrieval",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol2.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
        public_lookup=sol2.evaluate_lookup_coverage(public_model, valid_examples),
    )


def evaluate_solution_3(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    augmented_train = sol3.augment_with_generated_sequences(train)
    model = sol1.HybridRetrievalModel(augmented_train)
    task1_rows = sol1.predict_task1(model, valid_examples)
    task2_rows = sol1.predict_task2(model, valid_examples)
    task3_rows = sol1.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_3_synthetic_augmented_retrieval",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol3.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


def evaluate_solution_4(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol4.LengthAwareCompletionModel(train)
    task1_rows = sol4.predict_task1(model, valid_examples)
    task2_rows = sol4.predict_task2(model, valid_examples)
    task3_rows = sol2.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_4_length_aware_completion",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol4.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


def evaluate_solution_5(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol5.TunedRankEnsembleModel(train)
    task1_rows = sol5.predict_task1(model, valid_examples)
    task2_rows = sol5.predict_task2(model, valid_examples)
    task3_rows = sol1.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_5_tuned_rank_ensemble",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol5.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


def evaluate_solution_6(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol6.AliasCalibratedRetrievalModel(train)
    task1_rows = sol6.predict_task1(model, valid_examples)
    task2_rows = sol6.predict_task2(model, valid_examples)
    task3_rows = sol1.predict_task3(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_6_alias_calibrated_retrieval",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol6.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


_COMPLETION_GENERATED_CACHE: dict[str, list[str]] | None = None
_SOLUTION7_PER_SEED_CACHE: dict[int, dict[str, object]] = {}


def completion_generated_cache() -> dict[str, list[str]]:
    """Reuse Solution 7's deterministic generated suffix library across seeds."""
    global _COMPLETION_GENERATED_CACHE
    if _COMPLETION_GENERATED_CACHE is None:
        generated_sequences: dict[str, list[str]] = {}
        for family_index, family in enumerate(sol0.FAMILIES):
            generated = generate_dataset(
                family,
                count=sol7.COMPLETION_SYNTHETIC_PER_FAMILY,
                seed=sol7.COMPLETION_SYNTHETIC_SEED_BASE + family_index,
                validate=True,
            )
            for i, sequence in enumerate(generated):
                generated_sequences[f"{family}:solution7_completion_mc:{i:05d}"] = sequence
        _COMPLETION_GENERATED_CACHE = generated_sequences
    return _COMPLETION_GENERATED_CACHE


def evaluate_solution_7_like(seed: int) -> dict[str, object]:
    if seed in _SOLUTION7_PER_SEED_CACHE:
        return _SOLUTION7_PER_SEED_CACHE[seed]

    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    next_model = sol2.EvalAwareRetrievalModel(train, train)
    completion_library = dict(train)
    completion_library.update(completion_generated_cache())
    completion_model = sol1.HybridRetrievalModel(completion_library)

    task1_rows = sol2.predict_task1(next_model, valid_examples)
    task2_rows = sol1.predict_task2(completion_model, valid_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)

    result = {
        "train": train,
        "valid_examples": valid_examples,
        "anomaly_examples": anomaly_examples,
        "by_family": by_family,
        "task1_rows": task1_rows,
        "task2_rows": task2_rows,
        "task1": task1,
        "task2": sol0.evaluate_task2(task2_rows, valid_examples),
        "canonical": sol2.evaluate_canonical_task1(task1_rows, valid_examples),
        "ood": sol2.evaluate_ood_proxy(by_family),
    }
    _SOLUTION7_PER_SEED_CACHE[seed] = result
    return result


def evaluate_solution_7(seed: int) -> MetricDict:
    cached = evaluate_solution_7_like(seed)
    anomaly_examples = cached["anomaly_examples"]
    task3_rows = sol2.predict_task3(anomaly_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_7_monte_carlo_suffix_ensemble",
        task1=cached["task1"],
        task2=cached["task2"],
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=cached["ood"],
        canonical=cached["canonical"],
    )


def evaluate_solution_8(seed: int) -> MetricDict:
    cached = evaluate_solution_7_like(seed)
    anomaly_examples = cached["anomaly_examples"]
    task3_rows = sol8.predict_task3_semantic(anomaly_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_8_semantic_conformance_ensemble",
        task1=cached["task1"],
        task2=cached["task2"],
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=cached["ood"],
        canonical=cached["canonical"],
    )


def evaluate_solution_9(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    augmented_train = sol3.augment_with_generated_sequences(train)
    task1_model = sol1.HybridRetrievalModel(augmented_train)
    task1_rows = sol1.predict_task1(task1_model, valid_examples)
    cached_completion = evaluate_solution_7_like(seed)
    task3_rows = sol8.predict_task3_semantic(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_9_judge_aware_portfolio",
        task1=task1,
        task2=cached_completion["task2"],
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol3.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


class CachedCompletionTask2Model:
    """Solution 7-compatible completion model using the evaluator's cache."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        completion_library = dict(train_sequences)
        completion_library.update(completion_generated_cache())
        self.completion_model = sol1.HybridRetrievalModel(completion_library)

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


def evaluate_solution_10(seed: int) -> MetricDict:
    train, _heldout, valid_examples, anomaly_examples, by_family = load_seed_data(seed)
    model = sol10.ConfidenceGatedConsensusPortfolio.__new__(
        sol10.ConfidenceGatedConsensusPortfolio
    )
    augmented_train = sol3.augment_with_generated_sequences(train)
    model.task1_model = sol1.HybridRetrievalModel(augmented_train)
    model.task2_model = CachedCompletionTask2Model(train)
    model.consensus_used = 0
    model.fallback_used = 0
    model.avg_consensus_share_sum = 0.0

    task1_rows = model.predict_task1(valid_examples)
    task2_rows = model.predict_task2(valid_examples)
    task3_rows = sol8.predict_task3_semantic(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_10_confidence_gated_consensus",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=sol3.evaluate_ood_proxy(by_family),
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


def evaluate_solution_11(seed: int) -> MetricDict:
    cached_task1 = evaluate_solution_7_like(seed)
    train = cached_task1["train"]
    valid_examples = cached_task1["valid_examples"]
    anomaly_examples = cached_task1["anomaly_examples"]
    by_family = cached_task1["by_family"]

    model = sol11.OODGuardedConsensusPortfolio.__new__(sol11.OODGuardedConsensusPortfolio)
    model.task1_model = sol2.EvalAwareRetrievalModel(train, train)
    model.task2_model = CachedCompletionTask2Model(train)
    model.consensus_used = 0
    model.fallback_used = 0
    model.avg_consensus_share_sum = 0.0

    task1_rows = sol2.predict_task1(model.task1_model, valid_examples)
    task2_rows = model.predict_task2(valid_examples)
    task3_rows = sol8.predict_task3_semantic(anomaly_examples)
    task1 = sol0.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    return flatten_common_metrics(
        seed=seed,
        solution_name="solution_11_ood_guarded_consensus",
        task1=task1,
        task2=sol0.evaluate_task2(task2_rows, valid_examples),
        task3=sol0.evaluate_task3(task3_rows, anomaly_examples),
        ood=cached_task1["ood"],
        canonical=sol2.evaluate_canonical_task1(task1_rows, valid_examples),
    )


SOLUTIONS: tuple[tuple[str, Callable[[int], MetricDict]], ...] = (
    ("solution_0_rule_mock", evaluate_solution_0),
    ("solution_1_hybrid_retrieval", evaluate_solution_1),
    ("solution_2_eval_aware_retrieval", evaluate_solution_2),
    ("solution_3_synthetic_augmented_retrieval", evaluate_solution_3),
    ("solution_4_length_aware_completion", evaluate_solution_4),
    ("solution_5_tuned_rank_ensemble", evaluate_solution_5),
    ("solution_6_alias_calibrated_retrieval", evaluate_solution_6),
    ("solution_7_monte_carlo_suffix_ensemble", evaluate_solution_7),
    ("solution_8_semantic_conformance_ensemble", evaluate_solution_8),
    ("solution_9_judge_aware_portfolio", evaluate_solution_9),
    ("solution_10_confidence_gated_consensus", evaluate_solution_10),
    ("solution_11_ood_guarded_consensus", evaluate_solution_11),
)


HIGHER_IS_BETTER = {
    "task1_top1",
    "task1_top2",
    "task1_top3",
    "task1_top5",
    "task1_mrr",
    "task2_exact_match",
    "task2_token_accuracy",
    "task2_block_accuracy",
    "task3_accuracy",
    "task3_f1_valid",
    "task3_roc_auc_valid_probability",
    "task3_rule_attribution_accuracy",
    "ood_avg_top1",
    "ood_ic_top1",
    "ood_igbt_top1",
    "ood_mosfet_top1",
    "ood_ic_mrr",
    "ood_igbt_mrr",
    "ood_mosfet_mrr",
    "task1_canonical_top1",
    "task1_canonical_top2",
    "task1_same_canonical_miss_rate",
    "public_lookup_coverage",
    "public_lookup_exact_next_when_covered",
}

LOWER_IS_BETTER = {
    "task2_normalized_edit_distance",
}


def numeric(value: object) -> float | None:
    if value == "" or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[MetricDict]) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["solution"])].append(row)

    metric_names = sorted((HIGHER_IS_BETTER | LOWER_IS_BETTER))
    summary_rows: list[dict[str, object]] = []
    for solution, solution_rows in sorted(grouped.items()):
        for metric in metric_names:
            values = [numeric(row.get(metric)) for row in solution_rows]
            values = [value for value in values if value is not None]
            if not values:
                summary_rows.append(
                    {
                        "solution": solution,
                        "metric": metric,
                        "direction": "not_applicable",
                        "seeds": len(solution_rows),
                        "mean": "",
                        "std": "",
                        "best": "",
                        "best_seed": "",
                        "worst": "",
                        "worst_seed": "",
                        "determinism_note": "not_reported_for_this_solution",
                    }
                )
                continue

            direction = "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better"
            best_fn = max if direction == "higher_is_better" else min
            worst_fn = min if direction == "higher_is_better" else max
            best_value = best_fn(values)
            worst_value = worst_fn(values)
            best_seed = next(
                row["seed"]
                for row in solution_rows
                if numeric(row.get(metric)) == best_value
            )
            worst_seed = next(
                row["seed"]
                for row in solution_rows
                if numeric(row.get(metric)) == worst_value
            )
            deterministic = len(set(round(value, 12) for value in values)) == 1
            note = (
                "constant_across_split_seeds"
                if deterministic
                else "varies_by_local_split_seed_model_itself_deterministic"
            )
            summary_rows.append(
                {
                    "solution": solution,
                    "metric": metric,
                    "direction": direction,
                    "seeds": len(values),
                    "mean": mean(values),
                    "std": pstdev(values),
                    "best": best_value,
                    "best_seed": best_seed,
                    "worst": worst_value,
                    "worst_seed": worst_seed,
                    "determinism_note": note,
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    priority = ["solution", "seed", "metric", "direction"]
    fieldnames = [key for key in priority if key in fieldnames] + [
        key for key in fieldnames
        if key not in priority
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    number = numeric(value)
    if number is None:
        return str(value)
    return f"{number:.4f}"


def get_summary(summary_rows: list[dict[str, object]], solution: str, metric: str) -> dict[str, object]:
    for row in summary_rows:
        if row["solution"] == solution and row["metric"] == metric:
            return row
    raise KeyError((solution, metric))


def write_markdown(rows: list[MetricDict], summary_rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 10-Seed Solution Evaluation",
        "",
        "This report evaluates the current solutions across seeds `0` through `9`.",
        "",
        "Important interpretation: these solutions do not train stochastic neural weights. "
        "For a fixed local split, each one is deterministic. Here, the seed changes the "
        "local train/held-out split, anomaly shuffle, and OOD sample. Solution 3 uses "
        "deterministic public-generator augmentation inside each run; Solutions 9 and 10 use "
        "the same augmentation for Task 1. Solutions 7, 8, 9, 10, and 11 "
        "use a cached deterministic Monte Carlo suffix library in this evaluator to avoid "
        "regenerating the same 30,000 suffix candidates for every split seed. Metrics marked "
        "`constant_across_split_seeds` did not change at all across the 10 runs.",
        "",
        "## Main Judging Metrics",
        "",
        "| Solution | Task 1 Top-1 mean | best | worst | Task 1 MRR mean | Task 2 block mean | Task 2 edit mean | Task 3 acc mean | OOD avg Top-1 mean | Determinism note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for solution, _runner in SOLUTIONS:
        top1 = get_summary(summary_rows, solution, "task1_top1")
        mrr = get_summary(summary_rows, solution, "task1_mrr")
        block = get_summary(summary_rows, solution, "task2_block_accuracy")
        edit = get_summary(summary_rows, solution, "task2_normalized_edit_distance")
        task3 = get_summary(summary_rows, solution, "task3_accuracy")
        ood = get_summary(summary_rows, solution, "ood_avg_top1")
        lines.append(
            f"| `{solution}` | {fmt(top1['mean'])} | {fmt(top1['best'])} "
            f"(seed {top1['best_seed']}) | {fmt(top1['worst'])} "
            f"(seed {top1['worst_seed']}) | {fmt(mrr['mean'])} | "
            f"{fmt(block['mean'])} | {fmt(edit['mean'])} | {fmt(task3['mean'])} | "
            f"{fmt(ood['mean'])} | {top1['determinism_note']} |"
        )

    lines += [
        "",
        "## Alias / Canonical Diagnostic",
        "",
        "This is not an official judging metric, but it explains why exact Top-1 is much lower than process understanding.",
        "",
        "| Solution | Canonical Top-1 mean | best | worst | Canonical Top-2 mean | Same-canonical miss-rate mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for solution, _runner in SOLUTIONS:
        ctop1 = get_summary(summary_rows, solution, "task1_canonical_top1")
        ctop2 = get_summary(summary_rows, solution, "task1_canonical_top2")
        same = get_summary(summary_rows, solution, "task1_same_canonical_miss_rate")
        lines.append(
            f"| `{solution}` | {fmt(ctop1['mean'])} | {fmt(ctop1['best'])} "
            f"(seed {ctop1['best_seed']}) | {fmt(ctop1['worst'])} "
            f"(seed {ctop1['worst_seed']}) | {fmt(ctop2['mean'])} | {fmt(same['mean'])} |"
        )

    lines += [
        "",
        "## Full Summary Table",
        "",
        "| Solution | Metric | Direction | Mean | Std | Best | Best seed | Worst | Worst seed | Determinism note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['solution']}` | `{row['metric']}` | {row['direction']} | "
            f"{fmt(row['mean'])} | {fmt(row['std'])} | {fmt(row['best'])} | "
            f"{row['best_seed']} | {fmt(row['worst'])} | {row['worst_seed']} | "
            f"{row['determinism_note']} |"
        )

    lines += [
        "",
        "## Raw Per-Seed Results",
        "",
        "See `per_seed_metrics.csv` and `per_seed_metrics.json` in this folder.",
        "",
    ]
    (OUT_DIR / "seed_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows: list[MetricDict] = []
    for solution_name, runner in SOLUTIONS:
        print(f"Evaluating {solution_name} across seeds {list(SEEDS)}", flush=True)
        for seed in SEEDS:
            print(f"  seed {seed}", flush=True)
            all_rows.append(runner(seed))

    summary_rows = summarize(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_seed_metrics.json").write_text(
        json.dumps(all_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary_metrics.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(OUT_DIR / "per_seed_metrics.csv", all_rows)
    write_csv(OUT_DIR / "summary_metrics.csv", summary_rows)
    write_markdown(all_rows, summary_rows)
    print(f"Wrote seed evaluation outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

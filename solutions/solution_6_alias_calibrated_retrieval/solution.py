#!/usr/bin/env python3
"""Solution 6: canonical-context alias calibrated retrieval.

The earlier solutions showed a large gap between exact Top-1 and canonical
process-step Top-1. This solution treats that as a two-stage prediction problem:

1. Predict the likely next manufacturing operation with eval-aware retrieval.
2. Choose the exact label alias using family-specific canonical context counts.

Run from repo root:
    python -B solutions/solution_6_alias_calibrated_retrieval/solution.py
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
ALIAS_CONTEXT_LENGTHS = (8, 4, 2, 1, 0)

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_1_hybrid_retrieval import solution as sol1  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CanonicalAliasCalibrator:
    """Choose exact label aliases from canonical process context counts."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.counts: dict[tuple[str, int, tuple[str, ...], str], Counter[str]] = defaultdict(Counter)
        self._fit(train_sequences)

    def _fit(self, train_sequences: dict[str, list[str]]) -> None:
        for key, sequence in train_sequences.items():
            family = key.split(":", 1)[0]
            canonical_sequence = [sol2.canonical(step) for step in sequence]
            for pos, exact_next in enumerate(sequence):
                canonical_next = canonical_sequence[pos]
                for context_len in ALIAS_CONTEXT_LENGTHS:
                    if context_len and pos < context_len:
                        continue
                    context = tuple(canonical_sequence[pos - context_len:pos]) if context_len else ()
                    self.counts[(family, context_len, context, canonical_next)][exact_next] += 1
                    self.counts[("*", context_len, context, canonical_next)][exact_next] += 1

    def choose(self, prefix: list[str], family: str, canonical_next: str, fallback: str) -> str:
        canonical_prefix = [sol2.canonical(step) for step in prefix]
        for family_key in (family, "*"):
            for context_len in ALIAS_CONTEXT_LENGTHS:
                if context_len and len(canonical_prefix) < context_len:
                    continue
                context = tuple(canonical_prefix[-context_len:]) if context_len else ()
                counter = self.counts.get((family_key, context_len, context, canonical_next))
                if counter:
                    return counter.most_common(1)[0][0]
        return fallback


class AliasCalibratedRetrievalModel:
    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.fallback = sol2.EvalAwareRetrievalModel(train_sequences, train_sequences)
        self.calibrator = CanonicalAliasCalibrator(train_sequences)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        base_ranks = self.fallback.next_step_ranking(
            prefix,
            family=family,
            completion_fraction=completion_fraction,
            k=20,
        )
        ranked: list[str] = []
        seen: set[str] = set()

        for step in base_ranks:
            if not step:
                continue
            canonical_next = sol2.canonical(step)
            calibrated = self.calibrator.choose(prefix, family, canonical_next, step)
            # Keep the retriever's exact first choice first. The calibrated alias
            # is added as a backup so alias calibration cannot collapse Top-k
            # coverage when the exact label choice is genuinely random.
            for candidate in (step, calibrated):
                if candidate and candidate not in seen:
                    ranked.append(candidate)
                    seen.add(candidate)
                if len(ranked) >= k:
                    return ranked

        return (ranked + [""] * k)[:k]

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        suffix = self.fallback.complete(
            prefix,
            family=family,
            completion_fraction=completion_fraction,
        )
        calibrated_suffix: list[str] = []
        history = list(prefix)
        for step in suffix:
            canonical_step = sol2.canonical(step)
            exact = self.calibrator.choose(history, family, canonical_step, step)
            calibrated_suffix.append(exact)
            history.append(exact)
        return calibrated_suffix


def predict_task1(
    model: AliasCalibratedRetrievalModel,
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
    model: AliasCalibratedRetrievalModel,
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

        model = AliasCalibratedRetrievalModel(train)
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
        "# Solution 6 Metrics",
        "",
        "## Method",
        "",
        "- Model: eval-aware retrieval plus canonical-context alias calibration.",
        f"- Alias context lengths: {', '.join(str(v) for v in ALIAS_CONTEXT_LENGTHS)}",
        "- Task 2 applies the same alias calibration to the retrieved suffix.",
        "",
        "## Task 1",
        "",
        f"- Top-1: {task1['top1']:.4f}",
        f"- Top-3: {task1['top3']:.4f}",
        f"- Top-5: {task1['top5']:.4f}",
        f"- MRR: {task1['mrr']:.4f}",
        f"- Canonical Top-1: {canonical['canonical_top1']:.4f}",
        f"- Same-canonical misses: {canonical['same_canonical_misses']} / {canonical['exact_top1_misses']}",
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
        "This solution tests whether exact-label misses can be reduced by learning alias choice as a separate deterministic calibration problem.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = AliasCalibratedRetrievalModel(train)

    task1_rows = predict_task1(model, valid_examples)
    task2_rows = predict_task2(model, valid_examples)
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
        "solution": "solution_6_alias_calibrated_retrieval",
        "seed": SEED,
        "method": {
            "name": "canonical-context alias calibrated eval-aware retrieval",
            "alias_context_lengths": ALIAS_CONTEXT_LENGTHS,
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

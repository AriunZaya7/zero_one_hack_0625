#!/usr/bin/env python3
"""Solution 1: Hybrid retrieval + n-gram + rule validator.

This solution implements the first research-informed step beyond Solution 0.

Core idea:
- Task 1: combine a trigram model with a retrieval index over historical
  process contexts near the official 60% and 80% cut points.
- Task 2: retrieve the most similar historical continuation instead of rolling
  out a local n-gram greedily.
- Task 3: keep the public validator oracle for validity/rule attribution.

The retrieval completion path deliberately does not use the true hidden
remainder length. It only uses the visible prefix, FAMILY, and
COMPLETION_FRACTION fields that are present in the official eval input.

Run from repo root:
    PYTHONDONTWRITEBYTECODE=1 python solutions/solution_1_hybrid_retrieval/solution.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
CONTEXT_LENGTHS = (12, 6, 3, 1)
CANDIDATE_FRACTIONS = (0.6, 0.8)
POSITION_WINDOW = 3
RETRIEVAL_MIN_ITEMS = 80
RETRIEVAL_COMPLETION_ITEMS = 120

sys.path.insert(0, str(ROOT))

from ngram import NGramModel  # noqa: E402
from solutions.solution_0_rule_mock import solution as base  # noqa: E402


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    context_len: int
    key: str
    family: str
    position: int
    sequence: list[str]


class HybridRetrievalModel:
    """Indexed historical-context retriever plus trigram fallback."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.train_sequences = train_sequences
        self.ngram = NGramModel(n=3, alpha=0.4).fit(train_sequences.values())
        self.index: dict[tuple[float, int], dict[tuple[str, ...], list[tuple[str, str, int, list[str]]]]] = {
            (frac, context_len): defaultdict(list)
            for frac in CANDIDATE_FRACTIONS
            for context_len in CONTEXT_LENGTHS
        }
        self._build_index()

    def _build_index(self) -> None:
        """Index contexts around the official 60% and 80% truncation points."""
        for key, sequence in self.train_sequences.items():
            family = key.split(":", 1)[0]
            for fraction in CANDIDATE_FRACTIONS:
                center = max(1, min(len(sequence) - 1, int(len(sequence) * fraction)))
                start = max(1, center - POSITION_WINDOW)
                stop = min(len(sequence), center + POSITION_WINDOW + 1)
                for position in range(start, stop):
                    for context_len in CONTEXT_LENGTHS:
                        if position < context_len:
                            continue
                        context = tuple(sequence[position - context_len:position])
                        self.index[(fraction, context_len)][context].append(
                            (key, family, position, sequence)
                        )

    @staticmethod
    def _fraction_key(completion_fraction: float) -> float:
        return min(
            CANDIDATE_FRACTIONS,
            key=lambda fraction: abs(fraction - completion_fraction),
        )

    def retrieve(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        min_items: int = RETRIEVAL_MIN_ITEMS,
    ) -> list[RetrievalHit]:
        """Return historical continuations whose recent context matches `prefix`."""
        fraction_key = self._fraction_key(completion_fraction)
        hits: list[RetrievalHit] = []
        seen: set[tuple[str, int]] = set()

        for context_len in CONTEXT_LENGTHS:
            if len(prefix) < context_len:
                continue
            context = tuple(prefix[-context_len:])
            bucket = self.index[(fraction_key, context_len)].get(context, [])
            for key, candidate_family, position, sequence in bucket:
                if (key, position) in seen:
                    continue
                seen.add((key, position))
                score = context_len * 20.0
                if candidate_family == family:
                    score += 20.0
                score -= abs(position / len(sequence) - completion_fraction) * 15.0
                score -= abs(position - len(prefix)) * 0.03
                hits.append(
                    RetrievalHit(
                        score=score,
                        context_len=context_len,
                        key=key,
                        family=candidate_family,
                        position=position,
                        sequence=sequence,
                    )
                )
            if len(hits) >= min_items:
                break

        hits.sort(key=lambda hit: (-hit.score, hit.key, hit.position))
        return hits

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        """Rank next steps using retrieval votes plus trigram fallback votes."""
        scores: dict[str, float] = defaultdict(float)

        for rank, hit in enumerate(self.retrieve(prefix, family, completion_fraction)):
            if hit.position >= len(hit.sequence):
                continue
            next_step = hit.sequence[hit.position]
            scores[next_step] += hit.score / (1.0 + rank * 0.2)

        for rank, step in enumerate(self.ngram.next_step_ranking(prefix, k=20)):
            scores[step] += 10.0 / (rank + 1)

        ranked = [
            step for step, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if step != "<eos>"
        ]
        return (ranked + [""] * k)[:k]

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        """Return a retrieved suffix from the best aligned historical route."""
        best: tuple[float, list[str]] | None = None
        hits = self.retrieve(
            prefix,
            family,
            completion_fraction,
            min_items=RETRIEVAL_COMPLETION_ITEMS,
        )

        for rank, hit in enumerate(hits):
            suffix = hit.sequence[hit.position:]
            if not suffix:
                continue
            score = hit.score
            score += hit.context_len * 3.0
            if hit.family == family:
                score += 10.0
            if suffix[-1] == "SHIP LOT":
                score += 8.0
            score -= rank * 0.02
            if best is None or score > best[0]:
                best = (score, suffix)

        if best is not None:
            return best[1]

        # Fallback should be rare. It keeps the output bounded if retrieval misses.
        total_estimate = max(len(prefix) + 1, round(len(prefix) / max(completion_fraction, 0.01)))
        max_new_steps = max(1, total_estimate - len(prefix) + 20)
        return base.rollout_completion(self.ngram, prefix, max_new_steps=max_new_steps)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_inputs_and_ground_truth(
    valid_examples: list[base.ValidExample],
    anomaly_examples: list[base.AnomalyExample],
) -> None:
    write_csv(
        OUT_DIR / "self_eval_valid_input.csv",
        ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
        (
            {
                "EXAMPLE_ID": ex.example_id,
                "FAMILY": ex.family,
                "COMPLETION_FRACTION": ex.completion_fraction,
                "PARTIAL_SEQUENCE": "|".join(ex.partial),
            }
            for ex in valid_examples
        ),
    )
    write_csv(
        OUT_DIR / "self_eval_valid_ground_truth.csv",
        ["EXAMPLE_ID", "TRUE_NEXT", "TRUE_REMAINDER"],
        (
            {
                "EXAMPLE_ID": ex.example_id,
                "TRUE_NEXT": ex.truth_next,
                "TRUE_REMAINDER": "|".join(ex.truth_remainder),
            }
            for ex in valid_examples
        ),
    )
    write_csv(
        OUT_DIR / "self_eval_anomaly_input.csv",
        ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
        (
            {
                "EXAMPLE_ID": ex.example_id,
                "FAMILY": ex.family,
                "SEQUENCE": "|".join(ex.sequence),
            }
            for ex in anomaly_examples
        ),
    )
    write_csv(
        OUT_DIR / "self_eval_anomaly_ground_truth.csv",
        ["EXAMPLE_ID", "IS_VALID", "RULE"],
        (
            {
                "EXAMPLE_ID": ex.example_id,
                "IS_VALID": ex.is_valid,
                "RULE": ex.rule,
            }
            for ex in anomaly_examples
        ),
    )


def predict_task1(
    model: HybridRetrievalModel,
    examples: list[base.ValidExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        ranks = model.next_step_ranking(
            ex.partial,
            family=ex.family,
            completion_fraction=ex.completion_fraction,
            k=5,
        )
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
    model: HybridRetrievalModel,
    examples: list[base.ValidExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        suffix = model.complete(
            ex.partial,
            family=ex.family,
            completion_fraction=ex.completion_fraction,
        )
        rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "PREDICTED_SEQUENCE": "|".join(suffix),
            }
        )
    return rows


def predict_task3(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
    """Validator-gated anomaly prediction.

    This keeps the same honest oracle stance as Solution 0: it maximizes the
    measurable task by using the public rule checker, but it is not evidence of
    learned anomaly reasoning.
    """
    rows: list[dict[str, object]] = []
    for ex in examples:
        violations = base.validate_sequence(ex.sequence)
        is_valid = 0 if violations else 1
        rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "IS_VALID": is_valid,
                "SCORE": 1.0 if is_valid else 0.0,
                "PREDICTED_RULE": "" if is_valid else violations[0].rule,
            }
        )
    return rows


def evaluate_ood_proxy(
    by_family: dict[str, dict[str, base.SequenceRecord]],
) -> dict[str, dict[str, float]]:
    """Leave-one-family-out next-step proxy for hidden Task 4."""
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

        model = HybridRetrievalModel(train)
        rng = random.Random(SEED)
        keys = sorted(test)
        rng.shuffle(keys)
        sampled_test = {key: test[key] for key in keys[:base.EVAL_PER_FAMILY]}
        examples = base.build_valid_examples({family: sampled_test})
        rows = predict_task1(model, examples)
        results[family] = base.evaluate_task1(rows, examples)
    return results


def write_asset_audit(audit: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = dict(audit)
    audit["solution_1_notes"] = [
        "Solution 1 uses the same data audit surface as Solution 0.",
        "The main modeling change is indexed retrieval near the official 60%/80% cut points.",
        "Task 2 prediction does not use hidden true remainder length.",
        "Task 3 still uses the public validator oracle for maximum measurable anomaly performance.",
    ]
    (OUT_DIR / "input_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Solution 1 Input Audit",
        "",
        "This file records the repo assets considered by `solution_1_hybrid_retrieval`.",
        "",
        "## Docs Reviewed",
        "",
    ]
    for item in audit["docs_reviewed"]:
        lines.append(f"- `{item}`")
    lines += ["", "## Scripts Reviewed", ""]
    for item in audit["scripts_reviewed"]:
        lines.append(f"- `{item}`")
    lines += [
        "",
        "## Sequence Sources Used By The Baseline",
        "",
        "| File | Family | Kind | Sequences | Valid | Invalid | Length min/mean/max |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in audit["sequence_sources_used"]:
        lines.append(
            f"| `{item['file']}` | {item['family']} | {item['kind']} | {item['sequences']} | "
            f"{item['valid_sequences']} | {item['invalid_sequences']} | "
            f"{item['min_len']}/{item['mean_len']:.1f}/{item['max_len']} |"
        )
    lines += ["", "## Solution 1 Notes", ""]
    for note in audit["solution_1_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    (OUT_DIR / "input_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task1 = metrics["task1_next_step"]
    task2 = metrics["task2_completion"]
    task3 = metrics["task3_anomaly"]
    ood = metrics["task4_ood_proxy_next_step"]
    self_eval = metrics["self_eval"]
    data_inventory = metrics["data_inventory"]
    lines = [
        "# Solution 1 Metrics",
        "",
        "## Data Coverage",
        "",
        f"- Train sequences: {self_eval['train_sequences']}",
        f"- Self-eval valid rows: {self_eval['valid_task_rows']}",
        f"- Self-eval anomaly rows: {self_eval['anomaly_task_rows']}",
        f"- Sequence source files used: {len(data_inventory)}",
        f"- Official eval available: {self_eval['official_eval_available']}",
        f"- Retrieval context lengths: {', '.join(str(x) for x in CONTEXT_LENGTHS)}",
        f"- Retrieval indexed cut fractions: {', '.join(str(x) for x in CANDIDATE_FRACTIONS)}",
        "",
        "## Task 1: Next-Step Prediction",
        "",
        f"- Top-1: {task1['top1']:.4f}",
        f"- Top-3: {task1['top3']:.4f}",
        f"- Top-5: {task1['top5']:.4f}",
        f"- MRR: {task1['mrr']:.4f}",
        f"- Examples: {task1['n_examples']}",
        "",
        "## Task 2: Sequence Completion",
        "",
        f"- Exact match: {task2['exact_match']:.4f}",
        f"- Normalized edit distance: {task2['normalized_edit_distance']:.4f}",
        f"- Token accuracy: {task2['token_accuracy']:.4f}",
        f"- Block accuracy: {task2['block_accuracy']:.4f}",
        f"- Examples: {task2['n_examples']}",
        "",
        "## Task 3: Anomaly Detection",
        "",
        f"- Accuracy: {task3['accuracy']:.4f}",
        f"- Precision(valid): {task3['precision_valid']:.4f}",
        f"- Recall(valid): {task3['recall_valid']:.4f}",
        f"- F1(valid): {task3['f1_valid']:.4f}",
        f"- ROC-AUC(valid probability): {task3['roc_auc_valid_probability']:.4f}",
        f"- Rule attribution accuracy: {task3['rule_attribution_accuracy']:.4f}",
        f"- Examples: {task3['n_examples']}",
        "",
        "## Task 4 Proxy: Leave-One-Family-Out Next-Step",
        "",
        "| Held-out family | Top-1 | Top-3 | Top-5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, family_metrics in ood.items():
        lines.append(
            f"| {family} | {family_metrics['top1']:.4f} | {family_metrics['top3']:.4f} | "
            f"{family_metrics['top5']:.4f} | {family_metrics['mrr']:.4f} |"
        )
    lines += [
        "",
        "## Honest Caveats",
        "",
        "- Task 3 still uses the public validator oracle.",
        "- These are deterministic local self-eval metrics, not official hidden eval results.",
        "- Solution 1 improves completion strongly over Solution 0, but it is still a retrieval baseline, not a trained transformer.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    audit = base.build_repo_asset_audit(sequence_inventory)

    model = HybridRetrievalModel(train)

    task1_rows = predict_task1(model, valid_examples)
    task2_rows = predict_task2(model, valid_examples)
    task3_rows = predict_task3(anomaly_examples)

    write_inputs_and_ground_truth(valid_examples, anomaly_examples)
    write_csv(OUT_DIR / "nextstep.csv", ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], task1_rows)
    write_csv(OUT_DIR / "completion.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], task2_rows)
    write_csv(OUT_DIR / "anomaly.csv", ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"], task3_rows)
    write_asset_audit(audit)

    metrics = {
        "solution": "solution_1_hybrid_retrieval",
        "seed": SEED,
        "method": {
            "name": "context retrieval + trigram fallback + validator oracle",
            "context_lengths": CONTEXT_LENGTHS,
            "candidate_fractions": CANDIDATE_FRACTIONS,
            "position_window": POSITION_WINDOW,
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
        "task2_completion": base.evaluate_task2(task2_rows, valid_examples),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_examples),
        "task4_ood_proxy_next_step": evaluate_ood_proxy(by_family),
    }
    write_metrics(metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

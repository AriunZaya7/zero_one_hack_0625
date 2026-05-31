#!/usr/bin/env python3
"""Solution 10: confidence-gated suffix consensus portfolio.

This solution keeps the strongest visible Task 1 and Task 3 specialists from
Solution 9, then replaces Task 2's single-nearest suffix copy with a
confidence-gated consensus decoder:

- If the top generated/retrieved suffixes strongly agree token-by-token, emit
  the weighted consensus suffix.
- If agreement is weak, fall back to Solution 7's valid single-suffix retrieval.

Run from repo root:
    python -B solutions/solution_10_confidence_gated_consensus/solution.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
CONSENSUS_TOP_N = 10
CONSENSUS_MIN_ITEMS = 120
CONSENSUS_AVG_SHARE_THRESHOLD = 0.80

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


def weighted_median_length(weighted_suffixes: list[tuple[float, list[str]]]) -> int:
    total = sum(weight for weight, _suffix in weighted_suffixes)
    running = 0.0
    for length, weight in sorted((len(suffix), weight) for weight, suffix in weighted_suffixes):
        running += weight
        if running >= total / 2:
            return length
    return len(weighted_suffixes[0][1])


class ConfidenceGatedConsensusPortfolio:
    """Task portfolio with a confidence-gated marginal suffix decoder."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        augmented_train = sol3.augment_with_generated_sequences(train_sequences)
        self.task1_model = sol1.HybridRetrievalModel(augmented_train)
        self.task2_model = sol7.TaskSpecializedMonteCarloModel(train_sequences)
        self.consensus_used = 0
        self.fallback_used = 0
        self.avg_consensus_share_sum = 0.0

    def predict_task1(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        return sol1.predict_task1(self.task1_model, examples)

    def consensus_suffix(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> tuple[list[str], float]:
        hits = self.task2_model.completion_model.retrieve(
            prefix,
            family,
            completion_fraction,
            min_items=CONSENSUS_MIN_ITEMS,
        )

        weighted_suffixes: list[tuple[float, list[str]]] = []
        for rank, hit in enumerate(hits[:CONSENSUS_TOP_N]):
            suffix = hit.sequence[hit.position:]
            if not suffix:
                continue
            weight = max(0.001, hit.score) / (1.0 + 0.15 * rank)
            if hit.family == family:
                weight *= 1.15
            weighted_suffixes.append((weight, suffix))

        if not weighted_suffixes:
            return self.task2_model.complete(prefix, family, completion_fraction), 0.0

        target_length = weighted_median_length(weighted_suffixes)
        consensus: list[str] = []
        token_shares: list[float] = []

        for offset in range(target_length):
            votes: Counter[str] = Counter()
            for weight, suffix in weighted_suffixes:
                if offset < len(suffix):
                    votes[suffix[offset]] += weight
                else:
                    votes["<end>"] += 0.5 * weight

            total = sum(votes.values())
            token, score = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0]
            token_shares.append(score / max(total, 1e-9))
            if token == "<end>":
                break
            consensus.append(token)

        if not consensus:
            consensus = weighted_suffixes[0][1]
        avg_share = sum(token_shares) / max(len(token_shares), 1)
        return consensus, avg_share

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        consensus, avg_share = self.consensus_suffix(prefix, family, completion_fraction)
        self.avg_consensus_share_sum += avg_share

        if avg_share >= CONSENSUS_AVG_SHARE_THRESHOLD:
            self.consensus_used += 1
            return consensus

        self.fallback_used += 1
        return self.task2_model.complete(prefix, family, completion_fraction)

    def predict_task2(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for ex in examples:
            suffix = self.complete(ex.partial, ex.family, ex.completion_fraction)
            rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
        return rows

    @staticmethod
    def predict_task3(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
        return sol8.predict_task3_semantic(examples)

    def consensus_stats(self, n_examples: int) -> dict[str, object]:
        return {
            "consensus_top_n": CONSENSUS_TOP_N,
            "consensus_min_items": CONSENSUS_MIN_ITEMS,
            "avg_share_threshold": CONSENSUS_AVG_SHARE_THRESHOLD,
            "consensus_rows_used": self.consensus_used,
            "fallback_rows_used": self.fallback_used,
            "mean_candidate_agreement": self.avg_consensus_share_sum / max(n_examples, 1),
        }


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
    consensus = metrics["task2_consensus"]
    lines = [
        "# Solution 10 Metrics",
        "",
        "## Method",
        "",
        "- Task 1 specialist: Solution 3 synthetic-augmented retrieval.",
        "- Task 2 specialist: confidence-gated weighted suffix consensus over Solution 7's generated retrieval library.",
        "- Task 3 specialist: Solution 8 semantic conformance checker.",
        f"- Consensus top-N suffixes: {consensus['consensus_top_n']}",
        f"- Consensus average-share threshold: {consensus['avg_share_threshold']:.2f}",
        f"- Consensus rows used: {consensus['consensus_rows_used']}",
        f"- Fallback rows used: {consensus['fallback_rows_used']}",
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
        f"- ROC-AUC: {task3['roc_auc_valid_probability']:.4f}",
        f"- Rule attribution accuracy: {task3['rule_attribution_accuracy']:.4f}",
        "",
        "## Honest Tradeoff",
        "",
        "The consensus gate improves local normalized edit distance versus Solution 9 on seed 42, but exact completion match is slightly lower. Task 1 keeps the same hidden-family caveat as Solution 9 because it uses the Solution 3 specialist.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = ConfidenceGatedConsensusPortfolio(train)

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
        "solution": "solution_10_confidence_gated_consensus",
        "seed": SEED,
        "method": {
            "name": "confidence-gated suffix consensus portfolio",
            "task1_specialist": "solution_3_synthetic_augmented_retrieval",
            "task2_specialist": "confidence_gated_consensus_over_solution_7_suffix_library",
            "task3_specialist": "solution_8_semantic_conformance_ensemble",
            "task2_uses_truth_remainder_length": False,
        },
        "task2_consensus": model.consensus_stats(len(valid_examples)),
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
            "visible_task_choice": "keep Solution 9 Task 1/3 strengths and improve Task 2 edit distance with a confidence-gated consensus.",
            "hidden_family_caveat": "Task 1 uses Solution 3, which is weaker than Solution 2/8 on IC leave-one-family-out.",
        },
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

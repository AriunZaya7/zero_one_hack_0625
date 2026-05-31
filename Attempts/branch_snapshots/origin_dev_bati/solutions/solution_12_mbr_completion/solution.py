#!/usr/bin/env python3
"""Solution 12: OOD-guarded Minimum Bayes Risk completion portfolio.

This solution keeps the conservative OOD-facing portfolio from Solution 11,
then changes Task 2 from confidence-gated token consensus to Minimum Bayes Risk
candidate selection:

- Task 1: use Solution 2's eval-aware retrieval, because it has stronger
  leave-one-family-out next-step behavior than the Solution 3 specialist.
- Task 2: retrieve a small cloud of generated/historical suffix candidates and
  select the candidate with the lowest weighted expected normalized edit
  distance to the other candidates.
- Task 3: keep Solution 8's semantic conformance checker.

Run from repo root:
    python -B solutions/solution_12_mbr_completion/solution.py
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
MBR_TOP_N = 15
MBR_MIN_ITEMS = 120
MBR_RETRIEVAL_SCORE_MIX = 0.03

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_7_monte_carlo_suffix_ensemble import solution as sol7  # noqa: E402
from solutions.solution_8_semantic_conformance_ensemble import solution as sol8  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class OODGuardedMBRCompletionPortfolio:
    """Task portfolio prioritizing hidden-family Task 1 and Task 2 edit distance."""

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.task1_model = sol2.EvalAwareRetrievalModel(train_sequences, train_sequences)
        self.task2_model = sol7.TaskSpecializedMonteCarloModel(train_sequences)
        self.mbr_used = 0
        self.fallback_used = 0
        self.candidate_count_sum = 0
        self.selected_risk_sum = 0.0

    def predict_task1(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        return sol2.predict_task1(self.task1_model, examples)

    @staticmethod
    def normalized_edit_distance(a: list[str], b: list[str]) -> float:
        return base.edit_distance(a, b) / max(len(a), len(b), 1)

    def suffix_candidates(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[tuple[float, list[str], float]]:
        hits = self.task2_model.completion_model.retrieve(
            prefix,
            family,
            completion_fraction,
            min_items=MBR_MIN_ITEMS,
        )

        candidates: list[tuple[float, list[str], float]] = []
        seen: set[tuple[str, ...]] = set()
        for rank, hit in enumerate(hits[:MBR_TOP_N]):
            suffix = hit.sequence[hit.position:]
            if not suffix:
                continue
            key = tuple(suffix)
            if key in seen:
                continue
            seen.add(key)

            weight = max(0.001, hit.score) / (1.0 + 0.15 * rank)
            if hit.family == family:
                weight *= 1.15
            candidates.append((weight, suffix, hit.score))
        return candidates

    def mbr_suffix(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> tuple[list[str], float, int]:
        candidates = self.suffix_candidates(prefix, family, completion_fraction)
        if not candidates:
            return self.task2_model.complete(prefix, family, completion_fraction), 0.0, 0

        distances = [[0.0 for _ in candidates] for _ in candidates]
        for i, (_weight_i, suffix_i, _score_i) in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                _weight_j, suffix_j, _score_j = candidates[j]
                distance = self.normalized_edit_distance(suffix_i, suffix_j)
                distances[i][j] = distance
                distances[j][i] = distance

        best: tuple[float, list[str]] | None = None
        for i, (_weight, suffix, retrieval_score) in enumerate(candidates):
            total_weight = 0.0
            risk = 0.0
            for j, (other_weight, _other_suffix, _other_score) in enumerate(candidates):
                if i == j:
                    continue
                total_weight += other_weight
                risk += other_weight * distances[i][j]
            expected_edit = risk / max(total_weight, 1e-9)
            adjusted_risk = expected_edit - MBR_RETRIEVAL_SCORE_MIX * (retrieval_score / 100.0)
            if best is None or adjusted_risk < best[0]:
                best = (adjusted_risk, suffix)

        assert best is not None
        return best[1], best[0], len(candidates)

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        suffix, selected_risk, candidate_count = self.mbr_suffix(
            prefix,
            family,
            completion_fraction,
        )
        if candidate_count == 0:
            self.fallback_used += 1
        else:
            self.mbr_used += 1
            self.candidate_count_sum += candidate_count
            self.selected_risk_sum += selected_risk
        return suffix

    def predict_task2(self, examples: list[base.ValidExample]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for ex in examples:
            suffix = self.complete(ex.partial, ex.family, ex.completion_fraction)
            rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
        return rows

    @staticmethod
    def predict_task3(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
        return sol8.predict_task3_semantic(examples)

    def mbr_stats(self, n_examples: int) -> dict[str, object]:
        return {
            "mbr_top_n": MBR_TOP_N,
            "mbr_min_items": MBR_MIN_ITEMS,
            "retrieval_score_mix": MBR_RETRIEVAL_SCORE_MIX,
            "mbr_rows_used": self.mbr_used,
            "fallback_rows_used": self.fallback_used,
            "mean_candidate_count": self.candidate_count_sum / max(self.mbr_used, 1),
            "mean_selected_adjusted_risk": self.selected_risk_sum / max(self.mbr_used, 1),
            "rows_total": n_examples,
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
    mbr = metrics["task2_mbr"]
    lines = [
        "# Solution 12 Metrics",
        "",
        "## Method",
        "",
        "- Task 1 specialist: Solution 2 eval-aware retrieval for stronger leave-one-family-out behavior.",
        "- Task 2 specialist: Minimum Bayes Risk suffix selection over Solution 7's generated retrieval library.",
        "- Task 3 specialist: Solution 8 semantic conformance checker.",
        f"- MBR candidate suffixes per row: up to {mbr['mbr_top_n']}",
        f"- MBR retrieval min-items: {mbr['mbr_min_items']}",
        f"- Retrieval score mix: {mbr['retrieval_score_mix']:.2f}",
        f"- MBR rows used: {mbr['mbr_rows_used']}",
        f"- Fallback rows used: {mbr['fallback_rows_used']}",
        f"- Mean candidate count: {mbr['mean_candidate_count']:.2f}",
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
        "This solution directly optimizes a proxy for the official normalized edit-distance completion metric. It is slower than the consensus decoder because it computes pairwise suffix edit distances, and it should be submitted only if edit distance, token accuracy, and block accuracy matter more than raw inference speed.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = base.load_split()
    valid_examples = base.build_valid_examples(heldout)
    anomaly_examples = base.build_anomaly_examples(heldout)
    model = OODGuardedMBRCompletionPortfolio(train)

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
        "solution": "solution_12_mbr_completion",
        "seed": SEED,
        "method": {
            "name": "OOD-guarded Minimum Bayes Risk completion portfolio",
            "task1_specialist": "solution_2_eval_aware_retrieval",
            "task2_specialist": "minimum_bayes_risk_suffix_selection_over_solution_7_suffix_library",
            "task3_specialist": "solution_8_semantic_conformance_ensemble",
            "task2_uses_truth_remainder_length": False,
            "mbr_top_n": MBR_TOP_N,
            "mbr_min_items": MBR_MIN_ITEMS,
            "mbr_retrieval_score_mix": MBR_RETRIEVAL_SCORE_MIX,
        },
        "task2_mbr": model.mbr_stats(len(valid_examples)),
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
        "task4_ood_proxy_next_step": sol2.evaluate_ood_proxy(by_family),
        "portfolio_tradeoff": {
            "visible_task_choice": "keep the hidden-family safer Task 1 specialist and spend the new complexity on Task 2 edit-distance risk.",
            "hidden_family_priority": "Task 1 uses the Solution 2/8 specialist, which is stronger than Solution 3 on leave-one-family-out.",
            "runtime_caveat": "Task 2 is slower than Solution 11 because it computes pairwise edit distances among retrieved candidate suffixes.",
        },
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

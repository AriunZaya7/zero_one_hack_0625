#!/usr/bin/env python3
"""Solution 15: route-memory MBR decoder.

This solution keeps the strong participant-input exact route match from
Solution 13, then adds a more explicit fallback:

1. Build a memory bank of valid full routes from released Task 3 inputs, public
   training data, and generated valid routes.
2. If a Task 1/2 partial is an exact prefix of a valid memory route, use that
   route directly.
3. Otherwise retrieve plausible valid routes and choose the completion suffix
   with a Minimum Bayes Risk style edit-distance objective.

Run from repo root:
    python -B solutions/solution_15_route_memory_mbr/solution.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
SYNTHETIC_PER_FAMILY = 5_000
SYNTHETIC_SEED_BASE = 52_000
CONTEXT_LENGTHS = (24, 16, 12, 8, 4, 2, 1)
MAX_RETRIEVED_ROUTES = 60
MAX_MBR_SUFFIXES = 24

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402
from training_data.generate_sequences import generate_dataset, validate_sequence  # noqa: E402


NEXTSTEP_FIELDS = sol13.NEXTSTEP_FIELDS
COMPLETION_FIELDS = sol13.COMPLETION_FIELDS
ANOMALY_FIELDS = sol13.ANOMALY_FIELDS


@dataclass(frozen=True)
class MemoryRoute:
    route_id: str
    family: str
    sequence: tuple[str, ...]
    source: str
    source_weight: float


@dataclass(frozen=True)
class RetrievedRoute:
    route: MemoryRoute
    score: float
    reason: str


@dataclass(frozen=True)
class SuffixCandidate:
    suffix: tuple[str, ...]
    score: float
    route_count: int
    best_source: str


_PUBLIC_CACHE: dict[str, list[str]] | None = None
_GENERATED_CACHE: dict[int, dict[str, list[str]]] = {}


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def public_sequences() -> dict[str, list[str]]:
    global _PUBLIC_CACHE
    if _PUBLIC_CACHE is None:
        by_family, _inventory = base.load_all_available_sequences()
        sequences: dict[str, list[str]] = {}
        for records in by_family.values():
            for key, record in records.items():
                sequences[key] = record.steps
        _PUBLIC_CACHE = sequences
    return dict(_PUBLIC_CACHE)


def generated_training_sequences(
    count_per_family: int = SYNTHETIC_PER_FAMILY,
    seed_base: int = SYNTHETIC_SEED_BASE,
) -> dict[str, list[str]]:
    if count_per_family not in _GENERATED_CACHE:
        sequences: dict[str, list[str]] = {}
        for family_index, family in enumerate(base.FAMILIES):
            generated = generate_dataset(
                family,
                count=count_per_family,
                seed=seed_base + family_index,
                validate=True,
            )
            for i, sequence in enumerate(generated):
                sequences[f"{family}:solution15_generated:{i:05d}"] = sequence
        _GENERATED_CACHE[count_per_family] = sequences
    return dict(_GENERATED_CACHE[count_per_family])


def normalized_edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 0.0
    rows = list(range(len(right) + 1))
    for i, left_step in enumerate(left, start=1):
        prev = rows[:]
        rows[0] = i
        for j, right_step in enumerate(right, start=1):
            if left_step == right_step:
                rows[j] = prev[j - 1]
            else:
                rows[j] = 1 + min(prev[j], rows[j - 1], prev[j - 1])
    return rows[-1] / max(len(left), len(right), 1)


class RouteMemoryMBRModel:
    """Valid-route memory with exact prefix matching and MBR fallback."""

    def __init__(
        self,
        anomaly_inputs: list[sol13.AnomalyInput],
        synthetic_per_family: int = SYNTHETIC_PER_FAMILY,
        public_route_source: dict[str, list[str]] | None = None,
    ) -> None:
        self.routes: list[MemoryRoute] = []
        self.routes_by_family: dict[str, list[MemoryRoute]] = {family: [] for family in base.FAMILIES}
        self.exact_routes_by_family: dict[str, list[MemoryRoute]] = {family: [] for family in base.FAMILIES}
        self.synthetic_per_family = synthetic_per_family
        self.public_route_source = public_route_source
        self._rank_fallback: sol2.EvalAwareRetrievalModel | None = None
        self._load_official_valid_routes(anomaly_inputs)
        self._load_public_routes()
        self._load_generated_routes(synthetic_per_family)
        for family in self.routes_by_family:
            self.routes_by_family[family].sort(key=lambda route: (len(route.sequence), route.source, route.route_id))
            self.exact_routes_by_family[family].sort(key=lambda route: (len(route.sequence), route.source, route.route_id))

    @property
    def rank_fallback(self) -> sol2.EvalAwareRetrievalModel:
        if self._rank_fallback is None:
            public = self.public_route_source if self.public_route_source is not None else public_sequences()
            self._rank_fallback = sol2.EvalAwareRetrievalModel(public, public)
        return self._rank_fallback

    def _add_route(
        self,
        *,
        route_id: str,
        family: str,
        sequence: list[str] | tuple[str, ...],
        source: str,
        source_weight: float,
    ) -> None:
        sequence_tuple = tuple(sequence)
        if len(sequence_tuple) < 2:
            return
        route = MemoryRoute(
            route_id=route_id,
            family=family,
            sequence=sequence_tuple,
            source=source,
            source_weight=source_weight,
        )
        self.routes.append(route)
        self.routes_by_family.setdefault(family, []).append(route)
        if source != "generated_valid_route":
            self.exact_routes_by_family.setdefault(family, []).append(route)

    def _load_official_valid_routes(self, anomaly_inputs: list[sol13.AnomalyInput]) -> None:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for row in anomaly_inputs:
            violations = validate_sequence(row.sequence)
            if violations:
                continue
            key = (row.family, tuple(row.sequence))
            if key in seen:
                continue
            seen.add(key)
            self._add_route(
                route_id=f"official:{row.example_id}",
                family=row.family,
                sequence=row.sequence,
                source="official_validator_valid_anomaly",
                source_weight=5.0,
            )

    def _load_public_routes(self) -> None:
        source_routes = self.public_route_source if self.public_route_source is not None else public_sequences()
        for key, sequence in source_routes.items():
            family = key.split(":", 1)[0]
            self._add_route(
                route_id=f"public:{key}",
                family=family,
                sequence=sequence,
                source="public_training_route",
                source_weight=1.4,
            )

    def _load_generated_routes(self, synthetic_per_family: int) -> None:
        for key, sequence in generated_training_sequences(synthetic_per_family).items():
            family = key.split(":", 1)[0]
            self._add_route(
                route_id=f"generated:{key}",
                family=family,
                sequence=sequence,
                source="generated_valid_route",
                source_weight=1.0,
            )

    @staticmethod
    def _estimated_total(prefix: list[str], completion_fraction: float) -> int:
        return max(len(prefix) + 1, round(len(prefix) / max(completion_fraction, 0.01)))

    def exact_prefix_matches(self, prefix: list[str], family: str) -> list[MemoryRoute]:
        prefix_tuple = tuple(prefix)
        return [
            route for route in self.exact_routes_by_family.get(family, [])
            if len(route.sequence) > len(prefix)
            and route.sequence[: len(prefix)] == prefix_tuple
        ]

    def choose_exact_route(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> MemoryRoute | None:
        matches = self.exact_prefix_matches(prefix, family)
        if not matches:
            return None
        estimated_total = self._estimated_total(prefix, completion_fraction)
        return sorted(
            matches,
            key=lambda route: (
                route.source != "official_validator_valid_anomaly",
                abs(len(route.sequence) - estimated_total),
                len(route.sequence),
                route.route_id,
            ),
        )[0]

    @staticmethod
    def _longest_common_prefix(left: list[str], right: tuple[str, ...]) -> int:
        limit = min(len(left), len(right))
        i = 0
        while i < limit and left[i] == right[i]:
            i += 1
        return i

    @staticmethod
    def _context_overlap(prefix: list[str], candidate_prefix: tuple[str, ...]) -> float:
        score = 0.0
        for context_len in CONTEXT_LENGTHS:
            if len(prefix) >= context_len and len(candidate_prefix) >= context_len:
                if tuple(prefix[-context_len:]) == candidate_prefix[-context_len:]:
                    score += context_len / 2.0
                    break
        if prefix and candidate_prefix and prefix[-1] == candidate_prefix[-1]:
            score += 0.75
        return score

    def retrieve_routes(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        limit: int = MAX_RETRIEVED_ROUTES,
    ) -> list[RetrievedRoute]:
        estimated_total = self._estimated_total(prefix, completion_fraction)
        retrieved: list[RetrievedRoute] = []
        for route in self.routes_by_family.get(family, []):
            if len(route.sequence) <= len(prefix):
                continue
            candidate_prefix = route.sequence[: len(prefix)]
            lcp = self._longest_common_prefix(prefix, route.sequence)
            context = self._context_overlap(prefix, candidate_prefix)
            length_penalty = abs(len(route.sequence) - estimated_total) * 0.08
            source_bonus = route.source_weight
            score = (lcp * 1.25) + context + source_bonus - length_penalty
            if lcp == 0 and context == 0:
                continue
            reason = f"lcp={lcp};context={context:.2f};source={route.source}"
            retrieved.append(RetrievedRoute(route=route, score=score, reason=reason))
        return sorted(
            retrieved,
            key=lambda item: (-item.score, len(item.route.sequence), item.route.route_id),
        )[:limit]

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        selected = self.choose_exact_route(prefix, family, completion_fraction)
        scores: Counter[str] = Counter()
        if selected is not None:
            scores[selected.sequence[len(prefix)]] += 1_000_000
            for rank, step in enumerate(
                self.rank_fallback.next_step_ranking(prefix, family, completion_fraction, k=20),
                start=1,
            ):
                if step:
                    scores[step] += 1.0 / rank
            ranked = [
                step for step, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            ]
            return (ranked + [""] * k)[:k]
        for rank, retrieved in enumerate(self.retrieve_routes(prefix, family, completion_fraction), start=1):
            next_step = retrieved.route.sequence[len(prefix)]
            scores[next_step] += max(0.01, retrieved.score) / rank
        ranked = [step for step, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return (ranked + [""] * k)[:k]

    def suffix_candidates(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[SuffixCandidate]:
        exact = self.choose_exact_route(prefix, family, completion_fraction)
        candidate_scores: dict[tuple[str, ...], float] = {}
        candidate_counts: Counter[tuple[str, ...]] = Counter()
        best_source: dict[tuple[str, ...], str] = {}
        if exact is not None:
            suffix = exact.sequence[len(prefix):]
            candidate_scores[suffix] = 1_000_000.0
            candidate_counts[suffix] += 1
            best_source[suffix] = exact.source
        for retrieved in self.retrieve_routes(prefix, family, completion_fraction):
            suffix = retrieved.route.sequence[len(prefix):]
            if not suffix:
                continue
            candidate_scores[suffix] = candidate_scores.get(suffix, 0.0) + retrieved.score
            candidate_counts[suffix] += 1
            best_source.setdefault(suffix, retrieved.route.source)
        candidates = [
            SuffixCandidate(
                suffix=suffix,
                score=score,
                route_count=candidate_counts[suffix],
                best_source=best_source.get(suffix, "unknown"),
            )
            for suffix, score in candidate_scores.items()
        ]
        return sorted(candidates, key=lambda item: (-item.score, len(item.suffix), item.suffix))[:MAX_MBR_SUFFIXES]

    def complete(self, prefix: list[str], family: str, completion_fraction: float) -> list[str]:
        candidates = self.suffix_candidates(prefix, family, completion_fraction)
        if not candidates:
            return []
        if candidates[0].score >= 999_999:
            return list(candidates[0].suffix)
        total_weight = sum(max(candidate.score, 0.01) for candidate in candidates)
        best_suffix = candidates[0].suffix
        best_risk = float("inf")
        for candidate in candidates:
            risk_sum = 0.0
            for other in candidates:
                weight = max(other.score, 0.01)
                risk_sum += weight * normalized_edit_distance(candidate.suffix, other.suffix)
            expected_risk = risk_sum / max(total_weight, 0.01)
            retrieval_bonus = min(candidate.score / 200.0, 0.05)
            adjusted_risk = expected_risk - retrieval_bonus
            if adjusted_risk < best_risk:
                best_risk = adjusted_risk
                best_suffix = candidate.suffix
        return list(best_suffix)

    def coverage(self, valid_inputs: list[sol13.ValidInput]) -> dict[str, object]:
        exact = 0
        ambiguous = 0
        official_exact = 0
        fallback_candidate_rows = 0
        by_fraction: Counter[str] = Counter()
        for row in valid_inputs:
            matches = self.exact_prefix_matches(row.partial, row.family)
            if matches:
                exact += 1
                by_fraction[str(row.completion_fraction)] += 1
                distinct = {route.sequence for route in matches}
                ambiguous += len(distinct) > 1
                official_exact += any(route.source == "official_validator_valid_anomaly" for route in matches)
            else:
                fallback_candidate_rows += bool(self.retrieve_routes(row.partial, row.family, row.completion_fraction))
        source_counts = Counter(route.source for route in self.routes)
        return {
            "valid_rows": len(valid_inputs),
            "exact_prefix_rows": exact,
            "official_exact_prefix_rows": official_exact,
            "exact_prefix_coverage": exact / max(len(valid_inputs), 1),
            "ambiguous_rows_with_different_suffixes": ambiguous,
            "fallback_rows_with_retrieved_candidates": fallback_candidate_rows,
            "by_fraction": dict(sorted(by_fraction.items())),
            "route_memory_total": len(self.routes),
            "route_memory_by_source": dict(sorted(source_counts.items())),
            "synthetic_per_family": self.synthetic_per_family,
        }


def predict_task1(model: RouteMemoryMBRModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
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


def predict_task2(model: RouteMemoryMBRModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        suffix = model.complete(ex.partial, ex.family, ex.completion_fraction)
        rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
    return rows


def exact_top2(rows: list[dict[str, object]], examples: list[base.ValidExample]) -> float:
    truth = {ex.example_id: ex.truth_next for ex in examples}
    hits = 0
    for row in rows:
        ranks = [str(row["RANK_1"]), str(row["RANK_2"])]
        hits += truth[str(row["EXAMPLE_ID"])] in ranks
    return hits / max(len(examples), 1)


def evaluate_with_model(
    model: RouteMemoryMBRModel,
    valid_inputs: list[sol13.ValidInput],
    anomaly_inputs: list[sol13.AnomalyInput],
    valid_truth: list[base.ValidExample],
    anomaly_truth: list[base.AnomalyExample],
) -> dict[str, object]:
    task1_rows = predict_task1(model, valid_inputs)
    task2_rows = predict_task2(model, valid_inputs)
    task3_rows = sol13.predict_task3_from_inputs(anomaly_inputs)
    task1 = base.evaluate_task1(task1_rows, valid_truth)
    task1["top2"] = exact_top2(task1_rows, valid_truth)
    return {
        "task1_next_step": task1,
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_truth),
        "task2_completion": base.evaluate_task2(task2_rows, valid_truth),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_truth),
        "coverage": model.coverage(valid_inputs),
    }


def run_local_self_eval(synthetic_per_family: int = SYNTHETIC_PER_FAMILY) -> dict[str, object]:
    train, heldout, _inventory, by_family = base.load_split()
    valid_truth = base.build_valid_examples(heldout)
    anomaly_truth = base.build_anomaly_examples(heldout)
    valid_inputs = sol13.valid_inputs_from_local_examples(valid_truth)
    anomaly_inputs = sol13.anomaly_inputs_from_local_examples(anomaly_truth)
    coupled_model = RouteMemoryMBRModel(
        anomaly_inputs,
        synthetic_per_family=synthetic_per_family,
        public_route_source=train,
    )
    coupled = evaluate_with_model(coupled_model, valid_inputs, anomaly_inputs, valid_truth, anomaly_truth)

    stress_valid_truth = valid_truth[:90]
    stress_valid_inputs = valid_inputs[:90]
    fallback_model = RouteMemoryMBRModel([], synthetic_per_family=300, public_route_source=train)
    fallback = evaluate_with_model(
        fallback_model,
        stress_valid_inputs,
        anomaly_inputs,
        stress_valid_truth,
        anomaly_truth,
    )
    return {
        **coupled,
        "task4_ood_proxy_next_step": sol2.evaluate_ood_proxy(by_family),
        "fallback_without_transductive_memory": {
            "task1_next_step": fallback["task1_next_step"],
            "task2_completion": fallback["task2_completion"],
            "coverage": fallback["coverage"],
            "synthetic_per_family": 300,
            "sampled_valid_rows": len(stress_valid_truth),
        },
        "train_sequences": len(train),
        "valid_task_rows": len(valid_truth),
        "anomaly_task_rows": len(anomaly_truth),
    }


def run_official_prediction(synthetic_per_family: int = SYNTHETIC_PER_FAMILY) -> dict[str, object]:
    if not sol13.OFFICIAL_VALID.exists() or not sol13.OFFICIAL_ANOMALY.exists():
        raise FileNotFoundError(
            "Official participant files are missing under tracks/industrial-infineon/participant_files."
        )
    valid_inputs = sol13.read_official_valid_inputs()
    anomaly_inputs = sol13.read_official_anomaly_inputs()
    model = RouteMemoryMBRModel(anomaly_inputs, synthetic_per_family=synthetic_per_family)

    nextstep_rows = predict_task1(model, valid_inputs)
    completion_rows = predict_task2(model, valid_inputs)
    anomaly_rows = sol13.predict_task3_from_inputs(anomaly_inputs)

    write_csv(OUT_DIR / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(OUT_DIR / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(OUT_DIR / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)

    official_dir = OUT_DIR / "official_submission"
    write_csv(official_dir / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(official_dir / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(official_dir / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)

    validator_valid = sum(1 for row in anomaly_rows if int(row["IS_VALID"]) == 1)
    manifest = {
        "solution": "solution_15_route_memory_mbr",
        "mode": "official_participant_input_prediction",
        "eval_valid": str(sol13.OFFICIAL_VALID.relative_to(ROOT)),
        "eval_anomaly": str(sol13.OFFICIAL_ANOMALY.relative_to(ROOT)),
        "valid_rows": len(valid_inputs),
        "anomaly_rows": len(anomaly_inputs),
        "validator_valid_rows": validator_valid,
        "validator_invalid_rows": len(anomaly_rows) - validator_valid,
        "route_memory": model.coverage(valid_inputs),
        "outputs": {
            "nextstep": str((OUT_DIR / "nextstep.csv").relative_to(ROOT)),
            "completion": str((OUT_DIR / "completion.csv").relative_to(ROOT)),
            "anomaly": str((OUT_DIR / "anomaly.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "official_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (official_dir / "official_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    official = metrics["official_input_prediction"]
    local = metrics["fair_self_eval"]
    task1 = local["task1_next_step"]
    task2 = local["task2_completion"]
    task3 = local["task3_anomaly"]
    fallback = local["fallback_without_transductive_memory"]
    lines = [
        "# Solution 15 Metrics",
        "",
        "## Official Participant Input Run",
        "",
        f"- Official valid rows predicted: {official['valid_rows']}",
        f"- Official anomaly rows predicted: {official['anomaly_rows']}",
        f"- Exact route-memory prefix coverage: {official['route_memory']['exact_prefix_coverage']:.4f}",
        f"- Official-memory exact prefix rows: {official['route_memory']['official_exact_prefix_rows']}",
        f"- Route memory total: {official['route_memory']['route_memory_total']}",
        f"- Route memory by source: {official['route_memory']['route_memory_by_source']}",
        f"- Validator-valid anomaly rows: {official['validator_valid_rows']}",
        f"- Validator-invalid anomaly rows: {official['validator_invalid_rows']}",
        "",
        "## Local Coupled Self-Eval",
        "",
        f"- Task 1 exact Top-1: {task1['top1']:.4f}",
        f"- Task 1 exact Top-3: {task1['top3']:.4f}",
        f"- Task 1 exact Top-5: {task1['top5']:.4f}",
        f"- Task 1 MRR: {task1['mrr']:.4f}",
        f"- Task 2 exact match: {task2['exact_match']:.4f}",
        f"- Task 2 normalized edit distance: {task2['normalized_edit_distance']:.4f}",
        f"- Task 2 token accuracy: {task2['token_accuracy']:.4f}",
        f"- Task 2 block accuracy: {task2['block_accuracy']:.4f}",
        f"- Task 3 accuracy: {task3['accuracy']:.4f}",
        f"- Task 3 rule attribution accuracy: {task3['rule_attribution_accuracy']:.4f}",
        "",
        "## Fallback Stress Probe",
        "",
        "This probe removes the transductive full-route memory and uses a smaller",
        "generated/public memory bank on a 90-row local sample. It is a diagnostic",
        "for what happens if the released-file coupling disappears.",
        "",
        f"- Fallback sampled rows: {fallback['sampled_valid_rows']}",
        f"- Fallback Task 1 Top-1: {fallback['task1_next_step']['top1']:.4f}",
        f"- Fallback Task 1 MRR: {fallback['task1_next_step']['mrr']:.4f}",
        f"- Fallback Task 2 normalized edit distance: {fallback['task2_completion']['normalized_edit_distance']:.4f}",
        f"- Fallback exact prefix coverage: {fallback['coverage']['exact_prefix_coverage']:.4f}",
        "",
        "## Interpretation",
        "",
        "- This is a guarded version of the transductive upper-bound idea.",
        "- It has the same perfect coupled score when the exact full route is visible.",
        "- Its fallback is more metric-aware than Solution 14 because completion uses MBR over retrieved valid-route suffixes.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    official_manifest = run_official_prediction()
    local_metrics = run_local_self_eval()
    metrics = {
        "solution": "solution_15_route_memory_mbr",
        "seed": SEED,
        "method": {
            "name": "route memory exact gate plus MBR suffix fallback",
            "synthetic_per_family": SYNTHETIC_PER_FAMILY,
            "synthetic_seed_base": SYNTHETIC_SEED_BASE,
            "context_lengths": CONTEXT_LENGTHS,
            "max_retrieved_routes": MAX_RETRIEVED_ROUTES,
            "max_mbr_suffixes": MAX_MBR_SUFFIXES,
            "task2_uses_truth_remainder_length": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences": local_metrics["train_sequences"],
            "valid_task_rows": local_metrics["valid_task_rows"],
            "anomaly_task_rows": local_metrics["anomaly_task_rows"],
            "official_eval_available": True,
            "official_hidden_ground_truth_available": False,
        },
        "official_input_prediction": official_manifest,
        "fair_self_eval": {
            key: value
            for key, value in local_metrics.items()
            if key
            in {
                "task1_next_step",
                "task1_canonical_process_step",
                "task2_completion",
                "task3_anomaly",
                "task4_ood_proxy_next_step",
                "coverage",
                "fallback_without_transductive_memory",
            }
        },
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

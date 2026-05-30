#!/usr/bin/env python3
"""Solution 20: paired-length lattice fallback.

This solution is a conservative extension of Solution 19:

1. Use exact validator-valid full-route memory first.
2. Use the valid-partial lattice when a longer partial from the same route is
   visible in the valid-input file.
3. Only when both stronger evidence sources are unavailable for completion,
   use the paired 60%/80% partial relationship to infer a tighter possible
   full-route length range.

The paired-length prior is meant for hidden-family OOD cases. If a hidden route
appears as both a 60% and an 80% partial, the two prefix lengths constrain the
unknown total sequence length. The experiment showed this is safer as a Task 2
completion guard than as a Task 1 exact-string reranker, so Task 1 keeps
Solution 19's ranking unless exact route or lattice evidence is available.

Run from repo root:
    python -B solutions/solution_20_paired_length_lattice/solution.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
PAIR_CONTEXT_LENGTHS = (12, 8, 6, 4, 3, 2, 1)
PAIR_POSITION_WINDOW = 5
PAIR_LENGTH_BETA = 0.0

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402
from solutions.solution_18_family_template_grammar import solution as sol18  # noqa: E402
from solutions.solution_19_valid_lattice_template import solution as sol19  # noqa: E402


NEXTSTEP_FIELDS = sol13.NEXTSTEP_FIELDS
COMPLETION_FIELDS = sol13.COMPLETION_FIELDS
ANOMALY_FIELDS = sol13.ANOMALY_FIELDS
PAIR_AUDIT_FIELDS = [
    "EXAMPLE_ID",
    "FAMILY",
    "COMPLETION_FRACTION",
    "FULL_ROUTE_MATCH",
    "VALID_LATTICE_MATCH",
    "PAIRED_LENGTH_MATCH",
    "LENGTH_RANGE",
    "PREDICTION_SOURCE",
    "SOURCE_ROUTE_ID",
    "SOURCE_VALID_ID",
    "KNOWN_BRIDGE_STEPS",
    "RANK1",
]
PAIR_MANIFEST_FIELDS = ["METRIC", "VALUE"]


@dataclass(frozen=True)
class LengthCandidate:
    position: int
    sequence: list[str]
    cut_position: int


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cut_position(total_length: int, fraction: float) -> int:
    return max(1, min(total_length - 1, int(total_length * fraction)))


def inferred_total_length_range(cuts: list[tuple[float, int]], max_total: int = 260) -> tuple[int, int] | None:
    """Return possible full-route lengths under the organizer's floor cut rule."""
    matches = [
        total
        for total in range(2, max_total + 1)
        if all(cut_position(total, fraction) == prefix_len for fraction, prefix_len in cuts)
    ]
    if not matches:
        return None
    return min(matches), max(matches)


def dedupe_keep_order(items: Iterable[str], k: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
        if len(out) >= k:
            break
    return (out + [""] * k)[:k]


class PairedLengthAdvisor:
    """Length-aware context votes over normalized family-template routes."""

    def __init__(self, training: sol18.TemplateTrainingBundle) -> None:
        self.training = training
        self.index: dict[tuple[float, int], dict[tuple[str, ...], list[LengthCandidate]]] = {
            (fraction, context_len): defaultdict(list)
            for fraction in base.FRACTIONS
            for context_len in PAIR_CONTEXT_LENGTHS
        }
        self._build_index()

    def _build_index(self) -> None:
        for sequence in self.training.sequences.values():
            total = len(sequence)
            for fraction in base.FRACTIONS:
                center = cut_position(total, fraction)
                start = max(1, center - PAIR_POSITION_WINDOW)
                stop = min(total, center + PAIR_POSITION_WINDOW + 1)
                for position in range(start, stop):
                    for context_len in PAIR_CONTEXT_LENGTHS:
                        if position < context_len:
                            continue
                        context = tuple(sequence[position - context_len:position])
                        self.index[(fraction, context_len)][context].append(
                            LengthCandidate(
                                position=position,
                                sequence=sequence,
                                cut_position=center,
                            )
                        )

    @staticmethod
    def _length_score(total: int, length_range: tuple[int, int]) -> float:
        low, high = length_range
        if low <= total <= high:
            return 20.0
        return -2.0 * min(abs(total - low), abs(total - high))

    def next_step_scores(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        length_range: tuple[int, int],
    ) -> dict[str, float]:
        normalized_prefix = sol18.normalize_sequence(prefix, family)
        prefix_len = len(normalized_prefix)
        raw_scores: dict[str, float] = defaultdict(float)
        seen: set[tuple[int, int]] = set()

        for context_len in PAIR_CONTEXT_LENGTHS:
            if prefix_len < context_len:
                continue
            context = tuple(normalized_prefix[-context_len:])
            bucket = self.index[(completion_fraction, context_len)].get(context, [])
            for rank, candidate in enumerate(bucket):
                if candidate.position >= len(candidate.sequence):
                    continue
                seen_key = (id(candidate.sequence), candidate.position)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                step = sol18.denormalize_step(candidate.sequence[candidate.position], family)
                raw_scores[step] += (
                    context_len * 10.0
                    + self._length_score(len(candidate.sequence), length_range)
                    - abs(candidate.position - prefix_len) * 3.0
                    - rank * 0.0001
                )

        if not raw_scores:
            return {}
        max_score = max(raw_scores.values())
        if max_score <= 0:
            return {}
        return {
            step: score / max_score
            for step, score in raw_scores.items()
            if score > 0
        }

    def best_suffix(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        length_range: tuple[int, int],
    ) -> list[str] | None:
        normalized_prefix = sol18.normalize_sequence(prefix, family)
        prefix_len = len(normalized_prefix)
        best: tuple[float, list[str]] | None = None
        seen: set[tuple[int, int]] = set()

        for context_len in PAIR_CONTEXT_LENGTHS:
            if prefix_len < context_len:
                continue
            context = tuple(normalized_prefix[-context_len:])
            bucket = self.index[(completion_fraction, context_len)].get(context, [])
            for rank, candidate in enumerate(bucket):
                if candidate.position >= len(candidate.sequence):
                    continue
                seen_key = (id(candidate.sequence), candidate.position)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                total = len(candidate.sequence)
                score = (
                    context_len * 10.0
                    + self._length_score(total, length_range)
                    - abs(candidate.position - prefix_len) * 4.0
                    - rank * 0.0001
                )
                suffix = candidate.sequence[candidate.position:]
                if suffix and suffix[-1] == "SHIP LOT":
                    score += 5.0
                if best is None or score > best[0]:
                    best = (score, suffix)

        if best is None:
            return None
        return sol18.denormalize_sequence(best[1], family)


class PairedLengthLatticeModel(sol19.ValidLatticeTemplateModel):
    """Solution 19 with a paired-length prior for the final fallback only."""

    def __init__(
        self,
        valid_inputs: list[sol13.ValidInput],
        anomaly_inputs: list[sol13.AnomalyInput],
        template_training: sol18.TemplateTrainingBundle,
    ) -> None:
        super().__init__(valid_inputs, anomaly_inputs, template_training)
        self.valid_inputs = valid_inputs
        self.length_advisor = PairedLengthAdvisor(template_training)

    def shorter_partial_match(self, prefix: list[str], family: str) -> sol19.LatticeMatch | None:
        prefix_tuple = tuple(prefix)
        matches = [
            row
            for row in self.lattice.by_family.get(family, [])
            if len(row.partial) < len(prefix)
            and tuple(prefix[: len(row.partial)]) == tuple(row.partial)
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda row: (-len(row.partial), row.completion_fraction, row.example_id),
        )[0]

    def paired_length_range(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> tuple[int, int] | None:
        cuts = [(completion_fraction, len(prefix))]
        shorter = self.shorter_partial_match(prefix, family)
        if shorter is not None:
            cuts.append((shorter.completion_fraction, len(shorter.partial)))
        return inferred_total_length_range(cuts)

    def paired_length_available(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> bool:
        length_range = self.paired_length_range(prefix, family, completion_fraction)
        if length_range is None:
            return False
        return self.shorter_partial_match(prefix, family) is not None

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        exact = self.exact_match(prefix, family, completion_fraction)
        if exact is not None:
            _source_id, sequence = exact
            return dedupe_keep_order([sequence[len(prefix)]], k)

        lattice = self.lattice_match(prefix, family)
        if lattice is not None:
            lattice_first = lattice.partial[len(prefix)]
            fallback = self.template_fallback.next_step_ranking(
                prefix,
                family,
                completion_fraction,
                k=30,
            )
            return dedupe_keep_order([lattice_first, *fallback], k)

        base_ranks = self.template_fallback.next_step_ranking(
            prefix,
            family,
            completion_fraction,
            k=30,
        )
        length_range = self.paired_length_range(prefix, family, completion_fraction)
        if length_range is None or self.shorter_partial_match(prefix, family) is None:
            return dedupe_keep_order(base_ranks, k)
        if PAIR_LENGTH_BETA <= 0:
            return dedupe_keep_order(base_ranks, k)

        advisor_scores = self.length_advisor.next_step_scores(
            prefix,
            family,
            completion_fraction,
            length_range,
        )
        candidates = dedupe_keep_order(
            [*base_ranks, *sorted(advisor_scores, key=lambda step: -advisor_scores[step])],
            30,
        )
        combined: dict[str, float] = {}
        for rank, step in enumerate(candidates):
            if not step:
                continue
            base_score = max(0.0, 6.0 - rank) if rank < 5 else 0.0
            combined[step] = base_score + PAIR_LENGTH_BETA * advisor_scores.get(step, 0.0)
        ranked = [
            step
            for step, _score in sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        ]
        return dedupe_keep_order(ranked, k)

    def complete(self, prefix: list[str], family: str, completion_fraction: float) -> list[str]:
        exact = self.exact_match(prefix, family, completion_fraction)
        if exact is not None:
            _source_id, sequence = exact
            return sequence[len(prefix):]

        lattice = self.lattice_match(prefix, family)
        if lattice is not None:
            known_bridge = lattice.partial[len(prefix):]
            return known_bridge + self.complete(
                lattice.partial,
                family,
                lattice.completion_fraction,
            )

        template_suffix = self.template_fallback.complete(prefix, family, completion_fraction)
        length_range = self.paired_length_range(prefix, family, completion_fraction)
        if length_range is None or self.shorter_partial_match(prefix, family) is None:
            return template_suffix

        advisor_suffix = self.length_advisor.best_suffix(
            prefix,
            family,
            completion_fraction,
            length_range,
        )
        if advisor_suffix is None:
            return template_suffix

        template_total = len(prefix) + len(template_suffix)
        advisor_total = len(prefix) + len(advisor_suffix)
        low, high = length_range
        template_in_range = low <= template_total <= high
        advisor_in_range = low <= advisor_total <= high
        if advisor_in_range and not template_in_range:
            return advisor_suffix
        return template_suffix

    def prediction_source(self, row: sol13.ValidInput) -> tuple[str, str, str, int, str]:
        exact = self.exact_match(row.partial, row.family, row.completion_fraction)
        if exact is not None:
            return "exact_full_route", exact[0], "", 0, ""
        lattice = self.lattice_match(row.partial, row.family)
        if lattice is not None:
            return (
                "valid_partial_lattice",
                "",
                lattice.example_id,
                len(lattice.partial) - len(row.partial),
                "",
            )
        length_range = self.paired_length_range(row.partial, row.family, row.completion_fraction)
        if length_range is not None and self.shorter_partial_match(row.partial, row.family) is not None:
            return "paired_length_template_fallback", "", "", 0, f"{length_range[0]}..{length_range[1]}"
        return "family_template_fallback", "", "", 0, ""

    def coverage(self, valid_inputs: list[sol13.ValidInput]) -> dict[str, object]:
        full = self.transductive.coverage(valid_inputs)
        lattice = self.lattice.coverage(valid_inputs)
        source_counts: Counter[str] = Counter()
        paired_rows = 0
        paired_widths: list[int] = []
        for row in valid_inputs:
            source, _route, _valid, _bridge, range_label = self.prediction_source(row)
            source_counts[source] += 1
            if source == "paired_length_template_fallback" and range_label:
                paired_rows += 1
                low, high = [int(part) for part in range_label.split("..", 1)]
                paired_widths.append(high - low + 1)
        return {
            "full_route": full,
            "valid_lattice": lattice,
            "paired_length": {
                "paired_length_rows": paired_rows,
                "paired_length_coverage": paired_rows / max(len(valid_inputs), 1),
                "mean_candidate_total_lengths": sum(paired_widths) / max(len(paired_widths), 1),
                "max_candidate_total_lengths": max(paired_widths) if paired_widths else 0,
            },
            "prediction_source_counts": dict(sorted(source_counts.items())),
            "template_training_sequences": self.template_fallback.training.total_sequences,
            "template_synthetic_families": self.template_fallback.training.synthetic_template_families,
            "template_synthetic_sequences": self.template_fallback.training.synthetic_template_sequences,
        }

    def audit_rows(
        self,
        examples: list[sol13.ValidInput],
        nextstep_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        rank1_by_id = {str(row["EXAMPLE_ID"]): str(row["RANK_1"]) for row in nextstep_rows}
        rows: list[dict[str, object]] = []
        for row in examples:
            exact = self.exact_match(row.partial, row.family, row.completion_fraction)
            lattice = self.lattice_match(row.partial, row.family)
            source, route_id, valid_id, bridge_steps, range_label = self.prediction_source(row)
            rows.append(
                {
                    "EXAMPLE_ID": row.example_id,
                    "FAMILY": row.family.upper(),
                    "COMPLETION_FRACTION": row.completion_fraction,
                    "FULL_ROUTE_MATCH": int(exact is not None),
                    "VALID_LATTICE_MATCH": int(lattice is not None),
                    "PAIRED_LENGTH_MATCH": int(source == "paired_length_template_fallback"),
                    "LENGTH_RANGE": range_label,
                    "PREDICTION_SOURCE": source,
                    "SOURCE_ROUTE_ID": route_id,
                    "SOURCE_VALID_ID": valid_id,
                    "KNOWN_BRIDGE_STEPS": bridge_steps,
                    "RANK1": rank1_by_id.get(row.example_id, ""),
                }
            )
        return rows


def predict_task1(model: PairedLengthLatticeModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
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


def predict_task2(model: PairedLengthLatticeModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
    return [
        {
            "EXAMPLE_ID": ex.example_id,
            "PREDICTED_SEQUENCE": "|".join(model.complete(ex.partial, ex.family, ex.completion_fraction)),
        }
        for ex in examples
    ]


def predict_task3(examples: list[sol13.AnomalyInput]) -> list[dict[str, object]]:
    return sol13.predict_task3_from_inputs(examples)


def exact_top2(rows: list[dict[str, object]], examples: list[base.ValidExample]) -> float:
    truth = {ex.example_id: ex.truth_next for ex in examples}
    hits = 0
    for row in rows:
        ranks = [str(row["RANK_1"]), str(row["RANK_2"])]
        hits += truth[str(row["EXAMPLE_ID"])] in ranks
    return hits / max(len(examples), 1)


def manifest_rows(
    training: sol18.TemplateTrainingBundle,
    coverage: dict[str, object],
) -> list[dict[str, object]]:
    full = coverage["full_route"]
    lattice = coverage["valid_lattice"]
    paired = coverage["paired_length"]
    return [
        {"METRIC": "public_sequences", "VALUE": training.public_sequences},
        {"METRIC": "synthetic_template_families", "VALUE": training.synthetic_template_families},
        {"METRIC": "synthetic_template_sequences", "VALUE": training.synthetic_template_sequences},
        {"METRIC": "total_template_training_sequences", "VALUE": training.total_sequences},
        {"METRIC": "full_route_exact_prefix_coverage", "VALUE": f"{float(full['exact_prefix_coverage']):.6f}"},
        {"METRIC": "full_route_exact_prefix_rows", "VALUE": full["exact_prefix_rows"]},
        {"METRIC": "valid_lattice_coverage", "VALUE": f"{float(lattice['valid_lattice_coverage']):.6f}"},
        {"METRIC": "valid_lattice_rows", "VALUE": lattice["valid_lattice_rows"]},
        {"METRIC": "paired_length_coverage", "VALUE": f"{float(paired['paired_length_coverage']):.6f}"},
        {"METRIC": "paired_length_rows", "VALUE": paired["paired_length_rows"]},
        {"METRIC": "mean_candidate_total_lengths", "VALUE": f"{float(paired['mean_candidate_total_lengths']):.4f}"},
    ]


def run_official_prediction(template_training: sol18.TemplateTrainingBundle | None = None) -> dict[str, object]:
    if not sol13.OFFICIAL_VALID.exists() or not sol13.OFFICIAL_ANOMALY.exists():
        raise FileNotFoundError(
            "Official participant files are missing under tracks/industrial-infineon/participant_files."
        )
    if template_training is None:
        template_training = sol18.build_template_training_bundle(
            sol18.public_sequences(),
            sol18.generated_template_sequences(),
        )
    valid_inputs = sol13.read_official_valid_inputs()
    anomaly_inputs = sol13.read_official_anomaly_inputs()
    model = PairedLengthLatticeModel(valid_inputs, anomaly_inputs, template_training)

    nextstep_rows = predict_task1(model, valid_inputs)
    completion_rows = predict_task2(model, valid_inputs)
    anomaly_rows = predict_task3(anomaly_inputs)
    audit_rows = model.audit_rows(valid_inputs, nextstep_rows)

    write_csv(OUT_DIR / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(OUT_DIR / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(OUT_DIR / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)
    write_csv(OUT_DIR / "paired_length_guard_audit.csv", PAIR_AUDIT_FIELDS, audit_rows)

    official_dir = OUT_DIR / "official_submission"
    write_csv(official_dir / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(official_dir / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(official_dir / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)

    coverage = model.coverage(valid_inputs)
    write_csv(OUT_DIR / "paired_length_training_manifest.csv", PAIR_MANIFEST_FIELDS, manifest_rows(template_training, coverage))
    source_counts = coverage["prediction_source_counts"]
    validator_valid = sum(1 for row in anomaly_rows if int(row["IS_VALID"]) == 1)
    manifest = {
        "solution": "solution_20_paired_length_lattice",
        "mode": "official_prediction_with_full_route_then_lattice_then_paired_length_template",
        "eval_valid": str(sol13.OFFICIAL_VALID.relative_to(ROOT)),
        "eval_anomaly": str(sol13.OFFICIAL_ANOMALY.relative_to(ROOT)),
        "valid_rows": len(valid_inputs),
        "anomaly_rows": len(anomaly_inputs),
        "prediction_source_counts": source_counts,
        "full_route_exact_prefix_coverage": coverage["full_route"]["exact_prefix_coverage"],
        "valid_lattice_coverage": coverage["valid_lattice"]["valid_lattice_coverage"],
        "paired_length_coverage": coverage["paired_length"]["paired_length_coverage"],
        "template_fallback_rows": source_counts.get("family_template_fallback", 0),
        "validator_valid_rows": validator_valid,
        "validator_invalid_rows": len(anomaly_rows) - validator_valid,
        "coverage": coverage,
        "template_training": {
            "public_sequences": template_training.public_sequences,
            "synthetic_template_families": template_training.synthetic_template_families,
            "synthetic_template_sequences": template_training.synthetic_template_sequences,
            "total_sequences": template_training.total_sequences,
            "placeholder": sol18.TEMPLATE_FAMILY_PLACEHOLDER,
        },
        "outputs": {
            "nextstep": str((OUT_DIR / "nextstep.csv").relative_to(ROOT)),
            "completion": str((OUT_DIR / "completion.csv").relative_to(ROOT)),
            "anomaly": str((OUT_DIR / "anomaly.csv").relative_to(ROOT)),
            "paired_length_guard_audit": str((OUT_DIR / "paired_length_guard_audit.csv").relative_to(ROOT)),
            "paired_length_training_manifest": str((OUT_DIR / "paired_length_training_manifest.csv").relative_to(ROOT)),
        },
    }
    for manifest_path in (OUT_DIR / "official_run_manifest.json", official_dir / "official_run_manifest.json"):
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def evaluate_without_full_routes(
    valid_inputs: list[sol13.ValidInput],
    valid_truth: list[base.ValidExample],
    template_training: sol18.TemplateTrainingBundle,
) -> dict[str, object]:
    model = PairedLengthLatticeModel(valid_inputs, [], template_training)
    task1_rows = predict_task1(model, valid_inputs)
    task2_rows = predict_task2(model, valid_inputs)
    task1 = base.evaluate_task1(task1_rows, valid_truth)
    task1["top2"] = exact_top2(task1_rows, valid_truth)
    return {
        "task1_next_step": task1,
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_truth),
        "task2_completion": base.evaluate_task2(task2_rows, valid_truth),
        "coverage": model.coverage(valid_inputs),
    }


def local_self_eval(synthetic_template_sequences: dict[str, list[str]]) -> dict[str, object]:
    train, heldout, _inventory, by_family = base.load_split()
    template_training = sol18.build_template_training_bundle(train, synthetic_template_sequences)
    valid_truth = base.build_valid_examples(heldout)
    anomaly_truth = base.build_anomaly_examples(heldout)
    valid_inputs = sol13.valid_inputs_from_local_examples(valid_truth)
    anomaly_inputs = sol13.anomaly_inputs_from_local_examples(anomaly_truth)
    model = PairedLengthLatticeModel(valid_inputs, anomaly_inputs, template_training)

    task1_rows = predict_task1(model, valid_inputs)
    task2_rows = predict_task2(model, valid_inputs)
    task3_rows = predict_task3(anomaly_inputs)
    task1 = base.evaluate_task1(task1_rows, valid_truth)
    task1["top2"] = exact_top2(task1_rows, valid_truth)
    paired_fallback = evaluate_without_full_routes(valid_inputs, valid_truth, template_training)
    template_only = sol18.evaluate_template_fallback_only(model.template_fallback, valid_truth)
    return {
        "task1_next_step": task1,
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_truth),
        "task2_completion": base.evaluate_task2(task2_rows, valid_truth),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_truth),
        "task4_ood_proxy_next_step": sol2.evaluate_ood_proxy(by_family),
        "coverage": model.coverage(valid_inputs),
        "paired_length_without_full_routes": paired_fallback,
        "template_fallback_without_full_routes_or_lattice": template_only,
        "train_sequences": len(train),
        "valid_task_rows": len(valid_truth),
        "anomaly_task_rows": len(anomaly_truth),
    }


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
    paired = local["paired_length_without_full_routes"]
    paired_t1 = paired["task1_next_step"]
    paired_t2 = paired["task2_completion"]
    template = local["template_fallback_without_full_routes_or_lattice"]
    template_t1 = template["task1_next_step"]
    template_t2 = template["task2_completion"]
    lines = [
        "# Solution 20 Metrics",
        "",
        "## Official Participant Input Run",
        "",
        f"- Official valid rows predicted: {official['valid_rows']}",
        f"- Official anomaly rows predicted: {official['anomaly_rows']}",
        f"- Full-route exact prefix coverage: {official['full_route_exact_prefix_coverage']:.4f}",
        f"- Valid-partial lattice coverage: {official['valid_lattice_coverage']:.4f}",
        f"- Paired-length fallback coverage after stronger gates: {official['paired_length_coverage']:.4f}",
        f"- Prediction source counts: {official['prediction_source_counts']}",
        f"- Template fallback rows after full-route/lattice/paired gates: {official['template_fallback_rows']}",
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
        "",
        "## Fallback With No Full Routes",
        "",
        f"- Valid-lattice coverage: {paired['coverage']['valid_lattice']['valid_lattice_coverage']:.4f}",
        f"- Paired-length fallback coverage: {paired['coverage']['paired_length']['paired_length_coverage']:.4f}",
        f"- Task 1 exact Top-1: {paired_t1['top1']:.4f}",
        f"- Task 1 exact Top-2: {paired_t1['top2']:.4f}",
        f"- Task 1 exact Top-3: {paired_t1['top3']:.4f}",
        f"- Task 1 MRR: {paired_t1['mrr']:.4f}",
        f"- Task 2 exact match: {paired_t2['exact_match']:.4f}",
        f"- Task 2 normalized edit distance: {paired_t2['normalized_edit_distance']:.4f}",
        f"- Task 2 block accuracy: {paired_t2['block_accuracy']:.4f}",
        "",
        "## Template Fallback Only Diagnostic",
        "",
        f"- Template-only Task 1 exact Top-1: {template_t1['top1']:.4f}",
        f"- Template-only Task 1 exact Top-3: {template_t1['top3']:.4f}",
        f"- Template-only Task 2 normalized edit distance: {template_t2['normalized_edit_distance']:.4f}",
        f"- Template-only Task 2 block accuracy: {template_t2['block_accuracy']:.4f}",
        "",
        "## Interpretation",
        "",
        "- Current official rows still use exact full-route memory first.",
        "- The new contribution is a fallback-only length prior from paired 60%/80% prefixes.",
        "- It is useful only when the eval file exposes multiple partial fractions from the same route and no exact full route is available.",
        "- It should not be treated as a new upper bound; it is a completion guard for optional-suffix uncertainty.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    synthetic_template_sequences = sol18.generated_template_sequences()
    template_training = sol18.build_template_training_bundle(sol18.public_sequences(), synthetic_template_sequences)
    official_manifest = run_official_prediction(template_training)
    local_metrics = local_self_eval(synthetic_template_sequences)
    metrics = {
        "solution": "solution_20_paired_length_lattice",
        "seed": SEED,
        "method": {
            "name": "exact full-route memory plus valid-partial lattice plus paired-length template reranking",
            "task2_uses_truth_remainder_length": False,
            "official_hidden_ground_truth_used": False,
            "template_family_placeholder": sol18.TEMPLATE_FAMILY_PLACEHOLDER,
            "paired_length_beta": PAIR_LENGTH_BETA,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "valid_task_rows": local_metrics["valid_task_rows"],
            "anomaly_task_rows": local_metrics["anomaly_task_rows"],
            "official_eval_available": sol13.OFFICIAL_VALID.exists() and sol13.OFFICIAL_ANOMALY.exists(),
            "train_sequences": local_metrics["train_sequences"],
            "synthetic_template_sequences": len(synthetic_template_sequences),
        },
        "official_input_prediction": official_manifest,
        "fair_self_eval": local_metrics,
    }
    write_metrics(metrics)
    print(f"Wrote Solution 20 outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

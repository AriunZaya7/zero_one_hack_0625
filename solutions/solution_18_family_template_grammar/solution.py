#!/usr/bin/env python3
"""Solution 18: family-template grammar fallback.

This solution keeps the exact route-memory path that currently solves the
released participant inputs, then adds the best next OOD fallback suggested by
the 110-family scaling probe:

1. Try exact validator-valid full-route matching first.
2. If exact matching is unavailable, normalize family-prefixed steps into a
   transferable `__FAMILY__ ...` template.
3. Predict with a template retrieval/grammar fallback.
4. Rewrite template-family steps back to the visible eval family name.

Run from repo root:
    python -B solutions/solution_18_family_template_grammar/solution.py
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
TEMPLATE_FAMILY_PLACEHOLDER = "__FAMILY__"
TEMPLATE_TRAIN_FAMILIES = tuple(f"SYNTHFAMILY{i:03d}" for i in range(1, 101))
TEMPLATE_SEQUENCES_PER_FAMILY = 20
TEMPLATE_SEED_BASE = 510_000

sys.path.insert(0, str(ROOT))

from solutions.ood_fourth_family_probe.fourth_family_probe import (  # noqa: E402
    generate_hidden_family_dataset,
)
from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402


NEXTSTEP_FIELDS = sol13.NEXTSTEP_FIELDS
COMPLETION_FIELDS = sol13.COMPLETION_FIELDS
ANOMALY_FIELDS = sol13.ANOMALY_FIELDS
TEMPLATE_AUDIT_FIELDS = [
    "EXAMPLE_ID",
    "FAMILY",
    "COMPLETION_FRACTION",
    "EXACT_ROUTE_MATCH",
    "PREDICTION_SOURCE",
    "SOURCE_ROUTE_ID",
    "PREFIX_FAMILY_TEMPLATE_STEPS",
    "RANK1",
]
TEMPLATE_MANIFEST_FIELDS = ["METRIC", "VALUE"]


@dataclass(frozen=True)
class TemplateTrainingBundle:
    public_sequences: int
    synthetic_template_families: int
    synthetic_template_sequences: int
    total_sequences: int
    sequences: dict[str, list[str]]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def public_sequences() -> dict[str, list[str]]:
    by_family, _inventory = base.load_all_available_sequences()
    sequences: dict[str, list[str]] = {}
    for records in by_family.values():
        for key, record in records.items():
            sequences[key] = record.steps
    return sequences


def normalize_step(step: str, family: str) -> str:
    prefix = f"{family} "
    if step.startswith(prefix):
        return f"{TEMPLATE_FAMILY_PLACEHOLDER} {step[len(prefix):]}"
    return step


def denormalize_step(step: str, family: str) -> str:
    prefix = f"{TEMPLATE_FAMILY_PLACEHOLDER} "
    if step.startswith(prefix):
        return f"{family} {step[len(prefix):]}"
    return step


def normalize_sequence(sequence: list[str], family: str) -> list[str]:
    return [normalize_step(step, family) for step in sequence]


def denormalize_sequence(sequence: list[str], family: str) -> list[str]:
    return [denormalize_step(step, family) for step in sequence]


def dedupe_keep_order(items: Iterable[str], k: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
        if len(out) >= k:
            break
    return (out + [""] * k)[:k]


def count_family_template_steps(sequence: list[str], family: str) -> int:
    return sum(1 for step in sequence if step.startswith(f"{family} "))


def generated_template_sequences() -> dict[str, list[str]]:
    """Generate many family-prefixed routes for template learning.

    The family names are synthetic and are not official. They give the fallback
    model examples of steps like `__FAMILY__ JTE DOSE VERIFICATION` so it can
    rewrite that template to a new visible family name at inference time.
    """
    sequences: dict[str, list[str]] = {}
    for index, family in enumerate(TEMPLATE_TRAIN_FAMILIES):
        generated = generate_hidden_family_dataset(
            family,
            TEMPLATE_SEQUENCES_PER_FAMILY,
            TEMPLATE_SEED_BASE + index * 97,
        )
        sequences.update(generated)
    return sequences


def build_template_training_bundle(
    public: dict[str, list[str]] | None = None,
    synthetic: dict[str, list[str]] | None = None,
) -> TemplateTrainingBundle:
    public = public_sequences() if public is None else public
    synthetic = generated_template_sequences() if synthetic is None else synthetic
    normalized: dict[str, list[str]] = {}
    for key, sequence in public.items():
        family = key.split(":", 1)[0]
        normalized[f"template_public:{key}"] = normalize_sequence(sequence, family)
    for key, sequence in synthetic.items():
        family = key.split(":", 1)[0]
        normalized[f"template_synthetic:{key}"] = normalize_sequence(sequence, family)
    return TemplateTrainingBundle(
        public_sequences=len(public),
        synthetic_template_families=len(TEMPLATE_TRAIN_FAMILIES),
        synthetic_template_sequences=len(synthetic),
        total_sequences=len(normalized),
        sequences=normalized,
    )


class FamilyTemplateFallback:
    """Retrieval fallback over normalized family-template routes."""

    def __init__(self, training: TemplateTrainingBundle) -> None:
        self.training = training
        self.model = sol2.EvalAwareRetrievalModel(training.sequences, training.sequences)

    def _normalize_prefix(self, prefix: list[str], family: str) -> list[str]:
        return normalize_sequence(prefix, family)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        normalized_prefix = self._normalize_prefix(prefix, family)
        raw_ranks = self.model.next_step_ranking(
            normalized_prefix,
            "template_public",
            completion_fraction,
            k=30,
        )
        denormalized = [denormalize_step(step, family) for step in raw_ranks]
        return dedupe_keep_order(denormalized, k)

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        normalized_prefix = self._normalize_prefix(prefix, family)
        suffix = self.model.complete(normalized_prefix, "template_public", completion_fraction)
        return denormalize_sequence(suffix, family)

    def lookup_coverage(self, examples: list[base.ValidExample]) -> float:
        hits = 0
        for ex in examples:
            hits += bool(
                self.model.lookup(
                    self._normalize_prefix(ex.partial, ex.family),
                    "template_public",
                    ex.completion_fraction,
                )
            )
        return hits / max(len(examples), 1)


class FamilyTemplateGrammarModel:
    """Exact route matcher with family-template fallback."""

    def __init__(
        self,
        anomaly_inputs: list[sol13.AnomalyInput],
        template_training: TemplateTrainingBundle,
    ) -> None:
        self.transductive = sol13.TransductiveGeneratorValidatorModel(anomaly_inputs)
        self.template_fallback = FamilyTemplateFallback(template_training)
        self.exact_route_calls = 0
        self.template_fallback_calls = 0

    def exact_match(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> tuple[str, list[str]] | None:
        return self.transductive.choose_full_sequence(prefix, family, completion_fraction)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        selected = self.exact_match(prefix, family, completion_fraction)
        ranked: list[str] = []
        if selected is not None:
            self.exact_route_calls += 1
            _source_id, sequence = selected
            ranked.append(sequence[len(prefix)])
        else:
            self.template_fallback_calls += 1
        for step in self.template_fallback.next_step_ranking(prefix, family, completion_fraction, k=30):
            if step and step not in ranked:
                ranked.append(step)
            if len(ranked) >= k:
                break
        return (ranked + [""] * k)[:k]

    def complete(self, prefix: list[str], family: str, completion_fraction: float) -> list[str]:
        selected = self.exact_match(prefix, family, completion_fraction)
        if selected is not None:
            _source_id, sequence = selected
            return sequence[len(prefix):]
        return self.template_fallback.complete(prefix, family, completion_fraction)

    def coverage(self, valid_inputs: list[sol13.ValidInput]) -> dict[str, object]:
        coverage = self.transductive.coverage(valid_inputs)
        coverage["template_training_sequences"] = self.template_fallback.training.total_sequences
        coverage["template_synthetic_families"] = self.template_fallback.training.synthetic_template_families
        coverage["template_synthetic_sequences"] = self.template_fallback.training.synthetic_template_sequences
        return coverage

    def audit_rows(
        self,
        examples: list[sol13.ValidInput],
        nextstep_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        rank1_by_id = {str(row["EXAMPLE_ID"]): str(row["RANK_1"]) for row in nextstep_rows}
        rows: list[dict[str, object]] = []
        for example in examples:
            selected = self.exact_match(example.partial, example.family, example.completion_fraction)
            rows.append(
                {
                    "EXAMPLE_ID": example.example_id,
                    "FAMILY": example.family.upper(),
                    "COMPLETION_FRACTION": example.completion_fraction,
                    "EXACT_ROUTE_MATCH": int(selected is not None),
                    "PREDICTION_SOURCE": "exact_route" if selected else "family_template_fallback",
                    "SOURCE_ROUTE_ID": selected[0] if selected else "",
                    "PREFIX_FAMILY_TEMPLATE_STEPS": count_family_template_steps(example.partial, example.family),
                    "RANK1": rank1_by_id.get(example.example_id, ""),
                }
            )
        return rows


def predict_task1(
    model: FamilyTemplateGrammarModel,
    examples: list[sol13.ValidInput],
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
    model: FamilyTemplateGrammarModel,
    examples: list[sol13.ValidInput],
) -> list[dict[str, object]]:
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


def evaluate_template_fallback_only(
    fallback: FamilyTemplateFallback,
    valid_truth: list[base.ValidExample],
) -> dict[str, object]:
    task1_rows = []
    task2_rows = []
    for ex in valid_truth:
        ranks = fallback.next_step_ranking(ex.partial, ex.family, ex.completion_fraction, k=5)
        task1_rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "RANK_1": ranks[0],
                "RANK_2": ranks[1],
                "RANK_3": ranks[2],
                "RANK_4": ranks[3],
                "RANK_5": ranks[4],
            }
        )
        task2_rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "PREDICTED_SEQUENCE": "|".join(
                    fallback.complete(ex.partial, ex.family, ex.completion_fraction)
                ),
            }
        )
    task1 = base.evaluate_task1(task1_rows, valid_truth)
    task1["top2"] = exact_top2(task1_rows, valid_truth)
    return {
        "task1_next_step": task1,
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_truth),
        "task2_completion": base.evaluate_task2(task2_rows, valid_truth),
        "template_lookup_coverage": fallback.lookup_coverage(valid_truth),
    }


def manifest_rows(training: TemplateTrainingBundle, coverage: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"METRIC": "public_sequences", "VALUE": training.public_sequences},
        {"METRIC": "synthetic_template_families", "VALUE": training.synthetic_template_families},
        {"METRIC": "synthetic_template_sequences", "VALUE": training.synthetic_template_sequences},
        {"METRIC": "total_template_training_sequences", "VALUE": training.total_sequences},
        {"METRIC": "exact_prefix_coverage", "VALUE": f"{float(coverage['exact_prefix_coverage']):.6f}"},
        {"METRIC": "exact_prefix_rows", "VALUE": coverage["exact_prefix_rows"]},
        {"METRIC": "valid_rows", "VALUE": coverage["valid_rows"]},
    ]


def run_official_prediction(template_training: TemplateTrainingBundle) -> dict[str, object]:
    if not sol13.OFFICIAL_VALID.exists() or not sol13.OFFICIAL_ANOMALY.exists():
        raise FileNotFoundError(
            "Official participant files are missing under tracks/industrial-infineon/participant_files."
        )
    valid_inputs = sol13.read_official_valid_inputs()
    anomaly_inputs = sol13.read_official_anomaly_inputs()
    model = FamilyTemplateGrammarModel(anomaly_inputs, template_training)

    nextstep_rows = predict_task1(model, valid_inputs)
    completion_rows = predict_task2(model, valid_inputs)
    anomaly_rows = predict_task3(anomaly_inputs)
    audit_rows = model.audit_rows(valid_inputs, nextstep_rows)

    write_csv(OUT_DIR / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(OUT_DIR / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(OUT_DIR / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)
    write_csv(OUT_DIR / "template_guard_audit.csv", TEMPLATE_AUDIT_FIELDS, audit_rows)

    official_dir = OUT_DIR / "official_submission"
    write_csv(official_dir / "nextstep.csv", NEXTSTEP_FIELDS, nextstep_rows)
    write_csv(official_dir / "completion.csv", COMPLETION_FIELDS, completion_rows)
    write_csv(official_dir / "anomaly.csv", ANOMALY_FIELDS, anomaly_rows)

    coverage = model.coverage(valid_inputs)
    write_csv(OUT_DIR / "template_training_manifest.csv", TEMPLATE_MANIFEST_FIELDS, manifest_rows(template_training, coverage))
    exact_rows = sum(int(row["EXACT_ROUTE_MATCH"]) for row in audit_rows)
    template_rows = len(audit_rows) - exact_rows
    validator_valid = sum(1 for row in anomaly_rows if int(row["IS_VALID"]) == 1)
    manifest = {
        "solution": "solution_18_family_template_grammar",
        "mode": "official_participant_input_prediction_with_family_template_fallback",
        "eval_valid": str(sol13.OFFICIAL_VALID.relative_to(ROOT)),
        "eval_anomaly": str(sol13.OFFICIAL_ANOMALY.relative_to(ROOT)),
        "valid_rows": len(valid_inputs),
        "anomaly_rows": len(anomaly_inputs),
        "exact_route_rows": exact_rows,
        "template_fallback_rows": template_rows,
        "exact_route_coverage": exact_rows / max(len(valid_inputs), 1),
        "validator_valid_rows": validator_valid,
        "validator_invalid_rows": len(anomaly_rows) - validator_valid,
        "route_coverage": coverage,
        "template_training": {
            "public_sequences": template_training.public_sequences,
            "synthetic_template_families": template_training.synthetic_template_families,
            "synthetic_template_sequences": template_training.synthetic_template_sequences,
            "total_sequences": template_training.total_sequences,
            "placeholder": TEMPLATE_FAMILY_PLACEHOLDER,
        },
        "outputs": {
            "nextstep": str((OUT_DIR / "nextstep.csv").relative_to(ROOT)),
            "completion": str((OUT_DIR / "completion.csv").relative_to(ROOT)),
            "anomaly": str((OUT_DIR / "anomaly.csv").relative_to(ROOT)),
            "template_guard_audit": str((OUT_DIR / "template_guard_audit.csv").relative_to(ROOT)),
            "template_training_manifest": str((OUT_DIR / "template_training_manifest.csv").relative_to(ROOT)),
        },
    }
    for manifest_path in (OUT_DIR / "official_run_manifest.json", official_dir / "official_run_manifest.json"):
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def local_self_eval(synthetic_template_sequences: dict[str, list[str]]) -> dict[str, object]:
    train, heldout, _inventory, by_family = base.load_split()
    template_training = build_template_training_bundle(train, synthetic_template_sequences)
    valid_truth = base.build_valid_examples(heldout)
    anomaly_truth = base.build_anomaly_examples(heldout)
    valid_inputs = sol13.valid_inputs_from_local_examples(valid_truth)
    anomaly_inputs = sol13.anomaly_inputs_from_local_examples(anomaly_truth)
    model = FamilyTemplateGrammarModel(anomaly_inputs, template_training)

    task1_rows = predict_task1(model, valid_inputs)
    task2_rows = predict_task2(model, valid_inputs)
    task3_rows = predict_task3(anomaly_inputs)
    task1 = base.evaluate_task1(task1_rows, valid_truth)
    task1["top2"] = exact_top2(task1_rows, valid_truth)
    fallback_only = evaluate_template_fallback_only(model.template_fallback, valid_truth)
    return {
        "task1_next_step": task1,
        "task1_canonical_process_step": sol2.evaluate_canonical_task1(task1_rows, valid_truth),
        "task2_completion": base.evaluate_task2(task2_rows, valid_truth),
        "task3_anomaly": base.evaluate_task3(task3_rows, anomaly_truth),
        "task4_ood_proxy_next_step": sol2.evaluate_ood_proxy(by_family),
        "coverage": model.coverage(valid_inputs),
        "template_fallback_without_exact_routes": fallback_only,
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
    fallback = local["template_fallback_without_exact_routes"]
    fallback_t1 = fallback["task1_next_step"]
    fallback_t2 = fallback["task2_completion"]
    lines = [
        "# Solution 18 Metrics",
        "",
        "## Official Participant Input Run",
        "",
        f"- Official valid rows predicted: {official['valid_rows']}",
        f"- Official anomaly rows predicted: {official['anomaly_rows']}",
        f"- Exact route rows: {official['exact_route_rows']}",
        f"- Exact route coverage: {official['exact_route_coverage']:.4f}",
        f"- Template fallback rows on official input: {official['template_fallback_rows']}",
        f"- Template synthetic training families: {official['template_training']['synthetic_template_families']}",
        f"- Template synthetic routes: {official['template_training']['synthetic_template_sequences']}",
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
        "## Template Fallback Without Exact Route Memory",
        "",
        f"- Fallback Task 1 exact Top-1: {fallback_t1['top1']:.4f}",
        f"- Fallback Task 1 exact Top-3: {fallback_t1['top3']:.4f}",
        f"- Fallback Task 1 MRR: {fallback_t1['mrr']:.4f}",
        f"- Fallback Task 2 exact match: {fallback_t2['exact_match']:.4f}",
        f"- Fallback Task 2 normalized edit distance: {fallback_t2['normalized_edit_distance']:.4f}",
        f"- Fallback Task 2 block accuracy: {fallback_t2['block_accuracy']:.4f}",
        f"- Template lookup coverage: {fallback['template_lookup_coverage']:.4f}",
        "",
        "## Interpretation",
        "",
        "- Current official rows still use exact route memory for every Task 1/2 row.",
        "- The new contribution is the fallback family-template grammar, not a change to the official CSV schema.",
        "- The fallback normalizes family-prefixed steps and can rewrite learned templates to a new visible family name.",
        "- This is the concrete next step suggested by the 110-family scaling probe.",
        "",
    ]
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    synthetic_template_sequences = generated_template_sequences()
    template_training = build_template_training_bundle(public_sequences(), synthetic_template_sequences)
    official_manifest = run_official_prediction(template_training)
    local_metrics = local_self_eval(synthetic_template_sequences)
    metrics = {
        "solution": "solution_18_family_template_grammar",
        "seed": SEED,
        "method": {
            "name": "exact route memory plus normalized family-template grammar fallback",
            "family_placeholder": TEMPLATE_FAMILY_PLACEHOLDER,
            "template_train_families": len(TEMPLATE_TRAIN_FAMILIES),
            "template_sequences_per_family": TEMPLATE_SEQUENCES_PER_FAMILY,
            "task2_uses_truth_remainder_length": False,
            "official_hidden_ground_truth_used": False,
        },
        "self_eval": {
            "families": base.FAMILIES,
            "eval_per_family": base.EVAL_PER_FAMILY,
            "train_sequences": local_metrics["train_sequences"],
            "valid_task_rows": local_metrics["valid_task_rows"],
            "anomaly_task_rows": local_metrics["anomaly_task_rows"],
            "template_training_sequences": local_metrics["coverage"]["template_training_sequences"],
            "template_synthetic_families": template_training.synthetic_template_families,
            "template_synthetic_sequences": template_training.synthetic_template_sequences,
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
                "template_fallback_without_exact_routes",
            }
        },
    }
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

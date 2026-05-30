#!/usr/bin/env python3
"""Generate official-input CSVs for every implemented solution.

Older solution scripts write local self-eval rows to `outputs/`. Now that the
participant input files are available, this helper writes official-row-count
submission CSVs to `outputs/official_submission/` for each solution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram import NGramModel  # noqa: E402
from solutions.solution_0_rule_mock import solution as sol0  # noqa: E402
from solutions.solution_1_hybrid_retrieval import solution as sol1  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_3_synthetic_augmented_retrieval import solution as sol3  # noqa: E402
from solutions.solution_4_length_aware_completion import solution as sol4  # noqa: E402
from solutions.solution_5_tuned_rank_ensemble import solution as sol5  # noqa: E402
from solutions.solution_6_alias_calibrated_retrieval import solution as sol6  # noqa: E402
from solutions.solution_7_monte_carlo_suffix_ensemble import solution as sol7  # noqa: E402
from solutions.solution_8_semantic_conformance_ensemble import solution as sol8  # noqa: E402
from solutions.solution_9_judge_aware_portfolio import solution as sol9  # noqa: E402
from solutions.solution_10_confidence_gated_consensus import solution as sol10  # noqa: E402
from solutions.solution_11_ood_guarded_consensus import solution as sol11  # noqa: E402
from solutions.solution_12_mbr_completion import solution as sol12  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402
from solutions.solution_14_synthetic_ml_generator_ensemble import solution as sol14  # noqa: E402
from solutions.solution_15_route_memory_mbr import solution as sol15  # noqa: E402
from solutions.solution_16_pseudolabel_metric_audit import solution as sol16  # noqa: E402
from solutions.solution_17_conformal_route_guard import solution as sol17  # noqa: E402
from solutions.solution_18_family_template_grammar import solution as sol18  # noqa: E402
from solutions.solution_19_valid_lattice_template import solution as sol19  # noqa: E402


OFFICIAL_VALID = sol13.OFFICIAL_VALID
OFFICIAL_ANOMALY = sol13.OFFICIAL_ANOMALY


def public_sequences() -> dict[str, list[str]]:
    by_family, _inventory = sol0.load_all_available_sequences()
    sequences: dict[str, list[str]] = {}
    for family_records in by_family.values():
        for key, record in family_records.items():
            sequences[key] = record.steps
    return sequences


def official_valid_examples() -> list[sol0.ValidExample]:
    return [
        sol0.ValidExample(
            example_id=row.example_id,
            family=row.family,
            completion_fraction=row.completion_fraction,
            partial=row.partial,
            truth_next="",
            truth_remainder=[],
        )
        for row in sol13.read_official_valid_inputs(OFFICIAL_VALID)
    ]


def official_anomaly_examples() -> list[sol0.AnomalyExample]:
    return [
        sol0.AnomalyExample(
            example_id=row.example_id,
            family=row.family,
            sequence=row.sequence,
            is_valid=-1,
            rule="",
        )
        for row in sol13.read_official_anomaly_inputs(OFFICIAL_ANOMALY)
    ]


def official_anomaly_inputs() -> list[sol13.AnomalyInput]:
    return sol13.read_official_anomaly_inputs(OFFICIAL_ANOMALY)


def write_outputs(solution_name: str, nextstep_rows, completion_rows, anomaly_rows, manifest: dict) -> None:
    out_dir = ROOT / "solutions" / solution_name / "outputs" / "official_submission"
    sol13.write_csv(out_dir / "nextstep.csv", sol13.NEXTSTEP_FIELDS, nextstep_rows)
    sol13.write_csv(out_dir / "completion.csv", sol13.COMPLETION_FIELDS, completion_rows)
    sol13.write_csv(out_dir / "anomaly.csv", sol13.ANOMALY_FIELDS, anomaly_rows)
    (out_dir / "official_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def estimated_ngram_completion(model: NGramModel, examples: list[sol0.ValidExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        estimated_total = round(len(ex.partial) / max(ex.completion_fraction, 0.01))
        max_new_steps = max(1, estimated_total - len(ex.partial) + 30)
        suffix = sol0.rollout_completion(model, ex.partial, max_new_steps=max_new_steps)
        rows.append({"EXAMPLE_ID": ex.example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})
    return rows


def validator_task3(examples: list[sol0.AnomalyExample]) -> list[dict[str, object]]:
    return sol2.predict_task3(examples)


def semantic_task3(examples: list[sol0.AnomalyExample]) -> list[dict[str, object]]:
    return sol8.predict_task3_semantic(examples)


def run_standard_solution(
    solution_name: str,
    build_predictors: Callable[[dict[str, list[str]], list[sol0.ValidExample], list[sol0.AnomalyExample]], tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict]],
) -> None:
    train = public_sequences()
    valid_examples = official_valid_examples()
    anomaly_examples = official_anomaly_examples()
    nextstep_rows, completion_rows, anomaly_rows, extra = build_predictors(
        train,
        valid_examples,
        anomaly_examples,
    )
    manifest = {
        "solution": solution_name,
        "mode": "official_participant_input_prediction",
        "eval_valid": str(OFFICIAL_VALID.relative_to(ROOT)),
        "eval_anomaly": str(OFFICIAL_ANOMALY.relative_to(ROOT)),
        "valid_rows": len(valid_examples),
        "anomaly_rows": len(anomaly_examples),
        **extra,
    }
    write_outputs(solution_name, nextstep_rows, completion_rows, anomaly_rows, manifest)
    print(f"Wrote {solution_name} official_submission")


def build_solution_0(train, valid_examples, anomaly_examples):
    model = NGramModel(n=3, alpha=0.4).fit(train.values())
    return (
        sol0.predict_task1(model, valid_examples),
        estimated_ngram_completion(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_1(train, valid_examples, anomaly_examples):
    model = sol1.HybridRetrievalModel(train)
    return (
        sol1.predict_task1(model, valid_examples),
        sol1.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_2(train, valid_examples, anomaly_examples):
    model = sol2.EvalAwareRetrievalModel(train, train)
    return (
        sol2.predict_task1(model, valid_examples),
        sol2.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_3(train, valid_examples, anomaly_examples):
    augmented = sol3.augment_with_generated_sequences(train)
    model = sol1.HybridRetrievalModel(augmented)
    return (
        sol1.predict_task1(model, valid_examples),
        sol1.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train), "augmented_train_sequences": len(augmented)},
    )


def build_solution_4(train, valid_examples, anomaly_examples):
    model = sol4.LengthAwareCompletionModel(train)
    return (
        sol4.predict_task1(model, valid_examples),
        sol4.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_5(train, valid_examples, anomaly_examples):
    model = sol5.TunedRankEnsembleModel(train)
    return (
        sol5.predict_task1(model, valid_examples),
        sol5.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_6(train, valid_examples, anomaly_examples):
    model = sol6.AliasCalibratedRetrievalModel(train)
    return (
        sol6.predict_task1(model, valid_examples),
        sol6.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_7(train, valid_examples, anomaly_examples):
    model = sol7.TaskSpecializedMonteCarloModel(train)
    return (
        sol7.predict_task1(model, valid_examples),
        sol7.predict_task2(model, valid_examples),
        validator_task3(anomaly_examples),
        {"train_sequences": len(train), "completion_library_sequences": model.completion_library_size},
    )


def build_solution_8(train, valid_examples, anomaly_examples):
    model = sol7.TaskSpecializedMonteCarloModel(train)
    return (
        sol7.predict_task1(model, valid_examples),
        sol7.predict_task2(model, valid_examples),
        semantic_task3(anomaly_examples),
        {"train_sequences": len(train), "completion_library_sequences": model.completion_library_size},
    )


def build_solution_9(train, valid_examples, anomaly_examples):
    model = sol9.JudgeAwarePortfolio(train)
    return (
        model.predict_task1(valid_examples),
        model.predict_task2(valid_examples),
        model.predict_task3(anomaly_examples),
        {"train_sequences": len(train)},
    )


def build_solution_10(train, valid_examples, anomaly_examples):
    model = sol10.ConfidenceGatedConsensusPortfolio(train)
    return (
        model.predict_task1(valid_examples),
        model.predict_task2(valid_examples),
        model.predict_task3(anomaly_examples),
        {"train_sequences": len(train), "task2_consensus": model.consensus_stats(len(valid_examples))},
    )


def build_solution_11(train, valid_examples, anomaly_examples):
    model = sol11.OODGuardedConsensusPortfolio(train)
    return (
        model.predict_task1(valid_examples),
        model.predict_task2(valid_examples),
        model.predict_task3(anomaly_examples),
        {"train_sequences": len(train), "task2_consensus": model.consensus_stats(len(valid_examples))},
    )


def build_solution_12(train, valid_examples, anomaly_examples):
    model = sol12.OODGuardedMBRCompletionPortfolio(train)
    return (
        model.predict_task1(valid_examples),
        model.predict_task2(valid_examples),
        model.predict_task3(anomaly_examples),
        {"train_sequences": len(train), "task2_mbr": model.mbr_stats(len(valid_examples))},
    )


def run_solution_13() -> None:
    sol13.run_official_prediction()
    print("Wrote solution_13_transductive_generator_validator official_submission")


def run_solution_14() -> None:
    sol14.run_official_prediction()
    print("Wrote solution_14_synthetic_ml_generator_ensemble official_submission")


def run_solution_15() -> None:
    sol15.run_official_prediction()
    print("Wrote solution_15_route_memory_mbr official_submission")


def run_solution_16() -> None:
    sol16.run_official_prediction()
    print("Wrote solution_16_pseudolabel_metric_audit official_submission")


def run_solution_17() -> None:
    sol17.run_official_prediction()
    print("Wrote solution_17_conformal_route_guard official_submission")


def run_solution_18() -> None:
    synthetic = sol18.generated_template_sequences()
    template_training = sol18.build_template_training_bundle(sol18.public_sequences(), synthetic)
    sol18.run_official_prediction(template_training)
    print("Wrote solution_18_family_template_grammar official_submission")


def run_solution_19() -> None:
    synthetic = sol18.generated_template_sequences()
    template_training = sol18.build_template_training_bundle(sol18.public_sequences(), synthetic)
    sol19.run_official_prediction(template_training)
    print("Wrote solution_19_valid_lattice_template official_submission")


STANDARD_BUILDERS = {
    "solution_0_rule_mock": build_solution_0,
    "solution_1_hybrid_retrieval": build_solution_1,
    "solution_2_eval_aware_retrieval": build_solution_2,
    "solution_3_synthetic_augmented_retrieval": build_solution_3,
    "solution_4_length_aware_completion": build_solution_4,
    "solution_5_tuned_rank_ensemble": build_solution_5,
    "solution_6_alias_calibrated_retrieval": build_solution_6,
    "solution_7_monte_carlo_suffix_ensemble": build_solution_7,
    "solution_8_semantic_conformance_ensemble": build_solution_8,
    "solution_9_judge_aware_portfolio": build_solution_9,
    "solution_10_confidence_gated_consensus": build_solution_10,
    "solution_11_ood_guarded_consensus": build_solution_11,
    "solution_12_mbr_completion": build_solution_12,
}


def main() -> None:
    if not OFFICIAL_VALID.exists() or not OFFICIAL_ANOMALY.exists():
        raise FileNotFoundError(
            "Official participant inputs are missing under tracks/industrial-infineon/participant_files"
        )
    for solution_name, builder in STANDARD_BUILDERS.items():
        run_standard_solution(solution_name, builder)
    run_solution_13()
    run_solution_14()
    run_solution_15()
    run_solution_16()
    run_solution_17()
    run_solution_18()
    run_solution_19()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hypothetical fourth-family OOD probe.

This is not an official benchmark. It creates a plausible hidden fourth family
(`sic_power`) with new exact strings and SiC/power-device-specific process
blocks, then tests the final strategy under several information settings.

Run from repo root:
    python -B solutions/ood_fourth_family_probe/fourth_family_probe.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEEDS = tuple(range(10))
FOURTH_FAMILY = "sic_power"
TEST_SEQUENCES_PER_SEED = 100
DECOY_VISIBLE_SEQUENCES = 160
ORACLE_TRAIN_SEQUENCES = 1200
FRACTIONS = (0.6, 0.8)

sys.path.insert(0, str(ROOT))

from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from solutions.solution_13_transductive_generator_validator import solution as sol13  # noqa: E402
from training_data.generate_sequences import validate_sequence  # noqa: E402


NEXTSTEP_FIELDS = ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
COMPLETION_FIELDS = ["EXAMPLE_ID", "PREDICTED_SEQUENCE"]
ANOMALY_FIELDS = ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]


def _opt(rng: random.Random, step: str, prob: float = 0.5) -> list[str]:
    return [step] if rng.random() < prob else []


def _litho(rng: random.Random, level: int, inspection: str | None = None) -> list[str]:
    steps = [
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        f"ALIGN MASK LEVEL {level}",
        f"EXPOSE LITHO LEVEL {level}",
    ]
    steps += _opt(rng, "POST EXPOSE BAKE", 0.35)
    steps += [
        "DEVELOP PHOTORESIST",
        inspection or f"INSPECT PATTERN LEVEL {level}",
    ]
    steps += _opt(rng, "HARD BAKE", 0.35)
    return steps


def generate_sic_power_sequence(rng: random.Random) -> list[str]:
    """Generate one valid hypothetical SiC power-device route.

    The route intentionally mixes known public-rule steps with new exact strings
    such as `SIC SUBSTRATE ORIENTATION CHECK` and `JTE DOSE VERIFICATION`.
    Known steps keep the public validator meaningful; new strings simulate the
    hidden-family exact-string problem.
    """
    seq: list[str] = [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
        rng.choice(["MEASURE THICKNESS", "MEASURE INITIAL THICKNESS"]),
        "SIC SUBSTRATE ORIENTATION CHECK",
        "MEASURE SUBSTRATE MICROPIPES",
        "PRE CLEAN WAFER",
        "BACKSIDE CLEAN",
        "FRONTSIDE CLEAN",
        rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]),
        rng.choice(["RCA CLEAN 2", "WET CLEAN RCA2"]),
        "HF DIP",
        rng.choice(["DRY WAFER", "DRY WAFER BACKSIDE"]),
        "SIC EPITAXY PREP",
        "EPITAXIAL DEPOSITION",
        "MEASURE DRIFT LAYER THICKNESS",
        "MEASURE RESISTIVITY",
        "EPITAXY ANNEAL",
        "WAFER SURFACE CLEAN",
        rng.choice(["THERMAL OXIDATION", "GATE OXIDE GROWTH"]),
        rng.choice(["MEASURE GATE OXIDE THICKNESS", "MEASURE OXIDE THICKNESS"]),
        "NITRIDATION ANNEAL",
        "JTE LAYOUT CHECK",
    ]

    level = 1
    cycles = rng.randint(4, 6)
    for cycle in range(cycles):
        role = rng.choice(["jte", "well", "source", "trench"])
        if role == "jte":
            seq += _litho(rng, level, "JTE WINDOW INSPECTION")
            seq += [
                rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
                "IMPLANT CHANNEL STOP",
                "JTE DOSE VERIFICATION",
                rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
                "CLEAN AFTER OXIDE ETCH",
                "RAPID THERMAL ANNEAL",
            ]
        elif role == "well":
            seq += _litho(rng, level, "P WELL WINDOW INSPECTION")
            seq += [
                rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
                rng.choice(["IMPLANT P BODY", "IMPLANT WELL"]),
                "HIGH TEMPERATURE ACTIVATION ANNEAL",
                rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
                "CLEAN AFTER OXIDE ETCH",
                "RAPID THERMAL ANNEAL",
            ]
        elif role == "source":
            seq += _litho(rng, level, "SOURCE WINDOW INSPECTION")
            seq += [
                rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
                rng.choice(["IMPLANT SOURCE REGION", "IMPLANT SOURCE DRAIN"]),
                "SOURCE IMPLANT DOSE CHECK",
                rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
                "CLEAN AFTER OXIDE ETCH",
                rng.choice(["RAPID THERMAL ANNEAL", "LIGHT ANNEAL"]),
            ]
        else:
            seq += _litho(rng, level, "TRENCH PATTERN INSPECTION")
            seq += [
                rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
                "TRENCH DEPTH MEASUREMENT",
                rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
                "CLEAN AFTER OXIDE ETCH",
                "GATE TRENCH CORNER ROUNDING",
                "RAPID THERMAL ANNEAL",
            ]
        level += 1
        seq += _opt(rng, "CARBON CAP STRIP CHECK", 0.45)

    seq += [
        "FRONTSIDE CLEAN",
        "DEPOSIT POLYSILICON",
        rng.choice(["POLYSILICON ANNEAL", "ANNEAL POLYSILICON"]),
        "MEASURE POLY THICKNESS",
    ]
    seq += _litho(rng, level, "POLY GATE PATTERN INSPECTION")
    seq += [
        rng.choice(["POLYSILICON ETCH", "POLYSILICON ETCH DRY"]),
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN AFTER POLY ETCH",
        "MEASURE GATE CD",
    ]
    level += 1

    seq += [
        "DEPOSIT INTERLAYER DIELECTRIC",
        rng.choice(["DENSIFY DIELECTRIC", "DENSIFY OXIDE"]),
        rng.choice(["MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"]),
        rng.choice(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"]),
        rng.choice(["MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"]),
    ]
    seq += _litho(rng, level, "CONTACT VIA INSPECTION")
    seq += [
        rng.choice(["VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"]),
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN AFTER VIA ETCH",
        "MEASURE VIA CD",
        "DEPOSIT BARRIER METAL",
        "DEPOSIT METAL SEED",
        "FILL VIA METAL",
        rng.choice(["CMP METAL", "CMP VIA FILL"]),
        "MEASURE CONTACT RESISTANCE",
        "FRONTSIDE CLEAN",
        rng.choice(["DEPOSIT METAL 1", "DEPOSIT TOP METAL"]),
        "ANNEAL METAL 1",
        "MEASURE METAL THICKNESS",
    ]
    level += 1

    seq += _litho(rng, level, "SOURCE METAL PATTERN INSPECTION")
    seq += [
        rng.choice(["METAL ETCH", "METAL ETCH DRY"]),
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN AFTER METAL ETCH",
        "MEASURE LINE WIDTH",
        rng.choice(["DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"]),
        "CURE PASSIVATION",
        rng.choice(["MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"]),
        rng.choice(["OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"]),
        rng.choice(["PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"]),
        "DEVELOP PHOTORESIST",
        rng.choice(["PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"]),
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN PAD OPENING",
        "MEASURE PAD OPENING",
        "BACKSIDE CLEAN FINAL",
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "DEPOSIT BACKSIDE METAL",
        "BACKSIDE ANNEAL",
        "MEASURE BACKSIDE CONTACT",
        "FINAL CLEAN",
        "FINAL PARTICLE INSPECTION",
        "FINAL GEOMETRY CHECK",
        "PARAMETRIC TEST",
        "LEAKAGE TEST",
        "BREAKDOWN VOLTAGE TEST",
        "SWITCHING TEST",
        "WAFER SORT TEST",
        "YIELD ANALYSIS",
        rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"]),
        "SHIP LOT",
    ]
    violations = validate_sequence(seq)
    if violations:
        raise RuntimeError(f"Generated invalid sic_power route: {violations[0]}")
    return seq


def generate_sic_power_dataset(count: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    sequences: dict[str, list[str]] = {}
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(sequences) < count:
        attempts += 1
        if attempts > count * 50:
            raise RuntimeError("Could not generate enough unique sic_power routes")
        seq = generate_sic_power_sequence(rng)
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        sequences[f"{FOURTH_FAMILY}:hypothetical:{seed}:{len(sequences):04d}"] = seq
    return sequences


def public_training_sequences() -> dict[str, list[str]]:
    by_family, _inventory = base.load_all_available_sequences()
    sequences: dict[str, list[str]] = {}
    for records in by_family.values():
        for key, record in records.items():
            sequences[key] = record.steps
    return sequences


def build_valid_examples(sequences: dict[str, list[str]], seed: int) -> list[base.ValidExample]:
    rows: list[base.ValidExample] = []
    counter = 1
    for _key, seq in sorted(sequences.items()):
        for fraction in FRACTIONS:
            cut = max(1, min(len(seq) - 1, int(len(seq) * fraction)))
            rows.append(
                base.ValidExample(
                    example_id=f"s{seed}_valid_{counter:04d}",
                    family=FOURTH_FAMILY,
                    completion_fraction=fraction,
                    partial=seq[:cut],
                    truth_next=seq[cut],
                    truth_remainder=seq[cut:],
                )
            )
            counter += 1
    return rows


def build_anomaly_examples(sequences: dict[str, list[str]], seed: int) -> list[base.AnomalyExample]:
    rows: list[base.AnomalyExample] = []
    counter = 1
    for i, (_key, seq) in enumerate(sorted(sequences.items())):
        rows.append(
            base.AnomalyExample(
                example_id=f"s{seed}_anom_{counter:04d}",
                family=FOURTH_FAMILY,
                sequence=seq,
                is_valid=1,
                rule="",
            )
        )
        counter += 1
        invalid_seq, rule = base.make_invalid(seq, i)
        rows.append(
            base.AnomalyExample(
                example_id=f"s{seed}_anom_{counter:04d}",
                family=FOURTH_FAMILY,
                sequence=invalid_seq,
                is_valid=0,
                rule=rule,
            )
        )
        counter += 1
    random.Random(seed).shuffle(rows)
    return rows


def to_valid_inputs(examples: list[base.ValidExample]) -> list[sol13.ValidInput]:
    return [
        sol13.ValidInput(
            example_id=ex.example_id,
            family=ex.family,
            completion_fraction=ex.completion_fraction,
            partial=ex.partial,
        )
        for ex in examples
    ]


def to_anomaly_inputs(examples: list[base.AnomalyExample]) -> list[sol13.AnomalyInput]:
    return [
        sol13.AnomalyInput(
            example_id=ex.example_id,
            family=ex.family,
            sequence=ex.sequence,
        )
        for ex in examples
    ]


def sequences_to_anomaly_inputs(
    sequences: dict[str, list[str]],
    seed: int,
    prefix: str,
) -> list[sol13.AnomalyInput]:
    return [
        sol13.AnomalyInput(
            example_id=f"{prefix}_{seed}_{i:04d}",
            family=FOURTH_FAMILY,
            sequence=seq,
        )
        for i, (_key, seq) in enumerate(sorted(sequences.items()))
    ]


def predict_task1(model: sol13.TransductiveGeneratorValidatorModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
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


def predict_task2(model: sol13.TransductiveGeneratorValidatorModel, examples: list[sol13.ValidInput]) -> list[dict[str, object]]:
    return [
        {
            "EXAMPLE_ID": ex.example_id,
            "PREDICTED_SEQUENCE": "|".join(
                model.complete(ex.partial, ex.family, ex.completion_fraction)
            ),
        }
        for ex in examples
    ]


def add_top2(rows: list[dict[str, object]], examples: list[base.ValidExample]) -> float:
    by_id = {row["EXAMPLE_ID"]: row for row in rows}
    hits = 0
    for ex in examples:
        ranks = [str(by_id[ex.example_id][f"RANK_{i}"]) for i in range(1, 3)]
        hits += ex.truth_next in ranks
    return hits / max(len(examples), 1)


def exact_coverage(
    model: sol13.TransductiveGeneratorValidatorModel,
    examples: list[sol13.ValidInput],
) -> float:
    hits = 0
    for ex in examples:
        hits += bool(model.exact_full_matches(ex.partial, ex.family))
    return hits / max(len(examples), 1)


def evaluate_scenario(
    *,
    seed: int,
    scenario: str,
    valid_examples: list[base.ValidExample],
    anomaly_examples: list[base.AnomalyExample],
    fit_anomaly_inputs: list[sol13.AnomalyInput],
    fallback_train: dict[str, list[str]],
) -> dict[str, object]:
    model = sol13.TransductiveGeneratorValidatorModel(fit_anomaly_inputs)
    model._fallback = sol2.EvalAwareRetrievalModel(fallback_train, fallback_train)
    valid_inputs = to_valid_inputs(valid_examples)
    task1_rows = predict_task1(model, valid_inputs)
    task2_rows = predict_task2(model, valid_inputs)
    task3_rows = sol13.predict_task3_from_inputs(to_anomaly_inputs(anomaly_examples))
    task1 = base.evaluate_task1(task1_rows, valid_examples)
    task1["top2"] = add_top2(task1_rows, valid_examples)
    task2 = base.evaluate_task2(task2_rows, valid_examples)
    task3 = base.evaluate_task3(task3_rows, anomaly_examples)
    return {
        "seed": seed,
        "scenario": scenario,
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
        "task3_rule_attribution_accuracy": task3["rule_attribution_accuracy"],
        "exact_route_coverage": exact_coverage(model, valid_inputs),
        "fit_valid_full_routes": sum(
            1 for row in fit_anomaly_inputs if not validate_sequence(row.sequence)
        ),
        "fallback_training_sequences": len(fallback_train),
        "valid_rows": len(valid_examples),
        "anomaly_rows": len(anomaly_examples),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "task1_top1",
        "task1_top2",
        "task1_top3",
        "task1_top5",
        "task1_mrr",
        "task2_exact_match",
        "task2_normalized_edit_distance",
        "task2_token_accuracy",
        "task2_block_accuracy",
        "task3_accuracy",
        "task3_rule_attribution_accuracy",
        "exact_route_coverage",
    ]
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario"])].append(row)

    summary: list[dict[str, object]] = []
    for scenario, scenario_rows in sorted(by_scenario.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in scenario_rows]
            higher = metric != "task2_normalized_edit_distance"
            best = max(values) if higher else min(values)
            worst = min(values) if higher else max(values)
            summary.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "direction": "higher_is_better" if higher else "lower_is_better",
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "best": best,
                    "worst": worst,
                }
            )
    return summary


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(per_seed: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    wanted = {
        "task1_top1",
        "task1_top3",
        "task1_mrr",
        "task2_exact_match",
        "task2_normalized_edit_distance",
        "task2_block_accuracy",
        "task3_accuracy",
        "exact_route_coverage",
    }
    by_scenario: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary:
        if row["metric"] in wanted:
            by_scenario[str(row["scenario"])][str(row["metric"])] = row

    lines = [
        "# Hypothetical Fourth-Family OOD Probe",
        "",
        "This is a local stress test, not an official benchmark.",
        "",
        "The probe creates a plausible `sic_power` fourth family with new exact",
        "strings such as `SIC SUBSTRATE ORIENTATION CHECK`, `JTE DOSE",
        "VERIFICATION`, and `GATE TRENCH CORNER ROUNDING`. It uses the public",
        "validator-compatible process structure so the task is still fair rather",
        "than random.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Meaning |",
        "| --- | --- |",
        "| `coupled_exact_route_memory` | Best case: the visible anomaly input contains the exact full valid routes that complete the partial rows. |",
        "| `decoupled_known_fallback_only` | Hard case: no matching fourth-family full routes are visible; fallback only knows MOSFET/IGBT/IC public routes. |",
        "| `decoupled_visible_family_routes` | Middle case: full valid fourth-family routes are visible, but they are different routes from the evaluated partial rows. The fallback may harvest new strings and contexts. |",
        "| `oracle_fourth_family_generator_training` | Upper comparator: the fourth-family generator spec is known and can generate training routes, but exact test routes are still not visible. |",
        "",
        "## 10-Seed Summary",
        "",
        "| Scenario | Exact route coverage | Task 1 Top-1 | Task 1 Top-3 | Task 1 MRR | Task 2 exact | Task 2 edit distance | Task 2 block | Task 3 accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in sorted(by_scenario):
        item = by_scenario[scenario]
        lines.append(
            f"| `{scenario}` | "
            f"{float(item['exact_route_coverage']['mean']):.4f} | "
            f"{float(item['task1_top1']['mean']):.4f} | "
            f"{float(item['task1_top3']['mean']):.4f} | "
            f"{float(item['task1_mrr']['mean']):.4f} | "
            f"{float(item['task2_exact_match']['mean']):.4f} | "
            f"{float(item['task2_normalized_edit_distance']['mean']):.4f} | "
            f"{float(item['task2_block_accuracy']['mean']):.4f} | "
            f"{float(item['task3_accuracy']['mean']):.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If the hidden family keeps the same visible partial/full-route coupling,",
        "  exact route memory handles new exact strings and new blocks perfectly,",
        "  because it reads them from the visible full route instead of inventing",
        "  them.",
        "- If the hidden family is decoupled and its exact strings are not visible,",
        "  exact-string Task 1 and exact Task 2 completion drop sharply.",
        "- If related fourth-family full routes are visible but not exact matches,",
        "  harvesting those routes helps only when local contexts overlap enough.",
        "- If the fourth-family generator specification is known, synthetic",
        "  training improves the fallback substantially, but it is still not the",
        "  same as seeing the exact test route.",
        "",
        "## Files",
        "",
        "- `per_seed_metrics.csv` / `.json`: raw rows for all seeds and scenarios.",
        "- `summary_metrics.csv` / `.json`: aggregate metrics.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    public_train = public_training_sequences()
    per_seed: list[dict[str, object]] = []

    for seed in SEEDS:
        print(f"evaluating hypothetical fourth family seed {seed}", flush=True)
        test_sequences = generate_sic_power_dataset(TEST_SEQUENCES_PER_SEED, 90_000 + seed)
        decoy_sequences = generate_sic_power_dataset(DECOY_VISIBLE_SEQUENCES, 120_000 + seed)
        oracle_sequences = generate_sic_power_dataset(ORACLE_TRAIN_SEQUENCES, 150_000 + seed)

        valid_examples = build_valid_examples(test_sequences, seed)
        anomaly_examples = build_anomaly_examples(test_sequences, seed)
        invalid_anomaly_inputs = [
            row for row in to_anomaly_inputs(anomaly_examples)
            if validate_sequence(row.sequence)
        ]
        coupled_anomaly_inputs = to_anomaly_inputs(anomaly_examples)
        decoy_anomaly_inputs = sequences_to_anomaly_inputs(decoy_sequences, seed, "decoy")
        visible_train = dict(public_train)
        visible_train.update(decoy_sequences)
        oracle_train = dict(public_train)
        oracle_train.update(oracle_sequences)

        scenarios = [
            (
                "coupled_exact_route_memory",
                coupled_anomaly_inputs,
                public_train,
            ),
            (
                "decoupled_known_fallback_only",
                invalid_anomaly_inputs,
                public_train,
            ),
            (
                "decoupled_visible_family_routes",
                decoy_anomaly_inputs,
                visible_train,
            ),
            (
                "oracle_fourth_family_generator_training",
                invalid_anomaly_inputs,
                oracle_train,
            ),
        ]
        for scenario, fit_inputs, fallback_train in scenarios:
            per_seed.append(
                evaluate_scenario(
                    seed=seed,
                    scenario=scenario,
                    valid_examples=valid_examples,
                    anomaly_examples=anomaly_examples,
                    fit_anomaly_inputs=fit_inputs,
                    fallback_train=fallback_train,
                )
            )

    summary = summarize(per_seed)
    write_csv(OUT_DIR / "per_seed_metrics.csv", list(per_seed[0].keys()), per_seed)
    write_csv(OUT_DIR / "summary_metrics.csv", list(summary[0].keys()), summary)
    (OUT_DIR / "per_seed_metrics.json").write_text(
        json.dumps(per_seed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(per_seed, summary)
    print(f"wrote fourth-family probe outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

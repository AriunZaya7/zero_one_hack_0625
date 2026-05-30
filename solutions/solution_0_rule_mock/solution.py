#!/usr/bin/env python3
"""Rule Mock Baseline for the Industrial AI track.

This solution is intentionally simple but complete:
- n-gram model for Task 1 next-step ranking
- n-gram greedy rollout for Task 2 completion
- public rule validator as an oracle for Task 3 anomaly detection
- deterministic self-evaluation because official eval files are not present
- all available provided process-sequence CSVs are considered

Run from repo root:
    python solutions/solution_0_rule_mock/solution.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TRAINING_DATA_DIR = ROOT / "training_data"
SEED = 42
EVAL_PER_FAMILY = 100
FRACTIONS = (0.6, 0.8)

sys.path.insert(0, str(ROOT))

from data import FAMILIES  # noqa: E402
from ngram import NGramModel  # noqa: E402
from training_data.generate_sequences import (  # noqa: E402
    CMP_STEPS,
    DEPOSITION_STEPS,
    ELECTRICAL_TEST_STEPS,
    ETCH_STEPS,
    FILL_STEPS,
    validate_sequence,
)


@dataclass(frozen=True)
class SequenceRecord:
    key: str
    family: str
    source_file: str
    source_kind: str
    steps: list[str]


@dataclass(frozen=True)
class ValidExample:
    example_id: str
    family: str
    completion_fraction: float
    partial: list[str]
    truth_next: str
    truth_remainder: list[str]


@dataclass(frozen=True)
class AnomalyExample:
    example_id: str
    family: str
    sequence: list[str]
    is_valid: int
    rule: str


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detect_family(path: Path) -> str | None:
    name = path.name.lower()
    for family in FAMILIES:
        if family in name:
            return family
    return None


def read_long_format(path: Path, family: str) -> list[SequenceRecord]:
    sequences: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {field.strip().upper() for field in (reader.fieldnames or [])}
        if "SEQUENCE_ID" not in fields or "STEP" not in fields:
            return []
        for row in reader:
            sid = (row.get("SEQUENCE_ID") or row.get("sequence_id") or "").strip()
            step = (row.get("STEP") or row.get("step") or "").strip()
            if not sid or not step:
                continue
            sequences.setdefault(sid, []).append(step)

    records: list[SequenceRecord] = []
    for sid, steps in sequences.items():
        records.append(
            SequenceRecord(
                key=f"{family}:{path.stem}:{sid}",
                family=family,
                source_file=str(path.relative_to(ROOT)),
                source_kind="long_format_sequence",
                steps=steps,
            )
        )
    return records


def read_single_step_sequence(path: Path, family: str) -> SequenceRecord | None:
    steps: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {field.strip().upper() for field in (reader.fieldnames or [])}
        if "SEQUENCE_ID" in fields or "STEP" not in fields:
            return None
        for row in reader:
            step = (row.get("STEP") or row.get("step") or "").strip()
            if step:
                steps.append(step)
    if not steps:
        return None
    return SequenceRecord(
        key=f"{family}:{path.stem}:canonical",
        family=family,
        source_file=str(path.relative_to(ROOT)),
        source_kind="canonical_single_sequence",
        steps=steps,
    )


def load_all_available_sequences() -> tuple[dict[str, dict[str, SequenceRecord]], list[dict[str, object]]]:
    """Load every provided process sequence source in training_data/.

    Long-format files supply most data. Canonical `synthetic*.csv` files are
    also loaded as training-only reference sequences. Description/parameter
    files are inventoried separately by `build_repo_asset_audit()`.
    """
    by_family: dict[str, dict[str, SequenceRecord]] = {family: {} for family in FAMILIES}
    inventory: list[dict[str, object]] = []

    for path in sorted(TRAINING_DATA_DIR.glob("*.csv")):
        family = detect_family(path)
        if family is None:
            continue

        records = read_long_format(path, family)
        if not records:
            single = read_single_step_sequence(path, family)
            records = [single] if single else []
        if not records:
            continue

        valid_count = 0
        invalid_count = 0
        lengths = []
        for record in records:
            violations = validate_sequence(record.steps)
            if violations:
                invalid_count += 1
            else:
                valid_count += 1
            lengths.append(len(record.steps))
            by_family[family][record.key] = record

        inventory.append(
            {
                "file": str(path.relative_to(ROOT)),
                "family": family,
                "kind": records[0].source_kind,
                "sequences": len(records),
                "valid_sequences": valid_count,
                "invalid_sequences": invalid_count,
                "min_len": min(lengths),
                "mean_len": sum(lengths) / len(lengths),
                "max_len": max(lengths),
            }
        )

    return by_family, inventory


def load_split() -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, list[str]]],
    list[dict[str, object]],
    dict[str, dict[str, SequenceRecord]],
]:
    """Return train sequences and held-out self-eval sequences per family."""
    rng = random.Random(SEED)
    train: dict[str, list[str]] = {}
    heldout: dict[str, dict[str, list[str]]] = {}
    by_family, inventory = load_all_available_sequences()

    for family in FAMILIES:
        records = by_family[family]
        eval_candidates = [
            key for key, record in records.items()
            if record.source_kind == "long_format_sequence"
        ]
        rng.shuffle(eval_candidates)
        eval_keys = set(eval_candidates[:EVAL_PER_FAMILY])

        keys = sorted(records)
        heldout[family] = {
            key: records[key].steps
            for key in keys
            if key in eval_keys
        }
        for key in keys:
            if key not in eval_keys:
                train[key] = records[key].steps

    return train, heldout, inventory, by_family


def build_valid_examples(heldout: dict[str, dict[str, list[str]]]) -> list[ValidExample]:
    examples: list[ValidExample] = []
    counter = 1
    families = [family for family in FAMILIES if family in heldout]
    for family in families:
        for _, seq in sorted(heldout[family].items()):
            for frac in FRACTIONS:
                cut = int(len(seq) * frac)
                cut = max(1, min(cut, len(seq) - 1))
                examples.append(
                    ValidExample(
                        example_id=f"valid_{counter:04d}",
                        family=family,
                        completion_fraction=frac,
                        partial=seq[:cut],
                        truth_next=seq[cut],
                        truth_remainder=seq[cut:],
                    )
                )
                counter += 1
    return examples


def _move_step_before(seq: list[str], step: str, anchor: str) -> list[str] | None:
    if step not in seq or anchor not in seq:
        return None
    out = list(seq)
    step_idx = out.index(step)
    moved = out.pop(step_idx)
    anchor_idx = out.index(anchor)
    out.insert(anchor_idx, moved)
    return out


def inject_ship_before_test(seq: list[str]) -> list[str] | None:
    return _move_step_before(seq, "SHIP LOT", "WAFER SORT TEST")


def inject_test_before_passivation(seq: list[str]) -> list[str] | None:
    if "CURE PASSIVATION" not in seq:
        return None
    for step in seq:
        if step in ELECTRICAL_TEST_STEPS:
            return _move_step_before(seq, step, "CURE PASSIVATION")
    return None


def inject_backside_metal_before_passivation(seq: list[str]) -> list[str] | None:
    return _move_step_before(seq, "DEPOSIT BACKSIDE METAL", "CURE PASSIVATION")


def inject_etch_no_mask(seq: list[str]) -> list[str] | None:
    """Remove a nearby develop step before a patterned etch."""
    for i, step in enumerate(seq):
        if step not in ETCH_STEPS:
            continue
        start = max(0, i - 12)
        for j in range(i - 1, start - 1, -1):
            if seq[j] in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}:
                out = list(seq)
                out.pop(j)
                return out
    return None


def inject_cmp_no_dep(seq: list[str]) -> list[str] | None:
    """Remove the nearest fill/deposition before a CMP step."""
    for i, step in enumerate(seq):
        if step not in CMP_STEPS:
            continue
        start = max(0, i - 6)
        for j in range(i - 1, start - 1, -1):
            if seq[j] in FILL_STEPS or seq[j] in DEPOSITION_STEPS:
                out = list(seq)
                out.pop(j)
                return out
    return None


INJECTORS: list[Callable[[list[str]], list[str] | None]] = [
    inject_ship_before_test,
    inject_test_before_passivation,
    inject_backside_metal_before_passivation,
    inject_etch_no_mask,
    inject_cmp_no_dep,
]


def make_invalid(seq: list[str], offset: int) -> tuple[list[str], str]:
    """Try multiple simple injectors until the validator confirms invalidity."""
    for i in range(len(INJECTORS)):
        injector = INJECTORS[(offset + i) % len(INJECTORS)]
        candidate = injector(seq)
        if not candidate:
            continue
        violations = validate_sequence(candidate)
        if violations:
            return candidate, violations[0].rule
    raise RuntimeError("Could not inject a validated anomaly into sequence")


def build_anomaly_examples(heldout: dict[str, dict[str, list[str]]]) -> list[AnomalyExample]:
    examples: list[AnomalyExample] = []
    counter = 1
    injector_offset = 0
    families = [family for family in FAMILIES if family in heldout]
    for family in families:
        for _, seq in sorted(heldout[family].items()):
            valid_violations = validate_sequence(seq)
            if valid_violations:
                raise RuntimeError(f"Expected held-out sequence to be valid: {valid_violations[0]}")

            examples.append(
                AnomalyExample(
                    example_id=f"anomaly_{counter:04d}",
                    family=family,
                    sequence=seq,
                    is_valid=1,
                    rule="",
                )
            )
            counter += 1

            invalid_seq, rule = make_invalid(seq, injector_offset)
            injector_offset += 1
            examples.append(
                AnomalyExample(
                    example_id=f"anomaly_{counter:04d}",
                    family=family,
                    sequence=invalid_seq,
                    is_valid=0,
                    rule=rule,
                )
            )
            counter += 1
    random.Random(SEED).shuffle(examples)
    return examples


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i]
        for j, bj in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ai == bj else 1),
                )
            )
        prev = curr
    return prev[-1]


def positional_accuracy(pred: list[str], truth: list[str]) -> float:
    denom = max(len(pred), len(truth), 1)
    return sum(1 for p, t in zip(pred, truth) if p == t) / denom


def block_label(step: str) -> str:
    """Coarse block labels aligned with the public grammar categories."""
    if step in {"RECEIVE WAFER LOT", "LOT IDENTIFICATION", "LOT RELEASE", "FINAL LOT RELEASE", "SHIP LOT"}:
        return "logistics"
    if "CLEAN" in step or step in {"HF DIP", "DRY WAFER", "DRY WAFER BACKSIDE", "BACKSIDE RINSE"}:
        return "clean"
    if any(k in step for k in ("EPITAX", "SUBSTRATE", "GRINDING", "BACKSIDE ROUGHNESS")):
        return "family_prep"
    if "OXID" in step or "PAD OXIDE" in step:
        return "oxidation"
    if any(k in step for k in ("LITHO", "PHOTORESIST", "MASK", "EXPOSE", "DEVELOP", "ETCH", "IMPLANT", "DIFFUSION", "ANNEAL")):
        return "litho_etch_implant"
    if "INTERLAYER" in step or "INTERLEVEL" in step or "DIELECTRIC" in step:
        return "ild"
    if "VIA" in step or "TUNGSTEN" in step:
        return "via"
    if "METAL" in step or "CONTACT" in step:
        return "metal"
    if "PASSIVATION" in step or "PAD WINDOW" in step or "BOND PAD" in step:
        return "passivation"
    if "BACKSIDE" in step:
        return "backside"
    if "FINAL" in step or "INSPECTION" in step or step.startswith("MEASURE"):
        return "inspection_measurement"
    if "TEST" in step or "YIELD" in step:
        return "test"
    return "other"


def roc_auc(labels_valid: list[int], scores_valid_probability: list[float]) -> float:
    """Pairwise AUC with labels using 1=valid as positive."""
    positives = [s for y, s in zip(labels_valid, scores_valid_probability) if y == 1]
    negatives = [s for y, s in zip(labels_valid, scores_valid_probability) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = ties = 0
    for ps in positives:
        for ns in negatives:
            if ps > ns:
                wins += 1
            elif ps == ns:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def evaluate_task1(rows: list[dict[str, object]], examples: list[ValidExample]) -> dict[str, float]:
    by_id = {row["EXAMPLE_ID"]: row for row in rows}
    top1 = top3 = top5 = 0
    mrr = 0.0
    for ex in examples:
        row = by_id[ex.example_id]
        ranks = [str(row[f"RANK_{i}"]) for i in range(1, 6)]
        if ex.truth_next in ranks:
            rank = ranks.index(ex.truth_next)
            if rank == 0:
                top1 += 1
            if rank < 3:
                top3 += 1
            top5 += 1
            mrr += 1.0 / (rank + 1)
    n = len(examples)
    return {
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "mrr": mrr / n,
        "n_examples": n,
    }


def evaluate_task2(rows: list[dict[str, object]], examples: list[ValidExample]) -> dict[str, float]:
    by_id = {row["EXAMPLE_ID"]: row for row in rows}
    exact = []
    norm_edits = []
    token_accs = []
    block_accs = []
    for ex in examples:
        pred_text = str(by_id[ex.example_id]["PREDICTED_SEQUENCE"])
        pred = pred_text.split("|") if pred_text else []
        truth = ex.truth_remainder
        exact.append(1.0 if pred == truth else 0.0)
        norm_edits.append(edit_distance(pred, truth) / max(len(pred), len(truth), 1))
        token_accs.append(positional_accuracy(pred, truth))
        block_accs.append(
            positional_accuracy(
                [block_label(s) for s in pred],
                [block_label(s) for s in truth],
            )
        )
    return {
        "exact_match": mean(exact),
        "normalized_edit_distance": mean(norm_edits),
        "token_accuracy": mean(token_accs),
        "block_accuracy": mean(block_accs),
        "n_examples": len(examples),
    }


def evaluate_task3(rows: list[dict[str, object]], examples: list[AnomalyExample]) -> dict[str, float]:
    by_id = {row["EXAMPLE_ID"]: row for row in rows}
    tp = fp = tn = fn = 0
    labels: list[int] = []
    scores: list[float] = []
    attributed_correct = 0
    attributed_total = 0

    for ex in examples:
        row = by_id[ex.example_id]
        pred_valid = int(row["IS_VALID"])
        score = float(row["SCORE"])
        labels.append(ex.is_valid)
        scores.append(score)

        if ex.is_valid == 1 and pred_valid == 1:
            tp += 1
        elif ex.is_valid == 1 and pred_valid == 0:
            fn += 1
        elif ex.is_valid == 0 and pred_valid == 0:
            tn += 1
            attributed_total += 1
            if str(row["PREDICTED_RULE"]) == ex.rule:
                attributed_correct += 1
        elif ex.is_valid == 0 and pred_valid == 1:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": accuracy,
        "precision_valid": precision,
        "recall_valid": recall,
        "f1_valid": f1,
        "roc_auc_valid_probability": roc_auc(labels, scores),
        "rule_attribution_accuracy": attributed_correct / max(attributed_total, 1),
        "tp_valid": tp,
        "tn_invalid": tn,
        "fp_invalid_as_valid": fp,
        "fn_valid_as_invalid": fn,
        "n_examples": len(examples),
    }


def evaluate_ood_proxy(by_family: dict[str, dict[str, SequenceRecord]]) -> dict[str, dict[str, float]]:
    """Leave-one-family-out next-step proxy for hidden Task 4."""
    results: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        train: dict[str, list[str]] = {}
        test: dict[str, list[str]] = {}
        for fam in FAMILIES:
            for key, record in by_family[fam].items():
                if fam == family:
                    if record.source_kind == "long_format_sequence":
                        test[key] = record.steps
                else:
                    train[key] = record.steps

        model = NGramModel(n=3, alpha=0.4).fit(train.values())
        rng = random.Random(SEED)
        keys = sorted(test)
        rng.shuffle(keys)
        sampled_test = {key: test[key] for key in keys[:EVAL_PER_FAMILY]}
        examples = build_valid_examples({family: sampled_test})
        rows = predict_task1(model, examples)
        results[family] = evaluate_task1(rows, examples)
    return results


def predict_task1(model: NGramModel, examples: list[ValidExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        ranks = model.next_step_ranking(ex.partial, k=5)
        ranks = ranks + [""] * (5 - len(ranks))
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


def rollout_completion(model: NGramModel, prefix: list[str], max_new_steps: int) -> list[str]:
    """Greedy n-gram rollout with simple loop prevention.

    The base n-gram can get trapped in repeating lithography cycles. That is a
    real baseline limitation, but the submission file should still stay bounded.
    """
    seq = list(prefix)
    predicted: list[str] = []
    seen_contexts: set[tuple[str, ...]] = set()
    context_width = max(1, model.n - 1)

    for _ in range(max_new_steps):
        context = tuple(seq[-context_width:])
        if context in seen_contexts:
            break
        seen_contexts.add(context)

        nxt = model.next_step_ranking(seq, k=1)[0]
        if nxt == "<eos>" or nxt == seq[-1]:
            break
        predicted.append(nxt)
        seq.append(nxt)
        if nxt == "SHIP LOT":
            break
    return predicted


def predict_task2(model: NGramModel, examples: list[ValidExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        predicted_remainder = rollout_completion(
            model,
            ex.partial,
            max_new_steps=len(ex.truth_remainder) + 20,
        )
        rows.append(
            {
                "EXAMPLE_ID": ex.example_id,
                "PREDICTED_SEQUENCE": "|".join(predicted_remainder),
            }
        )
    return rows


def describe_csv(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = sum(1 for _ in reader)
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": rows,
        "columns": fields,
    }


def build_repo_asset_audit(sequence_inventory: list[dict[str, object]]) -> dict[str, object]:
    docs = [
        "README.md",
        "CLAUDE.md",
        "PLAN.md",
        "submission/SUBMISSION.md",
        "submission/REPORT_TEMPLATE.md",
        "industrial-infineon/README.md",
        "industrial-infineon/Track_industrial_en.md",
        "industrial-infineon/Track_industrial.md",
        "training_data/README.md",
        "training_data/generation_rules.md",
    ]
    scripts = [
        "data.py",
        "tokenizer.py",
        "ngram.py",
        "run_baseline.py",
        "ablation_n.py",
        "training_data/generate_sequences.py",
        "small_transformer_model/data_generation/generate_more_sequence_data.py",
        "small_transformer_model/data_generation/prepare_dataset.py",
        "small_transformer_model/train/train_model.py",
        "small_transformer_model/train/setup_leonardo.sh",
        "small_transformer_model/train/job.slurm",
        "test_baseline_1/baseline_random.py",
        "test_baseline_1/baseline_ngram.py",
        "test_baseline_1/baseline_llm_zeroshot.py",
        "test_baseline_1/run_all_baselines.py",
        "test_baseline_1/visualize_baselines.py",
        "test_baseline_2/ngram_context_comparison.py",
    ]
    metadata_csvs = []
    for path in sorted(TRAINING_DATA_DIR.glob("*.csv")):
        if any(item["file"] == str(path.relative_to(ROOT)) for item in sequence_inventory):
            continue
        metadata_csvs.append(describe_csv(path))

    previous_artifacts = {
        "small_transformer_jsonl": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "small_transformer_model/data").glob("*.jsonl")),
        "small_transformer_vocab": str((ROOT / "small_transformer_model/data/vocab.json").relative_to(ROOT)),
        "baseline_plots": sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("test_baseline_*/**/*.png")),
        "wandb_summaries": sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/wandb-summary.json")),
    }

    return {
        "docs_reviewed": docs,
        "scripts_reviewed": scripts,
        "sequence_sources_used": sequence_inventory,
        "metadata_csvs_reviewed_not_modeled": metadata_csvs,
        "previous_artifacts_reviewed": previous_artifacts,
        "notes": [
            "Official eval inputs and eval_metrics.py are not present in reachable repo history.",
            "Solution 0 uses all available sequence sources for the baseline model.",
            "For enriched description/parameter CSVs, Solution 0 uses only the ordered STEP column; natural-language descriptions and fab parameters are not used by this baseline.",
            "CLAUDE.md and PLAN.md recommend the final winning model use step-level tokens and avoid family embeddings for OOD generalization.",
        ],
    }


def write_asset_audit(audit: dict[str, object]) -> None:
    (OUT_DIR / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Solution 0 Input Audit",
        "",
        "This file records the repo assets considered by `solution_0_rule_mock`.",
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
    lines += [
        "",
        "## Metadata-Only CSVs Reviewed But Not Modeled",
        "",
        "| File | Rows | Columns |",
        "| --- | ---: | --- |",
    ]
    for item in audit["metadata_csvs_reviewed_not_modeled"]:
        columns = ", ".join(f"`{column}`" for column in item["columns"])
        lines.append(f"| `{item['file']}` | {item['rows']} | {columns} |")
    lines += ["", "## Notes", ""]
    for note in audit["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    (OUT_DIR / "input_audit.md").write_text("\n".join(lines), encoding="utf-8")


def predict_task3(examples: list[AnomalyExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        violations = validate_sequence(ex.sequence)
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


def write_inputs_and_ground_truth(valid_examples: list[ValidExample], anomaly_examples: list[AnomalyExample]) -> None:
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


def write_metrics(metrics: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    task1 = metrics["task1_next_step"]
    task2 = metrics["task2_completion"]
    task3 = metrics["task3_anomaly"]
    ood = metrics["task4_ood_proxy_next_step"]
    self_eval = metrics["self_eval"]
    data_inventory = metrics["data_inventory"]
    lines = [
        "# Solution 0 Metrics",
        "",
        "## Data Coverage",
        "",
        f"- Train sequences: {self_eval['train_sequences']}",
        f"- Self-eval valid rows: {self_eval['valid_task_rows']}",
        f"- Self-eval anomaly rows: {self_eval['anomaly_task_rows']}",
        f"- Sequence source files used: {len(data_inventory)}",
        f"- Official eval available: {self_eval['official_eval_available']}",
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
    lines.append("")
    (OUT_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, heldout, sequence_inventory, by_family = load_split()
    valid_examples = build_valid_examples(heldout)
    anomaly_examples = build_anomaly_examples(heldout)
    audit = build_repo_asset_audit(sequence_inventory)

    model = NGramModel(n=3, alpha=0.4).fit(train.values())

    task1_rows = predict_task1(model, valid_examples)
    task2_rows = predict_task2(model, valid_examples)
    task3_rows = predict_task3(anomaly_examples)

    write_inputs_and_ground_truth(valid_examples, anomaly_examples)
    write_csv(OUT_DIR / "nextstep.csv", ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], task1_rows)
    write_csv(OUT_DIR / "completion.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], task2_rows)
    write_csv(OUT_DIR / "anomaly.csv", ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"], task3_rows)
    write_asset_audit(audit)

    metrics = {
        "solution": "solution_0_rule_mock",
        "seed": SEED,
        "self_eval": {
            "families": FAMILIES,
            "eval_per_family": EVAL_PER_FAMILY,
            "train_sequences": len(train),
            "valid_task_rows": len(valid_examples),
            "anomaly_task_rows": len(anomaly_examples),
            "official_eval_available": False,
        },
        "data_inventory": sequence_inventory,
        "task1_next_step": evaluate_task1(task1_rows, valid_examples),
        "task2_completion": evaluate_task2(task2_rows, valid_examples),
        "task3_anomaly": evaluate_task3(task3_rows, anomaly_examples),
        "task4_ood_proxy_next_step": evaluate_ood_proxy(by_family),
    }
    write_metrics(metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

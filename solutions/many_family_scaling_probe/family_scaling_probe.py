#!/usr/bin/env python3
"""Many-family OOD scaling probe.

This is a local synthetic benchmark, not an official score. It creates 110
plausible hidden families, trains on increasing numbers of families from the
first 100, and evaluates on the same held-out 10 families every time.

Run from repo root:
    python -B solutions/many_family_scaling_probe/family_scaling_probe.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
ANALYSIS_HTML = Path(__file__).resolve().parent / "analysis.html"

TOTAL_FAMILIES = 110
TRAIN_POOL_SIZE = 100
VALIDATION_FAMILY_COUNT = 10
TRAIN_FAMILY_COUNTS = (2, 5, 10, 20, 40, 60, 80, 100)
TRAIN_SEQUENCES_PER_FAMILY = 20
VALIDATION_SEQUENCES_PER_FAMILY = 16
FRACTIONS = (0.6, 0.8)
FAMILY_PLACEHOLDER = "__FAMILY__"

sys.path.insert(0, str(ROOT))

from solutions.ood_fourth_family_probe.fourth_family_probe import (  # noqa: E402
    generate_hidden_family_dataset,
)
from solutions.solution_0_rule_mock import solution as base  # noqa: E402
from solutions.solution_2_eval_aware_retrieval import solution as sol2  # noqa: E402
from training_data.generate_sequences import validate_sequence  # noqa: E402


NEXTSTEP_FIELDS = ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
COMPLETION_FIELDS = ["EXAMPLE_ID", "PREDICTED_SEQUENCE"]
ANOMALY_FIELDS = ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]


def family_name(index: int) -> str:
    return f"SYNTHFAMILY{index:03d}"


ALL_FAMILIES = tuple(family_name(i) for i in range(1, TOTAL_FAMILIES + 1))
TRAIN_FAMILIES = ALL_FAMILIES[:TRAIN_POOL_SIZE]
VALIDATION_FAMILIES = ALL_FAMILIES[TRAIN_POOL_SIZE:]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_family_sequences() -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]:
    train: dict[str, dict[str, list[str]]] = {}
    validation: dict[str, dict[str, list[str]]] = {}
    for idx, family in enumerate(TRAIN_FAMILIES, start=1):
        train[family] = generate_hidden_family_dataset(
            family,
            TRAIN_SEQUENCES_PER_FAMILY,
            310_000 + idx * 101,
        )
    for idx, family in enumerate(VALIDATION_FAMILIES, start=1):
        validation[family] = generate_hidden_family_dataset(
            family,
            VALIDATION_SEQUENCES_PER_FAMILY,
            410_000 + idx * 101,
        )
    return train, validation


def flatten(selected: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for family in sorted(selected):
        out.update(selected[family])
    return out


def build_valid_examples(validation: dict[str, dict[str, list[str]]]) -> list[base.ValidExample]:
    rows: list[base.ValidExample] = []
    counter = 1
    for family in VALIDATION_FAMILIES:
        for _key, seq in sorted(validation[family].items()):
            for fraction in FRACTIONS:
                cut = max(1, min(len(seq) - 1, int(len(seq) * fraction)))
                rows.append(
                    base.ValidExample(
                        example_id=f"scale_valid_{counter:05d}",
                        family=family,
                        completion_fraction=fraction,
                        partial=seq[:cut],
                        truth_next=seq[cut],
                        truth_remainder=seq[cut:],
                    )
                )
                counter += 1
    return rows


def build_anomaly_examples(validation: dict[str, dict[str, list[str]]]) -> list[base.AnomalyExample]:
    rows: list[base.AnomalyExample] = []
    counter = 1
    injector_offset = 0
    for family in VALIDATION_FAMILIES:
        for _key, seq in sorted(validation[family].items()):
            violations = validate_sequence(seq)
            if violations:
                raise RuntimeError(f"Expected valid synthetic sequence for {family}: {violations[0]}")
            rows.append(
                base.AnomalyExample(
                    example_id=f"scale_anomaly_{counter:05d}",
                    family=family,
                    sequence=seq,
                    is_valid=1,
                    rule="",
                )
            )
            counter += 1
            invalid_seq, rule = base.make_invalid(seq, injector_offset)
            injector_offset += 1
            rows.append(
                base.AnomalyExample(
                    example_id=f"scale_anomaly_{counter:05d}",
                    family=family,
                    sequence=invalid_seq,
                    is_valid=0,
                    rule=rule,
                )
            )
            counter += 1
    return rows


def normalize_step(step: str, family: str) -> str:
    prefix = f"{family} "
    if step.startswith(prefix):
        return FAMILY_PLACEHOLDER + " " + step[len(prefix):]
    return step


def denormalize_step(step: str, family: str) -> str:
    prefix = f"{FAMILY_PLACEHOLDER} "
    if step.startswith(prefix):
        return family + " " + step[len(prefix):]
    return step


def normalize_sequence(sequence: list[str], family: str) -> list[str]:
    return [normalize_step(step, family) for step in sequence]


def denormalize_sequence(sequence: list[str], family: str) -> list[str]:
    return [denormalize_step(step, family) for step in sequence]


def is_family_specific_step(step: str, family: str) -> bool:
    return step.startswith(f"{family} ")


def dedupe_keep_order(items: list[str], k: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
        if len(out) >= k:
            break
    return (out + [""] * k)[:k]


class RawExactRetrieval:
    name = "raw_exact_retrieval"

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        self.model = sol2.EvalAwareRetrievalModel(train_sequences, train_sequences)

    def next_step_ranking(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
        k: int = 5,
    ) -> list[str]:
        return self.model.next_step_ranking(prefix, family, completion_fraction, k=k)

    def complete(
        self,
        prefix: list[str],
        family: str,
        completion_fraction: float,
    ) -> list[str]:
        return self.model.complete(prefix, family, completion_fraction)

    def lookup_coverage(self, examples: list[base.ValidExample]) -> float:
        hits = sum(
            bool(self.model.lookup(ex.partial, ex.family, ex.completion_fraction))
            for ex in examples
        )
        return hits / max(len(examples), 1)


class TemplateAdaptedRetrieval:
    name = "template_adapted_retrieval"

    def __init__(self, train_sequences: dict[str, list[str]]) -> None:
        normalized: dict[str, list[str]] = {}
        for key, seq in train_sequences.items():
            family = key.split(":", 1)[0]
            normalized[f"template:{key}"] = normalize_sequence(seq, family)
        self.model = sol2.EvalAwareRetrievalModel(normalized, normalized)

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
            "template",
            completion_fraction,
            k=20,
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
        suffix = self.model.complete(normalized_prefix, "template", completion_fraction)
        return denormalize_sequence(suffix, family)

    def lookup_coverage(self, examples: list[base.ValidExample]) -> float:
        hits = sum(
            bool(
                self.model.lookup(
                    self._normalize_prefix(ex.partial, ex.family),
                    "template",
                    ex.completion_fraction,
                )
            )
            for ex in examples
        )
        return hits / max(len(examples), 1)


def predict_task1(model: RawExactRetrieval | TemplateAdaptedRetrieval, examples: list[base.ValidExample]) -> list[dict[str, object]]:
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


def predict_task2(model: RawExactRetrieval | TemplateAdaptedRetrieval, examples: list[base.ValidExample]) -> list[dict[str, object]]:
    return [
        {
            "EXAMPLE_ID": ex.example_id,
            "PREDICTED_SEQUENCE": "|".join(
                model.complete(ex.partial, ex.family, ex.completion_fraction)
            ),
        }
        for ex in examples
    ]


def predict_task3(examples: list[base.AnomalyExample]) -> list[dict[str, object]]:
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


def task1_family_specific_metrics(
    rows: list[dict[str, object]],
    examples: list[base.ValidExample],
) -> dict[str, float]:
    subset = [
        ex for ex in examples
        if is_family_specific_step(ex.truth_next, ex.family)
    ]
    if not subset:
        return {
            "family_specific_next_step_share": 0.0,
            "family_specific_task1_top1": float("nan"),
            "family_specific_task1_top3": float("nan"),
        }
    metrics = base.evaluate_task1(rows, subset)
    return {
        "family_specific_next_step_share": len(subset) / max(len(examples), 1),
        "family_specific_task1_top1": metrics["top1"],
        "family_specific_task1_top3": metrics["top3"],
    }


def family_average(
    rows: list[dict[str, object]],
    metric_names: list[str],
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["validation_family"])].append(row)
    out: dict[str, float] = {}
    for metric in metric_names:
        values = [
            float(items[0][metric])
            for items in grouped.values()
            if not math.isnan(float(items[0][metric]))
        ]
        if not values:
            out[f"{metric}_family_mean"] = float("nan")
            out[f"{metric}_family_std"] = float("nan")
            out[f"{metric}_family_best"] = float("nan")
            out[f"{metric}_family_worst"] = float("nan")
            continue
        out[f"{metric}_family_mean"] = mean(values)
        out[f"{metric}_family_std"] = pstdev(values) if len(values) > 1 else 0.0
        higher = metric != "task2_normalized_edit_distance"
        out[f"{metric}_family_best"] = max(values) if higher else min(values)
        out[f"{metric}_family_worst"] = min(values) if higher else max(values)
    return out


def evaluate_model(
    *,
    train_family_count: int,
    train_sequences: dict[str, list[str]],
    model: RawExactRetrieval | TemplateAdaptedRetrieval,
    valid_examples: list[base.ValidExample],
    anomaly_examples: list[base.AnomalyExample],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    task1_rows = predict_task1(model, valid_examples)
    task2_rows = predict_task2(model, valid_examples)
    task3_rows = predict_task3(anomaly_examples)
    task1 = base.evaluate_task1(task1_rows, valid_examples)
    task2 = base.evaluate_task2(task2_rows, valid_examples)
    task3 = base.evaluate_task3(task3_rows, anomaly_examples)
    specific = task1_family_specific_metrics(task1_rows, valid_examples)

    per_family: list[dict[str, object]] = []
    for family in VALIDATION_FAMILIES:
        fam_valid = [ex for ex in valid_examples if ex.family == family]
        fam_anom = [ex for ex in anomaly_examples if ex.family == family]
        fam_task1 = base.evaluate_task1(task1_rows, fam_valid)
        fam_task2 = base.evaluate_task2(task2_rows, fam_valid)
        fam_task3 = base.evaluate_task3(task3_rows, fam_anom)
        fam_specific = task1_family_specific_metrics(task1_rows, fam_valid)
        per_family.append(
            {
                "train_family_count": train_family_count,
                "model": model.name,
                "validation_family": family,
                "task1_top1": fam_task1["top1"],
                "task1_top3": fam_task1["top3"],
                "task1_mrr": fam_task1["mrr"],
                "task2_exact_match": fam_task2["exact_match"],
                "task2_normalized_edit_distance": fam_task2["normalized_edit_distance"],
                "task2_block_accuracy": fam_task2["block_accuracy"],
                "task3_accuracy": fam_task3["accuracy"],
                **fam_specific,
            }
        )

    family_avg = family_average(
        per_family,
        [
            "task1_top1",
            "task1_top3",
            "task1_mrr",
            "task2_exact_match",
            "task2_normalized_edit_distance",
            "task2_block_accuracy",
            "family_specific_task1_top1",
            "family_specific_task1_top3",
        ],
    )

    summary = {
        "train_family_count": train_family_count,
        "model": model.name,
        "train_sequences": len(train_sequences),
        "validation_families": len(VALIDATION_FAMILIES),
        "validation_sequences": VALIDATION_FAMILY_COUNT * VALIDATION_SEQUENCES_PER_FAMILY,
        "valid_rows": len(valid_examples),
        "anomaly_rows": len(anomaly_examples),
        "lookup_coverage": model.lookup_coverage(valid_examples),
        "task1_top1": task1["top1"],
        "task1_top3": task1["top3"],
        "task1_top5": task1["top5"],
        "task1_mrr": task1["mrr"],
        "task2_exact_match": task2["exact_match"],
        "task2_normalized_edit_distance": task2["normalized_edit_distance"],
        "task2_token_accuracy": task2["token_accuracy"],
        "task2_block_accuracy": task2["block_accuracy"],
        "task3_accuracy": task3["accuracy"],
        "task3_rule_attribution_accuracy": task3["rule_attribution_accuracy"],
        **specific,
        **family_avg,
    }
    return summary, per_family


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def svg_line_chart(
    title: str,
    rows: list[dict[str, object]],
    metric: str,
    *,
    lower_is_better: bool = False,
) -> str:
    width = 760
    height = 310
    pad_left = 58
    pad_right = 26
    pad_top = 42
    pad_bottom = 48
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    counts = list(TRAIN_FAMILY_COUNTS)
    series = []
    for model_name, color in [
        ("raw_exact_retrieval", "#8a4b0f"),
        ("template_adapted_retrieval", "#0d6b72"),
    ]:
        values = [
            float(next(r for r in rows if r["model"] == model_name and int(r["train_family_count"]) == c)[metric])
            for c in counts
        ]
        series.append((model_name, color, values))
    all_values = [v for _name, _color, values in series for v in values if not math.isnan(v)]
    y_min = min(all_values)
    y_max = max(all_values)
    if lower_is_better:
        y_min = min(0.0, y_min)
    else:
        y_max = max(1.0, y_max)
        y_min = min(0.0, y_min)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    def x_pos(count: int) -> float:
        return pad_left + (count - min(counts)) / (max(counts) - min(counts)) * plot_w

    def y_pos(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_h

    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title} chart">',
        f'<text x="{pad_left}" y="24" class="chart-title">{title}</text>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" class="axis"/>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" class="axis"/>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        pieces.append(f'<line x1="{pad_left - 4}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        pieces.append(f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    for count in counts:
        x = x_pos(count)
        pieces.append(f'<line x1="{x:.1f}" y1="{pad_top + plot_h}" x2="{x:.1f}" y2="{pad_top + plot_h + 5}" class="axis"/>')
        pieces.append(f'<text x="{x:.1f}" y="{pad_top + plot_h + 24}" text-anchor="middle" class="tick">{count}</text>')
    for model_name, color, values in series:
        points = " ".join(f"{x_pos(c):.1f},{y_pos(v):.1f}" for c, v in zip(counts, values))
        pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for count, value in zip(counts, values):
            pieces.append(f'<circle cx="{x_pos(count):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{color}"/>')
    legend_y = height - 14
    pieces += [
        f'<line x1="{pad_left}" y1="{legend_y}" x2="{pad_left + 24}" y2="{legend_y}" stroke="#8a4b0f" stroke-width="3"/>',
        f'<text x="{pad_left + 32}" y="{legend_y + 4}" class="legend">raw exact retrieval</text>',
        f'<line x1="{pad_left + 210}" y1="{legend_y}" x2="{pad_left + 234}" y2="{legend_y}" stroke="#0d6b72" stroke-width="3"/>',
        f'<text x="{pad_left + 242}" y="{legend_y + 4}" class="legend">template-adapted retrieval</text>',
        f'<text x="{pad_left + plot_w}" y="{pad_top + plot_h + 42}" text-anchor="end" class="tick">training families</text>',
        "</svg>",
    ]
    return "\n".join(pieces)


def write_outputs_readme(summary_rows: list[dict[str, object]]) -> None:
    template_rows = [
        row for row in summary_rows
        if row["model"] == "template_adapted_retrieval"
    ]
    raw_rows = [
        row for row in summary_rows
        if row["model"] == "raw_exact_retrieval"
    ]
    best_template = template_rows[-1]
    best_raw = raw_rows[-1]
    lines = [
        "# Many-Family Scaling Probe Results",
        "",
        "This is a synthetic local stress test, not an official benchmark.",
        "",
        "## Setup",
        "",
        f"- Total synthetic families: {TOTAL_FAMILIES}",
        f"- Training pool: {TRAIN_FAMILIES[0]} through {TRAIN_FAMILIES[-1]}",
        f"- Fixed validation families: {VALIDATION_FAMILIES[0]} through {VALIDATION_FAMILIES[-1]}",
        f"- Train-family counts: {', '.join(str(x) for x in TRAIN_FAMILY_COUNTS)}",
        f"- Training sequences per family: {TRAIN_SEQUENCES_PER_FAMILY}",
        f"- Validation sequences per family: {VALIDATION_SEQUENCES_PER_FAMILY}",
        f"- Validation task rows per train count: {best_template['valid_rows']}",
        "",
        "## Main Takeaway",
        "",
        "More training families help only when the model has a way to transfer",
        "family-specific exact strings. Raw exact retrieval mostly stays limited",
        "because validation-family strings are absent from training. The",
        "template-adapted model improves because it learns suffixes like",
        "`JTE DOSE VERIFICATION` behind a family placeholder and then rewrites",
        "the placeholder to the visible validation family name.",
        "",
        "At 100 training families:",
        "",
        f"- Raw Task 1 Top-1: {fmt(best_raw['task1_top1'])}",
        f"- Template Task 1 Top-1: {fmt(best_template['task1_top1'])}",
        f"- Raw family-specific next-step Top-1: {fmt(best_raw['family_specific_task1_top1'])}",
        f"- Template family-specific next-step Top-1: {fmt(best_template['family_specific_task1_top1'])}",
        f"- Raw Task 2 edit distance: {fmt(best_raw['task2_normalized_edit_distance'])}",
        f"- Template Task 2 edit distance: {fmt(best_template['task2_normalized_edit_distance'])}",
        "",
        "## Aggregate Curves",
        "",
        "| Train families | Model | Lookup coverage | Task 1 Top-1 | Task 1 Top-3 | Family-step Top-1 | Task 2 exact | Task 2 edit | Task 2 block | Task 3 acc |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['train_family_count']} | `{row['model']}` | "
            f"{fmt(row['lookup_coverage'])} | {fmt(row['task1_top1'])} | "
            f"{fmt(row['task1_top3'])} | {fmt(row['family_specific_task1_top1'])} | "
            f"{fmt(row['task2_exact_match'])} | {fmt(row['task2_normalized_edit_distance'])} | "
            f"{fmt(row['task2_block_accuracy'])} | {fmt(row['task3_accuracy'])} |"
        )
    lines += [
        "",
        "## How To Read This",
        "",
        "- `lookup_coverage` means the model found an exact 60%/80% cut-prefix",
        "  match in its training index. For template-adapted retrieval this is a",
        "  normalized template match, not a same-family exact route.",
        "- `family-specific next-step Top-1` isolates the hard rows where the true",
        "  next step starts with the validation family name.",
        "- Task 3 is flat because it is handled by the public process-rule",
        "  validator, not by learned family scaling.",
        "- This probe validates the final pitch caveat: exact strings need either",
        "  visible evidence or a valid derivation rule. More families help the",
        "  derivation rule, but raw memorization cannot invent unseen names.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_analysis_html(summary_rows: list[dict[str, object]]) -> None:
    rows_by_template = {
        int(row["train_family_count"]): row
        for row in summary_rows
        if row["model"] == "template_adapted_retrieval"
    }
    rows_by_raw = {
        int(row["train_family_count"]): row
        for row in summary_rows
        if row["model"] == "raw_exact_retrieval"
    }
    table_rows = []
    for count in TRAIN_FAMILY_COUNTS:
        raw = rows_by_raw[count]
        templ = rows_by_template[count]
        table_rows.append(
            f"""
            <tr>
              <td>{count}</td>
              <td>{fmt(raw['task1_top1'])}</td>
              <td>{fmt(templ['task1_top1'])}</td>
              <td>{fmt(raw['family_specific_task1_top1'])}</td>
              <td>{fmt(templ['family_specific_task1_top1'])}</td>
              <td>{fmt(raw['task2_normalized_edit_distance'])}</td>
              <td>{fmt(templ['task2_normalized_edit_distance'])}</td>
              <td>{fmt(templ['task2_block_accuracy'])}</td>
            </tr>
            """
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>110-Family Scaling Probe</title>
  <style>
    :root {{
      --ink: #182233;
      --muted: #5a6678;
      --line: #d8e0ea;
      --panel: #f6f8fb;
      --accent: #0d6b72;
      --warn: #8a4b0f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.58;
      background: #fff;
    }}
    .wrap {{ width: min(1160px, calc(100% - 40px)); margin: 0 auto; }}
    header {{ border-bottom: 1px solid var(--line); background: #f4fbfc; }}
    header .wrap {{ padding: 42px 0 34px; }}
    main.wrap {{ padding: 30px 0 70px; }}
    h1, h2, h3 {{ line-height: 1.18; margin: 0; color: #102033; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.1rem); max-width: 900px; }}
    h2 {{ margin-top: 38px; padding-top: 14px; border-top: 2px solid var(--accent); }}
    h3 {{ margin-top: 24px; font-size: 1.15rem; }}
    p, ul {{ max-width: 980px; }}
    code {{ background: #eef2f6; padding: 0.1em 0.3em; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0 28px; font-size: 0.92rem; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #edf3f7; }}
    .subtitle {{ color: var(--muted); font-size: 1.08rem; max-width: 900px; }}
    .callout {{ border-left: 5px solid var(--accent); background: #e9f5f6; padding: 14px 16px; border-radius: 6px; max-width: 1000px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .chart-card {{ border: 1px solid var(--line); border-radius: 6px; padding: 14px; overflow-x: auto; }}
    svg {{ width: 100%; min-width: 520px; height: auto; }}
    .axis {{ stroke: #2f3b4a; stroke-width: 1.2; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .tick, .legend {{ fill: #536173; font-size: 12px; }}
    .chart-title {{ fill: #102033; font-weight: 700; font-size: 16px; }}
    @media (max-width: 860px) {{ .chart-grid {{ grid-template-columns: 1fr; }} .wrap {{ width: min(100% - 24px, 1160px); }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>110-Family Scaling Probe</h1>
      <p class="subtitle">
        A standalone curve analysis for the hidden-family question: if we hold
        out the same 10 validation families and increase training from 2 to 100
        related families, which metrics improve and which hit an exact-string
        limit?
      </p>
    </div>
  </header>
  <main class="wrap">
    <section>
      <h2>1. TL;DR</h2>
      <div class="callout">
        More training families help when the model can transfer a learned
        family-step template to the visible validation family name. Raw exact
        retrieval cannot reliably invent validation-family strings that never
        appear in training.
      </div>
      <p>
        The probe uses 110 synthetic families. The first 100 form the training
        pool. The same final 10 families are always used for validation. We
        evaluate 2, 5, 10, 20, 40, 60, 80, and 100 training families.
      </p>
    </section>

    <section>
      <h2>2. Curves</h2>
      <div class="chart-grid">
        <div class="chart-card">{svg_line_chart("Task 1 Top-1", summary_rows, "task1_top1")}</div>
        <div class="chart-card">{svg_line_chart("Family-specific next-step Top-1", summary_rows, "family_specific_task1_top1")}</div>
        <div class="chart-card">{svg_line_chart("Task 2 edit distance", summary_rows, "task2_normalized_edit_distance", lower_is_better=True)}</div>
        <div class="chart-card">{svg_line_chart("Task 2 block accuracy", summary_rows, "task2_block_accuracy")}</div>
      </div>
    </section>

    <section>
      <h2>3. Results Table</h2>
      <table>
        <thead>
          <tr>
            <th>Train families</th>
            <th>Raw Task 1 Top-1</th>
            <th>Template Task 1 Top-1</th>
            <th>Raw family-step Top-1</th>
            <th>Template family-step Top-1</th>
            <th>Raw Task 2 edit</th>
            <th>Template Task 2 edit</th>
            <th>Template Task 2 block</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>4. Interpretation</h2>
      <p>
        The most important line is the family-specific next-step curve. Those
        are the rows where the true next step starts with the held-out validation
        family name. Raw exact retrieval performs poorly on that slice because
        the exact validation-family string was not in training. The
        template-adapted model can improve because it learns a placeholder step
        such as <code>__FAMILY__ JTE DOSE VERIFICATION</code> and then rewrites
        it to the visible family name.
      </p>
      <p>
        This does not remove the final-pitch caveat. It only works when the new
        exact strings follow a derivable pattern. If the organizers use entirely
        new names that are neither visible nor derivable, the exact-string limit
        remains.
      </p>
      <p>
        Raw data files are in <a href="outputs/README.md">outputs/README.md</a>,
        <a href="outputs/metrics_by_count.csv">metrics_by_count.csv</a>, and
        <a href="outputs/per_family_metrics.csv">per_family_metrics.csv</a>.
      </p>
    </section>
  </main>
</body>
</html>
"""
    ANALYSIS_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("generating 110 synthetic families", flush=True)
    train_by_family, validation_by_family = generate_family_sequences()
    valid_examples = build_valid_examples(validation_by_family)
    anomaly_examples = build_anomaly_examples(validation_by_family)

    summary_rows: list[dict[str, object]] = []
    per_family_rows: list[dict[str, object]] = []

    for train_count in TRAIN_FAMILY_COUNTS:
        selected_families = TRAIN_FAMILIES[:train_count]
        selected = {family: train_by_family[family] for family in selected_families}
        train_sequences = flatten(selected)
        print(f"evaluating {train_count} training families ({len(train_sequences)} routes)", flush=True)
        for model in [
            RawExactRetrieval(train_sequences),
            TemplateAdaptedRetrieval(train_sequences),
        ]:
            summary, per_family = evaluate_model(
                train_family_count=train_count,
                train_sequences=train_sequences,
                model=model,
                valid_examples=valid_examples,
                anomaly_examples=anomaly_examples,
            )
            summary_rows.append(summary)
            per_family_rows.extend(per_family)

    write_csv(OUT_DIR / "metrics_by_count.csv", list(summary_rows[0].keys()), summary_rows)
    write_csv(OUT_DIR / "per_family_metrics.csv", list(per_family_rows[0].keys()), per_family_rows)
    (OUT_DIR / "metrics_by_count.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "per_family_metrics.json").write_text(
        json.dumps(per_family_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_outputs_readme(summary_rows)
    write_analysis_html(summary_rows)
    print(f"wrote scaling outputs to {OUT_DIR.relative_to(ROOT)}")
    print(f"wrote scaling analysis to {ANALYSIS_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

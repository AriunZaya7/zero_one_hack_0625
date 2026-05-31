#!/usr/bin/env python3
"""Mine empirical support for description-derived sequence rules.

Attempt 1 is deliberately data-backed. It tests candidate ordering conditions
against the generated training traces and writes:

- empirical_rule_support.json
- empirical_rule_summary.md

Run from repo root:
    python LLM_SEQUENCE_RULING_AS_PER_DESCRIPTION_ATTEMPTS/attempt_1/mine_empirical_rule_support.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training_data"


def load_sequences() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for family, filename in [
        ("MOSFET", "MOSFET_variants.csv"),
        ("IGBT", "IGBT_variants.csv"),
        ("IC", "IC_variants.csv"),
    ]:
        grouped: dict[str, list[str]] = defaultdict(list)
        with (TRAINING_DIR / filename).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                grouped[row["SEQUENCE_ID"]].append(row["STEP"])
        for sid, steps in grouped.items():
            out[f"{family}:{sid}"] = steps
    return out


def before(seq: list[str], index: int, window: int) -> list[str]:
    return seq[max(0, index - window):index]


def after(seq: list[str], index: int, window: int) -> list[str]:
    return seq[index + 1:index + 1 + window]


def has_any(steps: list[str], predicate: Callable[[str], bool]) -> bool:
    return any(predicate(step) for step in steps)


def is_clean_like(step: str) -> bool:
    return (
        "CLEAN" in step
        or step in {"HF DIP", "DRY WAFER", "DRY WAFER BACKSIDE", "BACKSIDE RINSE"}
        or "SURFACE PREP" in step
        or step in {"OXIDE STRIP", "GATE OXIDE PREP"}
    )


def is_deposition_like(step: str) -> bool:
    return (
        step.startswith("DEPOSIT ")
        or step in {"THERMAL OXIDATION", "GATE OXIDE GROWTH", "EPITAXIAL DEPOSITION"}
    )


def is_patterned_etch(step: str) -> bool:
    if "ETCH" not in step:
        return False
    if step in {"BACKSIDE ETCH CLEAN", "ANISOTROPIC ETCH SPACER"}:
        return False
    if "CLEAN" in step:
        return False
    return True


def is_develop(step: str) -> bool:
    return step in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}


def is_strip(step: str) -> bool:
    return step in {"STRIP PHOTORESIST", "STRIP RESIST"}


def is_implant(step: str) -> bool:
    return step.startswith("IMPLANT ")


def is_anneal_or_diffusion(step: str) -> bool:
    return "ANNEAL" in step or step == "DRIVE IN DIFFUSION"


def is_cmp(step: str) -> bool:
    return step.startswith("CMP ")


def is_fill_or_deposition(step: str) -> bool:
    return is_deposition_like(step) or step in {"FILL VIA METAL", "FILL VIA TUNGSTEN"}


def is_electrical_test(step: str) -> bool:
    return step in {
        "PARAMETRIC TEST",
        "ELECTRICAL PARAMETRIC TEST",
        "THRESHOLD VOLTAGE TEST",
        "BREAKDOWN VOLTAGE TEST",
        "LEAKAGE TEST",
        "SWITCHING TEST",
        "WAFER SORT TEST",
    }


def measure_subject_ok(seq: list[str], i: int, step: str) -> bool | None:
    prev = before(seq, i, 10)
    if step in {"MEASURE OXIDE THICKNESS", "MEASURE GATE OXIDE THICKNESS", "MEASURE OXIDE QUALITY"}:
        return has_any(prev, lambda s: s in {"THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC", "ANNEAL OXIDE", "ANNEAL DIELECTRIC"})
    if step in {"MEASURE VIA CD", "MEASURE VIA RESISTANCE", "MEASURE CONTACT RESISTANCE"}:
        return has_any(prev, lambda s: "VIA" in s or s in {"DEPOSIT BARRIER METAL", "FILL VIA METAL", "FILL VIA TUNGSTEN"})
    if step == "MEASURE SHEET RESISTANCE":
        return has_any(prev, lambda s: is_implant(s) or is_anneal_or_diffusion(s))
    if step in {"MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"}:
        return has_any(prev, lambda s: "PASSIVATION" in s)
    if step == "MEASURE METAL THICKNESS":
        return has_any(prev, lambda s: s in {"DEPOSIT METAL 1", "DEPOSIT TOP METAL", "ANNEAL METAL 1", "ANNEAL METAL"})
    if step == "MEASURE BACKSIDE CONTACT":
        return has_any(prev, lambda s: s in {"DEPOSIT BACKSIDE METAL", "BACKSIDE ANNEAL"})
    return None


def add_result(results: list[dict], rule_id: str, total: int, satisfied: int, examples: list[dict], notes: str) -> None:
    rate = satisfied / total if total else None
    results.append(
        {
            "rule_id": rule_id,
            "total_trigger_occurrences": total,
            "satisfied_occurrences": satisfied,
            "support_rate": rate,
            "example_failures": examples[:8],
            "notes": notes,
        }
    )


def mine() -> dict:
    seqs = load_sequences()
    results: list[dict] = []

    total = ok = 0
    examples: list[dict] = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if is_deposition_like(step):
                total += 1
                satisfied = has_any(before(seq, i, 12), is_clean_like)
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "context": before(seq, i, 6)})
    add_result(results, "EMP_001_CLEAN_BEFORE_DEPOSITION_LIKE", total, ok, examples, "Tests clean-like context before oxidation/epitaxy/deposition.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if is_patterned_etch(step):
                total += 1
                satisfied = has_any(before(seq, i, 15), is_develop)
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "context": before(seq, i, 8)})
    add_result(results, "EMP_002_DEVELOP_BEFORE_PATTERNED_ETCH", total, ok, examples, "Patterned etches should have developed resist nearby.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if is_patterned_etch(step):
                total += 1
                nxt = after(seq, i, 8)
                satisfied = has_any(nxt, is_strip) and has_any(nxt, is_clean_like)
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "after": nxt})
    add_result(results, "EMP_003_STRIP_AND_CLEAN_AFTER_ETCH", total, ok, examples, "Checks whether etch is followed by strip and clean within 8 steps.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if is_implant(step):
                total += 1
                satisfied = has_any(after(seq, i, 10), is_anneal_or_diffusion)
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "after": after(seq, i, 10)})
    add_result(results, "EMP_004_IMPLANT_FOLLOWED_BY_ACTIVATION", total, ok, examples, "Looks for drive-in diffusion, RTA, or anneal after implant.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if is_cmp(step):
                total += 1
                satisfied = has_any(before(seq, i, 8), is_fill_or_deposition)
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "context": before(seq, i, 8)})
    add_result(results, "EMP_005_CMP_AFTER_DEPOSITION_OR_FILL", total, ok, examples, "Checks CMP has material to planarize nearby.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        cure_positions = [i for i, s in enumerate(seq) if s == "CURE PASSIVATION"]
        cure_pos = min(cure_positions) if cure_positions else None
        for i, step in enumerate(seq):
            if is_electrical_test(step):
                total += 1
                satisfied = cure_pos is not None and cure_pos < i
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "step": step, "cure_pos": cure_pos})
    add_result(results, "EMP_006_TEST_AFTER_PASSIVATION_CURE", total, ok, examples, "Electrical tests should appear after passivation cure.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            if step == "SHIP LOT":
                total += 1
                satisfied = "WAFER SORT TEST" in before(seq, i, len(seq))
                ok += int(satisfied)
                if not satisfied:
                    examples.append({"sequence_id": key, "index": i, "context": before(seq, i, 10)})
    add_result(results, "EMP_007_WAFER_SORT_BEFORE_SHIP", total, ok, examples, "Terminal order sanity check.")

    total = ok = 0
    examples = []
    for key, seq in seqs.items():
        for i, step in enumerate(seq):
            subject = measure_subject_ok(seq, i, step)
            if subject is not None:
                total += 1
                ok += int(subject)
                if not subject:
                    examples.append({"sequence_id": key, "index": i, "step": step, "context": before(seq, i, 10)})
    add_result(results, "EMP_008_MEASUREMENT_SUBJECT_ALIGNMENT", total, ok, examples, "Soft heuristic for whether a measurement follows what it measures.")

    transition_counts: dict[str, Counter] = defaultdict(Counter)
    for seq in seqs.values():
        for a, b in zip(seq, seq[1:]):
            transition_counts[a][b] += 1
    top_transitions = {
        step: [{"next_step": nxt, "count": count} for nxt, count in counts.most_common(8)]
        for step, counts in sorted(transition_counts.items())
        if sum(counts.values()) >= 500
    }

    return {
        "attempt_id": "attempt_1",
        "title": "Empirical support for description-derived sequence rules",
        "status": "generated_from_training_traces",
        "sequence_count": len(seqs),
        "rule_results": results,
        "top_transitions_for_frequent_steps": top_transitions,
    }


def write_summary(data: dict) -> None:
    lines = [
        "# Attempt 1 - Empirical Rule Support",
        "",
        "This attempt tests description-derived ordering rules against the generated training traces.",
        "It is not online research; it is corpus evidence from the repo itself.",
        "",
        f"Sequences analyzed: `{data['sequence_count']}`",
        "",
        "## Rule Support",
        "",
        "| Rule | Triggers | Satisfied | Support | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for result in data["rule_results"]:
        rate = result["support_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(
            f"| `{result['rule_id']}` | {result['total_trigger_occurrences']} | "
            f"{result['satisfied_occurrences']} | {rate_text} | {result['notes']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- High support means the heuristic matches this synthetic corpus well.",
        "- Lower support does not automatically mean the rule is wrong; it may be too strict for optional or context-dependent steps.",
        "- Use low-support rules as soft reranking features, not hard validators.",
        "",
        "## Files",
        "",
        "- `mine_empirical_rule_support.py` - reproducible miner.",
        "- `empirical_rule_support.json` - machine-readable output.",
        "- `empirical_rule_summary.md` - this summary.",
    ]
    (OUT_DIR / "empirical_rule_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = mine()
    (OUT_DIR / "empirical_rule_support.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_summary(data)
    print(f"wrote {OUT_DIR / 'empirical_rule_support.json'}")
    print(f"wrote {OUT_DIR / 'empirical_rule_summary.md'}")


if __name__ == "__main__":
    main()

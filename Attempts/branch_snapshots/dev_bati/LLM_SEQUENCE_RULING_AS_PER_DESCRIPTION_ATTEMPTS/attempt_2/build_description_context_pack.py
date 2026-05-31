#!/usr/bin/env python3
"""Build a description-aware context pack for future solutions.

Attempt 2 is not a scoring model. It converts the long descriptions and
parameter-description rows into reusable semantic features:

- operation-family labels,
- duplicate step warnings,
- context-aware join guidance,
- sample per-occurrence features.

Run from repo root:
    python LLM_SEQUENCE_RULING_AS_PER_DESCRIPTION_ATTEMPTS/attempt_2/build_description_context_pack.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training_data"


PARAM_FILES = {
    "MOSFET": "MOSFET_longdescription_parameters.csv",
    "IGBT": "IGBT_longdescription_parameters.csv",
    "IC": "IC_longdescription_parameters.csv",
}

VARIANT_FILES = {
    "MOSFET": "MOSFET_variants.csv",
    "IGBT": "IGBT_variants.csv",
    "IC": "IC_variants.csv",
}


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = (
            key.replace("\ufeff", "")
            .replace("\u2011", "-")
            .replace("\u2013", "-")
            .strip()
            .strip('"')
        )
        normalized_value = (
            (value or "")
            .replace("\u202f", " ")
            .replace("\u00b5", "u")
            .replace("\u2011", "-")
            .replace("\u2013", "-")
            .strip()
        )
        out[normalized_key] = normalized_value
    return out


def classify_step(step: str, description: str = "", params: str = "") -> str:
    text = f"{step} {description} {params}".upper()
    if step in {"RECEIVE WAFER LOT", "LOT IDENTIFICATION", "LOT RELEASE", "FINAL LOT RELEASE", "SHIP LOT", "PACKAGE PREPARATION"}:
        return "logistics"
    if "WAFER SORT" in text or "TEST" in text or "YIELD ANALYSIS" in text:
        return "test"
    if step.startswith("MEASURE") or "METROLOGY" in text or "INSPECTION" in step:
        return "measurement_inspection"
    if "CMP" in step or "PLANARIZATION" in text:
        return "planarization"
    if step.startswith("IMPLANT"):
        return "implant"
    if "ANNEAL" in step or step == "DRIVE IN DIFFUSION":
        return "anneal_diffusion"
    if step.startswith("STRIP"):
        return "strip_resist"
    if "ETCH" in step and "CLEAN" not in step:
        return "etch"
    if (
        "PHOTORESIST" in step
        or "LITHO" in step
        or step.startswith("ALIGN MASK")
        or step.startswith("EXPOSE LITHO")
        or step in {"SOFT BAKE", "HARD BAKE", "POST EXPOSE BAKE", "DEVELOP PAD WINDOW"}
    ):
        return "lithography"
    if "CLEAN" in step or step in {"HF DIP", "DRY WAFER", "DRY WAFER BACKSIDE", "BACKSIDE RINSE", "OXIDE STRIP"}:
        return "clean_surface_prep"
    if step.startswith("DEPOSIT") or "OXIDATION" in step or "EPITAXIAL DEPOSITION" in step or "GROWTH" in step:
        return "deposition_growth"
    if "BACKSIDE" in step or "GRINDING WAFER BACKSIDE" in step:
        return "backside"
    if "VIA" in step:
        return "via_interconnect"
    return "other_process"


def mask_level(step: str) -> int | None:
    match = re.search(r"LEVEL (\d+)", step)
    if match:
        return int(match.group(1))
    return None


def extract_units(params: str) -> list[str]:
    units = []
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:nm|um|mm|C|Torr|mTorr|W|sccm|slm|rpm|s|min|keV|cm|mJ/cm)",
        r"\b\d+(?:\.\d+)?(?:-)\d+(?:\.\d+)?\s*(?:nm|um|mm|C|Torr|mTorr|W|sccm|slm|rpm|s|min|keV|cm)",
    ]
    for pattern in patterns:
        units.extend(re.findall(pattern, params))
    return units[:12]


def load_parameter_rows() -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {}
    for family, filename in PARAM_FILES.items():
        rows: list[dict] = []
        with (TRAINING_DIR / filename).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                norm = normalize_row(row)
                step = norm.get("STEP", "")
                desc = norm.get("DESCRIPTION", "")
                params = norm.get("REALISTIC FAB-LEVEL PARAMETERS", "")
                rows.append(
                    {
                        "step": step,
                        "description": desc,
                        "parameters": params,
                        "operation_family": classify_step(step, desc, params),
                        "mask_level": mask_level(step),
                        "numeric_unit_hints": extract_units(params),
                    }
                )
        by_family[family] = rows
    return by_family


def load_first_sequences() -> dict[str, tuple[str, list[str]]]:
    out: dict[str, tuple[str, list[str]]] = {}
    for family, filename in VARIANT_FILES.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        with (TRAINING_DIR / filename).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                grouped[row["SEQUENCE_ID"]].append(row["STEP"])
        first_id = sorted(grouped)[0]
        out[family] = (first_id, grouped[first_id])
    return out


def infer_macro_block(step: str, index: int, sequence: list[str]) -> str:
    if index < 3:
        return "prefix"
    if "PASSIVATION" in step or "PAD WINDOW" in step:
        return "passivation"
    if "BACKSIDE" in step or "GRINDING WAFER BACKSIDE" in step:
        return "backside"
    if step in {"PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST", "LEAKAGE TEST", "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST", "SWITCHING TEST", "WAFER SORT TEST", "YIELD ANALYSIS"}:
        return "test_suite"
    if step in {"LOT RELEASE", "FINAL LOT RELEASE", "SHIP LOT", "PACKAGE PREPARATION"}:
        return "suffix"
    if "VIA" in step:
        return "via"
    if "METAL" in step:
        return "metal"
    if "INTERLAYER" in step or "INTERLEVEL" in step or "DIELECTRIC" in step or "CMP" in step:
        return "ild_or_planarization"
    if mask_level(step) is not None or step in {"SPIN COAT PHOTORESIST", "SOFT BAKE", "DEVELOP PHOTORESIST", "HARD BAKE", "POST EXPOSE BAKE"}:
        return "lithography_cycle"
    if index < len(sequence) * 0.18:
        return "prep"
    return "device_fabrication"


def distance_since(sequence: list[str], index: int, predicate) -> int | None:
    for j in range(index - 1, -1, -1):
        if predicate(sequence[j]):
            return index - j
    return None


def build_occurrence_samples() -> list[dict]:
    samples = []
    first_sequences = load_first_sequences()
    for family, (seq_id, sequence) in first_sequences.items():
        for index, step in enumerate(sequence[:35]):
            samples.append(
                {
                    "family": family,
                    "sequence_id": seq_id,
                    "index": index,
                    "step": step,
                    "operation_family": classify_step(step),
                    "macro_block_guess": infer_macro_block(step, index, sequence),
                    "mask_level": mask_level(step),
                    "normalized_position": round(index / max(len(sequence) - 1, 1), 4),
                    "previous_3": sequence[max(0, index - 3):index],
                    "next_3_training_only": sequence[index + 1:index + 4],
                    "distance_since_clean": distance_since(sequence, index, lambda s: classify_step(s) == "clean_surface_prep"),
                    "distance_since_develop": distance_since(sequence, index, lambda s: s in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}),
                }
            )
    return samples


def build_pack() -> dict:
    rows_by_family = load_parameter_rows()
    catalog: dict[str, dict] = {}
    duplicates: list[dict] = []
    family_counts: dict[str, dict[str, int]] = {}

    for family, rows in rows_by_family.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        op_counts = Counter()
        for row in rows:
            grouped[row["step"]].append(row)
            op_counts[row["operation_family"]] += 1
        family_counts[family] = dict(sorted(op_counts.items()))
        for step, items in sorted(grouped.items()):
            catalog[f"{family}:{step}"] = {
                "family": family,
                "step": step,
                "row_count": len(items),
                "operation_families": sorted({item["operation_family"] for item in items}),
                "rows": items,
            }
            if len(items) > 1:
                duplicates.append(
                    {
                        "family": family,
                        "step": step,
                        "row_count": len(items),
                        "operation_families": sorted({item["operation_family"] for item in items}),
                        "why_it_matters": "A future solution should resolve this step using local context, not a single STEP lookup.",
                    }
                )

    return {
        "attempt_id": "attempt_2",
        "title": "Description-aware semantic context pack",
        "status": "generated_from_description_and_parameter_files",
        "feature_schema": {
            "family": "Known product family in training/eval input.",
            "step": "Raw step token.",
            "operation_family": "Semantic class inferred from step string and descriptions.",
            "macro_block_guess": "Approximate high-level process block inferred from local sequence state.",
            "mask_level": "Integer lithography level parsed from step string when available.",
            "normalized_position": "Step index divided by sequence length.",
            "previous_3": "Visible local history for scoring a candidate occurrence.",
            "distance_since_clean": "Number of steps since a clean/surface-prep event.",
            "distance_since_develop": "Number of steps since photoresist develop.",
            "numeric_unit_hints": "Recipe/parameter snippets from reference rows, not per-lot measurements.",
        },
        "family_operation_counts": family_counts,
        "duplicate_step_warnings": duplicates,
        "step_catalog": catalog,
        "sample_occurrence_features": build_occurrence_samples(),
    }


def write_summary(pack: dict) -> None:
    lines = [
        "# Attempt 2 - Description-Aware Context Pack",
        "",
        "This attempt turns long descriptions and generic parameter references into a reusable semantic feature pack.",
        "",
        "## Why this exists",
        "",
        "The official traces are step sequences, but the repo also contains text descriptions and parameter hints. Those hints can enrich each step occurrence with operation family, macro block, mask level, and local-rule features.",
        "",
        "## Generated Files",
        "",
        "- `description_context_pack.json` - machine-readable catalog and feature schema.",
        "- `sample_occurrence_features.json` - small readable sample of occurrence-level features.",
        "- `build_description_context_pack.py` - reproducible builder.",
        "",
        "## Operation Counts",
        "",
        "| Family | Operation family | Rows |",
        "|---|---|---:|",
    ]
    for family, counts in pack["family_operation_counts"].items():
        for op, count in counts.items():
            lines.append(f"| {family} | `{op}` | {count} |")
    lines += [
        "",
        "## Duplicate Step Warnings",
        "",
        "Repeated step names are expected in lithography and measurement. This is why future joins should use local context.",
        "",
        "| Family | Step | Rows |",
        "|---|---|---:|",
    ]
    for warning in pack["duplicate_step_warnings"][:40]:
        lines.append(f"| {warning['family']} | `{warning['step']}` | {warning['row_count']} |")
    lines += [
        "",
        "## Integration Idea",
        "",
        "Use this pack to compute features for candidate next steps, then combine them with n-gram/retrieval scores:",
        "",
        "```text",
        "score = model_score",
        "      + block_continuation_bonus",
        "      + description_family_match_bonus",
        "      - distance_since_clean_penalty_for_deposition",
        "      - distance_since_develop_penalty_for_etch",
        "```",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pack = build_pack()
    (OUT_DIR / "description_context_pack.json").write_text(
        json.dumps(pack, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "sample_occurrence_features.json").write_text(
        json.dumps(pack["sample_occurrence_features"], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_summary(pack)
    print(f"wrote {OUT_DIR / 'description_context_pack.json'}")
    print(f"wrote {OUT_DIR / 'sample_occurrence_features.json'}")
    print(f"wrote {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()

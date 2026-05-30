#!/usr/bin/env python3
"""Verify the final Industrial AI submission bundle.

Run from the repository root:
    python -B final_submission/industrial_ai_infineon_solution_21/VERIFY_FINAL_SUBMISSION.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
SCORED = BUNDLE / "scored_csvs"
SOURCE = ROOT / "solutions" / "solution_21_template_boosted_bridge" / "outputs" / "official_submission"

CONTRACTS = {
    "nextstep.csv": {
        "header": ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"],
        "rows": 600,
    },
    "completion.csv": {
        "header": ["EXAMPLE_ID", "PREDICTED_SEQUENCE"],
        "rows": 600,
    },
    "anomaly.csv": {
        "header": ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"],
        "rows": 987,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_shape(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return header, rows


def main() -> None:
    manifest = {
        "solution": "solution_21_template_boosted_bridge",
        "scored_csvs": {},
        "attachments": sorted(path.name for path in (BUNDLE / "attachments").iterdir() if path.is_file()),
    }

    actual_scored_files = sorted(path.name for path in SCORED.iterdir() if path.is_file())
    expected_scored_files = sorted(CONTRACTS)
    if actual_scored_files != expected_scored_files:
        raise SystemExit(f"scored_csvs contains {actual_scored_files}, expected {expected_scored_files}")

    for name, contract in CONTRACTS.items():
        path = SCORED / name
        source_path = SOURCE / name
        if not path.exists():
            raise SystemExit(f"Missing scored CSV: {path}")
        if not source_path.exists():
            raise SystemExit(f"Missing source official CSV: {source_path}")
        header, rows = csv_shape(path)
        if header != contract["header"]:
            raise SystemExit(f"{name} header {header!r}, expected {contract['header']!r}")
        if rows != contract["rows"]:
            raise SystemExit(f"{name} row count {rows}, expected {contract['rows']}")
        scored_hash = sha256(path)
        source_hash = sha256(source_path)
        if scored_hash != source_hash:
            raise SystemExit(f"{name} differs from Solution 21 official_submission source")
        manifest["scored_csvs"][name] = {
            "rows": rows,
            "header": header,
            "sha256": scored_hash,
            "matches_solution21_official_submission": True,
        }

    manifest_path = BUNDLE / "FINAL_SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("FINAL SUBMISSION OK")


if __name__ == "__main__":
    main()

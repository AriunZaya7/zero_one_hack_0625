"""Score self-eval predictions with the OFFICIAL scorer (participant_files/eval_metrics.py).

We hold ground truth only for the *self-eval* split (built by make_selfeval.py with
injected violations). The organisers score the real submission; these are clearly-labelled
**self-eval estimates** for the report — but produced by the *same* official metric code,
so they are directly comparable in shape (Task 1 Top-k/MRR, Task 2 NED/Exact/Token/Block,
Task 3 Acc/P/R/F1/AUC/rule-attribution).

This adapter converts our self_eval/ground_truth_*.csv into the column names the official
scorer expects, then shells out to eval_metrics.py for each task.

Usage:
    python score_selfeval.py --tag gpt_large_ckpt           # scores submissions/<task>_<tag>.csv
    python score_selfeval.py --tag gpt_large_ckpt --sub-dir submissions
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SCORER = ROOT / "participant_files" / "eval_metrics.py"
SELF_EVAL = ROOT / "self_eval"
GT_DIR = SELF_EVAL / "official_gt"


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_official_gt() -> dict[str, Path]:
    """Translate self_eval ground truth → official scorer column names."""
    GT_DIR.mkdir(parents=True, exist_ok=True)
    valid = _read(SELF_EVAL / "ground_truth_valid.csv")
    anom = _read(SELF_EVAL / "ground_truth_anomaly.csv")

    # next-step GT: EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, NEXT_STEP
    ns_path = GT_DIR / "gt_nextstep.csv"
    with ns_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY",
                                          "COMPLETION_FRACTION", "NEXT_STEP"])
        w.writeheader()
        for r in valid:
            remaining = [s for s in r["_REMAINING"].split("|") if s]
            if not remaining:
                continue
            w.writerow({"EXAMPLE_ID": r["EXAMPLE_ID"], "FAMILY": r["FAMILY"],
                        "COMPLETION_FRACTION": r["COMPLETION_FRACTION"],
                        "NEXT_STEP": remaining[0]})

    # completion GT: EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, FULL_SEQUENCE
    comp_path = GT_DIR / "gt_completion.csv"
    with comp_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION",
                                          "PARTIAL_SEQUENCE", "FULL_SEQUENCE"])
        w.writeheader()
        for r in valid:
            w.writerow({"EXAMPLE_ID": r["EXAMPLE_ID"], "FAMILY": r["FAMILY"],
                        "COMPLETION_FRACTION": r["COMPLETION_FRACTION"],
                        "PARTIAL_SEQUENCE": r["PARTIAL_SEQUENCE"],
                        "FULL_SEQUENCE": r["_FULL_SEQUENCE"]})

    # anomaly GT: forbidden rows only (label 0) with VIOLATION_RULE; + valid supplement
    forb_path = GT_DIR / "gt_anomaly_forbidden.csv"
    valid_sup = GT_DIR / "gt_anomaly_valid.csv"
    with forb_path.open("w", newline="") as ff, valid_sup.open("w", newline="") as vf:
        fw = csv.DictWriter(ff, fieldnames=["EXAMPLE_ID", "VIOLATION_RULE"])
        vw = csv.DictWriter(vf, fieldnames=["EXAMPLE_ID"])
        fw.writeheader()
        vw.writeheader()
        for r in anom:
            if int(r["IS_VALID"]) == 0:
                fw.writerow({"EXAMPLE_ID": r["EXAMPLE_ID"],
                             "VIOLATION_RULE": r["RULE_VIOLATED"]})
            else:
                vw.writerow({"EXAMPLE_ID": r["EXAMPLE_ID"]})

    return {"nextstep": ns_path, "completion": comp_path,
            "anomaly_forbidden": forb_path, "anomaly_valid": valid_sup}


def _run(task: str, gt: Path, pred: Path, supplement: Path | None = None):
    if not pred.exists():
        print(f"[skip] {task}: predictions not found at {pred}")
        return
    cmd = [sys.executable, str(SCORER), "--task", task,
           "--ground-truth", str(gt), "--predictions", str(pred)]
    if supplement:
        cmd += ["--valid-supplement", str(supplement)]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt_large_ckpt",
                    help="suffix of the prediction files: task{1,2,3}_<tag>.csv")
    ap.add_argument("--sub-dir", default="submissions",
                    help="directory holding the self-eval prediction CSVs")
    args = ap.parse_args()

    gt = build_official_gt()
    sub = ROOT / args.sub_dir

    print("=" * 60)
    print(f"SELF-EVAL SCORING (official metric code) — tag={args.tag}")
    print("NOTE: self-eval estimates; the organisers score the real submission.")
    print("=" * 60)

    _run("next-step", gt["nextstep"], sub / f"task1_{args.tag}.csv")
    _run("completion", gt["completion"], sub / f"task2_{args.tag}.csv")
    _run("anomaly", gt["anomaly_forbidden"], sub / f"task3_{args.tag}.csv",
         supplement=gt["anomaly_valid"])


if __name__ == "__main__":
    main()

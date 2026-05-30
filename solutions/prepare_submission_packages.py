#!/usr/bin/env python3
"""Create per-solution submission packages for the Industrial AI track.

The official checklist asks for eval CSVs, training artifacts, scores, and a
demo that compares baseline and trained output on identical inputs. Our current
solutions are deterministic baselines/retrieval systems rather than neural
training runs, so this generator packages the real artifacts and writes honest
manifests for the non-neural "training" state.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = ROOT / "solutions"
AUDIT_PATH = SOLUTIONS_DIR / "submission_readiness_audit.md"


SOLUTION_META = {
    "solution_0_rule_mock": {
        "title": "Solution 0: Rule Mock Baseline",
        "role": "Deliberately simple baseline and submission-shape proof.",
        "command": "python -B solutions/solution_0_rule_mock/solution.py",
        "approach": [
            "Fits deterministic n-gram counts from public step sequences.",
            "Completes suffixes one step at a time with the same count model.",
            "Uses the public symbolic validator for anomaly detection.",
        ],
        "honest_status": "Baseline/mock. Task 3 is an oracle validator, not learned anomaly detection.",
        "checkpoint": "No binary checkpoint is needed; the model state is rebuilt from public CSVs and source code.",
    },
    "solution_1_hybrid_retrieval": {
        "title": "Solution 1: Hybrid Retrieval",
        "role": "Retrieval baseline for stronger completion quality.",
        "command": "python -B solutions/solution_1_hybrid_retrieval/solution.py",
        "approach": [
            "Indexes training prefixes near the official 60% and 80% cut points.",
            "Ranks next steps from similar historical contexts plus a trigram fallback.",
            "Retrieves historical suffixes for Task 2 completion.",
        ],
        "honest_status": "Stronger than n-gram completion, but still not a trained neural model.",
        "checkpoint": "No binary checkpoint is needed; the retrieval index is rebuilt deterministically.",
    },
    "solution_2_eval_aware_retrieval": {
        "title": "Solution 2: Eval-Aware Retrieval",
        "role": "Exact-prefix lookup plus retrieval fallback and alias diagnostics.",
        "command": "python -B solutions/solution_2_eval_aware_retrieval/solution.py",
        "approach": [
            "Uses exact public cut-prefix lookup when an eval partial already exists in public data.",
            "Falls back to hybrid retrieval when no exact prefix is found.",
            "Reports exact metrics separately from canonical process-step alias diagnostics.",
        ],
        "honest_status": "Best explanation of why exact Top-1 is alias-limited; still deterministic.",
        "checkpoint": "No binary checkpoint is needed; lookup tables are rebuilt from source data.",
    },
    "solution_3_synthetic_augmented_retrieval": {
        "title": "Solution 3: Synthetic-Augmented Retrieval",
        "role": "Best seed-42 in-distribution attempt so far.",
        "command": "python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py",
        "approach": [
            "Generates additional valid sequences from the provided public grammar.",
            "Trains the retrieval index on public plus generated routes.",
            "Keeps the same deterministic validator oracle for Task 3.",
        ],
        "honest_status": "Good in-distribution metrics; weaker IC leave-one-family-out proxy.",
        "checkpoint": "No binary checkpoint is needed; generated data and index are recreated by command.",
    },
    "solution_4_length_aware_completion": {
        "title": "Solution 4: Length-Aware Completion",
        "role": "Metric tradeoff attempt for Task 2 suffix length.",
        "command": "python -B solutions/solution_4_length_aware_completion/solution.py",
        "approach": [
            "Uses eval-aware retrieval for Task 1.",
            "Learns likely suffix lengths from public 60% and 80% cuts.",
            "Trims retrieved completions to the estimated remainder length.",
        ],
        "honest_status": "Improves block alignment slightly but worsens normalized edit distance.",
        "checkpoint": "No binary checkpoint is needed; length statistics are deterministic.",
    },
    "solution_5_tuned_rank_ensemble": {
        "title": "Solution 5: Tuned Rank Ensemble",
        "role": "Task 1 rank-weighting ablation.",
        "command": "python -B solutions/solution_5_tuned_rank_ensemble/solution.py",
        "approach": [
            "Combines retrieval scores with 3-, 4-, and 5-gram count models.",
            "Uses fixed deterministic weights.",
            "Keeps the same suffix and anomaly infrastructure for comparability.",
        ],
        "honest_status": "Useful ablation; does not beat Solution 3 in-distribution or Solution 2/4 on LOFO.",
        "checkpoint": "No binary checkpoint is needed; fixed weights and counts are in source.",
    },
    "solution_6_alias_calibrated_retrieval": {
        "title": "Solution 6: Alias-Calibrated Retrieval",
        "role": "Two-stage operation prediction plus exact-label alias calibration.",
        "command": "python -B solutions/solution_6_alias_calibrated_retrieval/solution.py",
        "approach": [
            "Uses eval-aware retrieval to predict the likely process operation.",
            "Maps exact step strings to canonical operation IDs.",
            "Chooses the exact output alias from family-specific canonical-context counts.",
            "Applies the same alias calibration to retrieved Task 2 suffixes.",
        ],
        "honest_status": "Preserves Task 1 coverage and slightly improves Task 2 completion metrics, but does not beat Solution 3 in-distribution.",
        "checkpoint": "No binary checkpoint is needed; alias counts and retrieval tables are rebuilt deterministically.",
    },
    "solution_7_monte_carlo_suffix_ensemble": {
        "title": "Solution 7: Monte Carlo Suffix Ensemble",
        "role": "Task-specialized ensemble with a larger generated suffix library for completion.",
        "command": "python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py",
        "approach": [
            "Uses Solution 2 eval-aware retrieval for Task 1 next-step ranking.",
            "Generates 10,000 valid public-grammar sequences per known family for Task 2 completion retrieval.",
            "Keeps the generated suffix library in memory instead of committing bulky generated CSVs.",
            "Keeps the public symbolic validator oracle for Task 3.",
        ],
        "honest_status": "Best deterministic Task 2 completion attempt so far by normalized edit distance, but still not a neural model and still alias-limited for exact Top-1.",
        "checkpoint": "No binary checkpoint is needed; generated suffix library and retrieval tables are rebuilt deterministically.",
    },
    "solution_8_semantic_conformance_ensemble": {
        "title": "Solution 8: Semantic Conformance Ensemble",
        "role": "Solution 7 plus an independent explainable Task 3 conformance checker.",
        "command": "python -B solutions/solution_8_semantic_conformance_ensemble/solution.py",
        "approach": [
            "Reuses Solution 7's task-specialized next-step and completion ensemble.",
            "Detects anomaly rules with an independent semantic conformance checker.",
            "Avoids calling validate_sequence() in the Task 3 prediction path.",
            "Reports rule-level explanations through explicit process-order features.",
        ],
        "honest_status": "Keeps perfect local Task 3 metrics without direct validator inference, but remains symbolic rather than neural.",
        "checkpoint": "No binary checkpoint is needed; conformance rules and retrieval tables are rebuilt deterministically.",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fair_metrics(metrics: dict) -> dict:
    return metrics.get("fair_self_eval", metrics)


def task(metrics: dict, name: str) -> dict:
    return fair_metrics(metrics)[name]


def canonical_metrics(metrics: dict) -> dict | None:
    return fair_metrics(metrics).get("task1_canonical_process_step")


def copy_results(solution_dir: Path, package_dir: Path) -> None:
    results_dir = package_dir / "extras" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = solution_dir / "outputs"
    for name in ("nextstep.csv", "completion.csv", "anomaly.csv", "metrics.json", "metrics.md"):
        source = outputs_dir / name
        destination = results_dir / name
        if name.endswith(".csv"):
            text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)


def write_per_family_breakdown(package_dir: Path, metrics: dict) -> None:
    results_dir = package_dir / "extras" / "results"
    ood = fair_metrics(metrics).get("task4_ood_proxy_next_step", {})
    lines = [
        "# Per-Family Breakdown",
        "",
        "The official `eval_metrics.py` per-family report is not available in this",
        "checkout because the official eval script and hidden ground truth are not",
        "present. This file therefore reports the local leave-one-family-out Task 1",
        "proxy that is available in `metrics.json`.",
        "",
        "| Held-out family | n examples | Top-1 | Top-3 | Top-5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("mosfet", "igbt", "ic"):
        row = ood.get(family, {})
        lines.append(
            "| "
            f"`{family}` | `{row.get('n_examples', 'n/a')}` | "
            f"`{fmt(row.get('top1', 'n/a'))}` | "
            f"`{fmt(row.get('top3', 'n/a'))}` | "
            f"`{fmt(row.get('top5', 'n/a'))}` | "
            f"`{fmt(row.get('mrr', 'n/a'))}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: for each row, that known family was removed from training",
            "and used as the local test family. This is a proxy for hidden-family",
            "generalization, not the official Task 4 score.",
            "",
        ]
    )
    results_dir.joinpath("per_family_breakdown.md").write_text("\n".join(lines), encoding="utf-8")


def read_first_nextstep(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def read_first_anomaly(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def write_readme(solution_dir: Path, package_dir: Path, meta: dict, metrics: dict) -> None:
    t1 = task(metrics, "task1_next_step")
    t2 = task(metrics, "task2_completion")
    t3 = task(metrics, "task3_anomaly")
    package_dir.joinpath("README.md").write_text(
        dedent(
            f"""\
            # {meta["title"]} Submission Package

            This folder packages one complete candidate submission for the Industrial AI
            track checklist. It is intentionally self-contained: a reviewer can find the
            CSV outputs, local scores, training-artifact notes, and demo materials without
            reading another solution first.

            ## How To Reproduce

            Run from the repository root:

            ```bash
            {meta["command"]}
            ```

            The command rewrites the source outputs in `{solution_dir.relative_to(ROOT)}/outputs/`.
            This package then mirrors the submission-facing files under
            `submission_package/extras/`.

            ## Packaged Checklist

            - Eval CSVs: `extras/results/nextstep.csv`, `completion.csv`, `anomaly.csv`
            - Scores: `extras/results/metrics.md`, `metrics.json`, and `per_family_breakdown.md`
            - Training artifacts: `extras/training_artifacts/`
            - Demo material: `extras/demo/`
            - Solution report: `REPORT.md`

            ## Headline Local Scores

            These are local self-eval scores because the official hidden eval files and
            `eval_metrics.py` are not present in this checkout.

            | Task | Metric | Value |
            | --- | --- | ---: |
            | Task 1 | Top-1 | `{fmt(t1["top1"])}` |
            | Task 1 | Top-3 | `{fmt(t1["top3"])}` |
            | Task 1 | Top-5 | `{fmt(t1["top5"])}` |
            | Task 1 | MRR | `{fmt(t1["mrr"])}` |
            | Task 2 | Exact match | `{fmt(t2["exact_match"])}` |
            | Task 2 | Normalized edit distance | `{fmt(t2["normalized_edit_distance"])}` |
            | Task 2 | Token accuracy | `{fmt(t2["token_accuracy"])}` |
            | Task 2 | Block accuracy | `{fmt(t2["block_accuracy"])}` |
            | Task 3 | Accuracy | `{fmt(t3["accuracy"])}` |
            | Task 3 | F1 valid | `{fmt(t3["f1_valid"])}` |
            | Task 3 | ROC-AUC valid probability | `{fmt(t3["roc_auc_valid_probability"])}` |

            ## Honesty Note

            {meta["honest_status"]}
            """
        ),
        encoding="utf-8",
    )


def write_report(solution_dir: Path, package_dir: Path, meta: dict, metrics: dict) -> None:
    t1 = task(metrics, "task1_next_step")
    t2 = task(metrics, "task2_completion")
    t3 = task(metrics, "task3_anomaly")
    canon = canonical_metrics(metrics)
    self_eval = metrics.get("self_eval", {})
    approach = "\n".join(f"- {item}" for item in meta["approach"])
    score_lines = [
        f"- **Task 1 exact Top-1:** `{fmt(t1['top1'])}`",
        f"- **Task 1 exact Top-3:** `{fmt(t1['top3'])}`",
        f"- **Task 1 exact Top-5:** `{fmt(t1['top5'])}`",
        f"- **Task 1 MRR:** `{fmt(t1['mrr'])}`",
    ]
    if canon:
        score_lines.append(
            f"- **Canonical process-step Top-1:** `{fmt(canon['canonical_top1'])}` local diagnostic"
        )
    score_lines.extend(
        [
            f"- **Task 2 exact match:** `{fmt(t2['exact_match'])}`",
            f"- **Task 2 normalized edit distance:** `{fmt(t2['normalized_edit_distance'])}`",
            f"- **Task 2 token accuracy:** `{fmt(t2['token_accuracy'])}`",
            f"- **Task 2 block accuracy:** `{fmt(t2['block_accuracy'])}`",
            f"- **Task 3 accuracy:** `{fmt(t3['accuracy'])}`",
            f"- **Task 3 F1 valid:** `{fmt(t3['f1_valid'])}`",
            f"- **Task 3 ROC-AUC valid probability:** `{fmt(t3['roc_auc_valid_probability'])}`",
            f"- **Task 3 rule attribution accuracy:** `{fmt(t3['rule_attribution_accuracy'])}`",
        ]
    )
    scores = "\n".join(score_lines)
    package_dir.joinpath("REPORT.md").write_text(
        dedent(
            f"""\
            # {meta["title"]} - Industrial AI Report

            ## TL;DR

            {meta["role"]} It produces the three required Industrial AI eval CSV
            shapes and reports local scores for next-step prediction, sequence
            completion, and anomaly detection. {meta["honest_status"]}

            ## Problem

            The Industrial AI track asks us to model semiconductor process-flow
            sequences. Given a partial route, the system must rank the next process
            step, complete the remaining suffix, and flag full routes that violate
            process rules. The hidden organizer eval is not present in this checkout,
            so this package uses the same task shapes on a deterministic public-data
            self-eval split.

            ## Approach

            __APPROACH__

            ## How To Run It

            ```bash
            {meta["command"]}
            ```

            The command uses only files already in this repository and writes outputs
            to `{solution_dir.relative_to(ROOT)}/outputs/`.

            ## Results

            Local self-eval rows:

            - Valid Task 1/2 rows: `{self_eval.get("valid_task_rows", "unknown")}`
            - Anomaly Task 3 rows: `{self_eval.get("anomaly_task_rows", "unknown")}`
            - Official hidden eval available in this checkout: `{self_eval.get("official_eval_available", False)}`

            Headline scores:

            __SCORES__

            The raw files are in `extras/results/`. The official `eval_metrics.py`
            and hidden ground truth are not in this checkout, so these are local
            scorer outputs rather than official leaderboard numbers. A local
            leave-one-family-out proxy breakdown is included in
            `extras/results/per_family_breakdown.md`.

            ## What Worked

            - The solution emits all three required CSV shapes.
            - The local metrics are reproducible from a clean checkout.
            - The result is directly comparable against the other solution folders.

            ## What Did Not Work

            - This is not a neural Leonardo training run.
            - There is no official hidden eval score yet because the official eval
              files are not present.
            - Task 3 uses the public validator oracle in the current solution family.

            ## What We Would Do With Another 36 Hours

            - Train a sequence model on canonical operation IDs, then learn exact alias
              choice as a second-stage problem.
            - Add a learned anomaly detector trained from validator-generated labels.
            - Run the final model on Leonardo and store real checkpoints, loss curves,
              and cluster logs.

            ## Track-Specific Deliverables

            - [x] Eval submission files in `extras/results/`
            - [x] Scores for all three tasks in `extras/results/metrics.md`
            - [x] Training-artifact folder present with deterministic-fit manifest
            - [x] Demo material present in `extras/demo/`
            - [ ] Official `eval_metrics.py` scores: blocked until organizers provide
              the official eval script and hidden ground truth
            - [ ] Real neural checkpoint/loss curve: not applicable to this deterministic
              solution; see `extras/training_artifacts/checkpoint_manifest.md`

            ## Credits & Dependencies

            - Python standard library only for this solution package.
            - Data: public synthetic Infineon track CSVs in `training_data/`.
            - External APIs: none.
            - AI coding assistants: Codex and earlier generated repo notes were used.
            """
        ).replace("__APPROACH__", approach).replace("__SCORES__", scores),
        encoding="utf-8",
    )


def write_training_artifacts(package_dir: Path, meta: dict, metrics: dict) -> None:
    artifact_dir = package_dir / "extras" / "training_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    self_eval = metrics.get("self_eval", {})
    t1 = task(metrics, "task1_next_step")
    t2 = task(metrics, "task2_completion")
    t3 = task(metrics, "task3_anomaly")
    artifact_dir.joinpath("training_log.md").write_text(
        dedent(
            f"""\
            # Training / Fitting Log

            This solution is deterministic. "Training" means fitting counts,
            lookup tables, retrieval indexes, generated-data indexes, or fixed
            statistics from the public CSV files. It does not run gradient descent.

            ## Command

            ```bash
            {meta["command"]}
            ```

            ## Data Summary

            - Families: `{", ".join(self_eval.get("families", []))}`
            - Valid self-eval rows: `{self_eval.get("valid_task_rows", "unknown")}`
            - Anomaly self-eval rows: `{self_eval.get("anomaly_task_rows", "unknown")}`
            - Training sequences: `{self_eval.get("train_sequences", self_eval.get("train_sequences_after_augmentation", "unknown"))}`
            - Local split seed: `{metrics.get("seed", "unknown")}`

            ## Result Snapshot

            - Task 1 Top-1: `{fmt(t1["top1"])}`
            - Task 1 MRR: `{fmt(t1["mrr"])}`
            - Task 2 normalized edit distance: `{fmt(t2["normalized_edit_distance"])}`
            - Task 2 block accuracy: `{fmt(t2["block_accuracy"])}`
            - Task 3 accuracy: `{fmt(t3["accuracy"])}`

            ## Checkpoint Status

            {meta["checkpoint"]} For a real neural submission, this folder should be
            extended with `.pt`/`.safetensors` checkpoints and cluster logs.
            """
        ),
        encoding="utf-8",
    )
    artifact_dir.joinpath("checkpoint_manifest.md").write_text(
        dedent(
            f"""\
            # Checkpoint Manifest

            Binary checkpoint: **not produced**.

            Reason: {meta["checkpoint"]}

            Rebuild command:

            ```bash
            {meta["command"]}
            ```

            Source-controlled state needed to rebuild:

            - `training_data/`
            - solution source file
            - shared helper solutions imported by this solution
            - `training_data/generate_sequences.py` for validator/generator behavior
            """
        ),
        encoding="utf-8",
    )
    artifact_dir.joinpath("loss_curve.csv").write_text(
        "\n".join(
            [
                "step,phase,loss,task1_top1,task1_mrr,task2_normalized_edit_distance,task2_block_accuracy,task3_accuracy,notes",
                (
                    "0,deterministic_fit,not_applicable,"
                    f"{fmt(t1['top1'])},{fmt(t1['mrr'])},"
                    f"{fmt(t2['normalized_edit_distance'])},{fmt(t2['block_accuracy'])},"
                    f"{fmt(t3['accuracy'])},"
                    '"No gradient training loss; row records final local metrics."'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_demo_material(solution_dir: Path, package_dir: Path, meta: dict) -> None:
    demo_dir = package_dir / "extras" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    solution_next = read_first_nextstep(solution_dir / "outputs" / "nextstep.csv")
    solution_anomaly = read_first_anomaly(solution_dir / "outputs" / "anomaly.csv")
    baseline_next_path = SOLUTIONS_DIR / "solution_0_rule_mock" / "outputs" / "nextstep.csv"
    baseline_next = read_first_nextstep(baseline_next_path)
    demo_dir.joinpath("baseline_vs_model_examples.md").write_text(
        dedent(
            f"""\
            # Baseline vs Model Demo Examples

            These examples are ready to use in the required two-minute demo video.
            Use the same `EXAMPLE_ID` for baseline and candidate output so the jury
            sees a fair side-by-side comparison.

            ## Task 1 Example

            Example ID: `{solution_next["EXAMPLE_ID"]}`

            | System | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
            | --- | --- | --- | --- | --- | --- |
            | Solution 0 baseline | `{baseline_next["RANK_1"]}` | `{baseline_next["RANK_2"]}` | `{baseline_next["RANK_3"]}` | `{baseline_next["RANK_4"]}` | `{baseline_next["RANK_5"]}` |
            | This solution | `{solution_next["RANK_1"]}` | `{solution_next["RANK_2"]}` | `{solution_next["RANK_3"]}` | `{solution_next["RANK_4"]}` | `{solution_next["RANK_5"]}` |

            ## Task 3 Example

            Example ID: `{solution_anomaly["EXAMPLE_ID"]}`

            - Predicted valid flag: `{solution_anomaly["IS_VALID"]}`
            - Score: `{solution_anomaly["SCORE"]}`
            - Predicted rule: `{solution_anomaly.get("PREDICTED_RULE", "") or "none"}`
            """
        ),
        encoding="utf-8",
    )
    demo_dir.joinpath("video_script.md").write_text(
        dedent(
            f"""\
            # Two-Minute Demo Script

            ## 0:00-0:15 Problem

            "The industrial track asks us to predict semiconductor process flow steps,
            complete partially observed routes, and detect rule violations."

            ## 0:15-0:50 Solution Running

            Run:

            ```bash
            {meta["command"]}
            ```

            Open `extras/results/nextstep.csv`, `completion.csv`, and `anomaly.csv`.

            ## 0:50-1:25 Concrete Result

            Show `extras/results/metrics.md` and quote the Task 1, Task 2, and Task 3
            headline metrics.

            ## 1:25-1:50 Reasoning Visible

            Open `baseline_vs_model_examples.md` and compare the baseline row against
            this solution's ranked next-step output.

            ## 1:50-2:00 Honesty

            State clearly that these are local self-eval scores because the official
            hidden eval files are not present in this checkout.
            """
        ),
        encoding="utf-8",
    )
    demo_dir.joinpath("pitch_slides_outline.md").write_text(
        dedent(
            f"""\
            # Pitch Slides Outline

            1. Team and one-sentence system summary.
            2. Industrial sequence modeling problem and why process logic matters.
            3. Approach: {meta["role"]}
            4. Required outputs: next-step, completion, anomaly.
            5. Local metrics and baseline comparison.
            6. Honest limitations and next 36-hour plan.
            """
        ),
        encoding="utf-8",
    )


def write_audit(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Submission Readiness Audit",
        "",
        "This audit maps the official `submission/SUBMISSION.md` checklist to each",
        "candidate solution package. The official hidden eval files and",
        "`eval_metrics.py` are not present in this checkout, so the score artifacts",
        "are local self-eval outputs unless explicitly noted otherwise.",
        "",
        "## Repo-Level Checklist",
        "",
        "- [x] Root `README.md` has setup/run instructions for the industrial work.",
        "- [x] Root `REPORT.md` summarizes the current solution suite.",
        "- [x] Root `requirements.txt` is present.",
        "- [x] Root `LICENSE` is present.",
        "- [x] Track-specific CSV outputs exist for every solution package.",
        "- [x] Each solution package includes a local per-family proxy breakdown.",
        "- [ ] Public repo visibility must be checked in GitHub before final Tally submission.",
        "- [ ] Slides PDF and demo video are external Tally uploads; this repo includes outlines/scripts, not the final uploaded media.",
        "- [ ] Official `eval_metrics.py` scores are blocked until organizers provide the script and hidden ground truth.",
        "",
        "## Per-Solution Checklist",
        "",
        "| Solution | Eval CSVs in `extras/results` | Scores | Training artifacts | Demo assets | Honest caveat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['solution']}` | yes | local metrics + LOFO family proxy | deterministic manifest/log/loss-curve note | video script + examples | {row['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## What Counts As Complete Here",
            "",
            "For the current deterministic solutions, a package is complete when it has:",
            "",
            "1. `submission_package/extras/results/nextstep.csv`",
            "2. `submission_package/extras/results/completion.csv`",
            "3. `submission_package/extras/results/anomaly.csv`",
            "4. `submission_package/extras/results/metrics.md` and `metrics.json`",
            "5. `submission_package/extras/results/per_family_breakdown.md`",
            "6. `submission_package/extras/training_artifacts/training_log.md`",
            "7. `submission_package/extras/training_artifacts/checkpoint_manifest.md`",
            "8. `submission_package/extras/training_artifacts/loss_curve.csv`",
            "9. `submission_package/extras/demo/baseline_vs_model_examples.md`",
            "10. `submission_package/extras/demo/video_script.md`",
            "11. `submission_package/REPORT.md`",
            "",
            "The checkpoint and loss files are explicit honesty artifacts for deterministic",
            "solutions. They do not pretend that neural training occurred. A future final",
            "trained solution should replace them with real cluster logs, checkpoints, and",
            "training curves.",
            "",
        ]
    )
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    audit_rows: list[dict[str, str]] = []
    for name, meta in SOLUTION_META.items():
        solution_dir = SOLUTIONS_DIR / name
        metrics_path = solution_dir / "outputs" / "metrics.json"
        package_dir = solution_dir / "submission_package"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = read_json(metrics_path)
        copy_results(solution_dir, package_dir)
        write_per_family_breakdown(package_dir, metrics)
        write_readme(solution_dir, package_dir, meta, metrics)
        write_report(solution_dir, package_dir, meta, metrics)
        write_training_artifacts(package_dir, meta, metrics)
        write_demo_material(solution_dir, package_dir, meta)
        audit_rows.append({"solution": name, "caveat": meta["honest_status"]})
    write_audit(audit_rows)


if __name__ == "__main__":
    main()

"""
Streamlit dashboard for comparing and probing process-sequence models.

Run from the repo root:
    streamlit run solution/app.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "solution" / "results"
SWEEP_DIR = RESULTS_DIR / "sweeps"
CKPT_DIR = REPO_ROOT / "solution" / "checkpoints"
TRAINING_DATA_DIR = REPO_ROOT / "training_data"

st.set_page_config(page_title="Zero One Hack - Industrial AI", layout="wide")
st.title("Industrial AI - Process Sequence Models")


@st.cache_data
def load_family_data():
    from solution.data.loader import load_all_families

    return load_all_families(str(TRAINING_DATA_DIR))


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def checkpoint_label(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).name


def checkpoint_labels(result: dict) -> str:
    paths = []
    if result.get("checkpoint"):
        paths.append(result["checkpoint"])
    paths.extend(result.get("checkpoints") or [])
    paths.extend(result.get("next_checkpoints") or [])
    for key in ("completion_checkpoint", "anomaly_checkpoint"):
        if result.get(key):
            paths.append(result[key])
    labels = []
    for path in paths:
        label = checkpoint_label(path)
        if label and label not in labels:
            labels.append(label)
    return ",".join(labels)


def topk_with_scores(adapter, steps, family, k=5):
    if hasattr(adapter, "m") and hasattr(adapter.m, "top_k_next"):
        try:
            rows = adapter.m.top_k_next(steps, k=k, family=family)
        except TypeError:
            rows = adapter.m.top_k_next(steps, k=k)
        if rows and isinstance(rows[0], tuple):
            return rows
    preds = adapter.top_k_next(steps, k=k, family=family)
    return [(step, None) for step in preds]


def rank_of(pred_rows, gold):
    for i, row in enumerate(pred_rows, 1):
        step = row[0] if isinstance(row, tuple) else row
        if step == gold:
            return i
    return None


tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Live demo", "Training curves", "XGBoost sweeps"])

with tab1:
    rows = []
    seen = set()
    for path_str in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        path = Path(path_str)
        result = load_json(path)
        t1, t2, t3 = result.get("task1", {}), result.get("task2", {}), result.get("task3", {})
        name = result.get("model") or path.stem
        label = name if name not in seen else f"{name} [{path.stem}]"
        seen.add(label)
        rows.append({
            "run": label,
            "file": path.name,
            "type": result.get("model_type") or result.get("model"),
            "checkpoint": checkpoint_labels(result),
            "strategy": result.get("eval_strategy"),
            "split": result.get("eval_split"),
            "train_families": ",".join(result.get("train_families", [])),
            "families": ",".join(result.get("families", [])),
            "top1": t1.get("top1"),
            "top5": t1.get("top5"),
            "mrr": t1.get("mrr"),
            "exact": t2.get("exact_match"),
            "token_acc": t2.get("token_acc"),
            "block_acc": t2.get("block_acc"),
            "edit_dist": t2.get("norm_edit_dist"),
            "f1": t3.get("f1"),
            "auc": t3.get("roc_auc"),
            "rule_attr": t3.get("rule_attribution_acc"),
            "elapsed_s": result.get("elapsed_s"),
            "timestamp": result.get("timestamp"),
        })

    if rows:
        try:
            import pandas as pd

            df = pd.DataFrame(rows).set_index("run")
            higher_better = [
                col for col in ("top1", "top5", "mrr", "exact", "token_acc", "block_acc", "f1", "auc", "rule_attr")
                if col in df.columns
            ]
            sty = df.style.highlight_max(subset=higher_better, color="#1b5e20")
            if "edit_dist" in df.columns:
                sty = sty.highlight_min(subset=["edit_dist"], color="#1b5e20")
            st.dataframe(sty, width="stretch")
        except Exception:
            st.table(rows)
    else:
        st.info(f"No result JSONs under {RESULTS_DIR}. Run `python -m solution.run_eval ... --out solution/results/name.json`.")

with tab2:
    st.caption("Use a labeled example for immediate right/wrong feedback, or switch it off and paste your own sequence.")
    families = load_family_data()

    left, right = st.columns([1, 1])
    with left:
        kind = st.selectbox("Model", ["ngram", "xgboost", "xgboost_ensemble", "xgboost_hybrid", "catboost", "transformer", "llm"])
        family = st.selectbox("Family", ["MOSFET", "IGBT", "IC"])
        use_example = st.checkbox("Use labeled example", value=True)
    with right:
        ckpt = None
        base = "Qwen/Qwen2-0.5B"
        if kind == "transformer":
            ckpt = st.text_input("Checkpoint .pt", str(CKPT_DIR / "transformer_small" / "best.pt"))
        elif kind == "xgboost":
            ckpt = st.text_input("XGBoost checkpoint dir", str(CKPT_DIR / "xgboost"))
        elif kind == "xgboost_ensemble":
            ckpt = st.text_area(
                "XGBoost checkpoint dirs",
                "\n".join([
                    str(CKPT_DIR / "xgboost"),
                    str(CKPT_DIR / "xgb_v2_reg_100k_300"),
                ]),
                height=80,
            )
            ensemble_weights = st.text_input("Ensemble weights", "1 1")
        elif kind == "xgboost_hybrid":
            next_ckpt = st.text_area(
                "Task 1 checkpoint dirs",
                str(CKPT_DIR / "xgboost"),
                height=80,
            )
            completion_ckpt = st.text_input("Task 2 checkpoint dir", str(CKPT_DIR / "xgboost"))
            anomaly_ckpt = st.text_input("Task 3 checkpoint dir", str(CKPT_DIR / "xgb_v2_reg_100k_300"))
            ensemble_weights = st.text_input("Task 1 weights", "1")
            ckpt = "\n".join([next_ckpt, completion_ckpt, anomaly_ckpt])
        elif kind == "catboost":
            ckpt = st.text_input("CatBoost checkpoint dir", str(CKPT_DIR / "catboost"))
        elif kind == "llm":
            base = st.text_input("Base model", "Qwen/Qwen2-0.5B")
            ckpt = st.text_input("LoRA adapter dir", str(CKPT_DIR / "llm_qwen"))

    gold_next = None
    gold_rest = None
    selected_full = None
    if use_example:
        seqs = families.get(family, [])
        max_idx = max(len(seqs) - 1, 0)
        ex_cols = st.columns([1, 1])
        example_idx = ex_cols[0].number_input("Example index", min_value=0, max_value=max_idx, value=0, step=1)
        cut_fraction = ex_cols[1].selectbox("Cut fraction", [0.6, 0.8], index=0)
        selected_full = seqs[int(example_idx)] if seqs else []
        cut = max(1, int(round(len(selected_full) * cut_fraction)))
        steps = selected_full[:cut]
        if cut < len(selected_full):
            gold_next = selected_full[cut]
            gold_rest = selected_full[cut:]
        text_value = " | ".join(steps)
    else:
        text_value = (
            "RECEIVE WAFER LOT | LOT IDENTIFICATION | PRE CLEAN WAFER | HF DIP | "
            "THERMAL OXIDATION | SPIN COAT PHOTORESIST | SOFT BAKE | ALIGN MASK LEVEL 1 | "
            "EXPOSE LITHO LEVEL 1 | DEVELOP PHOTORESIST | OXIDE ETCH"
        )
        text_value = st.session_state.get("free_sequence", text_value)

    text_key = f"example_sequence_{family}_{int(example_idx)}_{cut_fraction}" if use_example else "free_sequence"
    text = st.text_area(
        "Partial sequence",
        value=text_value,
        height=130,
        disabled=use_example,
        key=text_key,
    )
    steps = [s.strip() for s in text.split("|") if s.strip()]
    if kind in {"xgboost", "xgboost_ensemble", "xgboost_hybrid", "catboost"}:
        beam_cols = st.columns([1, 1])
        beam_width = beam_cols[0].number_input("Completion beam width", min_value=1, max_value=12, value=5, step=1)
        beam_branching = beam_cols[1].number_input("Beam branching", min_value=1, max_value=20, value=8, step=1)
    else:
        beam_width = beam_branching = None

    @st.cache_resource
    def get_adapter(kind, ckpt, base="Qwen/Qwen2-0.5B", weights_text=""):
        from solution.tasks.adapters import make_adapter

        if kind == "ngram":
            from solution.data.loader import train_val_split
            from solution.models.ngram import NGramModel

            train = []
            for seqs in load_family_data().values():
                part, _ = train_val_split(seqs, 0.1, 42)
                train.extend(part)
            return make_adapter("ngram", NGramModel(3).fit(train))
        if kind == "xgboost":
            from solution.models.boosting import XGBoostStepModel

            return make_adapter("xgboost", XGBoostStepModel.load(ckpt))
        if kind == "xgboost_ensemble":
            from solution.models.boosting import BoostingEnsembleModel

            paths = [line.strip() for line in str(ckpt).splitlines() if line.strip()]
            weights = [float(value) for value in str(weights_text).replace(",", " ").split()] if weights_text else None
            return make_adapter("xgboost", BoostingEnsembleModel.load_many(paths, weights=weights))
        if kind == "xgboost_hybrid":
            from solution.models.boosting import BoostingTaskHybridModel

            lines = [line.strip() for line in str(ckpt).splitlines() if line.strip()]
            if len(lines) < 3:
                raise ValueError("xgboost_hybrid needs at least two task-1 lines plus task-2 and task-3 checkpoints")
            next_paths = lines[:-2]
            weights = [float(value) for value in str(weights_text).replace(",", " ").split()] if weights_text else None
            return make_adapter(
                "xgboost",
                BoostingTaskHybridModel.load(
                    next_paths=next_paths,
                    completion_path=lines[-2],
                    anomaly_path=lines[-1],
                    next_weights=weights,
                ),
            )
        if kind == "catboost":
            from solution.models.boosting import CatBoostStepModel

            return make_adapter("catboost", CatBoostStepModel.load(ckpt))
        if kind == "transformer":
            import torch

            from solution.data.vocab import Vocab
            from solution.models.transformer import ProcessTransformer

            checkpoint = torch.load(ckpt, map_location="cpu")
            model = ProcessTransformer(**checkpoint["config"])
            model.load_state_dict(checkpoint["model_state"])
            vocab = Vocab.load(os.path.join(os.path.dirname(ckpt), "vocab.json"))
            return make_adapter("transformer", model, vocab, "cpu")
        from solution.models.llm import LLMModel

        return make_adapter("llm", LLMModel(base, checkpoint_path=ckpt))

    col1, col2, col3 = st.columns(3)
    if col1.button("Predict next step"):
        adapter = get_adapter(kind, ckpt, base, locals().get("ensemble_weights", ""))
        pred_rows = topk_with_scores(adapter, steps, family, 5)
        table_rows = []
        for i, (step, prob) in enumerate(pred_rows, 1):
            table_rows.append({
                "rank": i,
                "step": step,
                "prob": None if prob is None else round(float(prob), 4),
                "matches_gold": bool(gold_next and step == gold_next),
            })
        st.table(table_rows)
        if gold_next:
            rank = rank_of(pred_rows, gold_next)
            st.metric("Gold next step", gold_next)
            if rank:
                st.success(f"Correct in top-5 at rank {rank}. MRR contribution: {1 / rank:.3f}")
            else:
                st.error("Gold step is not in the top-5 predictions.")

    if col2.button("Complete sequence"):
        from solution.eval.metrics import completion_metrics

        adapter = get_adapter(kind, ckpt, base, locals().get("ensemble_weights", ""))
        if kind in {"xgboost", "xgboost_ensemble", "xgboost_hybrid", "catboost"} and hasattr(adapter, "m"):
            adapter.m.config.beam_width = int(beam_width)
            adapter.m.config.beam_branching = int(beam_branching)
        predicted = adapter.complete(steps, family=family)
        st.write("Predicted remainder:")
        st.code(" | ".join(predicted))
        if gold_rest:
            comparable = predicted[:len(gold_rest)]
            metrics = completion_metrics([comparable], [gold_rest])
            st.json({k: v for k, v in metrics.items() if k != "n"})
            st.caption(f"Gold remainder length: {len(gold_rest)} | predicted length: {len(predicted)}")

    if col3.button("Check anomalies"):
        from solution.eval.rules import first_violation

        adapter = get_adapter(kind, ckpt, base, locals().get("ensemble_weights", ""))
        anomaly_steps = selected_full if use_example and selected_full else steps
        score = -adapter.seq_log_prob(anomaly_steps, family=family)
        rule, idx = first_violation(anomaly_steps, family=family)
        st.metric("Anomaly score", f"{score:.3f}")
        if rule:
            st.error(f"Rule violation: {rule} at step {idx}: '{anomaly_steps[idx]}'")
        else:
            st.success("No rule violation detected by the symbolic checker.")
        if use_example:
            st.caption("The selected source sequence is expected to be valid; this button checks the full source sequence.")

with tab3:
    log_paths = sorted(glob.glob(str(CKPT_DIR / "*" / "training_log.jsonl")))
    if not log_paths:
        st.info(f"No training logs yet under {CKPT_DIR}/*/training_log.jsonl.")
    for log_path_str in log_paths:
        log_path = Path(log_path_str)
        run_dir = log_path.parent
        meta_path = run_dir / "training_meta.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        title = meta.get("run_name") or run_dir.name
        st.subheader(title)
        if meta:
            st.caption(
                f"{meta.get('model', '')} | examples={meta.get('max_train_examples')} | "
                f"iterations={meta.get('iterations')} | elapsed={meta.get('elapsed_s')}s"
            )
        records = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
        if not records:
            st.info("Training log exists but contains no curve data.")
            continue
        try:
            import pandas as pd

            df = pd.DataFrame(records)
            if "iteration" in df.columns:
                df = df.set_index("iteration")
            numeric_cols = [
                col for col in df.columns
                if col not in {"best_iteration"} and getattr(df[col], "dtype", None) is not None and df[col].dtype.kind in "if"
            ]
            if numeric_cols:
                st.line_chart(df[numeric_cols])
            st.dataframe(df.tail(10), width="stretch")
        except Exception:
            st.json(records[-10:])

with tab4:
    sweep_paths = sorted(glob.glob(str(SWEEP_DIR / "*.csv")))
    if not sweep_paths:
        st.info(f"No sweep summaries under {SWEEP_DIR}. Run `python -m solution.train.sweep_xgboost ...`.")
    else:
        selected = st.selectbox("Sweep summary", sweep_paths, format_func=lambda p: Path(p).name)
        try:
            import pandas as pd

            df = pd.read_csv(selected)
            if "run_name" in df.columns:
                df = df.set_index("run_name")
            metric_cols = [
                col for col in ("top1", "top5", "mrr", "completion_exact", "completion_token_acc", "anomaly_f1", "anomaly_auc")
                if col in df.columns
            ]
            sty = df.style.highlight_max(subset=metric_cols, color="#1b5e20")
            if "completion_edit_dist" in df.columns:
                sty = sty.highlight_min(subset=["completion_edit_dist"], color="#1b5e20")
            st.dataframe(sty, width="stretch")
        except Exception:
            rows = [json.loads(line) for line in open(str(selected).replace(".csv", ".jsonl"), encoding="utf-8") if line.strip()]
            st.table(rows)

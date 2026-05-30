"""
app.py — Streamlit dashboard
============================
Three tabs:
  1. Model comparison — table of all results/*.json, best per metric highlighted
  2. Live demo        — next-step / completion / anomaly on a typed sequence
  3. Training curves  — plot solution/checkpoints/*/training_log.jsonl

Run:
    pixi run streamlit run solution/app.py
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

RESULTS_DIR = str(REPO_ROOT / "solution" / "results")
CKPT_DIR = str(REPO_ROOT / "solution" / "checkpoints")
TRAINING_DATA_DIR = str(REPO_ROOT / "training_data")

st.set_page_config(page_title="Zero One Hack — Industrial AI", layout="wide")
st.title("Industrial AI — Process Sequence Models")

tab1, tab2, tab3 = st.tabs(["Model comparison", "Live demo", "Training curves"])

# ── Tab 1: comparison ────────────────────────────────────────────────────────
with tab1:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        t1, t2, t3 = r.get("task1", {}), r.get("task2", {}), r.get("task3", {})
        rows.append({
            "model": r.get("model", os.path.basename(path)),
            "families": ",".join(r.get("families", [])),
            "t1_top1": t1.get("top1"), "t1_top5": t1.get("top5"), "t1_mrr": t1.get("mrr"),
            "t2_exact": t2.get("exact_match"), "t2_tok_acc": t2.get("token_acc"),
            "t2_block_acc": t2.get("block_acc"), "t2_edit": t2.get("norm_edit_dist"),
            "t3_f1": t3.get("f1"), "t3_auc": t3.get("roc_auc"),
            "t3_rule_attr": t3.get("rule_attribution_acc"),
        })
    if rows:
        try:
            import pandas as pd
            df = pd.DataFrame(rows).set_index("model")
            higher_better = [c for c in df.columns if c not in ("families", "t2_edit")]
            sty = df.style.highlight_max(subset=higher_better, color="#1b5e20")
            if "t2_edit" in df.columns:
                sty = sty.highlight_min(subset=["t2_edit"], color="#1b5e20")
            st.dataframe(sty, width="stretch")
        except Exception:
            st.table(rows)
    else:
        st.info(f"No results yet. Run a model, e.g. "
                f"`pixi run python -m solution.run_baseline`, to populate {RESULTS_DIR}/.")

# ── Tab 2: live demo ─────────────────────────────────────────────────────────
with tab2:
    st.caption("Loads a model on demand. n-gram is fastest; trained models need checkpoints.")
    kind = st.selectbox("Model", ["ngram", "xgboost", "catboost", "transformer", "llm"])
    family = st.selectbox("Family", ["MOSFET", "IGBT", "IC"])
    default_seq = ("RECEIVE WAFER LOT | LOT IDENTIFICATION | PRE CLEAN WAFER | HF DIP | "
                   "THERMAL OXIDATION | SPIN COAT PHOTORESIST | SOFT BAKE | ALIGN MASK LEVEL 1 | "
                   "EXPOSE LITHO LEVEL 1 | DEVELOP PHOTORESIST | OXIDE ETCH")
    text = st.text_area("Partial sequence (steps separated by ' | ')", default_seq, height=120)
    steps = [s.strip() for s in text.split("|") if s.strip()]

    ckpt = None
    base = "Qwen/Qwen2-0.5B"
    if kind == "transformer":
        ckpt = st.text_input("Checkpoint .pt", f"{CKPT_DIR}/transformer_small/best.pt")
    elif kind == "xgboost":
        ckpt = st.text_input("XGBoost checkpoint dir", f"{CKPT_DIR}/xgboost")
    elif kind == "catboost":
        ckpt = st.text_input("CatBoost checkpoint dir", f"{CKPT_DIR}/catboost")
    elif kind == "llm":
        base = st.text_input("Base model", "Qwen/Qwen2-0.5B")
        ckpt = st.text_input("LoRA adapter dir", f"{CKPT_DIR}/llm_qwen")

    @st.cache_resource
    def get_adapter(kind, ckpt, base="Qwen/Qwen2-0.5B"):
        from solution.tasks.adapters import make_adapter
        if kind == "ngram":
            from solution.data.loader import load_all_families, train_val_split
            from solution.models.ngram import NGramModel
            fams = load_all_families(TRAINING_DATA_DIR)
            tr = []
            for fam, seqs in fams.items():
                a, _ = train_val_split(seqs, 0.1, 42); tr += a
            return make_adapter("ngram", NGramModel(3).fit(tr))
        if kind == "xgboost":
            from solution.models.boosting import XGBoostStepModel
            return make_adapter("xgboost", XGBoostStepModel.load(ckpt))
        if kind == "catboost":
            from solution.models.boosting import CatBoostStepModel
            return make_adapter("catboost", CatBoostStepModel.load(ckpt))
        if kind == "transformer":
            import torch
            from solution.models.transformer import ProcessTransformer
            from solution.data.vocab import Vocab
            c = torch.load(ckpt, map_location="cpu")
            m = ProcessTransformer(**c["config"]); m.load_state_dict(c["model_state"])
            v = Vocab.load(os.path.join(os.path.dirname(ckpt), "vocab.json"))
            return make_adapter("transformer", m, v, "cpu")
        from solution.models.llm import LLMModel
        return make_adapter("llm", LLMModel(base, checkpoint_path=ckpt))

    col1, col2, col3 = st.columns(3)
    if col1.button("Predict next step"):
        ad = get_adapter(kind, ckpt, base)
        preds = ad.top_k_next(steps, 5, family=family)
        st.write("Top-5 next steps:")
        st.bar_chart({p: 5 - i for i, p in enumerate(preds)})
    if col2.button("Complete sequence"):
        ad = get_adapter(kind, ckpt, base)
        st.write("Predicted remainder:")
        st.code(" | ".join(ad.complete(steps, family=family)))
    if col3.button("Check anomalies"):
        from solution.eval.rules import first_violation
        ad = get_adapter(kind, ckpt, base)
        score = -ad.seq_log_prob(steps, family=family)
        rule, idx = first_violation(steps, family=family)
        st.metric("Anomaly score (higher = more anomalous)", f"{score:.3f}")
        if rule:
            st.error(f"Rule violation: {rule} at step {idx}: '{steps[idx]}'")
        else:
            st.success("No rule violation detected by the symbolic checker.")

# ── Tab 3: training curves ───────────────────────────────────────────────────
with tab3:
    logs = glob.glob(os.path.join(CKPT_DIR, "*", "training_log.jsonl"))
    if not logs:
        st.info(f"No training logs yet under {CKPT_DIR}/*/training_log.jsonl.")
    for log in sorted(logs):
        st.subheader(os.path.basename(os.path.dirname(log)))
        recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        if recs:
            keys = [k for k in ("train_loss", "val_loss", "ood_loss") if k in recs[0]]
            st.line_chart({k: [r[k] for r in recs] for k in keys})

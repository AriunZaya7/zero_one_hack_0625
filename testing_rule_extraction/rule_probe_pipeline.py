"""
rule_probe_pipeline.py
=======================
Branch: angie
Folder: testing_rule_extraction/

Generic behavioral probing pipeline.
Any model that implements get_perplexity(sequence) -> float
can be plugged in and will get a full learned_rules HTML report.

Built-in model adapters:
  - NGramAdapter     (no dependencies)
  - XGBoostAdapter   (requires trained xgb model + class_map)

Usage:
    python rule_probe_pipeline.py --model ngram --train_dir ../training_data
    python rule_probe_pipeline.py --model xgboost --model_path ./xgb_model.json

To add your own model: subclass BaseModelAdapter and implement get_perplexity().
"""

import csv
import json
import math
import os
import argparse
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from openai import OpenAI


# ══════════════════════════════════════════════════════════════════════════════
# BASE ADAPTER — subclass this to plug in any model
# ══════════════════════════════════════════════════════════════════════════════

class BaseModelAdapter(ABC):
    """
    Any model that wants to use the probing pipeline must implement
    get_perplexity(sequence) -> float.

    Higher perplexity = model more surprised by the sequence.
    Lower perplexity  = model considers the sequence more natural/valid.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short display name e.g. 'N-gram (n=10)'"""
        pass

    @property
    @abstractmethod
    def model_info(self) -> dict:
        """Dict of metadata to include in report e.g. {'n': 10, 'vocab': 199}"""
        pass

    @abstractmethod
    def get_perplexity(self, sequence: list[str]) -> float:
        """
        Compute perplexity of a sequence.
        sequence: list of step name strings e.g. ['HF DIP', 'DRY WAFER', ...]
        returns: float, higher = more surprised
        """
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER 1 — N-gram
# ══════════════════════════════════════════════════════════════════════════════

class NGramAdapter(BaseModelAdapter):
    def __init__(self, n: int = 10, smoothing: float = 1e-6):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.vocab = set()
        self.unigram = Counter()
        self._trained = False

    @property
    def model_name(self) -> str:
        return f"N-gram (n={self.n})"

    @property
    def model_info(self) -> dict:
        return {
            "model_type":      "ngram",
            "n":               self.n,
            "vocab_size":      len(self.vocab),
            "unique_contexts": len(self.counts),
        }

    def train(self, sequences: list[list[str]]):
        for seq in sequences:
            padded = ["<START>"] * (self.n - 1) + seq + ["<END>"]
            for i in range(len(padded) - self.n + 1):
                context   = tuple(padded[i: i + self.n - 1])
                next_step = padded[i + self.n - 1]
                self.counts[context][next_step] += 1
                self.vocab.add(next_step)
            for step in seq:
                self.unigram[step] += 1
        self.vocab.discard("<START>")
        self._trained = True
        print(f"N-gram trained | n={self.n} | vocab={len(self.vocab)} | "
              f"contexts={len(self.counts)}")

    def get_perplexity(self, sequence: list[str]) -> float:
        if not self._trained:
            raise RuntimeError("Call train() before get_perplexity()")
        padded     = ["<START>"] * (self.n - 1) + sequence + ["<END>"]
        log_prob   = 0.0
        vocab_size = len(self.vocab) + 1
        for i in range(len(padded) - self.n + 1):
            context   = tuple(padded[i: i + self.n - 1])
            next_step = padded[i + self.n - 1]
            counter   = Counter()
            for length in range(self.n - 1, 0, -1):
                ctx = tuple(padded[i + (self.n - 1 - length): i + self.n - 1])
                counter = self.counts.get(ctx, Counter())
                if counter:
                    break
            if not counter:
                counter = self.unigram
            total    = sum(counter.values())
            count    = counter.get(next_step, 0)
            prob     = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            log_prob += math.log(prob)
        avg_log_prob = log_prob / max(len(sequence), 1)
        return math.exp(-avg_log_prob)


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER 2 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════

class ExternalModelAdapter(BaseModelAdapter):
    """
    Black-box adapter for any externally trained model.

    Your teammate implements ONE function:
        def get_perplexity(sequence: list[str]) -> float

    and passes it here. No imports, no shared feature code needed.

    Example — teammate creates my_model_adapter.py:
    ─────────────────────────────────────────────
    import xgboost as xgb
    import numpy as np

    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")

    def get_perplexity(sequence):
        # their own feature engineering here
        X = their_feature_function(sequence)
        probas = model.predict_proba(X)
        avg_conf = np.mean(np.max(probas, axis=1))
        return 1.0 / max(avg_conf, 1e-6)
    ─────────────────────────────────────────────

    Then in rule_probe_pipeline.py:
        from my_model_adapter import get_perplexity
        adapter = ExternalModelAdapter(
            perplexity_fn=get_perplexity,
            name="XGBoost (teammate)",
            info={"model_type": "xgboost", "n_features": 168}
        )
    """

    def __init__(self, perplexity_fn, name: str, info: dict):
        """
        perplexity_fn : callable(sequence: list[str]) -> float
        name          : display name e.g. "XGBoost (teammate)"
        info          : dict of metadata for the report
        """
        self._perplexity_fn = perplexity_fn
        self._name          = name
        self._info          = info

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_info(self) -> dict:
        return self._info

    def get_perplexity(self, sequence: list[str]) -> float:
        return float(self._perplexity_fn(sequence))


# ══════════════════════════════════════════════════════════════════════════════
# RULE PROBES (same for all models)
# ══════════════════════════════════════════════════════════════════════════════

RULE_PROBES = [
    {
        "rule": "RULE_DEP_NO_CLEAN",
        "description": "Deposition must always be preceded by a cleaning step",
        "valid_seq": [
            "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN INSPECTION",
            "MEASURE INITIAL THICKNESS", "HF DIP", "DRY WAFER",
            "THERMAL OXIDATION", "MEASURE OXIDE THICKNESS",
        ],
        "invalid_seq": [
            "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN INSPECTION",
            "MEASURE INITIAL THICKNESS", "MEASURE SURFACE PLANARITY",
            "THERMAL OXIDATION", "MEASURE OXIDE THICKNESS",
        ],
        "violation": "THERMAL OXIDATION (deposition) without prior clean step",
    },
    {
        "rule": "RULE_METAL_ETCH_NO_LITHO",
        "description": "Lithography must precede metal etching",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT METAL 1", "COAT PHOTORESIST",
            "SOFT BAKE", "EXPOSE LITHO LEVEL 3", "DEVELOP PHOTORESIST",
            "METAL ETCH", "STRIP RESIST",
        ],
        "invalid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT METAL 1", "COAT PHOTORESIST",
            "SOFT BAKE", "METAL ETCH", "STRIP RESIST",
        ],
        "violation": "METAL ETCH without EXPOSE + DEVELOP preceding it",
    },
    {
        "rule": "RULE_ETCH_NO_MASK",
        "description": "A developed mask must exist before any etch",
        "valid_seq": [
            "COAT PHOTORESIST", "SOFT BAKE", "EXPOSE LITHO LEVEL 1",
            "DEVELOP PHOTORESIST", "OXIDE ETCH", "STRIP RESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "SOFT BAKE", "EXPOSE LITHO LEVEL 1",
            "OXIDE ETCH", "STRIP RESIST",
        ],
        "violation": "OXIDE ETCH without DEVELOP PHOTORESIST first",
    },
    {
        "rule": "RULE_LITHO_LEVEL_SKIP",
        "description": "Lithography levels must increment sequentially",
        "valid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "STRIP RESIST", "CLEAN AFTER ETCH",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 2", "DEVELOP PHOTORESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "STRIP RESIST", "CLEAN AFTER ETCH",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 3", "DEVELOP PHOTORESIST",
        ],
        "violation": "Jumped from LITHO LEVEL 1 directly to LITHO LEVEL 3",
    },
    {
        "rule": "RULE_IMPLANT_NO_MASK",
        "description": "An implant window must be opened before ion implantation",
        "valid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "IMPLANT P BODY", "STRIP RESIST", "ANNEAL",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "IMPLANT P BODY", "STRIP RESIST", "ANNEAL",
        ],
        "violation": "IMPLANT P BODY without prior OXIDE ETCH to open window",
    },
    {
        "rule": "RULE_CMP_NO_DEP",
        "description": "CMP planarization requires a prior deposition step",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT INTERLAYER DIELECTRIC",
            "MEASURE ILD THICKNESS", "CMP DIELECTRIC", "MEASURE POST CMP",
        ],
        "invalid_seq": [
            "MEASURE SURFACE PLANARITY", "MEASURE INITIAL THICKNESS",
            "CMP DIELECTRIC", "MEASURE POST CMP",
        ],
        "violation": "CMP DIELECTRIC with no prior deposition step",
    },
    {
        "rule": "RULE_PAD_OPEN_BEFORE_DEP",
        "description": "Bond pad window only after passivation deposited and cured",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 4", "DEVELOP PHOTORESIST",
            "OPEN PAD WINDOW", "STRIP RESIST",
        ],
        "invalid_seq": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 4", "DEVELOP PHOTORESIST",
            "OPEN PAD WINDOW", "STRIP RESIST",
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
        ],
        "violation": "OPEN PAD WINDOW before DEPOSIT PASSIVATION and CURE PASSIVATION",
    },
    {
        "rule": "RULE_TEST_BEFORE_PASSIVATION",
        "description": "Electrical tests must come after passivation is cured",
        "valid_seq": [
            "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "OPEN PAD WINDOW", "PARAMETRIC TEST", "WAFER SORT TEST",
        ],
        "invalid_seq": [
            "DEPOSIT PASSIVATION", "PARAMETRIC TEST",
            "CURE PASSIVATION", "OPEN PAD WINDOW", "WAFER SORT TEST",
        ],
        "violation": "PARAMETRIC TEST before CURE PASSIVATION",
    },
    {
        "rule": "RULE_SHIP_BEFORE_TEST",
        "description": "Wafer sort test must complete before shipping",
        "valid_seq": [
            "CURE PASSIVATION", "OPEN PAD WINDOW",
            "WAFER SORT TEST", "FINAL INSPECTION", "SHIP LOT",
        ],
        "invalid_seq": [
            "CURE PASSIVATION", "OPEN PAD WINDOW",
            "FINAL INSPECTION", "SHIP LOT", "WAFER SORT TEST",
        ],
        "violation": "SHIP LOT appears before WAFER SORT TEST",
    },
    {
        "rule": "RULE_BACKSIDE_BEFORE_PASSIVATION",
        "description": "Backside metal only after frontside passivation is cured",
        "valid_seq": [
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
            "BACKSIDE GRIND", "BACKSIDE CLEAN", "DEPOSIT BACKSIDE METAL",
        ],
        "invalid_seq": [
            "BACKSIDE GRIND", "BACKSIDE CLEAN", "DEPOSIT BACKSIDE METAL",
            "HF DIP", "DRY WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
        ],
        "violation": "DEPOSIT BACKSIDE METAL before DEPOSIT PASSIVATION",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# PROBE RUNNER — works with any adapter
# ══════════════════════════════════════════════════════════════════════════════

def run_probes(adapter: BaseModelAdapter) -> dict:
    """Run all 10 rule probes on any model adapter. Returns structured results."""
    results = []

    print(f"\nRunning {len(RULE_PROBES)} rule probes on: {adapter.model_name}")
    print(f"{'─'*80}")
    print(f"{'Rule':<38} {'Valid PPL':>10} {'Invalid PPL':>12} "
          f"{'Ratio':>8} {'Learned?':>12}")
    print(f"{'─'*80}")

    for probe in RULE_PROBES:
        valid_ppl   = adapter.get_perplexity(probe["valid_seq"])
        invalid_ppl = adapter.get_perplexity(probe["invalid_seq"])
        ratio       = invalid_ppl / max(valid_ppl, 1e-6)

        if ratio >= 2.0:
            learned = "STRONG"
        elif ratio >= 1.3:
            learned = "PARTIAL"
        elif ratio >= 1.05:
            learned = "WEAK"
        else:
            learned = "NOT LEARNED"

        print(f"{probe['rule']:<38} {valid_ppl:>10.2f} {invalid_ppl:>12.2f} "
              f"{ratio:>8.2f}x  {learned:>12}")

        results.append({
            "rule":        probe["rule"],
            "description": probe["description"],
            "violation":   probe["violation"],
            "valid_seq":   probe["valid_seq"],
            "invalid_seq": probe["invalid_seq"],
            "valid_ppl":   round(valid_ppl,   4),
            "invalid_ppl": round(invalid_ppl, 4),
            "ppl_ratio":   round(ratio,       4),
            "learned":     learned,
        })

    strong  = sum(1 for r in results if r["learned"] == "STRONG")
    partial = sum(1 for r in results if r["learned"] == "PARTIAL")
    weak    = sum(1 for r in results if r["learned"] == "WEAK")
    none_   = sum(1 for r in results if r["learned"] == "NOT LEARNED")
    score   = round((strong + 0.5*partial + 0.2*weak) / len(results) * 100, 1)

    print(f"\nSummary: Strong={strong} Partial={partial} "
          f"Weak={weak} NotLearned={none_} Score={score}%")

    return {
        "model_name":  adapter.model_name,
        "model_info":  adapter.model_info,
        "probe_results": results,
        "summary": {
            "strong": strong, "partial": partial,
            "weak": weak, "not_learned": none_,
            "total": len(results), "score_pct": score,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLM RULE EXTRACTION — same for all models
# ══════════════════════════════════════════════════════════════════════════════

RULE_TEMPLATE = """## Learned Rule: [GIVE THE RULE A DESCRIPTIVE NAME]
**Observed behavior:** Model assigns X times higher perplexity when [describe the violation]
**Inferred constraint:** [step type A] must be preceded by [step type B] within [N] steps
**Confidence:** HIGH / MEDIUM / LOW
**Evidence:** valid_ppl=[X], invalid_ppl=[Y], ratio=[Z]x
**Rule type:** LOCAL (within a few steps) / GLOBAL (spans entire sequence)
"""

EXTRACTION_SYSTEM = """You are an expert in semiconductor manufacturing and ML interpretability.
Analyze perplexity evidence to infer what rules a model learned.
Do NOT invent rules not supported by evidence. Be honest."""


def get_featherless_client() -> OpenAI:
    api_key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "FEATHERLESS_API_KEY not set.\n"
            "PowerShell: $env:FEATHERLESS_API_KEY='your_key'\n"
            "CMD:        set FEATHERLESS_API_KEY=your_key"
        )
    return OpenAI(api_key=api_key, base_url="https://api.featherless.ai/v1")


def call_llm(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    client = get_featherless_client()
    model  = os.environ.get("FEATHERLESS_MODEL", "deepseek-ai/DeepSeek-V3-0324")
    print(f"  Calling: {model}")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=0.0,
    )
    return resp.choices[0].message.content


def build_extraction_prompt(probe_data: dict) -> str:
    info = probe_data["model_info"]
    results_text = ""
    for r in probe_data["probe_results"]:
        results_text += f"""
---
Rule: {r['rule']}
Description: {r['description']}
Violation: {r['violation']}
Valid sequence:   {' -> '.join(r['valid_seq'])}
Invalid sequence: {' -> '.join(r['invalid_seq'])}
Valid PPL: {r['valid_ppl']} | Invalid PPL: {r['invalid_ppl']} | Ratio: {r['ppl_ratio']}x
Assessment: {r['learned']}
"""
    return f"""N-gram behavioral probe results for semiconductor manufacturing sequences.

Model: {probe_data['model_name']}
Info: {json.dumps(info)}

PROBE RESULTS:
{results_text}

THRESHOLDS: >=2.0=STRONG, 1.3-2.0=PARTIAL, 1.05-1.3=WEAK, ~1.0=NOT LEARNED

For EACH rule use EXACTLY this template:
{RULE_TEMPLATE}

Then add:
## Overall Assessment
## Rules Learned (list)
## Rules NOT Learned (list with reason)
## Key Insight: Local vs Global Rules
## What This Tells Us (patterns vs genuine logic?)"""


def extract_learned_rules(probe_data: dict) -> str:
    print("\nExtracting learned rules via LLM...")
    print("(LLM never sees generation_rules.md — purely behavioral evidence)")
    learned_md = call_llm(build_extraction_prompt(probe_data), EXTRACTION_SYSTEM)
    header = f"""# Learned Rules — {probe_data['model_name']} Behavioral Probing

> **Model:** {probe_data['model_name']}
> **Method:** Perplexity probing on valid vs rule-violated sequences
> **Note:** LLM never saw generation_rules.md — inferred from behavior only

---

"""
    return header + learned_md


def compare_with_ground_truth(learned_md: str,
                               ground_truth_path: str,
                               probe_data: dict) -> str | None:
    if not os.path.exists(ground_truth_path):
        print(f"[skip] Ground truth not found: {ground_truth_path}")
        return None
    print("\nComparing with ground truth...")
    with open(ground_truth_path, encoding="utf-8") as f:
        gt_md = f.read()
    prompt = f"""Compare these two process rule documents.

LEARNED (from {probe_data['model_name']} probing, never saw ground truth):
{learned_md}

GROUND TRUTH:
{gt_md[:4000]}

Produce:
1. Summary table: | Rule | Ground Truth | Model Learned | Similarity (0-1) | Match? |
2. Per-rule gap analysis
3. Overall similarity score
4. Final verdict: patterns or genuine logic?"""
    comparison = call_llm(prompt, EXTRACTION_SYSTEM)
    return f"# Comparison: Ground Truth vs {probe_data['model_name']}\n\n---\n\n{comparison}"


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT BUILDER — works with any probe_data dict
# ══════════════════════════════════════════════════════════════════════════════

LEARNED_COLORS = {
    "STRONG":      "#2ecc71",
    "PARTIAL":     "#f39c12",
    "WEAK":        "#e67e22",
    "NOT LEARNED": "#e74c3c",
}
LEARNED_SCORES = {
    "STRONG": 1.0, "PARTIAL": 0.6, "WEAK": 0.2, "NOT LEARNED": 0.0
}


def build_html_report(probe_data: dict, learned_md: str,
                       comparison_md: str = "") -> str:
    results  = probe_data["probe_results"]
    summary  = probe_data["summary"]
    score    = summary["score_pct"]
    info     = probe_data["model_info"]

    bar_labels   = json.dumps([r["rule"].replace("RULE_","") for r in results])
    bar_ratios   = json.dumps([r["ppl_ratio"] for r in results])
    bar_colors   = json.dumps([LEARNED_COLORS[r["learned"]] for r in results])
    valid_ppls   = json.dumps([r["valid_ppl"] for r in results])
    invalid_ppls = json.dumps([r["invalid_ppl"] for r in results])
    learn_scores = json.dumps([LEARNED_SCORES[r["learned"]] for r in results])

    # Info rows for model card
    info_rows = "".join(
        f'<tr><td style="color:#666;padding:4px 0">{k}</td>'
        f'<td style="text-align:right;padding:4px 0;font-weight:500">{v}</td></tr>'
        for k, v in info.items()
    )

    # Rule cards
    rule_cards = ""
    for r in results:
        bg    = LEARNED_COLORS[r["learned"]]
        badge = (f'<span style="background:{bg};color:#fff;padding:3px 10px;'
                 f'border-radius:12px;font-size:12px;font-weight:600">'
                 f'{r["learned"]}</span>')
        rule_cards += f"""
        <div style="border:1px solid #e0e0e0;border-radius:10px;padding:1.2rem;
                    margin-bottom:1rem;border-left:4px solid {bg}">
          <div style="display:flex;justify-content:space-between;
                      align-items:flex-start;flex-wrap:wrap;gap:8px">
            <div>
              <p style="font-weight:600;margin:0 0 4px">{r['rule']}</p>
              <p style="color:#555;margin:0;font-size:13px">{r['description']}</p>
            </div>
            {badge}
          </div>
          <div style="margin-top:10px;background:#f8f8f8;border-radius:6px;
                      padding:10px;font-size:13px">
            <p style="margin:0 0 3px"><b>Violation:</b> {r['violation']}</p>
            <p style="margin:0 0 3px"><b>Valid PPL:</b> {r['valid_ppl']}
              &nbsp;|&nbsp; <b>Invalid PPL:</b> {r['invalid_ppl']}
              &nbsp;|&nbsp; <b>Ratio:</b> {r['ppl_ratio']}x</p>
            <p style="margin:0">
              {'✓ Violation detected' if r['ppl_ratio'] >= 1.3
               else '✗ Violation NOT detected'}
            </p>
          </div>
          <details style="margin-top:8px">
            <summary style="cursor:pointer;font-size:13px;color:#555">
              Show sequences</summary>
            <div style="margin-top:8px;font-size:12px;font-family:monospace">
              <p style="color:#2ecc71;margin:4px 0">
                <b>Valid:</b> {' → '.join(r['valid_seq'])}</p>
              <p style="color:#e74c3c;margin:4px 0">
                <b>Invalid:</b> {' → '.join(r['invalid_seq'])}</p>
            </div>
          </details>
        </div>"""

    # Simple markdown renderer
    def md_to_html(md: str) -> str:
        lines  = md.split("\n")
        html   = ""
        for line in lines:
            if line.startswith("## "):
                html += (f'<h2 style="margin-top:1.5rem;font-size:17px;'
                         f'border-bottom:1px solid #eee;padding-bottom:6px">'
                         f'{line[3:]}</h2>\n')
            elif line.startswith("### "):
                html += (f'<h3 style="margin-top:1rem;font-size:15px">'
                         f'{line[4:]}</h3>\n')
            elif line.startswith("| "):
                html += (f'<p style="font-family:monospace;font-size:12px;'
                         f'margin:2px 0">{line}</p>\n')
            elif line.startswith("- ") or line.startswith("* "):
                html += f'<li style="margin:3px 0;font-size:14px">{line[2:]}</li>\n'
            elif line.strip() == "":
                html += "<br>\n"
            else:
                bold = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
                html += (f'<p style="margin:4px 0;font-size:14px;'
                         f'line-height:1.6">{bold}</p>\n')
        return html

    learned_html    = md_to_html(learned_md)
    comparison_html = md_to_html(comparison_md) if comparison_md else (
        "<p style='color:#999'>Run with --ground_truth to enable comparison.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{probe_data['model_name']} — Rule Learning Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f5f5f5;color:#222;line-height:1.6}}
.wrap{{max-width:980px;margin:0 auto;padding:2rem 1.5rem}}
.card{{background:#fff;border-radius:12px;padding:1.5rem;
       box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1.5rem}}
h1{{font-size:26px;font-weight:700;margin-bottom:6px}}
h2{{font-size:19px;font-weight:600;margin:2rem 0 1rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}}
.stat{{background:#f0f0f0;border-radius:8px;padding:1rem;text-align:center}}
.stat .v{{font-size:22px;font-weight:700}}
.stat .l{{font-size:12px;color:#666;margin-top:4px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:12px}}
.dot{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:1rem}}
.tab{{padding:8px 18px;border-radius:20px;cursor:pointer;font-size:13px;
      border:1px solid #ddd;background:#fff}}
.tab.on{{background:#222;color:#fff;border-color:#222}}
.tc{{display:none}}.tc.on{{display:block}}
.ch{{position:relative;width:100%;height:340px}}
.ch2{{position:relative;width:100%;height:280px}}
</style>
</head>
<body>
<div class="wrap">

<div class="card">
  <p style="font-size:12px;color:#888;margin-bottom:6px">
    HACKATHON · INDUSTRIAL AI TRACK · BRANCH: ANGIE</p>
  <h1>{probe_data['model_name']} — Rule Learning Report</h1>
  <p style="color:#555;margin-top:6px">
    Behavioral probing: does this model reproduce known patterns
    or learn genuine process logic?</p>
</div>

<h2>Model Information</h2>
<div class="card">
  <table style="width:100%;font-size:14px">{info_rows}</table>
</div>

<h2>Overall Rule Learning Score</h2>
<div class="card">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div style="font-size:52px;font-weight:700">{score:.0f}%</div>
    <div style="flex:1;min-width:200px">
      <div style="background:#eee;border-radius:8px;height:20px;overflow:hidden">
        <div style="height:100%;border-radius:8px;width:{score}%;
             background:linear-gradient(90deg,#e74c3c,#f39c12,#2ecc71)"></div>
      </div>
      <p style="font-size:13px;color:#555;margin-top:6px">
        Strong:{summary['strong']} &nbsp;|&nbsp;
        Partial:{summary['partial']} &nbsp;|&nbsp;
        Weak:{summary['weak']} &nbsp;|&nbsp;
        Not learned:{summary['not_learned']}
      </p>
    </div>
  </div>
  <div class="legend" style="margin-top:12px">
    <span><span class="dot" style="background:#2ecc71"></span> Strong (≥2.0x)</span>
    <span><span class="dot" style="background:#f39c12"></span> Partial (1.3–2.0x)</span>
    <span><span class="dot" style="background:#e67e22"></span> Weak (1.05–1.3x)</span>
    <span><span class="dot" style="background:#e74c3c"></span> Not learned (~1.0x)</span>
  </div>
</div>

<h2>Charts</h2>
<div class="card">
  <div class="tabs">
    <div class="tab on" onclick="sw('ratios',this)">Perplexity ratios</div>
    <div class="tab" onclick="sw('ppls',this)">Valid vs invalid PPL</div>
    <div class="tab" onclick="sw('scores',this)">Learning scores</div>
  </div>
  <div id="t-ratios" class="tc on">
    <p style="font-size:13px;color:#555;margin-bottom:8px">
      Ratio = invalid / valid perplexity. Higher = model learned the rule.
      Anything below 1.3x (dashed line) is not detected.</p>
    <div class="ch">
      <canvas id="cR" role="img"
        aria-label="Horizontal bar chart of perplexity ratios">PPL ratios</canvas>
    </div>
  </div>
  <div id="t-ppls" class="tc">
    <p style="font-size:13px;color:#555;margin-bottom:8px">
      Blue = valid sequence PPL. Orange = invalid PPL. Larger gap = stronger rule detection.</p>
    <div class="ch">
      <canvas id="cP" role="img"
        aria-label="Grouped bar of valid vs invalid PPL">PPL values</canvas>
    </div>
  </div>
  <div id="t-scores" class="tc">
    <p style="font-size:13px;color:#555;margin-bottom:8px">
      Converted to 0–1 learning score.</p>
    <div class="ch2">
      <canvas id="cS" role="img"
        aria-label="Bar chart of learning scores">Learning scores</canvas>
    </div>
  </div>
</div>

<h2>Rule-by-Rule Evidence</h2>
<div class="card">{rule_cards}</div>

<h2>Extracted Learned Rules</h2>
<div class="card" style="font-size:14px;line-height:1.7">{learned_html}</div>

<h2>Comparison with Ground Truth</h2>
<div class="card" style="font-size:14px;line-height:1.7">{comparison_html}</div>

<p style="text-align:center;color:#aaa;font-size:12px;
          margin-top:2rem;padding-bottom:2rem">
  rule_probe_pipeline.py · Branch: angie · Industrial AI Hackathon
</p>
</div>

<script>
const L={bar_labels},R={bar_ratios},C={bar_colors},
      V={valid_ppls},I={invalid_ppls},S={learn_scores};
Chart.defaults.font.family='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';
new Chart(document.getElementById('cR'),{{type:'bar',
  data:{{labels:L,datasets:[{{label:'PPL ratio',data:R,
    backgroundColor:C,borderWidth:0}}]}},
  options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{x:{{beginAtZero:true,
      title:{{display:true,text:'Perplexity ratio (higher = learned)'}}}}}}
  }}
}});
new Chart(document.getElementById('cP'),{{type:'bar',
  data:{{labels:L,datasets:[
    {{label:'Valid PPL',data:V,backgroundColor:'#3266ad',borderWidth:0}},
    {{label:'Invalid PPL',data:I,backgroundColor:'#c97b4b',borderWidth:0}}
  ]}},
  options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:true,position:'top'}}}}
  }}
}});
new Chart(document.getElementById('cS'),{{type:'bar',
  data:{{labels:L,datasets:[{{label:'Score',data:S,
    backgroundColor:C,borderWidth:0}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{y:{{beginAtZero:true,max:1.1}}}}
  }}
}});
function sw(n,el){{
  document.querySelectorAll('.tc').forEach(t=>t.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.getElementById('t-'+n).classList.add('on');
  el.classList.add('on');
}}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_sequences(train_dir: str) -> list[list[str]]:
    sequences = []
    current_id, current_seq = None, []
    files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} CSV files in {train_dir}")
    for fname in sorted(files):
        fpath = os.path.join(train_dir, fname)
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.strip().upper() for c in (reader.fieldnames or [])]
            if "SEQUENCE_ID" not in cols or "STEP" not in cols:
                continue
            for row in reader:
                sid  = (row.get("SEQUENCE_ID") or "").strip()
                step = (row.get("STEP") or "").strip()
                if not step or not sid:
                    continue
                if sid != current_id:
                    if current_seq:
                        sequences.append(current_seq)
                    current_id, current_seq = sid, [step]
                else:
                    current_seq.append(step)
    if current_seq:
        sequences.append(current_seq)
    print(f"Loaded {len(sequences)} sequences")
    return sequences


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generic rule probing pipeline — plug in any model")
    parser.add_argument("--model",
                        choices=["ngram", "xgboost"],
                        default="ngram")
    parser.add_argument("--train_dir",
                        default="../training_data",
                        help="Training data folder (for ngram training)")
    parser.add_argument("--n",
                        type=int, default=10,
                        help="N for n-gram model")
    parser.add_argument("--adapter_path",
                        default="./my_model_adapter.py",
                        help="Path to adapter .py file defining get_perplexity()")
    parser.add_argument("--ground_truth",
                        default="../training_data/generation_rules.md")
    parser.add_argument("--out_dir",
                        default="./probe_results")
    parser.add_argument("--skip_llm",
                        action="store_true",
                        help="Skip LLM extraction (just save probe JSON + HTML shell)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Build adapter ──────────────────────────────────────────────────────────
    if args.model == "ngram":
        sequences = load_sequences(args.train_dir)
        random.seed(42)
        random.shuffle(sequences)
        train_seqs = sequences[:int(len(sequences) * 0.9)]
        adapter = NGramAdapter(n=args.n)
        adapter.train(train_seqs)

    elif args.model == "xgboost":
        # Dynamically import the teammate's adapter file
        # The file must define: get_perplexity(sequence: list[str]) -> float
        import importlib.util
        spec   = importlib.util.spec_from_file_location(
            "teammate_adapter", args.adapter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_perplexity"):
            raise ValueError(
                f"{args.adapter_path} must define: "
                "def get_perplexity(sequence: list[str]) -> float"
            )

        adapter = ExternalModelAdapter(
            perplexity_fn=module.get_perplexity,
            name=getattr(module, "MODEL_NAME", "External Model"),
            info=getattr(module, "MODEL_INFO", {"model_type": "external"}),
        )

    # ── Run probes ─────────────────────────────────────────────────────────────
    probe_data = run_probes(adapter)
    probe_data["train_sequences"] = (
        len(train_seqs) if args.model == "ngram" else "N/A"
    )

    # Save raw probe results
    probe_path = os.path.join(args.out_dir,
                              f"{args.model}_probe_results.json")
    with open(probe_path, "w") as f:
        json.dump(probe_data, f, indent=2)
    print(f"\nProbe results saved: {probe_path}")

    # ── LLM extraction ─────────────────────────────────────────────────────────
    learned_md    = ""
    comparison_md = ""

    if not args.skip_llm:
        learned_md = extract_learned_rules(probe_data)
        learned_path = os.path.join(args.out_dir, "learned_rules.md")
        with open(learned_path, "w", encoding="utf-8") as f:
            f.write(learned_md)
        print(f"Learned rules saved: {learned_path}")

        comparison_md_result = compare_with_ground_truth(
            learned_md, args.ground_truth, probe_data)
        if comparison_md_result:
            comparison_md = comparison_md_result
            comp_path = os.path.join(args.out_dir, "rules_comparison.md")
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comparison_md)
            print(f"Comparison saved: {comp_path}")
    else:
        learned_md = "*LLM extraction skipped. Run without --skip_llm to generate.*"

    # ── Build HTML report ──────────────────────────────────────────────────────
    html = build_html_report(probe_data, learned_md, comparison_md)
    html_path = os.path.join(args.out_dir,
                             f"{args.model}_learned_rules.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")
    print(f"  Probe JSON : {probe_path}")
    print(f"  HTML report: {html_path}")
    print(f"\n  Open: file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
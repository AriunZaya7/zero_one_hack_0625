"""
xgb_probe.py
=============
Branch: angie
Folder: testing_rule_extraction/

XGBoost-specific behavioral probing using PREDICTION-based violation detection.

Instead of perplexity (which doesn't work for XGBoost), we ask:
  "At the position where a rule is violated, what does the model EXPECT?"

If the model learned RULE_DEP_NO_CLEAN:
  - Given context ending in MEASURE THICKNESS (no clean step)
  - Model should predict HF DIP / DRY WAFER / CLEAN (a cleaning step)
  - But the actual next step is THERMAL OXIDATION (deposition without clean)
  - The model's top-k predictions reveal whether it knows a clean is needed

Metrics per rule:
  - top1_is_correct_type: did model predict the RIGHT category of step?
  - top1_on_violation:    did model predict a clean/mask/etc before the violation?
  - expected_in_top5:     is the expected step type anywhere in top-5?
  - violation_rank:       where does the violating step rank in model's predictions?

Usage:
    python xgb_probe.py --adapter_path ./adapter_xgboost.py
    python xgb_probe.py --adapter_path ./adapter_xgboost.py --skip_llm
"""

import json
import os
import argparse
import importlib.util
from openai import OpenAI


# ── Step category helper ───────────────────────────────────────────────────────

CLEAN_STEPS = {
    "HF DIP", "DRY WAFER", "RCA CLEAN 1", "RCA CLEAN 2",
    "CLEAN AFTER ETCH", "CLEAN AFTER CMP", "CLEAN WAFER SURFACE",
    "POST IMPLANT CLEAN", "PRE DIFFUSION CLEAN", "SOLVENT CLEAN",
}
DEPOSIT_STEPS = {
    "THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE",
    "EPITAXIAL DEPOSITION", "DEPOSIT POLYSILICON", "DEPOSIT METAL 1",
    "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT ILD OXIDE",
    "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER",
    "DEPOSIT BARRIER METAL", "DEPOSIT TOP METAL",
}
LITHO_EXPOSE_STEPS = {
    "EXPOSE LITHO LEVEL 1", "EXPOSE LITHO LEVEL 2",
    "EXPOSE LITHO LEVEL 3", "EXPOSE LITHO LEVEL 4",
}
LITHO_DEVELOP_STEPS = {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}
ETCH_STEPS = {
    "OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH",
    "VIA ETCH", "FIELD OXIDE ETCH", "METAL ETCH", "METAL ETCH DRY",
}
IMPLANT_STEPS = {
    "IMPLANT P BODY", "IMPLANT WELL", "IMPLANT SOURCE DRAIN",
    "IMPLANT LDD", "IMPLANT N BUFFER", "IMPLANT CHANNEL STOP",
}
CMP_STEPS = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC",
             "CMP METAL", "CMP VIA FILL"}
TEST_STEPS = {
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
}


def step_category(step: str) -> str:
    s = step.strip().upper()
    if s in CLEAN_STEPS:         return "CLEAN"
    if s in DEPOSIT_STEPS:       return "DEPOSIT"
    if s in LITHO_EXPOSE_STEPS:  return "LITHO_EXPOSE"
    if s in LITHO_DEVELOP_STEPS: return "LITHO_DEVELOP"
    if s in ETCH_STEPS:          return "ETCH"
    if s in IMPLANT_STEPS:       return "IMPLANT"
    if s in CMP_STEPS:           return "CMP"
    if s in TEST_STEPS:          return "TEST"
    if "BACKSIDE" in s:          return "BACKSIDE"
    if "PASSIVATION" in s:       return "PASSIVATION"
    if "SHIP" in s:              return "SHIP"
    if "WAFER SORT" in s:        return "TEST_SORT"
    return "OTHER"


# ── Prediction-based rule probes ───────────────────────────────────────────────
# Each probe defines:
#   context:          steps given to model as input history
#   actual_next:      what appears next in the VIOLATED sequence
#   expected_cats:    what category the model SHOULD predict if it learned the rule
#   rule_description: plain English explanation

PREDICTION_PROBES = [
    {
        "rule": "RULE_DEP_NO_CLEAN",
        "description": "Deposition must always be preceded by a cleaning step",
        "violation": "THERMAL OXIDATION follows MEASURE THICKNESS with no clean",
        "context": [
            "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN INSPECTION",
            "MEASURE INITIAL THICKNESS", "MEASURE SURFACE PLANARITY",
        ],
        "actual_next": "THERMAL OXIDATION",
        "expected_cats": ["CLEAN"],
        "explanation": "Model should predict a CLEAN step before deposition",
    },
    {
        "rule": "RULE_METAL_ETCH_NO_LITHO",
        "description": "Lithography must precede metal etching",
        "violation": "METAL ETCH follows SOFT BAKE without EXPOSE + DEVELOP",
        "context": [
            "HF DIP", "DRY WAFER", "DEPOSIT METAL 1",
            "COAT PHOTORESIST", "SOFT BAKE",
        ],
        "actual_next": "METAL ETCH",
        "expected_cats": ["LITHO_EXPOSE"],
        "explanation": "Model should predict EXPOSE LITHO before METAL ETCH",
    },
    {
        "rule": "RULE_ETCH_NO_MASK",
        "description": "A developed mask must exist before any etch",
        "violation": "OXIDE ETCH follows EXPOSE LITHO without DEVELOP first",
        "context": [
            "COAT PHOTORESIST", "SOFT BAKE", "EXPOSE LITHO LEVEL 1",
        ],
        "actual_next": "OXIDE ETCH",
        "expected_cats": ["LITHO_DEVELOP"],
        "explanation": "Model should predict DEVELOP PHOTORESIST before OXIDE ETCH",
    },
    {
        "rule": "RULE_LITHO_LEVEL_SKIP",
        "description": "Lithography levels must increment sequentially",
        "violation": "EXPOSE LITHO LEVEL 3 follows LEVEL 1 skipping LEVEL 2",
        "context": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "STRIP RESIST", "CLEAN AFTER ETCH",
            "COAT PHOTORESIST",
        ],
        "actual_next": "EXPOSE LITHO LEVEL 3",
        "expected_cats": ["LITHO_EXPOSE"],
        "expected_step": "EXPOSE LITHO LEVEL 2",
        "explanation": "Model should predict EXPOSE LITHO LEVEL 2 not LEVEL 3",
    },
    {
        "rule": "RULE_IMPLANT_NO_MASK",
        "description": "An implant window must be opened before ion implantation",
        "violation": "IMPLANT P BODY follows DEVELOP PHOTORESIST without OXIDE ETCH",
        "context": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
        ],
        "actual_next": "IMPLANT P BODY",
        "expected_cats": ["ETCH"],
        "explanation": "Model should predict OXIDE ETCH to open window before implant",
    },
    {
        "rule": "RULE_CMP_NO_DEP",
        "description": "CMP planarization requires a prior deposition step",
        "violation": "CMP DIELECTRIC follows MEASURE steps with no deposition",
        "context": [
            "MEASURE SURFACE PLANARITY", "MEASURE INITIAL THICKNESS",
            "MEASURE THICKNESS",
        ],
        "actual_next": "CMP DIELECTRIC",
        "expected_cats": ["DEPOSIT"],
        "explanation": "Model should predict a DEPOSIT step before CMP",
    },
    {
        "rule": "RULE_PAD_OPEN_BEFORE_DEP",
        "description": "Bond pad window only after passivation deposited and cured",
        "violation": "OPEN PAD WINDOW appears before passivation block",
        "context": [
            "COAT PHOTORESIST", "EXPOSE LITHO LEVEL 4", "DEVELOP PHOTORESIST",
        ],
        "actual_next": "OPEN PAD WINDOW",
        "expected_cats": ["ETCH", "OTHER"],
        "unexpected_step": "OPEN PAD WINDOW",
        "explanation": "Model should NOT predict OPEN PAD WINDOW before passivation",
    },
    {
        "rule": "RULE_TEST_BEFORE_PASSIVATION",
        "description": "Electrical tests must come after passivation is cured",
        "violation": "PARAMETRIC TEST appears before CURE PASSIVATION",
        "context": [
            "DEPOSIT PASSIVATION",
        ],
        "actual_next": "PARAMETRIC TEST",
        "expected_cats": ["PASSIVATION"],
        "expected_step": "CURE PASSIVATION",
        "explanation": "Model should predict CURE PASSIVATION not PARAMETRIC TEST",
    },
    {
        "rule": "RULE_SHIP_BEFORE_TEST",
        "description": "Wafer sort test must complete before shipping",
        "violation": "SHIP LOT appears before WAFER SORT TEST",
        "context": [
            "CURE PASSIVATION", "OPEN PAD WINDOW", "FINAL INSPECTION",
        ],
        "actual_next": "SHIP LOT",
        "expected_cats": ["TEST_SORT"],
        "expected_step": "WAFER SORT TEST",
        "explanation": "Model should predict WAFER SORT TEST before SHIP LOT",
    },
    {
        "rule": "RULE_BACKSIDE_BEFORE_PASSIVATION",
        "description": "Backside metal only after frontside passivation is cured",
        "violation": "DEPOSIT BACKSIDE METAL appears before passivation block",
        "context": [
            "BACKSIDE GRIND", "BACKSIDE CLEAN",
        ],
        "actual_next": "DEPOSIT BACKSIDE METAL",
        "expected_cats": ["OTHER"],
        "unexpected_step": "DEPOSIT BACKSIDE METAL",
        "explanation": "Model should not allow backside metal before passivation",
    },
]


# ── Load adapter ───────────────────────────────────────────────────────────────

def load_adapter(adapter_path: str):
    spec   = importlib.util.spec_from_file_location("adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build_features") or not hasattr(module, "model"):
        raise ValueError(
            f"{adapter_path} must define 'model' and 'build_features()'"
        )
    return module


# ── Run prediction probes ──────────────────────────────────────────────────────

def run_prediction_probes(module) -> list[dict]:
    import numpy as np

    results = []
    print(f"\nRunning {len(PREDICTION_PROBES)} prediction-based probes")
    print(f"Model: {module.MODEL_NAME}")
    print(f"{'─'*80}")
    print(f"{'Rule':<35} {'Top-1 pred':>20} {'Expected cat':>14} {'Learned?':>12}")
    print(f"{'─'*80}")

    for probe in PREDICTION_PROBES:
        context     = probe["context"]
        actual_next = probe["actual_next"]
        expected_cats = probe["expected_cats"]

        # Build features for predicting next step after context
        try:
            X      = module.build_features(context + [actual_next])
            # Use the last row — features at the final context position
            X_last = X[[-1]]
            probas = module.model.predict_proba(X_last)[0]

            # Get top-5 predicted class indices
            top5_idx   = probas.argsort()[-5:][::-1]
            top5_probs = probas[top5_idx]

            # Map indices back to step names using class_map
            idx_to_step = {v: k for k, v in module.class_map.items()}
            top5_steps  = [idx_to_step.get(int(i), f"class_{i}")
                           for i in top5_idx]
            top5_cats   = [step_category(s) for s in top5_steps]

            top1_step = top5_steps[0]
            top1_cat  = top5_cats[0]
            top1_prob = float(top5_probs[0])

            # Did the model predict the expected category?
            top1_correct = top1_cat in expected_cats

            # Is the expected category in top-5?
            expected_in_top5 = any(c in expected_cats for c in top5_cats)

            # Where does the actual (violating) step rank?
            actual_cat = step_category(actual_next)
            actual_rank = next(
                (i+1 for i, c in enumerate(top5_cats) if c == actual_cat),
                99
            )

            # Check for specific expected step if defined
            specific_step = probe.get("expected_step", "")
            specific_in_top5 = (
                specific_step in top5_steps if specific_step else None
            )

            # Check for unexpected step if defined
            unexpected_step = probe.get("unexpected_step", "")
            unexpected_in_top5 = (
                unexpected_step in top5_steps if unexpected_step else False
            )

            # Determine learning status
            if top1_correct and top1_prob > 0.3:
                learned = "STRONG"
            elif top1_correct or expected_in_top5:
                learned = "PARTIAL"
            elif actual_rank > 3:
                learned = "WEAK"
            else:
                learned = "NOT LEARNED"

            # Special case: if model predicts the unexpected step, it's wrong
            if unexpected_step and top1_step == unexpected_step:
                learned = "NOT LEARNED"

        except Exception as e:
            print(f"  [error] {probe['rule']}: {e}")
            top1_step = "ERROR"
            top1_cat  = "ERROR"
            top1_prob = 0.0
            top5_steps = []
            top5_cats  = []
            expected_in_top5 = False
            actual_rank = 99
            specific_in_top5 = None
            learned = "NOT LEARNED"

        print(f"{probe['rule']:<35} {top1_step:>20} {','.join(expected_cats):>14} "
              f"{learned:>12}")

        results.append({
            "rule":             probe["rule"],
            "description":      probe["description"],
            "violation":        probe["violation"],
            "explanation":      probe["explanation"],
            "context":          probe["context"],
            "actual_next":      actual_next,
            "expected_cats":    expected_cats,
            "top1_prediction":  top1_step,
            "top1_category":    top1_cat,
            "top1_probability": round(top1_prob, 4),
            "top5_predictions": top5_steps,
            "top5_categories":  top5_cats,
            "expected_in_top5": expected_in_top5,
            "actual_step_rank": actual_rank,
            "specific_in_top5": specific_in_top5,
            "learned":          learned,
        })

    return results


def summarize(results: list[dict]) -> dict:
    strong  = sum(1 for r in results if r["learned"] == "STRONG")
    partial = sum(1 for r in results if r["learned"] == "PARTIAL")
    weak    = sum(1 for r in results if r["learned"] == "WEAK")
    none_   = sum(1 for r in results if r["learned"] == "NOT LEARNED")
    score   = round((strong + 0.5*partial + 0.2*weak) / len(results) * 100, 1)

    print(f"\n{'='*55}")
    print(f"  XGBOOST PREDICTION PROBE SUMMARY")
    print(f"{'='*55}")
    print(f"  STRONG      : {strong}/10")
    print(f"  PARTIAL     : {partial}/10")
    print(f"  WEAK        : {weak}/10")
    print(f"  NOT LEARNED : {none_}/10")
    print(f"  Score       : {score}%")
    print(f"{'='*55}")

    return {
        "strong": strong, "partial": partial,
        "weak": weak, "not_learned": none_,
        "total": len(results), "score_pct": score,
    }


# ── HTML Report ────────────────────────────────────────────────────────────────

LEARNED_COLORS = {
    "STRONG":      "#2ecc71",
    "PARTIAL":     "#f39c12",
    "WEAK":        "#e67e22",
    "NOT LEARNED": "#e74c3c",
}
LEARNED_SCORES = {
    "STRONG": 1.0, "PARTIAL": 0.6, "WEAK": 0.2, "NOT LEARNED": 0.0
}


def build_html(module, results: list[dict],
               summary: dict, learned_md: str = "") -> str:
    import json as _json

    bar_labels  = _json.dumps([r["rule"].replace("RULE_","") for r in results])
    bar_scores  = _json.dumps([LEARNED_SCORES[r["learned"]] for r in results])
    bar_colors  = _json.dumps([LEARNED_COLORS[r["learned"]] for r in results])
    score       = summary["score_pct"]

    # Rule cards
    rule_cards = ""
    for r in results:
        bg    = LEARNED_COLORS[r["learned"]]
        badge = (f'<span style="background:{bg};color:#fff;padding:3px 10px;'
                 f'border-radius:12px;font-size:12px;font-weight:600">'
                 f'{r["learned"]}</span>')

        top5_html = "".join(
            f'<span style="background:#f0f0f0;border-radius:4px;padding:2px 8px;'
            f'margin:2px;display:inline-block;font-size:12px;font-family:monospace">'
            f'{s} <span style="color:#888">({c})</span></span>'
            for s, c in zip(r["top5_predictions"], r["top5_categories"])
        )

        rule_cards += f"""
        <div style="border:1px solid #e0e0e0;border-radius:10px;
                    padding:1.2rem;margin-bottom:1rem;border-left:4px solid {bg}">
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
            <p style="margin:0 0 4px"><b>What we tested:</b> {r['explanation']}</p>
            <p style="margin:0 0 4px"><b>Violation:</b> {r['violation']}</p>
            <p style="margin:0 0 4px">
              <b>Model top-1 prediction:</b>
              <span style="font-family:monospace;background:#fff;
                           border:1px solid #ddd;padding:1px 6px;border-radius:4px">
                {r['top1_prediction']}
              </span>
              (category: {r['top1_category']}, prob: {r['top1_probability']})
            </p>
            <p style="margin:0 0 4px">
              <b>Expected category:</b> {', '.join(r['expected_cats'])} &nbsp;
              {'✓ in top-5' if r['expected_in_top5'] else '✗ not in top-5'}
            </p>
            <p style="margin:0">
              <b>Violating step rank:</b>
              {'#' + str(r['actual_step_rank']) if r['actual_step_rank'] < 99
               else 'not in top-5'}
            </p>
          </div>
          <details style="margin-top:8px">
            <summary style="cursor:pointer;font-size:13px;color:#555">
              Top-5 predictions</summary>
            <div style="margin-top:8px">{top5_html}</div>
          </details>
          <details style="margin-top:6px">
            <summary style="cursor:pointer;font-size:13px;color:#555">
              Context given to model</summary>
            <p style="margin-top:6px;font-size:12px;font-family:monospace;
                      color:#333">
              {' → '.join(r['context'])} → <b>[predict]</b>
            </p>
          </details>
        </div>"""

    # Simple markdown renderer for learned rules
    def md_to_html(md: str) -> str:
        if not md or md.startswith("*"):
            return f'<p style="color:#999">{md}</p>'
        html = ""
        for line in md.split("\n"):
            if line.startswith("## "):
                html += (f'<h2 style="margin-top:1.5rem;font-size:17px;'
                         f'border-bottom:1px solid #eee;padding-bottom:6px">'
                         f'{line[3:]}</h2>\n')
            elif line.startswith("- ") or line.startswith("* "):
                html += f'<li style="margin:3px 0;font-size:14px">{line[2:]}</li>\n'
            elif line.strip() == "":
                html += "<br>\n"
            else:
                bold = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
                html += (f'<p style="margin:4px 0;font-size:14px;'
                         f'line-height:1.6">{bold}</p>\n')
        return html

    info = {
        "model_type":  "XGBoost",
        "n_features":  module.MODEL_INFO.get("n_features", "?"),
        "n_classes":   module.MODEL_INFO.get("n_classes", "?"),
        "trained_on":  module.MODEL_INFO.get("trained_on", "?"),
        "ood_family":  module.MODEL_INFO.get("ood_family", "?"),
        "probe_method":"Prediction-based (not perplexity)",
    }
    info_rows = "".join(
        f'<tr><td style="color:#666;padding:4px 0">{k}</td>'
        f'<td style="text-align:right;font-weight:500;padding:4px 0">{v}</td></tr>'
        for k, v in info.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>XGBoost Rule Learning Report</title>
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
.legend{{display:flex;flex-wrap:wrap;gap:12px;font-size:12px}}
.dot{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.ch{{position:relative;width:100%;height:300px}}
details summary{{cursor:pointer;font-size:13px;color:#555;padding:4px 0}}
</style>
</head>
<body>
<div class="wrap">

<div class="card">
  <p style="font-size:12px;color:#888;margin-bottom:6px">
    HACKATHON · INDUSTRIAL AI TRACK · BRANCH: ANGIE</p>
  <h1>XGBoost — Rule Learning Report</h1>
  <p style="color:#555;margin-top:6px">
    Prediction-based probing: at each rule violation point, what does
    the model predict should come next?</p>
  <p style="margin-top:8px;font-size:13px;background:#fff8e1;
            border-left:3px solid #f39c12;padding:8px 12px;border-radius:4px">
    <b>Note:</b> XGBoost was probed using prediction accuracy, not perplexity.
    Perplexity is not meaningful for XGBoost because it outputs similar
    confidence (~0.47) for almost all sequences due to 179 classes.
    Instead we ask: at the violation point, does the model predict the
    correct step type?
  </p>
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
    <span><span class="dot" style="background:#2ecc71"></span>
      Strong (top-1 = correct category, prob &gt; 30%)</span>
    <span><span class="dot" style="background:#f39c12"></span>
      Partial (correct in top-5)</span>
    <span><span class="dot" style="background:#e67e22"></span>
      Weak (violation ranks low)</span>
    <span><span class="dot" style="background:#e74c3c"></span>
      Not learned</span>
  </div>
</div>

<h2>Learning Score per Rule</h2>
<div class="card">
  <div class="ch">
    <canvas id="cS" role="img"
      aria-label="Bar chart of learning scores per rule">Learning scores</canvas>
  </div>
</div>

<h2>Rule-by-Rule Evidence</h2>
<div class="card">{rule_cards}</div>

<h2>Extracted Learned Rules</h2>
<div class="card" style="font-size:14px;line-height:1.7">
  {md_to_html(learned_md)}
</div>

<p style="text-align:center;color:#aaa;font-size:12px;
          margin-top:2rem;padding-bottom:2rem">
  xgb_probe.py · Branch: angie · Industrial AI Hackathon
</p>
</div>

<script>
const L={bar_labels}, S={bar_scores}, C={bar_colors};
Chart.defaults.font.family='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';
new Chart(document.getElementById('cS'),{{
  type:'bar',
  data:{{labels:L,datasets:[{{
    label:'Learning score',data:S,
    backgroundColor:C,borderWidth:0
  }}]}},
  options:{{
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      y:{{beginAtZero:true,max:1.1,
        title:{{display:true,text:'0 = not learned, 1 = strongly learned'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""


# ── LLM extraction ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are an expert in semiconductor manufacturing and ML interpretability.
Analyze prediction evidence to infer what rules an XGBoost model learned.
Do NOT invent rules not supported by evidence. Be honest."""


def build_prompt(module, results: list[dict], summary: dict) -> str:
    results_text = ""
    for r in results:
        results_text += f"""
---
Rule: {r['rule']}
What we tested: {r['explanation']}
Violation: {r['violation']}
Context given: {' -> '.join(r['context'])}
Actual violating next step: {r['actual_next']}
Model top-1 prediction: {r['top1_prediction']} (category={r['top1_category']}, prob={r['top1_probability']})
Expected category: {', '.join(r['expected_cats'])}
Expected category in top-5: {r['expected_in_top5']}
Violating step rank in top-5: {r['actual_step_rank']}
Assessment: {r['learned']}
"""
    return f"""XGBoost model probed for semiconductor process rule learning.
Model: {module.MODEL_NAME} | Classes: {module.MODEL_INFO.get('n_classes')} | Features: {module.MODEL_INFO.get('n_features')}
Trained on: {module.MODEL_INFO.get('trained_on')} | OOD family: {module.MODEL_INFO.get('ood_family')}

PROBE METHOD: At each rule violation point, we ask what the model predicts
should come next. If it predicts the correct step type, it learned the rule.

RESULTS:
{results_text}

SUMMARY: Strong={summary['strong']} Partial={summary['partial']} NotLearned={summary['not_learned']} Score={summary['score_pct']}%

For each rule write:
## Rule: [name]
**Prediction evidence:** Model predicted [X] when [Y] was expected
**Conclusion:** LEARNED / PARTIALLY LEARNED / NOT LEARNED
**Why:** [brief reason based on the evidence]

Then add:
## Overall Verdict
Does this XGBoost model understand process logic or just predict common next steps?
What do the predictions reveal about what it learned from the training data?"""


def call_llm(prompt: str, max_tokens: int = 3000) -> str:
    api_key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set FEATHERLESS_API_KEY environment variable")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.featherless.ai/v1")
    model  = os.environ.get("FEATHERLESS_MODEL", "deepseek-ai/DeepSeek-V3-0324")
    print(f"  Calling: {model}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost prediction-based rule probing")
    parser.add_argument("--adapter_path", default="./adapter_xgboost.py")
    parser.add_argument("--out_dir",      default="./probe_results")
    parser.add_argument("--skip_llm",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load adapter
    print(f"Loading adapter: {args.adapter_path}")
    module = load_adapter(args.adapter_path)

    # Run probes
    results = run_prediction_probes(module)
    summary = summarize(results)

    # Save JSON
    probe_data = {
        "model_name":    module.MODEL_NAME,
        "model_info":    module.MODEL_INFO,
        "probe_method":  "prediction-based",
        "probe_results": results,
        "summary":       summary,
    }
    json_path = os.path.join(args.out_dir, "xgboost_prediction_probe.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe_data, f, indent=2)
    print(f"\nProbe results saved: {json_path}")

    # LLM extraction
    learned_md = "*Run without --skip_llm to generate learned rules interpretation.*"
    if not args.skip_llm:
        print("\nExtracting learned rules via LLM...")
        learned_md = call_llm(build_prompt(module, results, summary))
        md_path = os.path.join(args.out_dir, "xgboost_learned_rules.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(learned_md)
        print(f"Learned rules saved: {md_path}")

    # Build HTML
    html      = build_html(module, results, summary, learned_md)
    html_path = os.path.join(args.out_dir, "xgboost_prediction_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")
    print(f"  JSON   : {json_path}")
    print(f"  HTML   : {html_path}")
    print(f"\n  Open: file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
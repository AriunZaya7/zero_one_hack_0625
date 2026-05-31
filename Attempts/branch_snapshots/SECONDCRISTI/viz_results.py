"""
viz_results.py — Build a standalone HTML results report from the GPT training run.

Reads: results/*.json, submissions/task1_*.csv, self_eval/ground_truth_valid.csv
Writes: results_report.html

Run:
    python viz_results.py
    scp LOGIN:~/zero_one_hack_0625/results_report.html .
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time
from pathlib import Path

# ── Hard-coded training curves from the SLURM log ─────────────────────────────
# (parsed from outputs/slurm/gpt_43114539.log)
LOSS_CURVES = {
    "ID — all families": [
        2.694, 0.411, 0.356, 0.342, 0.337, 0.334, 0.331, 0.330, 0.329, 0.328,
        0.327, 0.326, 0.326, 0.325, 0.325, 0.324, 0.324, 0.324, 0.323, 0.323,
    ],
    "LOO — hold-out MOSFET": [
        3.489, 0.604, 0.410, 0.378, 0.364, 0.356, 0.353, 0.350, 0.348, 0.346,
        0.345, 0.343, 0.343, 0.341, 0.341, 0.341, 0.340, 0.340, 0.339, 0.339,
    ],
    "LOO — hold-out IGBT": [
        2.734, 0.412, 0.356, 0.343, 0.337, 0.334, 0.332, 0.330, 0.329, 0.328,
        0.327, 0.327, 0.326, 0.326, 0.325, 0.325, 0.324, 0.324, 0.324, 0.324,
    ],
    "LOO — hold-out IC": [
        2.739, 0.399, 0.341, 0.327, 0.321, 0.318, 0.316, 0.315, 0.314, 0.313,
        0.312, 0.311, 0.311, 0.310, 0.310, 0.310, 0.310, 0.309, 0.309, 0.309,
    ],
}

EPOCHS = list(range(1, 21))

COLORS = {
    "ID — all families":       "#6C8EBF",
    "LOO — hold-out MOSFET":   "#D6A520",
    "LOO — hold-out IGBT":     "#82B366",
    "LOO — hold-out IC":       "#AE85C9",
    "GPT-2 small (4.9M)":      "#6C8EBF",
    "n-gram (n=3)":            "#8B93B0",
}

BG, CARD, BORDER, TEXT, SUB = "#0E1117", "#13161F", "#2A3149", "#E8EAF6", "#8B93B0"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    """Load flat per-run result dicts (skip guided_*.json which has a different schema)."""
    rows = []
    for p in sorted(glob.glob("results/*.json")):
        if os.path.basename(p).startswith("guided_"):
            continue
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def load_guided_results() -> list[dict]:
    """Load the grammar-guide comparison results."""
    rows = []
    for p in sorted(glob.glob("results/guided_*.json")):
        with open(p) as f:
            data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
    return rows


def load_predictions(path: str) -> dict[str, list[str]]:
    """task1_*.csv → {example_id: [rank1, rank2, ...]}"""
    preds = {}
    if not Path(path).exists():
        return preds
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            preds[r["EXAMPLE_ID"]] = [r.get(f"RANK_{i}", "") for i in range(1, 6)]
    return preds


def load_gt(path: str) -> dict[str, dict]:
    gt = {}
    if not Path(path).exists():
        return gt
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt[r["EXAMPLE_ID"]] = r
    return gt


# ── JSON serialiser ───────────────────────────────────────────────────────────

import math as _math
def _J(obj):
    return json.dumps(obj, allow_nan=False,
        default=lambda x: None if isinstance(x, float) and _math.isnan(x) else x)


# ── Dark layout helper ────────────────────────────────────────────────────────

def _layout(title="", **extra):
    return {
        "title": {"text": title, "font": {"size": 15}},
        "template": "plotly_dark",
        "paper_bgcolor": CARD, "plot_bgcolor": BG,
        "font": {"family": "Inter, sans-serif", "color": TEXT},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 50, "r": 20, "t": 45, "b": 50},
        "xaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER},
        "yaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER},
        **extra,
    }


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_loss_curves():
    traces = []
    for name, losses in LOSS_CURVES.items():
        traces.append({
            "x": EPOCHS, "y": losses,
            "mode": "lines+markers",
            "name": name,
            "line": {"color": COLORS[name], "width": 2},
            "marker": {"size": 4},
        })
    return {"data": traces, "layout": _layout(
        "Training Loss — GPT-2 small (4.9M params) · 20 epochs",
        xaxis={"title": "Epoch", "gridcolor": BORDER},
        yaxis={"title": "Cross-entropy loss", "gridcolor": BORDER},
    )}


def fig_top1_comparison(results: list[dict]):
    """Grouped bar: n-gram vs GPT for each run."""
    labels, ng_vals, gpt_vals, deltas = [], [], [], []
    run_order = ["id", "ood"]
    holdout_order = ["", "mosfet", "igbt", "ic"]

    def sort_key(r):
        return (run_order.index(r["mode"]) if r["mode"] in run_order else 99,
                holdout_order.index(r["holdout"]) if r["holdout"] in holdout_order else 99)

    for r in sorted(results, key=sort_key):
        lbl = "ID — all families" if r["mode"] == "id" else f"OOD — hold-out {r['holdout'].upper()}"
        labels.append(lbl)
        ng_vals.append(r["ngram_top1"])
        gpt_vals.append(r["top1"])
        deltas.append(r["top1"] - r["ngram_top1"])

    traces = [
        {
            "type": "bar", "name": "n-gram baseline",
            "x": labels, "y": ng_vals,
            "marker": {"color": "#8B93B0", "opacity": 0.75},
            "text": [f"{v:.3f}" for v in ng_vals], "textposition": "outside",
        },
        {
            "type": "bar", "name": "GPT-2 small (4.9M)",
            "x": labels, "y": gpt_vals,
            "marker": {"color": "#6C8EBF"},
            "text": [f"{v:.3f}" for v in gpt_vals], "textposition": "outside",
        },
    ]

    # Annotate delta above each pair
    annotations = []
    for i, (lbl, d) in enumerate(zip(labels, deltas)):
        color = "#82B366" if d > 0 else "#D1453B"
        sign  = "▲" if d > 0 else "▼"
        annotations.append({
            "x": lbl, "y": max(ng_vals[i], gpt_vals[i]) + 0.04,
            "text": f"{sign}{abs(d):.3f}",
            "showarrow": False,
            "font": {"color": color, "size": 11, "family": "Inter"},
        })

    return {"data": traces, "layout": _layout(
        "Next-Step Top-1 Accuracy: GPT vs n-gram baseline",
        barmode="group",
        yaxis={"title": "Top-1 accuracy", "range": [0, 1.0], "gridcolor": BORDER},
        annotations=annotations,
    )}


def fig_metrics_radar(results: list[dict]):
    """Grouped bar of all 4 metrics for each run."""
    metrics = ["top1", "top3", "top5", "mrr"]
    labels  = ["Top-1", "Top-3", "Top-5", "MRR"]

    traces = []
    run_colors = [COLORS["ID — all families"],
                  COLORS["LOO — hold-out MOSFET"],
                  COLORS["LOO — hold-out IGBT"],
                  COLORS["LOO — hold-out IC"]]

    run_labels = {
        ("id", ""):       "ID — all families",
        ("ood", "mosfet"): "OOD — MOSFET holdout",
        ("ood", "igbt"):   "OOD — IGBT holdout",
        ("ood", "ic"):     "OOD — IC holdout",
    }

    for i, r in enumerate(sorted(results,
                                  key=lambda x: (0 if x["mode"]=="id" else 1, x["holdout"]))):
        name = run_labels.get((r["mode"], r["holdout"]),
                              f"{r['mode']} {r['holdout']}")
        vals = [r[m] for m in metrics]
        traces.append({
            "type": "bar",
            "name": f"GPT — {name}",
            "x": labels, "y": vals,
            "marker": {"color": run_colors[i % len(run_colors)]},
            "text": [f"{v:.3f}" for v in vals],
            "textposition": "outside",
            "opacity": 0.85,
        })
        # Add n-gram reference for the ID run only (to keep chart readable)
        if r["mode"] == "id":
            traces.append({
                "type": "bar",
                "name": "n-gram — ID baseline",
                "x": labels,
                "y": [r["ngram_top1"]] + [None, None, None],
                "marker": {"color": "#8B93B0", "opacity": 0.6},
                "text": [f"{r['ngram_top1']:.3f}", "", "", ""],
                "textposition": "outside",
            })

    return {"data": traces, "layout": _layout(
        "All 4 Metrics — GPT-2 small across all runs",
        barmode="group",
        yaxis={"title": "Score", "range": [0, 1.1], "gridcolor": BORDER},
    )}


def fig_id_vs_ood_drop(results: list[dict]):
    """Scatter: ID top1 vs OOD top1 with arrows showing the drop."""
    ood_results = [r for r in results if r["mode"] == "ood"]

    gpt_ood_vals   = [r["top1"]        for r in ood_results]
    ng_ood_vals    = [r["ngram_top1"]  for r in ood_results]
    holdouts       = [r["holdout"].upper() for r in ood_results]

    id_result = next((r for r in results if r["mode"] == "id"), None)
    gpt_id    = id_result["top1"]       if id_result else None
    ng_id     = id_result["ngram_top1"] if id_result else None

    traces = []

    # GPT OOD bars
    traces.append({
        "type": "bar", "name": "GPT OOD top-1",
        "x": holdouts, "y": gpt_ood_vals,
        "marker": {"color": "#6C8EBF"},
        "text": [f"{v:.3f}" for v in gpt_ood_vals],
        "textposition": "outside",
    })
    # n-gram OOD bars
    traces.append({
        "type": "bar", "name": "n-gram OOD top-1",
        "x": holdouts, "y": ng_ood_vals,
        "marker": {"color": "#8B93B0", "opacity": 0.7},
        "text": [f"{v:.3f}" for v in ng_ood_vals],
        "textposition": "outside",
    })

    shapes = []
    if gpt_id is not None:
        shapes.append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1, "y0": gpt_id, "y1": gpt_id,
            "line": {"color": "#6C8EBF", "dash": "dot", "width": 1.5},
        })
    if ng_id is not None:
        shapes.append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1, "y0": ng_id, "y1": ng_id,
            "line": {"color": "#8B93B0", "dash": "dot", "width": 1.5},
        })

    annotations = []
    if gpt_id:
        annotations.append({"x": 1.01, "xref": "paper", "y": gpt_id, "yref": "y",
                              "text": f"GPT ID {gpt_id:.3f}", "showarrow": False,
                              "font": {"color": "#6C8EBF", "size": 10},
                              "xanchor": "left"})
    if ng_id:
        annotations.append({"x": 1.01, "xref": "paper", "y": ng_id, "yref": "y",
                              "text": f"n-gram ID {ng_id:.3f}", "showarrow": False,
                              "font": {"color": "#8B93B0", "size": 10},
                              "xanchor": "left"})

    return {"data": traces, "layout": _layout(
        "OOD Top-1 per Held-out Family  (dotted lines = ID performance)",
        barmode="group",
        yaxis={"title": "Top-1 accuracy", "range": [0, 1.0], "gridcolor": BORDER},
        shapes=shapes, annotations=annotations,
        margin={"l": 50, "r": 100, "t": 45, "b": 50},
    )}


def fig_top1_improvement():
    """Simple before/after bullet chart per run."""
    runs = [
        ("ID — all families",    0.735, 0.807),
        ("OOD — MOSFET holdout", 0.501, 0.551),
        ("OOD — IGBT holdout",   0.481, 0.473),
        ("OOD — IC holdout",     0.425, 0.467),
    ]
    labels  = [r[0] for r in runs]
    ng_vals = [r[1] for r in runs]
    gp_vals = [r[2] for r in runs]
    deltas  = [g - n for n, g in zip(ng_vals, gp_vals)]
    bar_colors = ["#82B366" if d >= 0 else "#D1453B" for d in deltas]

    traces = [
        {
            "type": "bar", "orientation": "h",
            "name": "n-gram", "x": ng_vals, "y": labels,
            "marker": {"color": "#8B93B0", "opacity": 0.7},
            "text": [f"{v:.3f}" for v in ng_vals], "textposition": "auto",
        },
        {
            "type": "bar", "orientation": "h",
            "name": "GPT", "x": gp_vals, "y": labels,
            "marker": {"color": "#6C8EBF"},
            "text": [f"{v:.3f}" for v in gp_vals], "textposition": "auto",
        },
    ]
    # Delta annotations
    annotations = []
    for i, (lbl, ng, gp, d) in enumerate(zip(labels, ng_vals, gp_vals, deltas)):
        color = "#82B366" if d >= 0 else "#D1453B"
        annotations.append({
            "x": max(ng, gp) + 0.02, "y": lbl,
            "text": f"{'+'if d>=0 else ''}{d:.3f}",
            "showarrow": False,
            "font": {"color": color, "size": 12, "family": "Inter, sans-serif"},
            "xanchor": "left",
        })

    return {"data": traces, "layout": _layout(
        "Top-1 Accuracy: n-gram → GPT  (green = improvement)",
        barmode="overlay",
        xaxis={"title": "Top-1 accuracy", "range": [0, 1.0], "gridcolor": BORDER},
        yaxis={"gridcolor": BORDER, "automargin": True},
        annotations=annotations,
        margin={"l": 180, "r": 100, "t": 45, "b": 50},
    )}


# ── Demo table ────────────────────────────────────────────────────────────────

def build_demo_table() -> str:
    """Show 8 example predictions: partial sequence → n-gram vs GPT."""
    ngram_preds = load_predictions("submissions/task1_ngram.csv")
    gpt_preds   = load_predictions("submissions/task1_gpt_small_ckpt.csv")
    gt          = load_gt("self_eval/ground_truth_valid.csv")

    if not ngram_preds or not gpt_preds or not gt:
        return "<p style='color:#8B93B0'>Submission files not found.</p>"

    examples = list(gt.keys())[:10]
    rows = ""
    shown = 0
    for eid in examples:
        if shown >= 8:
            break
        g   = gt.get(eid, {})
        ng  = ngram_preds.get(eid, ["?"] * 5)
        gp  = gpt_preds.get(eid, ["?"] * 5)
        remaining = g.get("_REMAINING", "")
        if not remaining:
            continue
        true_next = remaining.split("|")[0] if "|" in remaining else remaining
        partial = g.get("PARTIAL_SEQUENCE", "")
        last3   = " → ".join(partial.split("|")[-3:]) if partial else "…"

        ng_correct = true_next in ng[:5]
        gp_correct = true_next in gp[:5]

        def _cell(preds, correct):
            color = "#82B366" if correct else "#D1453B"
            check = "✓" if correct else "✗"
            return f'<td style="color:{color}">{check} {", ".join(p for p in preds[:3] if p)}</td>'

        rows += f"""
        <tr>
          <td style="color:{SUB};font-size:.8rem">…{last3}</td>
          <td style="color:#AE85C9;font-weight:600">{true_next}</td>
          {_cell(ng, ng_correct)}
          {_cell(gp, gp_correct)}
        </tr>"""
        shown += 1

    return f"""
    <table>
      <thead><tr>
        <th>Prefix (last 3 steps)</th>
        <th>True next step</th>
        <th>n-gram top-3 (✓/✗)</th>
        <th>GPT top-3 (✓/✗)</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(results: list[dict]) -> str:
    figs = [
        ("headline",  fig_top1_improvement()),
        ("headline",  fig_top1_comparison(results)),
        ("curves",    fig_loss_curves()),
        ("detail",    fig_id_vs_ood_drop(results)),
        ("detail",    fig_metrics_radar(results)),
    ]

    divs = {"headline": "", "curves": "", "detail": ""}
    js   = ""
    for idx, (section, fig) in enumerate(figs):
        did = f"fig_{idx}"
        divs[section] += f'<div class="chart-card"><div id="{did}"></div></div>\n'
        js += (f"Plotly.newPlot('{did}',{_J(fig['data'])},{_J(fig['layout'])}"
               f",{{responsive:true}});\n")

    demo = build_demo_table()
    ts   = time.strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT Process Grammar Results</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{BG};color:{TEXT};font-family:'Inter',-apple-system,sans-serif;padding:28px 32px;line-height:1.5}}
  h1{{font-size:1.9rem;font-weight:700;margin-bottom:4px}}
  .sub{{color:{SUB};font-size:.9rem;margin-bottom:24px}}
  hr{{border:none;border-top:1px solid {BORDER};margin:24px 0}}
  .sec{{font-size:.85rem;font-weight:600;color:{SUB};text-transform:uppercase;
        letter-spacing:.1em;margin:24px 0 12px}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  .grid-1{{display:grid;grid-template-columns:1fr;gap:14px}}
  .chart-card{{background:{CARD};border:1px solid {BORDER};border-radius:10px;
               padding:12px;overflow:hidden}}
  /* kpi cards */
  .kpi-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
  .kpi{{background:{CARD};border:1px solid {BORDER};border-radius:10px;
         padding:16px 22px;flex:1;min-width:140px}}
  .kpi-label{{color:{SUB};font-size:.72rem;text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:4px}}
  .kpi-value{{font-size:1.8rem;font-weight:700}}
  .green{{color:#82B366}} .blue{{color:#6C8EBF}}
  .red{{color:#D1453B}}   .gold{{color:#D6A520}}
  /* demo table */
  .tbl{{overflow-x:auto;border-radius:10px;border:1px solid {BORDER}}}
  table{{width:100%;border-collapse:collapse;background:{CARD}}}
  th{{background:#1A1F2E;color:{SUB};font-size:.72rem;text-transform:uppercase;
      letter-spacing:.07em;padding:10px 14px;text-align:left;
      border-bottom:1px solid {BORDER}}}
  td{{padding:9px 14px;border-bottom:1px solid #1e2233;font-size:.86rem}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#1A1F2E}}
  footer{{color:#444;font-size:.75rem;text-align:center;
          margin-top:28px;padding-top:16px;border-top:1px solid {BORDER}}}
  @media(max-width:800px){{.grid-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<h1>⚡ GPT-2 vs n-gram — Process Grammar Results</h1>
<p class="sub">Industrial AI Hackathon · Semiconductor Process Sequences · {ts}</p>

<!-- KPI row -->
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">ID Top-1 Accuracy</div>
    <div class="kpi-value blue">0.807</div>
    <div class="green" style="font-size:.85rem">▲ +0.073 vs n-gram</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">OOD Top-1 (avg 3 families)</div>
    <div class="kpi-value gold">0.497</div>
    <div class="green" style="font-size:.85rem">▲ +0.028 vs n-gram avg</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Task 3 — Anomaly F1</div>
    <div class="kpi-value green">1.000</div>
    <div style="color:{SUB};font-size:.85rem">Symbolic validator — perfect</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">ID Top-5 Accuracy</div>
    <div class="kpi-value blue">1.000</div>
    <div style="color:{SUB};font-size:.85rem">n-gram: 0.994</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">ID MRR</div>
    <div class="kpi-value blue">0.901</div>
    <div class="green" style="font-size:.85rem">▲ +0.049 vs n-gram</div>
  </div>
</div>

<hr>
<div class="sec">§1 — The Headline: Does GPT Beat n-gram?</div>
<div class="grid-2">
{divs["headline"]}
</div>

<hr>
<div class="sec">§2 — Training Convergence</div>
<div class="grid-1">
{divs["curves"]}
</div>

<hr>
<div class="sec">§3 — Detailed Breakdown</div>
<div class="grid-2">
{divs["detail"]}
</div>

<hr>
<div class="sec">§4 — Example Predictions: n-gram vs GPT</div>
<div class="tbl">
{demo}
</div>

<footer>
  GPT-2 small (4.9M params) · from scratch · step-level tokenizer (198 steps) ·
  12 000 training sequences · 20 epochs ·
  <code>scp LOGIN:~/zero_one_hack_0625/results_report.html .</code>
</footer>

<script>{js}</script>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results/*.json found — run the training job first.")
        raise SystemExit(1)

    html = build_html(results)
    out  = "results_report.html"
    Path(out).write_text(html)
    print(f"Report → {out}  ({len(html)//1024} KB)")
    print(f"Copy   → scp LOGIN:~/zero_one_hack_0625/{out} .")

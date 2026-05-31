"""Collect per-model loss curves (saved by train.py into each checkpoint dir) into a
single git-tracked markdown file, so the reproducibility evidence survives even though
outputs/ is gitignored.

Run AFTER training:  python collect_loss_curves.py
Writes: reports/loss_curves.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "reports" / "loss_curves.md"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    curves = sorted(ROOT.glob("outputs/*/loss_curve.json"))
    lines = ["# Training loss curves (Leonardo A100, reproducible run)", ""]
    if not curves:
        lines.append("_No loss_curve.json files found under outputs/ — run training first._")
    for p in curves:
        d = json.loads(p.read_text())
        name = p.parent.name
        lines.append(f"## {name}")
        lines.append(f"- model `{d.get('model')}` · mode `{d.get('mode')}` "
                     f"· holdout `{d.get('holdout') or '-'}` · epochs {d.get('epochs')} "
                     f"· lr {d.get('lr')}")
        lines.append("")
        lines.append("| epoch | loss | ppl |")
        lines.append("|---|---|---|")
        for row in d.get("loss_curve", []):
            lines.append(f"| {row['epoch']} | {row['loss']} | {row['ppl']} |")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"[collect] wrote {OUT}  ({len(curves)} curves)")


if __name__ == "__main__":
    main()

"""Export a standalone XGBoost adapter bundle for rule_probe_pipeline.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solution.eval import rules as R
from solution.models.boosting import XGBoostStepModel


def _normalized(values) -> list[str]:
    return sorted({str(value).strip().upper() for value in values})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Source XGBoost checkpoint directory")
    parser.add_argument("--out_dir", default="solution", help="Directory for teammate adapter artifacts")
    parser.add_argument("--ood_family", default="MOSFET")
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = XGBoostStepModel.load(args.checkpoint)
    if model.estimator is None or model.vocab is None:
        raise RuntimeError("checkpoint is missing its fitted estimator or vocabulary")

    adapter_metadata = {
        **model._class_map(),
        **model._native_metadata(),
        "default_family_id": model.family_to_id["<UNK_FAMILY>"],
        "trained_on": sorted(key for key in model.family_to_id if key != "<UNK_FAMILY>"),
        "ood_family": args.ood_family,
        "rule_sets": {
            "clean": _normalized(R.CLEAN_STEPS),
            "deposit": _normalized(R.DEPOSITION_STEPS),
            "develop": _normalized(R.DEVELOP_STEPS),
            "etch": _normalized(R.ETCH_STEPS),
            "metal_etch": _normalized(R.METAL_ETCH_STEPS),
            "implant": _normalized(R.IMPLANT_STEPS),
            "cmp": _normalized(R.CMP_STEPS),
            "fill": _normalized(R.FILL_STEPS),
            "pad_window": _normalized(R.PAD_WINDOW_STEPS),
            "electrical_test": _normalized(R.ELECTRICAL_TEST_STEPS),
            "passivation": _normalized(R.PASSIVATION_STEPS),
            "backside_metal": _normalized(R.BACKSIDE_METAL_STEPS),
            "oxidation": _normalized(R.OXIDATION_STEPS),
        },
    }

    model.estimator.save_model(str(out_dir / "xgb_model.json"))
    with open(out_dir / "xgb_class_map.json", "w", encoding="utf-8") as f:
        json.dump(adapter_metadata, f, indent=2)
    print(f"exported standalone adapter artifacts -> {out_dir}")


if __name__ == "__main__":
    main()

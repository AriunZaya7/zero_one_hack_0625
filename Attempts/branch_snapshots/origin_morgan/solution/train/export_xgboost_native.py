"""Export existing pickle-based XGBoost checkpoints to native JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from solution.models.boosting import XGBoostStepModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="XGBoost checkpoint directories to upgrade")
    return parser.parse_args()


def main():
    args = get_args()
    for checkpoint in args.checkpoints:
        path = Path(checkpoint)
        model = XGBoostStepModel.load(str(path))
        model.save_native(str(path))
        print(f"exported native XGBoost artifacts -> {path}")


if __name__ == "__main__":
    main()

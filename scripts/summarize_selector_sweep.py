#!/usr/bin/env python3
"""Summarize frozen best-checkpoint metrics from a selector sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

METRICS = (
    "mean_chosen_utility",
    "regret",
    "predicted_acquire_rate",
    "false_call_rate",
    "harmful_call_fraction",
    "acquire_recall",
    "exact_acquire_recall",
)


def summarize(root: Path) -> list[dict[str, Any]]:
    rows = []
    for checkpoint_path in sorted(root.glob("*/best.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint["val_metrics"]
        last_checkpoint = torch.load(
            checkpoint_path.parent / "last.pt", map_location="cpu", weights_only=False
        )
        final_record = last_checkpoint["history"][-1]
        rows.append(
            {
                "run": checkpoint_path.parent.name,
                "epoch": int(checkpoint["epoch"]),
                "eligible": bool(checkpoint.get("checkpoint_eligible", False)),
                **{name: float(metrics[name]) for name in METRICS},
                "final_train_predicted_acquire_rate": float(
                    final_record["train"]["predicted_acquire_rate"]
                ),
                "final_train_harmful_call_fraction": float(
                    final_record["train"]["harmful_call_fraction"]
                ),
                "final_train_exact_acquire_recall": float(
                    final_record["train"]["exact_acquire_recall"]
                ),
                "final_train_utility_regression_loss": float(
                    final_record["train"]["loss_utility_regression"]
                ),
                "calibration_acquire_rate": float(
                    final_record["val"]["calibration_train_acquire_rate"]
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no best.pt checkpoints below {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root)
    payload = json.dumps(rows, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

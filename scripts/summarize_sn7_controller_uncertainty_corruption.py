#!/usr/bin/env python3
"""Summarize train-only uncertainty gates under prior-input corruption."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

SEVERITIES = (0, 4, 8, 16)
QUANTILES = ("05", "15", "30")
SEEDS = (20260730, 20260731, 20260801)
CONTROLLER_KEYS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_tool_calls",
    "tool_call_episode_rate",
    "mean_tool_belief_l1_delta",
    "mean_quality_cost_utility",
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _controller_means(
    run_root: Path,
    severity: int,
    quantile: str,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, float]:
    metrics = [
        _read(
            run_root
            / f"severity{severity}"
            / f"seed{seed}"
            / f"q{quantile}"
            / "closed_loop"
            / "summary.json"
        )["policies"]["edit_utility"]["metrics"]
        for seed in seeds
    ]
    return {
        key: fmean(float(row[key]) for row in metrics)
        for key in CONTROLLER_KEYS
    }


def summarize(run_root: Path) -> dict[str, Any]:
    rows = []
    for severity in SEVERITIES:
        quantiles = {}
        for quantile in QUANTILES:
            q_vs_notool = _read(
                run_root
                / f"severity{severity}"
                / f"q{quantile}_vs_notool.json"
            )
            active_vs_q = _read(
                run_root
                / f"severity{severity}"
                / f"active_vs_q{quantile}.json"
            )
            if q_vs_notool["candidate"] != active_vs_q["baseline"]:
                raise ValueError(
                    f"severity {severity} q{quantile}: uncertainty means differ"
                )
            quantiles[f"q{quantile}"] = {
                "controller": _controller_means(
                    run_root,
                    severity,
                    quantile,
                ),
                "writeback": {
                    "notool": q_vs_notool["baseline"],
                    "uncertainty": q_vs_notool["candidate"],
                    "active": active_vs_q["candidate"],
                },
                "paired_writeback": {
                    "uncertainty_vs_notool": q_vs_notool["paired_delta"],
                    "active_vs_uncertainty": active_vs_q["paired_delta"],
                },
            }
        rows.append({"severity": severity, "quantiles": quantiles})
    return {
        "schema_version": "sn7-controller-uncertainty-corruption-summary-v1",
        "rows": rows,
        "protocol": {
            "split": "val",
            "episodes": 2123,
            "states": 6369,
            "severities": list(SEVERITIES),
            "train_only_gate": True,
            "outcome_labels_used": False,
            "severity_specific_recalibration": False,
            "test_assets_read": False,
        },
    }


def write_csv(payload: dict[str, Any], output: Path) -> None:
    columns = (
        "severity",
        "quantile",
        "call_rate",
        "uncertainty_iou_auc",
        "active_iou_auc",
        "active_minus_uncertainty_iou",
        "uncertainty_false_edit_auc",
        "active_false_edit_auc",
        "active_minus_uncertainty_false_edit",
        "uncertainty_cost_auc",
        "active_cost_auc",
        "active_minus_uncertainty_cost",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in payload["rows"]:
            for label, item in row["quantiles"].items():
                uncertainty = item["writeback"]["uncertainty"]
                active = item["writeback"]["active"]
                paired = item["paired_writeback"]["active_vs_uncertainty"]
                writer.writerow(
                    {
                        "severity": row["severity"],
                        "quantile": label,
                        "call_rate": item["controller"]["tool_call_episode_rate"],
                        "uncertainty_iou_auc": uncertainty["raster_iou_auc"],
                        "active_iou_auc": active["raster_iou_auc"],
                        "active_minus_uncertainty_iou": paired["raster_iou_auc"][
                            "delta"
                        ],
                        "uncertainty_false_edit_auc": uncertainty[
                            "false_edit_auc"
                        ],
                        "active_false_edit_auc": active["false_edit_auc"],
                        "active_minus_uncertainty_false_edit": paired[
                            "false_edit_auc"
                        ]["delta"],
                        "uncertainty_cost_auc": uncertainty["spent_cost_auc"],
                        "active_cost_auc": active["spent_cost_auc"],
                        "active_minus_uncertainty_cost": paired["spent_cost_auc"][
                            "delta"
                        ],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(payload, args.output.with_suffix(".csv"))
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()

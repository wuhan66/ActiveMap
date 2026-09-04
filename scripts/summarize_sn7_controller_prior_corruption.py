#!/usr/bin/env python3
"""Build one auditable table for controller-level prior corruption."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _controller_means(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in payload["per_seed"]:
        rows.setdefault(str(row["variant"]), []).append(row)
    keys = (
        "terminal_accuracy",
        "false_edit_rate",
        "missed_edit_rate",
        "mean_quality_gain",
        "mean_quality_cost_utility",
        "mean_tool_calls",
        "tool_call_episode_rate",
        "mean_tool_belief_l1_delta",
    )
    return {
        variant: {
            key: fmean(float(row[key]) for row in values)
            for key in keys
        }
        for variant, values in rows.items()
    }


def _writeback_variants(
    benefit_notool: dict[str, Any],
    benefit_forced: dict[str, Any],
) -> dict[str, dict[str, float]]:
    if benefit_notool["candidate"] != benefit_forced["candidate"]:
        raise ValueError("benefit writeback means differ across paired comparisons")
    return {
        "notool": benefit_notool["baseline"],
        "forced": benefit_forced["baseline"],
        "benefit": benefit_notool["candidate"],
    }


def summarize(
    run_root: Path,
    clean_controller: Path,
    clean_benefit_notool: Path,
    clean_benefit_forced: Path,
) -> dict[str, Any]:
    inputs = {
        0: {
            "controller": clean_controller,
            "benefit_vs_notool": clean_benefit_notool,
            "benefit_vs_forced": clean_benefit_forced,
        },
        **{
            severity: {
                "controller": run_root / f"severity{severity}" / "controller_summary.json",
                "benefit_vs_notool": (
                    run_root / f"severity{severity}" / "benefit_vs_notool.json"
                ),
                "benefit_vs_forced": (
                    run_root / f"severity{severity}" / "benefit_vs_forced.json"
                ),
            }
            for severity in (4, 8, 16)
        },
    }
    rows = []
    for severity, paths in inputs.items():
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing severity {severity} summaries: {missing}")
        controller = json.loads(paths["controller"].read_text(encoding="utf-8"))
        benefit_notool = json.loads(
            paths["benefit_vs_notool"].read_text(encoding="utf-8")
        )
        benefit_forced = json.loads(
            paths["benefit_vs_forced"].read_text(encoding="utf-8")
        )
        rows.append(
            {
                "severity": severity,
                "controller": _controller_means(controller),
                "writeback": _writeback_variants(
                    benefit_notool,
                    benefit_forced,
                ),
                "paired_writeback": {
                    "benefit_vs_notool": benefit_notool["paired_delta"],
                    "benefit_vs_forced": benefit_forced["paired_delta"],
                },
                "inputs": {name: str(path.resolve()) for name, path in paths.items()},
            }
        )
    clean = rows[0]["writeback"]["benefit"]
    for row in rows:
        current = row["writeback"]["benefit"]
        row["benefit_degradation_vs_clean"] = {
            key: float(current[key]) - float(clean[key])
            for key in (
                "raster_iou_auc",
                "false_edit_auc",
                "missed_edit_auc",
                "spent_cost_auc",
                "episode_utility_v2_balanced_auc",
            )
        }
    return {
        "schema_version": "sn7-controller-prior-corruption-summary-v1",
        "severities": [0, 4, 8, 16],
        "rows": rows,
        "protocol": {
            "split": "val",
            "episodes": 2123,
            "states": 6369,
            "model_input_size": 512,
            "corruption_seed": 20260729,
            "model_input_only": True,
            "frozen_observable_anchor": True,
            "severity_specific_recalibration": False,
            "test_assets_read": False,
        },
    }


def write_csv(payload: dict[str, Any], output: Path) -> None:
    columns = (
        "severity",
        "variant",
        "raster_iou_auc",
        "false_edit_auc",
        "missed_edit_auc",
        "spent_cost_auc",
        "episode_utility_v2_balanced_auc",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in payload["rows"]:
            for variant in ("notool", "forced", "benefit"):
                metrics = row["writeback"][variant]
                writer.writerow(
                    {
                        "severity": row["severity"],
                        "variant": variant,
                        **{key: metrics[key] for key in columns[2:]},
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("clean_controller", type=Path)
    parser.add_argument("clean_benefit_notool", type=Path)
    parser.add_argument("clean_benefit_forced", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = summarize(
        args.run_root,
        args.clean_controller,
        args.clean_benefit_notool,
        args.clean_benefit_forced,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(payload, args.output.with_suffix(".csv"))
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()

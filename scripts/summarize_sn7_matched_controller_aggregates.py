#!/usr/bin/env python3
"""Render the matched three-seed SN7 controller evidence as paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    "react",
    "plan_execute",
    "geommagent_style",
    "sensesearch_style",
    "active_map",
)
METRICS = (
    "raster_iou_auc",
    "raster_iou_gain_auc",
    "episode_utility_v2_balanced_auc",
    "episode_utility_v2_safety_auc",
    "episode_utility_v2_cost_aware_auc",
    "false_edit_auc",
    "missed_edit_auc",
    "spent_cost_auc",
    "vector_replay_iou_auc",
    "vector_delta_topology_valid_auc",
)


def _mean_metrics(
    per_seed: dict[str, Any],
    branch: str,
) -> dict[str, float]:
    return {
        metric: statistics.fmean(
            float(seed_metrics[branch][metric])
            for seed_metrics in per_seed.values()
        )
        for metric in METRICS
    }


def build(input_dir: Path) -> dict[str, Any]:
    payloads = {
        method: json.loads(
            (input_dir / f"{method}_vs_direct.json").read_text(
                encoding="utf-8"
            )
        )
        for method in METHODS
    }
    first = payloads[METHODS[0]]
    rows = [
        {
            "method": "direct_sft",
            "seed_count": first["seed_count"],
            "record_count_per_seed": first["record_count_per_seed"],
            **_mean_metrics(first["per_seed_metrics"], "sft"),
        }
    ]
    for method, payload in payloads.items():
        intervals = payload["candidate_minus_seed_matched_sft"]
        row = {
            "method": method,
            "seed_count": payload["seed_count"],
            "record_count_per_seed": payload["record_count_per_seed"],
            **_mean_metrics(payload["per_seed_metrics"], "candidate"),
        }
        for metric in METRICS:
            interval = intervals[metric]
            row[f"delta_{metric}"] = interval["observed_delta"]
            row[f"ci_low_{metric}"] = interval["ci95_low"]
            row[f"ci_high_{metric}"] = interval["ci95_high"]
        rows.append(row)
    return {
        "schema_version": "sn7-matched-controller-paper-table-v1",
        "split": "val",
        "test_assets_read": False,
        "reference": "direct_sft",
        "methods": list(METHODS),
        "rows": rows,
        "sources": {
            method: str((input_dir / f"{method}_vs_direct.json").resolve())
            for method in METHODS
        },
    }


def _number(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SN7 Matched Three-Seed Controller Table",
        "",
        (
            "Validation-only; 512 matched task-budget records per seed. "
            "Deltas and AOI-bootstrap intervals are relative to Direct SFT."
        ),
        "",
        "| Controller | Raster IoU | Delta (95% CI) | Balanced U | "
        "Delta (95% CI) | Safety U | False edit | Missed edit | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        raster_delta = ""
        utility_delta = ""
        if row["method"] != "direct_sft":
            raster_delta = (
                f"{_number(row['delta_raster_iou_auc'])} "
                f"[{_number(row['ci_low_raster_iou_auc'])}, "
                f"{_number(row['ci_high_raster_iou_auc'])}]"
            )
            utility_delta = (
                f"{_number(row['delta_episode_utility_v2_balanced_auc'])} "
                f"[{_number(row['ci_low_episode_utility_v2_balanced_auc'])}, "
                f"{_number(row['ci_high_episode_utility_v2_balanced_auc'])}]"
            )
        lines.append(
            f"| {row['method']} | {_number(row['raster_iou_auc'])} | "
            f"{raster_delta} | "
            f"{_number(row['episode_utility_v2_balanced_auc'])} | "
            f"{utility_delta} | "
            f"{_number(row['episode_utility_v2_safety_auc'])} | "
            f"{_number(row['false_edit_auc'])} | "
            f"{_number(row['missed_edit_auc'])} | "
            f"{_number(row['spent_cost_auc'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    payload = build(args.input_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = payload["rows"]
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "table.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {"rows": len(rows), "output": str(args.output_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build paper tables from the balanced-tool multi-seed aggregate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONTROLLER_METRICS = (
    "episode_utility_v2_balanced_auc",
    "quality_cost_utility_auc",
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_cost",
    "mean_acquisitions",
    "mean_tool_calls",
    "mean_tool_belief_l1_delta",
    "mean_tool_action_flips",
)
WRITEBACK_METRICS = (
    "raster_iou_gain_auc",
    "episode_utility_v2_balanced_auc",
    "false_edit_auc",
    "missed_edit_auc",
    "spent_cost_auc",
    "vector_delta_topology_valid_auc",
    "component_count_absolute_error_auc",
)
LOWER_IS_BETTER = {
    "false_edit_rate",
    "missed_edit_rate",
    "mean_cost",
    "false_edit_auc",
    "missed_edit_auc",
    "spent_cost_auc",
    "component_count_absolute_error_auc",
}


def _load(path: Path, expected_schema: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected schema in {path}: {data.get('schema_version')}")
    if data.get("test_assets_read") is not False and data.get("protocol", {}).get(
        "test_assets_read"
    ) is not False:
        raise ValueError(f"table input is not audited validation evidence: {path}")
    return data


def _interval_text(interval: dict[str, float], *, key: str) -> str:
    return (
        f"{interval[key]:+.5f} "
        f"[{interval['ci95_low']:+.5f}, {interval['ci95_high']:+.5f}]"
    )


def _direction(metric: str) -> str:
    return "lower" if metric in LOWER_IS_BETTER else "higher"


def _controller_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline, comparison in data["comparisons"].items():
        for metric in CONTROLLER_METRICS:
            interval = comparison["paired_delta"].get(metric)
            if interval is None:
                continue
            rows.append(
                {
                    "comparison": f"{data['candidate']} - {baseline}",
                    "metric": metric,
                    "direction": _direction(metric),
                    "delta": interval["delta"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                    "strictly_better": (
                        interval["ci95_high"] < 0
                        if metric in LOWER_IS_BETTER
                        else interval["ci95_low"] > 0
                    ),
                }
            )
    return rows


def _writeback_rows(
    datasets: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for comparison_name, data in datasets:
        for metric in WRITEBACK_METRICS:
            interval = data["candidate_minus_seed_matched_sft"].get(metric)
            if interval is None:
                continue
            rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "direction": _direction(metric),
                    "delta": interval["observed_delta"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                    "strictly_better": (
                        interval["ci95_high"] < 0
                        if metric in LOWER_IS_BETTER
                        else interval["ci95_low"] > 0
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(
    title: str, rows: list[dict[str, Any]], selected_metrics: tuple[str, ...]
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Comparison | Metric | Direction | Delta (95% CI) | Strict win |",
        "|---|---|:---:|---:|:---:|",
    ]
    for row in rows:
        if row["metric"] not in selected_metrics:
            continue
        interval = {
            "delta": row["delta"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
        }
        lines.append(
            "| {comparison} | {metric} | {direction} | {interval} | {win} |".format(
                comparison=row["comparison"],
                metric=row["metric"],
                direction=row["direction"],
                interval=_interval_text(interval, key="delta"),
                win="PASS" if row["strictly_better"] else "-",
            )
        )
    lines.append("")
    return lines


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def _tex_rows(
    rows: list[dict[str, Any]], selected_metrics: tuple[str, ...]
) -> list[str]:
    lines = []
    for row in rows:
        if row["metric"] not in selected_metrics:
            continue
        interval = (
            f"{row['delta']:+.5f} "
            f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]"
        )
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                _tex_escape(row["comparison"]),
                _tex_escape(row["metric"]),
                _tex_escape(row["direction"]),
                r"\textbf{" + interval + "}" if row["strictly_better"] else interval,
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", type=Path)
    parser.add_argument("raw_vs_sft", type=Path)
    parser.add_argument("safe_vs_sft", type=Path)
    parser.add_argument("safe_vs_raw", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    controller = _load(
        args.controller, "activemap-agent-three-seed-bootstrap-v2"
    )
    writebacks = [
        (
            "Raw Tool-to-Belief - SFT",
            _load(args.raw_vs_sft, "agent-writeback-seed-matched-aggregate-v1"),
        ),
        (
            "Safe Delta - SFT",
            _load(args.safe_vs_sft, "agent-writeback-seed-matched-aggregate-v1"),
        ),
        (
            "Safe Delta - Raw Tool-to-Belief",
            _load(args.safe_vs_raw, "agent-writeback-seed-matched-aggregate-v1"),
        ),
    ]
    seed_sets = [tuple(controller["protocol"]["model_seeds"])]
    seed_sets.extend(tuple(data["model_seeds"]) for _, data in writebacks)
    if len(set(seed_sets)) != 1:
        raise ValueError(f"aggregate inputs use different model seeds: {seed_sets}")

    controller_rows = _controller_rows(controller)
    writeback_rows = _writeback_rows(writebacks)
    if not controller_rows or not writeback_rows:
        raise ValueError("aggregate inputs do not contain table metrics")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "controller.csv", controller_rows)
    _write_csv(args.output_dir / "writeback.csv", writeback_rows)

    lines = [
        "# MUNO21 Balanced Tool Multi-Seed Results",
        "",
        f"Model seeds: `{', '.join(str(seed) for seed in seed_sets[0])}`.",
        "Validation-only; frozen test assets were not read.",
        "",
    ]
    lines.extend(
        _markdown_table(
            "Structured Controller",
            controller_rows,
            (
                "episode_utility_v2_balanced_auc",
                "quality_cost_utility_auc",
                "terminal_accuracy",
                "false_edit_rate",
                "mean_cost",
                "mean_tool_calls",
                "mean_tool_belief_l1_delta",
            ),
        )
    )
    lines.extend(
        _markdown_table(
            "Executable Writeback",
            writeback_rows,
            (
                "raster_iou_gain_auc",
                "episode_utility_v2_balanced_auc",
                "false_edit_auc",
                "missed_edit_auc",
                "spent_cost_auc",
                "vector_delta_topology_valid_auc",
            ),
        )
    )
    (args.output_dir / "table.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    controller_selected = (
        "episode_utility_v2_balanced_auc",
        "quality_cost_utility_auc",
        "terminal_accuracy",
        "false_edit_rate",
        "mean_cost",
        "mean_tool_calls",
        "mean_tool_belief_l1_delta",
    )
    writeback_selected = (
        "raster_iou_gain_auc",
        "episode_utility_v2_balanced_auc",
        "false_edit_auc",
        "missed_edit_auc",
        "spent_cost_auc",
        "vector_delta_topology_valid_auc",
    )
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{MUNO21 balanced-tool validation results across four model seeds. Bold intervals denote strict paired improvements in the favorable direction.}",
        r"\label{tab:muno21-balanced-tool}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Comparison & Metric & Better & Delta (95\% CI) \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Structured controller}} \\",
        *_tex_rows(controller_rows, controller_selected),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Executable writeback}} \\",
        *_tex_rows(writeback_rows, writeback_selected),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    (args.output_dir / "table.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )

    manifest = {
        "schema_version": "muno21-balanced-tool-multiseed-table-v1",
        "model_seeds": list(seed_sets[0]),
        "controller_rows": controller_rows,
        "writeback_rows": writeback_rows,
        "split": "val",
        "test_assets_read": False,
    }
    (args.output_dir / "table.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

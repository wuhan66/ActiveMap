#!/usr/bin/env python3
"""Summarize the current MUNO21 gate and SN7 controller round."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["sample_count"]) for row in rows)
    return sum(float(row[key]) * int(row["sample_count"]) for row in rows) / total


def _muno_rows(storage: Path) -> list[dict[str, Any]]:
    root = (
        storage
        / "artifacts/paper_rollouts"
        / "agent_v12_proactive_train_counterfactual_features_v2"
    )
    output = []
    for threshold in ("0p03", "0p05", "0p07", "0p09"):
        summary = root / f"threshold_{threshold}" / "summary.json"
        if not summary.is_file():
            output.append(
                {
                    "dataset": "MUNO21",
                    "family": "counterfactual_gate",
                    "method": "all",
                    "seed": 20260822,
                    "setting": threshold.replace("p", "."),
                    "status": "running",
                    "sample_count": None,
                }
            )
            continue
        payload = _read(summary)
        for method in (
            "edit_conditioned_proactive_tools",
            "qwen3_4b_sft_calibrated_tool_to_belief",
        ):
            rows = [row for row in payload["results"] if row["method"] == method]
            output.append(
                {
                    "dataset": "MUNO21",
                    "family": "counterfactual_gate",
                    "method": method,
                    "seed": 20260822,
                    "setting": threshold.replace("p", "."),
                    "status": "complete",
                    "sample_count": sum(int(row["sample_count"]) for row in rows),
                    "terminal_accuracy": statistics.fmean(
                        float(row["terminal_accuracy"]) for row in rows
                    ),
                    "false_edit_rate": statistics.fmean(
                        float(row["false_edit_rate"]) for row in rows
                    ),
                    "mean_tool_calls": statistics.fmean(
                        float(row["mean_tool_calls"]) for row in rows
                    ),
                    "belief_l1_delta": statistics.fmean(
                        float(row["mean_tool_belief_l1_delta"]) for row in rows
                    ),
                    "tool_action_flip_rate": statistics.fmean(
                        float(row["tool_action_flip_rate"]) for row in rows
                    ),
                    "quality_cost_utility": statistics.fmean(
                        float(row["mean_quality_cost_utility"]) for row in rows
                    ),
                }
            )
    return output


def _sn7_rows(storage: Path) -> list[dict[str, Any]]:
    root = storage / "runs/sn7_active_catalog"
    methods = (
        ("direct_sft", "closed_loop_sft_seed{seed}_n512_v2"),
        ("react", "react_style_qwen_seed{seed}_n512_v2"),
        ("plan_execute", "plan_execute_qwen_seed{seed}_n512_v2"),
        (
            "geommagent_style",
            "geommagent_style_qwen_seed{seed}_full_v3",
        ),
        (
            "sensesearch_style",
            "sensesearch_style_qwen_seed{seed}_full_v3",
        ),
    )
    output = []
    for method, run_pattern in methods:
        for seed in (20260717, 20260718, 20260719):
            run_name = run_pattern.format(seed=seed)
            run = root / run_name
            summary = run / "evaluation/summary.json"
            row: dict[str, Any] = {
                "dataset": "SN7",
                "family": "controller_baseline",
                "method": method,
                "seed": seed,
                "setting": "val_n512",
                "status": "missing",
                "sample_count": None,
            }
            if summary.is_file():
                payload = _read(summary)
                metrics = payload["metrics"]
                row.update(
                    {
                        "status": "rollout_complete",
                        "sample_count": payload["sample_count"],
                        "terminal_accuracy": metrics["terminal_accuracy"],
                        "false_edit_rate": metrics["false_edit_rate"],
                        "mean_tool_calls": metrics["mean_tool_calls"],
                        "belief_l1_delta": metrics["mean_tool_belief_l1_delta"],
                        "quality_cost_utility": metrics[
                            "mean_quality_cost_utility"
                        ],
                        "valid_action_rate": metrics["valid_action_rate"],
                        "fallback_episode_rate": metrics["fallback_episode_rate"],
                    }
                )
            writeback = root / f"{run_name}_writeback"
            writeback_summary = writeback / "evaluation/summary.json"
            if writeback_summary.is_file():
                payload = _read(writeback_summary)
                budgets = payload["budgets"]
                row.update(
                    {
                        "status": "writeback_complete",
                        "raster_iou_gain": _weighted(
                            budgets, "mean_raster_iou_gain"
                        ),
                        "balanced_utility": _weighted(
                            budgets, "mean_episode_utility_v2_balanced"
                        ),
                        "component_count_error": _weighted(
                            budgets, "mean_component_count_absolute_error"
                        ),
                        "topology_valid_rate": _weighted(
                            budgets, "vector_delta_topology_valid_rate"
                        ),
                    }
                )
            output.append(row)
    return output


def _active_map_rows(storage: Path) -> list[dict[str, Any]]:
    root = (
        storage
        / "runs/sn7_active_catalog"
        / "final_tool_belief_causal_v1"
    )
    paired_path = root / "paired_tool_belief_causal.json"
    promotion_path = root / "promotion.json"
    if not paired_path.is_file():
        return []
    paired = _read(paired_path)
    promoted = (
        _read(promotion_path).get("promote", False)
        if promotion_path.is_file()
        else False
    )
    writeback_paths = {
        "no_tool": root / "writeback_selective_vs_no_tool.json",
        "forced": root / "writeback_selective_vs_forced_recurrent.json",
        "selective_frozen_prior": (
            root / "writeback_selective_vs_selective_frozen_prior.json"
        ),
    }
    writeback_metrics: dict[str, dict[str, Any]] = {}
    selective_writeback: dict[str, Any] = {}
    for method, path in writeback_paths.items():
        if not path.is_file():
            continue
        payload = _read(path)
        writeback_metrics[method] = payload["baseline"]
        selective_writeback = payload["candidate"]

    output = []
    for method in (
        "no_tool",
        "forced",
        "selective_frozen_prior",
        "selective",
    ):
        metrics = paired["metrics"][method]
        writeback = (
            selective_writeback
            if method == "selective"
            else writeback_metrics.get(method, {})
        )
        output.append(
            {
                "dataset": "SN7",
                "family": "qwen_active_catalog_ablation",
                "method": method,
                "seed": "3-seed",
                "setting": "val_n512",
                "status": (
                    "promotion_passed"
                    if method == "selective" and promoted
                    else "promotion_failed"
                    if method == "selective"
                    else "complete"
                ),
                "sample_count": paired["record_count_per_seed"]
                * paired["seed_count"],
                "terminal_accuracy": metrics["terminal_accuracy"],
                "false_edit_rate": metrics["false_edit_rate"],
                "mean_tool_calls": metrics["mean_tool_calls"],
                "belief_l1_delta": metrics["mean_tool_belief_l1_delta"],
                "quality_cost_utility": metrics[
                    "mean_quality_cost_utility"
                ],
                "valid_action_rate": metrics["valid_action_rate"],
                "fallback_episode_rate": metrics["fallback_episode_rate"],
                "raster_iou_gain": writeback.get("raster_iou_gain_auc"),
                "balanced_utility": writeback.get(
                    "episode_utility_v2_balanced_auc"
                ),
                "component_count_error": writeback.get(
                    "component_count_absolute_error_auc"
                ),
                "topology_valid_rate": writeback.get(
                    "vector_delta_topology_valid_auc"
                ),
            }
        )
    return output


def _markdown(rows: list[dict[str, Any]]) -> str:
    def value(row: dict[str, Any], key: str) -> str:
        return "" if row.get(key) is None else f"{float(row[key]):.6f}"

    lines = [
        "# Current Counterfactual and Controller Round",
        "",
        "Validation-only snapshot. Frozen test remains locked.",
        "",
        "| Dataset | Family | Method | Seed | Setting | Status | Accuracy | "
        "False edit | Calls | Belief delta | Raster-IoU gain | Balanced utility |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['method']} | "
            f"{row['seed']} | {row['setting']} | {row['status']} | "
            f"{value(row, 'terminal_accuracy')} | "
            f"{value(row, 'false_edit_rate')} | "
            f"{value(row, 'mean_tool_calls')} | "
            f"{value(row, 'belief_l1_delta')} | "
            f"{value(row, 'raster_iou_gain')} | "
            f"{value(row, 'balanced_utility')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = (
        _muno_rows(args.storage_root)
        + _sn7_rows(args.storage_root)
        + _active_map_rows(args.storage_root)
    )
    args.output_dir.mkdir(parents=True)
    payload = {
        "schema_version": "activemap-current-round-summary-v1",
        "rows": rows,
        "test_assets_read": False,
    }
    (args.output_dir / "table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "table.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

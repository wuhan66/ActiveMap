#!/usr/bin/env python3
"""Assess the strict MUNO21 balanced-tool multi-seed promotion gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CONTROLLER_SCHEMA = "activemap-agent-three-seed-bootstrap-v2"
WRITEBACK_SCHEMA = "agent-writeback-seed-matched-aggregate-v1"


def _read(path: Path, schema: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != schema:
        raise ValueError(f"unexpected schema in {path}: {data.get('schema_version')}")
    protocol_no_test = data.get("protocol", {}).get("test_assets_read") is False
    top_level_no_test = data.get("test_assets_read") is False
    if not (protocol_no_test or top_level_no_test):
        raise ValueError(f"input is not audited validation evidence: {path}")
    return data


def _controller_interval(
    controller: dict[str, Any], baseline: str, metric: str
) -> dict[str, float]:
    try:
        return controller["comparisons"][baseline]["paired_delta"][metric]
    except KeyError as error:
        raise ValueError(f"missing controller interval {baseline}/{metric}") from error


def _writeback_interval(
    writeback: dict[str, Any], metric: str
) -> dict[str, float]:
    try:
        return writeback["candidate_minus_seed_matched_sft"][metric]
    except KeyError as error:
        raise ValueError(f"missing writeback interval {metric}") from error


def _primary_metric(comparison: dict[str, Any]) -> str:
    metrics = comparison["paired_delta"]
    for name in (
        "episode_utility_v2_balanced_auc",
        "quality_cost_utility_auc",
    ):
        if name in metrics:
            return name
    raise ValueError("controller comparison has no supported primary utility")


def _mean_candidate_metric(
    controller: dict[str, Any], baseline: str, metric: str
) -> float:
    values = []
    for row in controller["comparisons"][baseline]["per_seed"]:
        if metric not in row["candidate"]:
            raise ValueError(f"candidate metric missing: {baseline}/{metric}")
        values.append(float(row["candidate"][metric]))
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", type=Path)
    parser.add_argument("safe_vs_sft", type=Path)
    parser.add_argument("safe_vs_raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--max-false-edit-delta", type=float, default=0.01)
    args = parser.parse_args()

    controller = _read(args.controller, CONTROLLER_SCHEMA)
    safe_vs_sft = _read(args.safe_vs_sft, WRITEBACK_SCHEMA)
    safe_vs_raw = _read(args.safe_vs_raw, WRITEBACK_SCHEMA)
    seeds = tuple(int(seed) for seed in controller["protocol"]["model_seeds"])
    if tuple(safe_vs_sft["model_seeds"]) != seeds or tuple(
        safe_vs_raw["model_seeds"]
    ) != seeds:
        raise ValueError("controller and writeback aggregates use different seeds")
    if len(args.run_dir) != len(seeds):
        raise ValueError("one run directory is required per model seed")

    selections = {}
    for run_dir in args.run_dir:
        seed = int(run_dir.name.rsplit("seed", 1)[-1])
        decision = json.loads(
            (
                run_dir
                / "evaluation/selection/static_checkpoint_decision.json"
            ).read_text(encoding="utf-8")
        )
        if decision.get("protocol", {}).get("test_assets_read") is not False:
            raise ValueError(f"selection is not validation-only: {run_dir}")
        selections[seed] = {
            "passed": decision.get("selection_passed") is True,
            "selected_checkpoint": decision.get("selected_checkpoint"),
        }
    if tuple(sorted(selections)) != tuple(sorted(seeds)):
        raise ValueError("run-directory seeds do not match aggregates")

    edit_baseline = "edit_conditioned_selector"
    no_belief_baseline = "qwen3_4b_sft_tools_no_belief"
    edit_comparison = controller["comparisons"][edit_baseline]
    no_belief_comparison = controller["comparisons"][no_belief_baseline]
    edit_primary_name = _primary_metric(edit_comparison)
    no_belief_primary_name = _primary_metric(no_belief_comparison)
    edit_primary = _controller_interval(
        controller, edit_baseline, edit_primary_name
    )
    no_belief_primary = _controller_interval(
        controller, no_belief_baseline, no_belief_primary_name
    )
    edit_false = _controller_interval(
        controller, edit_baseline, "false_edit_rate"
    )
    mean_tool_calls = _mean_candidate_metric(
        controller, no_belief_baseline, "mean_tool_calls"
    )
    mean_belief_delta = _mean_candidate_metric(
        controller, no_belief_baseline, "mean_tool_belief_l1_delta"
    )
    mean_action_flips = _mean_candidate_metric(
        controller, no_belief_baseline, "mean_tool_action_flips"
    )

    safe_raster = _writeback_interval(safe_vs_sft, "raster_iou_gain_auc")
    safe_false = _writeback_interval(safe_vs_sft, "false_edit_auc")
    safe_missed = _writeback_interval(safe_vs_sft, "missed_edit_auc")
    filter_raster = _writeback_interval(safe_vs_raw, "raster_iou_gain_auc")

    checks = {
        "all_model_seeds_pass_checkpoint_gate": all(
            row["passed"] and row["selected_checkpoint"]
            for row in selections.values()
        ),
        "agent_beats_edit_selector_primary_ci": edit_primary["ci95_low"] > 0,
        "agent_false_edit_delta_within_margin": (
            edit_false["ci95_high"] <= args.max_false_edit_delta
        ),
        "belief_beats_no_belief_primary_ci": no_belief_primary["ci95_low"] > 0,
        "nonzero_tool_calls": mean_tool_calls > 0,
        "nonzero_belief_revision": mean_belief_delta > 0,
        "nonzero_tool_action_flips": mean_action_flips > 0,
        "safe_writeback_beats_sft_raster_ci": safe_raster["ci95_low"] > 0,
        "safe_writeback_no_false_edit_regression": safe_false["ci95_high"] <= 0,
        "safe_writeback_no_missed_edit_regression": safe_missed["ci95_high"] <= 0,
        "safe_delta_beats_raw_belief_raster_ci": filter_raster["ci95_low"] > 0,
    }
    groups = {
        "checkpoint_stability": (
            "all_model_seeds_pass_checkpoint_gate",
        ),
        "controller_claim": (
            "agent_beats_edit_selector_primary_ci",
            "agent_false_edit_delta_within_margin",
        ),
        "tool_belief_causality": (
            "belief_beats_no_belief_primary_ci",
            "nonzero_tool_calls",
            "nonzero_belief_revision",
            "nonzero_tool_action_flips",
        ),
        "safe_writeback_claim": (
            "safe_writeback_beats_sft_raster_ci",
            "safe_writeback_no_false_edit_regression",
            "safe_writeback_no_missed_edit_regression",
            "safe_delta_beats_raw_belief_raster_ci",
        ),
    }
    group_pass = {
        group: all(checks[name] for name in names)
        for group, names in groups.items()
    }
    result = {
        "schema_version": "muno21-balanced-tool-multiseed-promotion-v1",
        "status": "pass" if all(group_pass.values()) else "fail",
        "model_seeds": list(seeds),
        "selected_checkpoints": selections,
        "primary_metrics": {
            "edit_selector_comparison": edit_primary_name,
            "no_belief_comparison": no_belief_primary_name,
        },
        "mechanism_means": {
            "mean_tool_calls": mean_tool_calls,
            "mean_tool_belief_l1_delta": mean_belief_delta,
            "mean_tool_action_flips": mean_action_flips,
        },
        "checks": checks,
        "groups": group_pass,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "split": "val",
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the predeclared closed-loop and executable-writeback promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(
    closed_loop: dict[str, Any],
    writeback: dict[str, Any],
    *,
    candidate: str,
    baseline: str,
    false_edit_margin: float = 0.02,
    topology_margin: float = 0.01,
    utility_margin: float = 0.01,
) -> dict[str, Any]:
    comparison_key = f"{candidate}_minus_{baseline}"
    try:
        loop_intervals = closed_loop["paired_aoi_comparisons"][comparison_key][
            "intervals"
        ]
    except KeyError as error:
        raise ValueError(f"missing closed-loop comparison: {comparison_key}") from error
    if writeback.get("group_key") != "aoi_id":
        raise ValueError("writeback promotion requires AOI-grouped bootstrap")
    writeback_intervals = writeback.get("paired_delta", {})
    required_writeback = (
        "raster_iou_gain_auc",
        "vector_replay_iou_auc",
        "vector_delta_topology_valid_auc",
        "episode_utility_v2_balanced_auc",
        "episode_utility_v2_safety_auc",
        "episode_utility_v2_cost_aware_auc",
        "false_edit_auc",
    )
    missing = [name for name in required_writeback if name not in writeback_intervals]
    if missing:
        raise ValueError(f"missing writeback intervals: {missing}")

    checks = {
        "proxy_quality_cost_utility_significant": (
            loop_intervals["mean_quality_cost_utility"]["ci95_low"] > 0.0
        ),
        "executable_utility_v2_balanced_significant": (
            writeback_intervals["episode_utility_v2_balanced_auc"]["ci95_low"]
            > 0.0
        ),
        "executable_utility_v2_safety_noninferior": (
            writeback_intervals["episode_utility_v2_safety_auc"]["ci95_low"]
            >= -utility_margin
        ),
        "executable_utility_v2_cost_aware_noninferior": (
            writeback_intervals["episode_utility_v2_cost_aware_auc"]["ci95_low"]
            >= -utility_margin
        ),
        "executable_false_edit_noninferior": (
            writeback_intervals["false_edit_auc"]["ci95_high"]
            <= false_edit_margin
        ),
        "proxy_terminal_accuracy_noninferior": (
            loop_intervals["terminal_accuracy"]["ci95_low"] >= -0.01
        ),
        "raster_iou_gain_significant": (
            writeback_intervals["raster_iou_gain_auc"]["ci95_low"] > 0.0
        ),
        "vector_replay_improves_observed": (
            writeback_intervals["vector_replay_iou_auc"]["delta"] > 0.0
        ),
        "topology_noninferior": (
            writeback_intervals["vector_delta_topology_valid_auc"]["ci95_low"]
            >= -topology_margin
        ),
    }
    required = (
        "executable_utility_v2_balanced_significant",
        "executable_utility_v2_safety_noninferior",
        "executable_utility_v2_cost_aware_noninferior",
        "executable_false_edit_noninferior",
        "raster_iou_gain_significant",
        "topology_noninferior",
    )
    return {
        "schema_version": "active-catalog-closed-loop-promotion-v2",
        "candidate": candidate,
        "baseline": baseline,
        "promote": all(checks[name] for name in required),
        "checks": checks,
        "required_checks": list(required),
        "diagnostic_checks": ["vector_replay_improves_observed"],
        "margins": {
            "false_edit_rate": false_edit_margin,
            "topology_valid_rate": topology_margin,
            "utility_v2_robustness": utility_margin,
        },
        "claim_boundary": (
            "Promotion supports recurrent evidence acquisition under executable "
            "episode-utility-v2 and vector writeback, not explicit geospatial "
            "tool-use gains."
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("closed_loop_comparison", type=Path)
    parser.add_argument("writeback_comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", default="qwen")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--false-edit-margin", type=float, default=0.02)
    parser.add_argument("--topology-margin", type=float, default=0.01)
    parser.add_argument("--utility-margin", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = assess(
        json.loads(args.closed_loop_comparison.read_text(encoding="utf-8")),
        json.loads(args.writeback_comparison.read_text(encoding="utf-8")),
        candidate=args.candidate,
        baseline=args.baseline,
        false_edit_margin=args.false_edit_margin,
        topology_margin=args.topology_margin,
        utility_margin=args.utility_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

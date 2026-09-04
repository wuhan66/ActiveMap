#!/usr/bin/env python3
"""Promote selective tool use only after executable-map paired evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(
    tool_branch: dict[str, Any],
    versus_no_tool: dict[str, Any],
    versus_forced: dict[str, Any],
    *,
    replay_margin: float = 0.01,
    topology_margin: float = 0.01,
    false_edit_margin: float = 0.02,
    forced_map_margin: float = 0.001,
    min_seeds: int = 1,
    allow_failed_upstream: bool = False,
) -> dict[str, Any]:
    if tool_branch.get("schema_version") != "active-catalog-tool-branch-promotion-v1":
        raise ValueError("unexpected tool-branch promotion schema")
    branch_promoted = bool(tool_branch.get("promote"))
    if not branch_promoted and not allow_failed_upstream:
        raise ValueError("closed-loop quality-cost-safety gate did not pass")

    def intervals(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("group_key") != "aoi_id":
            raise ValueError("tool writeback promotion requires AOI-grouped bootstrap")
        values = payload.get("paired_delta", {})
        required = {
            "raster_iou_gain_auc",
            "vector_replay_iou_auc",
            "vector_delta_topology_valid_auc",
            "episode_utility_v2_balanced_auc",
            "episode_utility_v2_cost_aware_auc",
            "false_edit_auc",
            "spent_cost_auc",
        }
        if missing := sorted(required - values.keys()):
            raise ValueError(f"missing writeback intervals: {missing}")
        return values

    no_tool = intervals(versus_no_tool)
    forced = intervals(versus_forced)
    provenance = {
        bool(tool_branch.get("test_assets_read")),
        bool(versus_no_tool.get("test_assets_read")),
        bool(versus_forced.get("test_assets_read")),
    }
    if len(provenance) != 1:
        raise ValueError("tool branch and writeback evidence use mixed validation/test provenance")
    test_assets_read = provenance.pop()
    no_tool_seed_count = int(versus_no_tool.get("seed_count", 1))
    forced_seed_count = int(versus_forced.get("seed_count", 1))
    checks = {
        "closed_loop_tool_gate_passed": branch_promoted,
        "required_seed_count": (
            no_tool_seed_count == forced_seed_count and no_tool_seed_count >= min_seeds
        ),
        "map_gain_vs_no_tool": no_tool["raster_iou_gain_auc"]["ci95_low"] > 0.0,
        "map_noninferior_vs_forced": (
            forced["raster_iou_gain_auc"]["ci95_low"] >= -forced_map_margin
        ),
        "utility_v2_gain_vs_no_tool": (
            no_tool["episode_utility_v2_balanced_auc"]["ci95_low"] > 0.0
        ),
        "cost_aware_utility_gain_vs_forced": (
            forced["episode_utility_v2_cost_aware_auc"]["ci95_low"] > 0.0
        ),
        "cost_reduction_vs_forced": forced["spent_cost_auc"]["ci95_high"] < 0.0,
        "false_edit_noninferior_vs_no_tool": (
            no_tool["false_edit_auc"]["ci95_high"] <= false_edit_margin
        ),
        "false_edit_noninferior_vs_forced": (
            forced["false_edit_auc"]["ci95_high"] <= false_edit_margin
        ),
        "replay_noninferior_vs_no_tool": (
            no_tool["vector_replay_iou_auc"]["ci95_low"] >= -replay_margin
        ),
        "replay_noninferior_vs_forced": (
            forced["vector_replay_iou_auc"]["ci95_low"] >= -replay_margin
        ),
        "topology_noninferior_vs_no_tool": (
            no_tool["vector_delta_topology_valid_auc"]["ci95_low"]
            >= -topology_margin
        ),
        "topology_noninferior_vs_forced": (
            forced["vector_delta_topology_valid_auc"]["ci95_low"]
            >= -topology_margin
        ),
    }
    return {
        "schema_version": "active-catalog-tool-writeback-promotion-v2",
        "promote": all(checks.values()),
        "checks": checks,
        "margins": {
            "vector_replay": replay_margin,
            "topology": topology_margin,
            "false_edit": false_edit_margin,
            "forced_map": forced_map_margin,
        },
        "minimum_seed_count": min_seeds,
        "claim_boundary": (
            "Three-seed frozen-test evidence for selective grounded tool use in executable "
            "map writeback."
            if test_assets_read and min_seeds >= 3
            else "Validation-only evidence for selective grounded tool use in executable map "
            "writeback; multi-seed frozen-test evidence is still required."
        ),
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool_branch_promotion", type=Path)
    parser.add_argument("versus_no_tool", type=Path)
    parser.add_argument("versus_forced", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-margin", type=float, default=0.01)
    parser.add_argument("--topology-margin", type=float, default=0.01)
    parser.add_argument("--false-edit-margin", type=float, default=0.02)
    parser.add_argument("--forced-map-margin", type=float, default=0.001)
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--record-failed-upstream", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = assess(
        json.loads(args.tool_branch_promotion.read_text(encoding="utf-8")),
        json.loads(args.versus_no_tool.read_text(encoding="utf-8")),
        json.loads(args.versus_forced.read_text(encoding="utf-8")),
        replay_margin=args.replay_margin,
        topology_margin=args.topology_margin,
        false_edit_margin=args.false_edit_margin,
        forced_map_margin=args.forced_map_margin,
        min_seeds=args.min_seeds,
        allow_failed_upstream=args.record_failed_upstream,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

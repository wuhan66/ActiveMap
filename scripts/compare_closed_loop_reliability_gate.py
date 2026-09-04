#!/usr/bin/env python3
"""Paired closed-loop reliability-gate ablation with a fixed Tool-Need gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_active_catalog_closed_loop import load_rows, paired_aoi_bootstrap


FIXED_PROTOCOL_KEYS = (
    "model_selected_state_transitions",
    "oracle_next_state_replay",
    "tool_mode",
    "selective_tool_calling",
    "max_candidates",
    "max_acquisitions",
    "stochastic_policy_sampling",
    "temperature",
    "top_p",
)
FIXED_SOURCE_KEYS = ("model", "adapter", "states", "episodes", "tool_gate")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_controlled_difference(
    ungated: dict[str, Any], gated: dict[str, Any]
) -> dict[str, Any]:
    if ungated.get("schema_version") != "active-catalog-closed-loop-evaluation-v1":
        raise ValueError("unexpected ungated summary schema")
    if gated.get("schema_version") != "active-catalog-closed-loop-evaluation-v1":
        raise ValueError("unexpected gated summary schema")
    left_protocol, right_protocol = ungated["protocol"], gated["protocol"]
    mismatches = [
        key for key in FIXED_PROTOCOL_KEYS if left_protocol.get(key) != right_protocol.get(key)
    ]
    if mismatches:
        raise ValueError(f"closed-loop protocol mismatch: {mismatches}")
    if left_protocol.get("tool_mode") != "selective":
        raise ValueError("reliability ablation requires selective tool mode")
    if left_protocol.get("tool_belief_reliability_gate") is not False:
        raise ValueError("ungated summary does not identify an ungated updater")
    if right_protocol.get("tool_belief_reliability_gate") is not True:
        raise ValueError("gated summary does not identify a gated updater")
    left_sources, right_sources = ungated["sources"], gated["sources"]
    source_mismatches = [
        key for key in FIXED_SOURCE_KEYS if left_sources.get(key) != right_sources.get(key)
    ]
    if source_mismatches:
        raise ValueError(f"fixed source mismatch: {source_mismatches}")
    left_updater = left_sources.get("tool_belief_checkpoint") or {}
    right_updater = right_sources.get("tool_belief_checkpoint") or {}
    if not left_updater.get("sha256") or not right_updater.get("sha256"):
        raise ValueError("Tool-Belief checkpoint hashes are required")
    if left_updater["sha256"] == right_updater["sha256"]:
        raise ValueError("gated and ungated runs use the same updater checkpoint")
    return {
        "controlled_difference": "tool_belief_reliability_gate",
        "fixed_tool_gate_sha256": left_sources["tool_gate"]["sha256"],
        "ungated_updater_sha256": left_updater["sha256"],
        "gated_updater_sha256": right_updater["sha256"],
    }


def compare(
    ungated_summary: dict[str, Any],
    gated_summary: dict[str, Any],
    ungated_rows: dict[tuple[str, float], dict[str, Any]],
    gated_rows: dict[tuple[str, float], dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    accuracy_margin: float = 0.005,
    false_edit_margin: float = 0.005,
) -> dict[str, Any]:
    control = verify_controlled_difference(ungated_summary, gated_summary)
    paired = paired_aoi_bootstrap(
        gated_rows, ungated_rows, repetitions=repetitions, seed=seed
    )
    intervals = paired["intervals"]
    ungated_call_rate = float(ungated_summary["metrics"]["tool_call_episode_rate"])
    gated_call_rate = float(gated_summary["metrics"]["tool_call_episode_rate"])
    checks = {
        "tool_calls_nonzero": min(ungated_call_rate, gated_call_rate) > 0.0,
        "utility_gain_ci95": intervals["mean_quality_cost_utility"]["ci95_low"] > 0.0,
        "accuracy_noninferior": intervals["terminal_accuracy"]["ci95_low"] >= -accuracy_margin,
        "false_edit_noninferior": intervals["false_edit_rate"]["ci95_high"] <= false_edit_margin,
    }
    return {
        "schema_version": "active-catalog-closed-loop-reliability-gate-ablation-v1",
        **control,
        "paired_aoi_bootstrap": paired,
        "tool_call_episode_rate": {"ungated": ungated_call_rate, "gated": gated_call_rate},
        "margins": {
            "accuracy": accuracy_margin,
            "false_edit_rate": false_edit_margin,
        },
        "promotion_checks": {**checks, "passed": all(checks.values())},
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--ungated-summary", type=Path, required=True)
    parser.add_argument("--gated-summary", type=Path, required=True)
    parser.add_argument("--ungated-traces", type=Path, required=True)
    parser.add_argument("--gated-traces", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--accuracy-margin", type=float, default=0.005)
    parser.add_argument("--false-edit-margin", type=float, default=0.005)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(
        _read(args.ungated_summary),
        _read(args.gated_summary),
        load_rows(args.ungated_traces),
        load_rows(args.gated_traces),
        repetitions=args.repetitions,
        seed=args.seed,
        accuracy_margin=args.accuracy_margin,
        false_edit_margin=args.false_edit_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

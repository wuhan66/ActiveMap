#!/usr/bin/env python3
"""Summarize static Agent predictions on frozen tool-opportunity strata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_tool_opportunity_strata(
    manifest: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if manifest.get("scope") != "diagnostic_only":
        raise ValueError("opportunity manifest must be diagnostic_only")
    if manifest.get("split") == "test" or manifest.get("test_assets_read") is not False:
        raise ValueError("test opportunity manifests are forbidden")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("opportunity manifest has no records")
    stratum_by_trajectory: dict[str, str] = {}
    for record in records:
        trajectory_id = str(record["sequence_id"])
        if trajectory_id in stratum_by_trajectory:
            raise ValueError(f"duplicate sequence_id in manifest: {trajectory_id}")
        stratum_by_trajectory[trajectory_id] = str(record["stratum"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_prediction_count = 0
    for prediction in predictions:
        trajectory_id = str(prediction["trajectory_id"])
        stratum = stratum_by_trajectory.get(trajectory_id)
        if stratum is None:
            # The composed validation corpus also contains acquisition and terminal
            # controller trajectories that are outside the post-acquisition tool audit.
            excluded_prediction_count += 1
            continue
        grouped[stratum].append(prediction)
    missing = set(stratum_by_trajectory) - {
        str(prediction["trajectory_id"]) for prediction in predictions
    }
    if missing:
        raise ValueError(f"manifest trajectories missing predictions: {sorted(missing)[:3]}")

    summaries: dict[str, Any] = {}
    for stratum in ("positive", "near_boundary", "harmful"):
        rows = grouped.get(stratum, [])
        trajectory_ids = {str(row["trajectory_id"]) for row in rows}
        executable_rows = [row for row in rows if bool(row.get("executable"))]
        predicted_tool_rows = [
            row
            for row in executable_rows
            if str(row.get("prediction_class")) == "USE_TOOL"
        ]
        target_tool_rows = [
            row for row in rows if str(row.get("target_class")) == "USE_TOOL"
        ]
        no_tool_rows = [
            row for row in rows if str(row.get("target_class")) != "USE_TOOL"
        ]
        false_tool_rows = [
            row
            for row in no_tool_rows
            if bool(row.get("executable"))
            and str(row.get("prediction_class")) == "USE_TOOL"
        ]
        exact_rows = [
            row
            for row in rows
            if bool(row.get("executable")) and row.get("prediction") == row.get("target")
        ]
        exact_tool_rows = [
            row
            for row in target_tool_rows
            if bool(row.get("executable")) and row.get("prediction") == row.get("target")
        ]
        episodes_with_tool = {
            str(row["trajectory_id"]) for row in predicted_tool_rows
        }
        utilities = [
            float(row["predicted_utility"])
            for row in rows
            if row.get("predicted_utility") is not None
        ]
        regrets = [
            float(row["oracle_utility"]) - float(row["predicted_utility"])
            for row in rows
            if row.get("oracle_utility") is not None
            and row.get("predicted_utility") is not None
        ]
        summaries[stratum] = {
            "episode_count": len(trajectory_ids),
            "state_count": len(rows),
            "schema_valid_rate": (
                sum(bool(row.get("schema_valid")) for row in rows) / max(len(rows), 1)
            ),
            "executable_valid_rate": len(executable_rows) / max(len(rows), 1),
            "exact_action_accuracy": len(exact_rows) / max(len(rows), 1),
            "predicted_tool_call_rate": len(predicted_tool_rows) / max(len(rows), 1),
            "episode_any_tool_call_rate": len(episodes_with_tool)
            / max(len(trajectory_ids), 1),
            "target_tool_state_count": len(target_tool_rows),
            "grounded_tool_exact_recall": len(exact_tool_rows)
            / max(len(target_tool_rows), 1),
            "false_call_rate": len(false_tool_rows) / max(len(no_tool_rows), 1),
            "mean_policy_utility": _mean(utilities),
            "mean_regret": _mean(regrets),
        }

    return {
        "schema_version": "tool-opportunity-static-eval-v1",
        "scope": "diagnostic_only",
        "allowed_for_checkpoint_selection": False,
        "allowed_as_primary_result": False,
        "manifest_episode_id_sha256": manifest.get("episode_id_sha256"),
        "prediction_state_count": len(predictions),
        "evaluated_prediction_state_count": len(predictions) - excluded_prediction_count,
        "excluded_non_opportunity_state_count": excluded_prediction_count,
        "strata": summaries,
        "test_assets_read": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    result = evaluate_tool_opportunity_strata(
        _read_json(args.manifest), _read_jsonl(args.predictions)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

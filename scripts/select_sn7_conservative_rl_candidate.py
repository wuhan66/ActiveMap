#!/usr/bin/env python3
"""Select at most one conservative RL candidate using frozen paired gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_INTERVALS = (
    "terminal_accuracy",
    "false_edit_rate",
    "mean_quality_gain",
    "mean_cost",
    "mean_quality_cost_utility",
)


def select_candidate(
    table: dict[str, Any],
    candidates: list[str],
    *,
    min_utility_ci_low: float = 0.0,
    min_quality_ci_low: float = 0.0,
    max_false_edit_ci_high: float = 0.0,
    min_accuracy_observed: float = 0.0,
) -> dict[str, Any]:
    if table.get("schema_version") != "sn7-common-controller-validation-table-v1":
        raise ValueError("unexpected controller table schema")
    if table.get("split") != "val" or table.get("test_assets_read") is not False:
        raise ValueError("RL selection is validation-only")
    comparisons = table.get("paired_vs_reference", {})
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate RL candidates")
    missing = [label for label in candidates if label not in comparisons]
    if missing:
        raise ValueError(f"missing paired RL candidates: {missing}")

    assessments = {}
    for label in candidates:
        comparison = comparisons[label]
        intervals = comparison.get("intervals", {})
        absent = [metric for metric in REQUIRED_INTERVALS if metric not in intervals]
        if absent:
            raise ValueError(f"{label} lacks required intervals: {absent}")
        gates = {
            "utility_ci_positive": (
                float(intervals["mean_quality_cost_utility"]["ci95_low"])
                > min_utility_ci_low
            ),
            "quality_ci_nonnegative": (
                float(intervals["mean_quality_gain"]["ci95_low"])
                >= min_quality_ci_low
            ),
            "false_edit_ci_noninferior": (
                float(intervals["false_edit_rate"]["ci95_high"])
                <= max_false_edit_ci_high
            ),
            "accuracy_observed_nonnegative": (
                float(intervals["terminal_accuracy"]["observed_delta"])
                >= min_accuracy_observed
            ),
            "observed_non_dominated": bool(
                comparison.get("non_dominated_observed", False)
            ),
        }
        eligible = all(gates.values())
        assessments[label] = {
            "eligible": eligible,
            "gates": gates,
            "ranking_values": {
                "utility_ci_low": float(
                    intervals["mean_quality_cost_utility"]["ci95_low"]
                ),
                "quality_ci_low": float(
                    intervals["mean_quality_gain"]["ci95_low"]
                ),
                "false_edit_ci_high": float(
                    intervals["false_edit_rate"]["ci95_high"]
                ),
                "cost_observed_delta": float(
                    intervals["mean_cost"]["observed_delta"]
                ),
            },
        }
    eligible = [label for label in candidates if assessments[label]["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda label: (
                assessments[label]["ranking_values"]["utility_ci_low"],
                assessments[label]["ranking_values"]["quality_ci_low"],
                -assessments[label]["ranking_values"]["false_edit_ci_high"],
                -assessments[label]["ranking_values"]["cost_observed_delta"],
                label,
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": "sn7-conservative-rl-selection-v1",
        "decision": "promote_one" if selected is not None else "stop_rl",
        "selected_candidate": selected,
        "reference": table["reference"],
        "record_count": table["record_count"],
        "thresholds": {
            "min_utility_ci_low_exclusive": min_utility_ci_low,
            "min_quality_ci_low_inclusive": min_quality_ci_low,
            "max_false_edit_ci_high_inclusive": max_false_edit_ci_high,
            "min_accuracy_observed_inclusive": min_accuracy_observed,
        },
        "candidate_assessments": assessments,
        "test_assets_read": False,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SN7 Conservative RL Selection",
        "",
        f"Decision: **{result['decision']}**",
        f"Selected candidate: `{result['selected_candidate']}`",
        "",
        "| Candidate | Eligible | Utility CI low | Quality CI low | False-edit CI high | Cost delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in result["candidate_assessments"].items():
        values = row["ranking_values"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["eligible"]),
                    f"{values['utility_ci_low']:.6f}",
                    f"{values['quality_ci_low']:.6f}",
                    f"{values['false_edit_ci_high']:.6f}",
                    f"{values['cost_observed_delta']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "At most one candidate is promoted. If none passes every frozen gate,",
            "RL stops and the SFT controller remains the final policy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller_table", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--min-utility-ci-low", type=float, default=0.0)
    parser.add_argument("--min-quality-ci-low", type=float, default=0.0)
    parser.add_argument("--max-false-edit-ci-high", type=float, default=0.0)
    parser.add_argument("--min-accuracy-observed", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    table = json.loads(args.controller_table.read_text(encoding="utf-8"))
    result = select_candidate(
        table,
        args.candidate,
        min_utility_ci_low=args.min_utility_ci_low,
        min_quality_ci_low=args.min_quality_ci_low,
        max_false_edit_ci_high=args.max_false_edit_ci_high,
        min_accuracy_observed=args.min_accuracy_observed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selection.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

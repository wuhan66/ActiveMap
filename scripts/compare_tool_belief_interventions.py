#!/usr/bin/env python3
"""Compare full and ablated Tool-to-Belief intervention evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol", {}).get("schema_version") != "tool-belief-intervention-eval-v1":
        raise ValueError(f"unsupported intervention report: {path}")
    return payload


def compare(
    candidates: dict[str, tuple[Path, dict[str, Any]]], *, reference: str
) -> dict[str, Any]:
    if reference not in candidates:
        raise ValueError(f"missing reference candidate: {reference}")
    protocol_keys = ("split", "episode_count", "one_step_count", "tools_per_episode")
    reference_protocol = candidates[reference][1]["protocol"]
    reference_identity = candidates[reference][1]["summaries"]["identity_one_step"]
    rows = []
    for name, (path, report) in candidates.items():
        protocol = report["protocol"]
        mismatches = {
            key: (reference_protocol.get(key), protocol.get(key))
            for key in protocol_keys
            if reference_protocol.get(key) != protocol.get(key)
        }
        if mismatches:
            raise ValueError(f"protocol mismatch for {name}: {mismatches}")
        identity = report["summaries"]["identity_one_step"]
        for metric in ("accuracy", "macro_f1", "false_edit_rate", "missed_edit_rate"):
            if abs(float(identity[metric]) - float(reference_identity[metric])) > 1e-10:
                raise ValueError(f"identity baseline mismatch for {name}: {metric}")
        learned = report["summaries"]["learned_one_step"]
        recurrent = report["summaries"]["learned_temporal_3"]
        paired = report["summaries"]["learned_quality_temporal_3"]
        quality_only = report["summaries"]["learned_quality_3"]
        no_current = report["summaries"]["learned_no_current_one_step"]
        noop = report["quality_noop_delta_max"]
        rows.append(
            {
                "name": name,
                "path": str(path),
                "gates_passed": report["gates"]["passed"],
                "one_step_accuracy": learned["accuracy"],
                "one_step_macro_f1": learned["macro_f1"],
                "one_step_macro_f1_gain": learned["macro_f1"] - identity["macro_f1"],
                "one_step_false_edit_rate": learned["false_edit_rate"],
                "one_step_missed_edit_rate": learned["missed_edit_rate"],
                "one_step_joint_utility": learned["mean_joint_utility"],
                "recurrent_3_macro_f1": recurrent["macro_f1"],
                "recurrent_3_joint_utility": recurrent["mean_joint_utility"],
                "paired_3_macro_f1": paired["macro_f1"],
                "paired_3_joint_utility": paired["mean_joint_utility"],
                "no_quality_3_macro_f1": recurrent["macro_f1"],
                "no_temporal_3_macro_f1": quality_only["macro_f1"],
                "no_current_one_step_macro_f1": no_current["macro_f1"],
                "noop_probability_max": noop["probability_l1"],
                "noop_confidence_max": noop["confidence_absolute"],
                "noop_geometry_max": noop["geometry_mae"],
                "failed_checks": sorted(
                    key for key, passed in report["gates"]["checks"].items() if not passed
                ),
            }
        )

    by_name = {row["name"]: row for row in rows}
    reference_row = by_name[reference]
    deltas = {}
    delta_metrics = (
        "one_step_macro_f1",
        "one_step_false_edit_rate",
        "one_step_missed_edit_rate",
        "one_step_joint_utility",
        "recurrent_3_macro_f1",
        "recurrent_3_joint_utility",
        "noop_probability_max",
        "noop_confidence_max",
        "noop_geometry_max",
    )
    for row in rows:
        if row["name"] == reference:
            continue
        deltas[row["name"]] = {
            metric: float(row[metric]) - float(reference_row[metric])
            for metric in delta_metrics
        }

    passing = [row for row in rows if row["gates_passed"]]
    selected = (
        max(
            passing,
            key=lambda row: (
                row["one_step_joint_utility"],
                row["recurrent_3_joint_utility"],
                row["one_step_macro_f1"],
            ),
        )["name"]
        if passing
        else None
    )
    return {
        "schema_version": "tool-belief-intervention-comparison-v1",
        "reference": reference,
        "protocol": {key: reference_protocol[key] for key in protocol_keys},
        "candidates": rows,
        "deltas_from_reference": deltas,
        "selection": {
            "selected": selected,
            "eligible_count": len(passing),
            "criterion": (
                "gates pass, then one-step joint utility, recurrent joint utility, macro F1"
            ),
            "promoted": selected is not None,
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=SUMMARY")
    parser.add_argument("--reference", default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for raw in args.candidate:
        name, separator, path_value = raw.partition("=")
        if not separator or not name or not path_value:
            parser.error(f"invalid --candidate: {raw}")
        if name in candidates:
            parser.error(f"duplicate candidate: {name}")
        path = Path(path_value)
        candidates[name] = (path, _load(path))
    report = compare(candidates, reference=args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

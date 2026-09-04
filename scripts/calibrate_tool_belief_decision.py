#!/usr/bin/env python3
"""Freeze a validation-only KEEP boundary under false-edit safety constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_tool_belief_stopping import EDIT_ORDER, _summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty sequence details: {path}")
    return rows


def _thresholds(values: np.ndarray) -> list[float]:
    unique = np.unique(values)
    if unique.size == 1:
        return [float(unique[0] - 1e-6), float(unique[0] + 1e-6)]
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return [
        float(unique[0] - 1e-6),
        *[float(value) for value in midpoints],
        float(unique[-1] + 1e-6),
    ]


def calibrate_decision(
    report: dict[str, Any],
    details: list[dict[str, Any]],
    *,
    minimum_macro_f1_gain: float = 0.01,
    max_false_edit_increase: float = 0.02,
    max_missed_edit_increase: float = 0.02,
) -> dict[str, Any]:
    final_rows = sorted(
        (row for row in details if int(row["step"]) == 3),
        key=lambda row: str(row["sequence_id"]),
    )
    if len(final_rows) != report["protocol"]["sequence_count"]:
        raise ValueError("details do not contain one final row per sequence")
    probabilities = np.asarray(
        [row["paired_probabilities"] for row in final_rows], dtype=np.float64
    )
    if probabilities.shape != (len(final_rows), len(EDIT_ORDER)):
        raise ValueError("paired probabilities have an invalid shape")
    targets = [EDIT_ORDER.index(str(row["target"])) for row in final_rows]
    confidences = [float(row["paired_confidence"]) for row in final_rows]
    costs = [float(row["spent_cost"]) for row in final_rows]
    keep_index = EDIT_ORDER.index("KEEP")
    nonkeep_indices = np.asarray(
        [index for index in range(len(EDIT_ORDER)) if index != keep_index]
    )
    best_nonkeep_offsets = np.argmax(probabilities[:, nonkeep_indices], axis=1)
    best_nonkeep = nonkeep_indices[best_nonkeep_offsets]
    keep_probability = probabilities[:, keep_index]
    keep_margin = keep_probability - np.max(probabilities[:, nonkeep_indices], axis=1)
    identity = report["summaries"]["identity"]
    limits = {
        "false_edit_rate": identity["false_edit_rate"] + max_false_edit_increase,
        "missed_edit_rate": identity["missed_edit_rate"] + max_missed_edit_increase,
    }
    candidates = []
    for mode, scores in (
        ("keep_probability", keep_probability),
        ("keep_margin", keep_margin),
    ):
        for threshold in _thresholds(scores):
            predictions = np.where(scores >= threshold, keep_index, best_nonkeep)
            metrics = _summary(
                f"calibrated_{mode}",
                targets,
                predictions.tolist(),
                confidences,
                costs,
            )
            feasible = (
                metrics["false_edit_rate"] <= limits["false_edit_rate"] + 1e-12
                and metrics["missed_edit_rate"] <= limits["missed_edit_rate"] + 1e-12
            )
            candidates.append(
                {
                    "mode": mode,
                    "threshold": threshold,
                    "feasible": feasible,
                    "metrics": metrics,
                    "predictions": predictions.tolist(),
                }
            )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise ValueError("no decision threshold satisfies false/missed-edit constraints")
    best = max(
        feasible,
        key=lambda candidate: (
            candidate["metrics"]["macro_f1"],
            candidate["metrics"]["accuracy"],
            -candidate["metrics"]["false_edit_rate"],
        ),
    )
    metrics = best["metrics"]
    checks = {
        "macro_f1_gain": (
            metrics["macro_f1"] - identity["macro_f1"] >= minimum_macro_f1_gain
        ),
        "false_edit_safety": (
            metrics["false_edit_rate"] - identity["false_edit_rate"]
            <= max_false_edit_increase
        ),
        "missed_edit_safety": (
            metrics["missed_edit_rate"] - identity["missed_edit_rate"]
            <= max_missed_edit_increase
        ),
        "recurrent_not_worse": (
            metrics["macro_f1"] >= report["summaries"]["paired_1"]["macro_f1"]
        ),
        "noop_probability_safety": report["gates"]["checks"][
            "noop_probability_safety"
        ],
        "noop_confidence_safety": report["gates"]["checks"][
            "noop_confidence_safety"
        ],
        "noop_geometry_safety": report["gates"]["checks"]["noop_geometry_safety"],
    }
    prediction_rows = [
        {
            "sequence_id": row["sequence_id"],
            "target": row["target"],
            "uncalibrated": row["paired"],
            "calibrated": EDIT_ORDER[prediction],
        }
        for row, prediction in zip(final_rows, best["predictions"], strict=True)
    ]
    return {
        "protocol": {
            "schema_version": "tool-belief-decision-calibration-v1",
            "split": "val",
            "selection_uses_ground_truth": True,
            "test_assets_read": False,
        },
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "selected_mode": best["mode"],
        "selected_threshold": best["threshold"],
        "identity": identity,
        "uncalibrated": report["summaries"]["paired_3"],
        "calibrated": metrics,
        "limits": limits,
        "gates": {"checks": checks, "passed": all(checks.values())},
        "predictions": prediction_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_summary", type=Path)
    parser.add_argument("sequence_details", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-macro-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-missed-edit-increase", type=float, default=0.02)
    args = parser.parse_args()
    report = json.loads(args.sequence_summary.read_text(encoding="utf-8"))
    result = calibrate_decision(
        report,
        _read_jsonl(args.sequence_details),
        minimum_macro_f1_gain=args.minimum_macro_f1_gain,
        max_false_edit_increase=args.max_false_edit_increase,
        max_missed_edit_increase=args.max_missed_edit_increase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in result.items() if key != "predictions"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()

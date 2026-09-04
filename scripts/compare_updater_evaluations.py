from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "macro_f1",
    "edit_accuracy",
    "false_edit_rate",
    "missed_update_rate",
    "mean_raster_iou",
    "mean_polygon_iou",
    "topology_valid_rate",
    "ece",
    "brier",
    "nll",
    "aurc",
)


def _load_evaluation(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("validation_evaluation", payload)
    if not isinstance(evaluation, dict):
        raise ValueError(f"missing validation evaluation: {path}")
    return evaluation


def compare_evaluations(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_metrics = {name: reference.get(name) for name in METRICS}
    candidate_metrics = {name: candidate.get(name) for name in METRICS}
    reference_metrics["delete_f1"] = reference["per_edit"]["DELETE"]["f1"]
    candidate_metrics["delete_f1"] = candidate["per_edit"]["DELETE"]["f1"]
    deltas = {}
    for name in (*METRICS, "delete_f1"):
        reference_value = reference_metrics[name]
        candidate_value = candidate_metrics[name]
        deltas[name] = (
            float(candidate_value) - float(reference_value)
            if candidate_value is not None and reference_value is not None
            else None
        )
    return {
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_reference": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two frozen updater validation evaluations."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {
        "scope": "validation_only",
        "reference_path": str(args.reference.resolve()),
        "candidate_path": str(args.candidate.resolve()),
        **compare_evaluations(_load_evaluation(args.reference), _load_evaluation(args.candidate)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

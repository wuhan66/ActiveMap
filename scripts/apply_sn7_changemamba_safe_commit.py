#!/usr/bin/env python3
"""Apply a train-only frozen ChangeMamba safe-commit calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.calibrate_sn7_changemamba_safe_commit import (
    FEATURE_NAMES,
    deserialize_logistic,
    feature_matrix,
    paper_prediction,
    policy_metrics,
    predict_logistic,
    read_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def apply_frozen_calibration(
    calibration: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tuple(calibration.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("safe-commit feature schema mismatch")
    details = calibration.get("calibration", {})
    if "frozen_logistic_model" not in details:
        raise ValueError("calibration does not contain a frozen logistic model")
    threshold = float(details["selected_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("safe-commit threshold must be in [0, 1]")
    model = deserialize_logistic(details["frozen_logistic_model"])
    scores = predict_logistic(model, feature_matrix(rows))
    direct, _ = policy_metrics(rows, scores, threshold=0.0)
    gated, accepted = policy_metrics(rows, scores, threshold=threshold)
    output_rows = [
        {
            **row,
            "safe_commit_score": float(scores[index]),
            "safe_commit_threshold": threshold,
            "safe_commit_accepted": bool(accepted[index]),
            "safe_commit_predicted_edit": (
                row["predicted_edit"] if accepted[index] else "KEEP"
            ),
            "safe_commit_map_iou": (
                row["committed_map_iou"]
                if accepted[index]
                else row["prior_map_iou"]
            ),
        }
        for index, row in enumerate(rows)
    ]
    return (
        {
            "schema_version": "sn7-frozen-safe-commit-application-v1",
            "sample_count": len(rows),
            "aoi_count": len({str(row["aoi_id"]) for row in rows}),
            "split": next(iter({row["split"] for row in rows})),
            "selected_threshold": threshold,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "direct": direct,
            "safe_commit": gated,
            "safe_minus_direct": {
                key: gated[key] - direct[key] for key in direct if key in gated
            },
            "test_assets_read": False,
        },
        output_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_summary", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    calibration = json.loads(
        args.calibration_summary.read_text(encoding="utf-8")
    )
    rows = read_rows(args.predictions, expected_split="val")
    result, output_rows = apply_frozen_calibration(calibration, rows)
    result["calibration_summary_sha256"] = _sha256(args.calibration_summary)
    result["input_predictions_sha256"] = _sha256(args.predictions)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_sample.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(paper_prediction(row)) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report the validation-only ChangeMamba safe-commit operating frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.calibrate_sn7_changemamba_safe_commit import policy_metrics

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "accepted_commit_rate",
    "candidate_acceptance_rate",
)
DEFAULT_THRESHOLDS = tuple(np.linspace(0.0, 1.0, 21))


def read_run(run_dir: Path) -> tuple[list[dict[str, Any]], float]:
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"frontier input is not test-free: {run_dir}")
    rows = [
        json.loads(line)
        for line in (run_dir / "per_sample.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows or {row["split"] for row in rows} != {"val"}:
        raise ValueError(f"frontier requires validation rows: {run_dir}")
    return rows, float(summary["calibration"]["selected_threshold"])


def analyze(
    run_dirs: list[Path],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("frontier analysis requires replicated seeds")
    if (
        not thresholds
        or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds)
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ValueError("thresholds must be unique, sorted, and in [0, 1]")
    runs = []
    identity = None
    for run_dir in run_dirs:
        rows, selected_threshold = read_run(run_dir)
        current_identity = [
            (row["sample_id"], row["aoi_id"], row["target_edit"])
            for row in rows
        ]
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"sample identity mismatch: {run_dir}")
        scores = np.asarray(
            [row["safe_commit_score"] for row in rows], dtype=np.float64
        )
        runs.append(
            {
                "run_dir": str(run_dir),
                "selected_threshold": selected_threshold,
                "rows": rows,
                "scores": scores,
            }
        )

    points = []
    for threshold in thresholds:
        per_run = [
            policy_metrics(
                run["rows"], run["scores"], threshold=threshold
            )[0]
            for run in runs
        ]
        points.append(
            {
                "threshold": threshold,
                "metrics": {
                    metric: {
                        "mean": float(
                            np.mean([value[metric] for value in per_run])
                        ),
                        "std": float(
                            np.std(
                                [value[metric] for value in per_run], ddof=1
                            )
                        ),
                    }
                    for metric in METRICS
                },
            }
        )
    return {
        "schema_version": "sn7-changemamba-commit-frontier-v1",
        "run_count": len(runs),
        "sample_count": len(identity or []),
        "selected_thresholds": [
            run["selected_threshold"] for run in runs
        ],
        "threshold_source": "fixed diagnostic grid; primary thresholds remain train-OOF",
        "points": points,
        "test_assets_read": False,
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["threshold"]
            + [
                f"{metric}_{stat}"
                for metric in METRICS
                for stat in ("mean", "std")
            ]
        )
        for point in result["points"]:
            writer.writerow(
                [point["threshold"]]
                + [
                    point["metrics"][metric][stat]
                    for metric in METRICS
                    for stat in ("mean", "std")
                ]
            )


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Threshold | Map-IoU delta | False edit | Missed edit | Candidate acceptance |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in result["points"]:
        metrics = point["metrics"]
        lines.append(
            f"| {point['threshold']:.2f} | "
            f"{metrics['map_iou_delta']['mean']:.5f} | "
            f"{metrics['false_edit_rate']['mean']:.2%} | "
            f"{metrics['missed_edit_rate']['mean']:.2%} | "
            f"{metrics['candidate_acceptance_rate']['mean']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--threshold-count", type=int, default=21)
    args = parser.parse_args()
    if args.threshold_count < 2:
        raise ValueError("threshold count must be at least two")
    result = analyze(
        args.run_dirs,
        thresholds=tuple(float(value) for value in np.linspace(0.0, 1.0, args.threshold_count)),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_csv is not None:
        write_csv(result, args.output_csv)
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

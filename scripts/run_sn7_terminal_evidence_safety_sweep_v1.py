#!/usr/bin/env python3
"""Run a common three-seed safety sweep for terminal-evidence writeback."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Any


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STORAGE = Path("/home/wh/ActiveMap")
CHECKPOINT = STORAGE / (
    "runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/"
    "best_quality.pt"
)
EPISODES = STORAGE / (
    "processed/sn7_v1/agent/executable_selector_v3_512_sharded/"
    "closed_loop_val_bundle_v1/episodes_val.jsonl"
)
RAW_ROOT = STORAGE / (
    "runs/selector/sn7_value_aware_h18_terminal_evidence_writeback_v1/"
    "writebacks"
)
INPUT_ROOT = STORAGE / (
    "runs/selector/sn7_value_aware_h18_terminal_evidence_writeback_v1/inputs"
)
RUN_ROOT = STORAGE / (
    "runs/selector/sn7_value_aware_h18_terminal_evidence_safety_sweep_v1"
)
SEEDS = ("generic_value_h18_s1", "generic_value_h18_s2", "generic_value_h18_s3")
CANDIDATES = (
    "m15_p16:0.15:0:16",
    "m15_p64:0.15:0:64",
    "m20_p16:0.20:0:16",
    "m25_p32:0.25:0:32",
)
GPU_IDS = (1, 2, 3, 4)


def auc(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in sorted(rows, key=lambda row: row["budget"])]
    if len(values) != 3:
        raise ValueError("expected exactly three predeclared budgets")
    return (values[0] + 2.0 * values[1] + values[2]) / 4.0


def summary(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))["budgets"]
    return {
        key: auc(rows, key)
        for key in (
            "mean_raster_iou",
            "mean_component_count_absolute_error",
            "mean_episode_utility_v2_balanced",
            "mean_episode_utility_v2_safety",
            "mean_episode_utility_v2_cost_aware",
        )
    }


def aggregate() -> dict[str, Any]:
    labels = (
        "always_stop",
        "raw_m15_p0",
        *(spec.split(":", 1)[0] for spec in CANDIDATES),
    )
    rows = []
    for label in labels:
        seed_metrics = []
        for seed in SEEDS:
            if label == "always_stop":
                path = RAW_ROOT / "always_stop" / "evaluation/summary.json"
            elif label == "raw_m15_p0":
                path = RAW_ROOT / seed / "evaluation/summary.json"
            else:
                path = RUN_ROOT / seed / label / "evaluation/summary.json"
            seed_metrics.append(summary(path))
        rows.append(
            {
                "label": label,
                "per_seed": seed_metrics,
                "mean": {
                    key: fmean(item[key] for item in seed_metrics)
                    for key in seed_metrics[0]
                },
            }
        )
    stop = rows[0]["mean"]
    raw = rows[1]["mean"]
    for row in rows:
        row["delta_vs_raw"] = {
            key: row["mean"][key] - raw[key] for key in row["mean"]
        }
        row["quality_feasible"] = (
            row["mean"]["mean_raster_iou"] - stop["mean_raster_iou"] >= -0.001
            and row["mean"]["mean_component_count_absolute_error"]
            - stop["mean_component_count_absolute_error"]
            <= 1.0
        )
        row["utility_feasible"] = (
            row["mean"]["mean_episode_utility_v2_balanced"]
            > stop["mean_episode_utility_v2_balanced"]
        )
    feasible = [
        row
        for row in rows[2:]
        if row["quality_feasible"] and row["utility_feasible"]
    ]
    winner = max(
        feasible,
        key=lambda row: row["mean"]["mean_episode_utility_v2_balanced"],
        default=None,
    )
    return {
        "schema_version": "sn7-terminal-evidence-safety-sweep-v1",
        "selection_split": "val",
        "seeds": list(SEEDS),
        "candidates": rows,
        "promotion_rule": (
            "relative to always-STOP: balanced Utility-v2 AUC must improve, mean "
            "raster-IoU delta must be >= -0.001, and mean component-count-error "
            "delta must be <= 1.0; then maximize balanced Utility-v2 AUC"
        ),
        "promoted": winner is not None,
        "winner": winner["label"] if winner is not None else None,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if RUN_ROOT.exists() and not args.summarize_only:
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    if not args.summarize_only:
        RUN_ROOT.mkdir(parents=True)
        for seed in SEEDS:
            command = [
                PYTHON,
                "scripts/launch_sn7_writeback_safety_sweep.py",
                str(CHECKPOINT),
                str(EPISODES),
                str(INPUT_ROOT / f"{seed}.jsonl"),
                str(RUN_ROOT / seed),
                "--python",
                PYTHON,
                "--asset-root-map",
                "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
            ]
            for spec in CANDIDATES:
                command.extend(["--candidate", spec])
            for gpu in GPU_IDS:
                command.extend(["--gpu", str(gpu)])
            subprocess.run(command, cwd=REPO, check=True)
    result = aggregate()
    (RUN_ROOT / "three_seed_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

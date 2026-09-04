#!/usr/bin/env python3
"""Build no-test, group-audited support for C5 residual safety correction."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


AUDIT = Path(__file__).with_name("audit_updater_conditioned_disagreements.py")
SPEC = importlib.util.spec_from_file_location("updater_conditioned_audit", AUDIT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _selected_vector(sample: Any, choice: dict[str, Any]) -> list[float]:
    """Deployment-observable features for one selected candidate or STOP."""

    width = len(sample.evidence_features[0])
    if choice["stop"]:
        return [0.0] * width + [1.0, 1.0, 0.0, float(choice["score"])]
    index = int(choice["index"])
    return [
        *[float(value) for value in sample.evidence_features[index]],
        0.0,
        float(sample.false_edit_risks[index]),
        float(sample.evidence_costs[index]),
        float(choice["score"]),
    ]


def _record(sample: Any, stale: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    stale_outcome = MODULE._outcome(sample, stale["evidence_id"])
    refreshed_outcome = MODULE._outcome(sample, refreshed["evidence_id"])
    utility = lambda choice: float(sample.stop_utility) if choice["stop"] else float(sample.oracle_utilities[choice["index"]])
    stale_utility = utility(stale)
    refreshed_utility = utility(refreshed)
    differs = stale["evidence_id"] != refreshed["evidence_id"]
    features = [
        *[float(value) for value in sample.hypothesis_features],
        *[float(value) for value in sample.state_features],
        *_selected_vector(sample, stale),
        *_selected_vector(sample, refreshed),
    ]
    return {
        "sample_id": str(sample.sample_id),
        "split": str(sample.split),
        "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
        "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
        "edit_type": str(sample.edit_type),
        "differs": differs,
        "features": features,
        "stale": {
            "stop": bool(stale["stop"]), "utility": stale_utility,
            "raster_iou": float(stale_outcome["final_raster_iou"]),
            "false_edit": bool(stale_outcome["false_edit"]),
            "missed_edit": bool(stale_outcome["missed_edit"]),
            "cost": 0.0 if stale["stop"] else float(sample.evidence_costs[stale["index"]]),
        },
        "refreshed": {
            "stop": bool(refreshed["stop"]), "utility": refreshed_utility,
            "raster_iou": float(refreshed_outcome["final_raster_iou"]),
            "false_edit": bool(refreshed_outcome["false_edit"]),
            "missed_edit": bool(refreshed_outcome["missed_edit"]),
            "cost": 0.0 if refreshed["stop"] else float(sample.evidence_costs[refreshed["index"]]),
        },
        "targets": {
            "utility_delta": refreshed_utility - stale_utility,
            "raster_iou_delta": float(refreshed_outcome["final_raster_iou"]) - float(stale_outcome["final_raster_iou"]),
            "false_edit_regression": bool((not stale_outcome["false_edit"]) and refreshed_outcome["false_edit"]),
            "missed_edit_improvement": bool(stale_outcome["missed_edit"] and (not refreshed_outcome["missed_edit"])),
        },
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(row["split"]) for row in rows)
    groups = {split: {row["source_episode"] for row in rows if row["split"] == split} for split in by_split}
    overlap = sorted(groups.get("train", set()) & groups.get("val", set()))
    disagreements = [row for row in rows if row["differs"]]
    return {
        "schema_version": "updater-conditioned-residual-support-v1",
        "sample_count": len(rows),
        "feature_dim": len(rows[0]["features"]) if rows else 0,
        "split_counts": dict(sorted(by_split.items())),
        "group_counts": {key: len(value) for key, value in sorted(groups.items())},
        "cross_split_group_overlap": len(overlap),
        "disagreement_count": len(disagreements),
        "false_edit_regression_count": sum(row["targets"]["false_edit_regression"] for row in disagreements),
        "missed_edit_improvement_count": sum(row["targets"]["missed_edit_improvement"] for row in disagreements),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stale_checkpoint", type=Path)
    parser.add_argument("refreshed_checkpoint", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    samples, stale = MODULE._choice_records(
        args.stale_checkpoint, args.states, device=device, batch_size=args.batch_size, split=None
    )
    refreshed_samples, refreshed = MODULE._choice_records(
        args.refreshed_checkpoint, args.states, device=device, batch_size=args.batch_size, split=None
    )
    if [sample.sample_id for sample in samples] != [sample.sample_id for sample in refreshed_samples]:
        raise ValueError("stale and refreshed sample ordering differs")
    rows = [_record(sample, stale[str(sample.sample_id)], refreshed[str(sample.sample_id)]) for sample in samples]
    summary = _summary(rows)
    summary.update({
        "stale_checkpoint": str(args.stale_checkpoint.resolve()),
        "refreshed_checkpoint": str(args.refreshed_checkpoint.resolve()),
        "states": str(args.states.resolve()),
    })
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "support.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

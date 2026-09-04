from __future__ import annotations

import csv
import json
from pathlib import Path

from activemap.data.argotweak_subset import build_argotweak_balanced_subset


def _lane() -> dict:
    return {
        "left_lane_boundary": [{"x": 0, "y": 1}, {"x": 1, "y": 1}],
        "right_lane_boundary": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
    }


def _write_annotation(path: Path, operation: str, feature: str) -> None:
    current = _lane()
    stale = _lane()
    if operation == "ADD":
        stale = None
    elif operation == "DELETE":
        current = None
    payload = {"laneSegments": {}, "pedCrossings": {}, "drivableAreas": {}}
    section = "laneSegments" if feature == "lane" else "pedCrossings"
    payload[section]["object-1"] = {
        "old": current,
        "new": stale,
        "changes": [{"ADD": 2, "DELETE": 1, "RESHAPE": 3}[operation]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_balanced_subset_is_frozen_and_does_not_read_test(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    splits = {"train": {}, "val": {}, "test": {"0": "hidden-test"}}
    index = 0
    for split, total in (("train", 9), ("val", 6)):
        for operation in ("ADD", "DELETE", "RESHAPE"):
            for feature in ("lane", "crossing"):
                segment_id = f"{split}-{index:02d}"
                index += 1
                splits[split][str(index)] = segment_id
                _write_annotation(annotations / f"{segment_id}.json", operation, feature)
        while len(splits[split]) < total:
            segment_id = f"{split}-{index:02d}"
            index += 1
            splits[split][str(index)] = segment_id
            _write_annotation(annotations / f"{segment_id}.json", "RESHAPE", "lane")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    summary = build_argotweak_balanced_subset(
        splits_path,
        annotations,
        tmp_path / "output",
        train_count=8,
        val_count=6,
        train_minimum_logs_per_key=1,
        val_minimum_logs_per_key=1,
    )

    assert summary["audited_logs"] == 15
    assert summary["selected_logs"] == {"train": 8, "val": 6}
    assert summary["test_assets_read"] is False
    assert all(
        summary["selected_support"][split]["log_counts"][key] >= 1
        for split in ("train", "val")
        for key in summary["support_targets_in_logs"][split]
    )
    with (tmp_path / "output" / "annotation_support.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    assert not any(row["segment_id"] == "hidden-test" for row in rows)

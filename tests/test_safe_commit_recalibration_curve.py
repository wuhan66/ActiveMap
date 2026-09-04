import json

import numpy as np

from scripts.evaluate_sn7_safe_commit_recalibration_curve import (
    parse_counts,
    run_curve,
)


def _rows(split):
    rows = []
    for aoi_index in range(4):
        for update in (False, True):
            beneficial = update
            rows.append(
                {
                    "sample_id": f"{split}-{aoi_index}-{int(update)}",
                    "aoi_id": f"aoi-{aoi_index}",
                    "split": split,
                    "target_edit": "ADD" if update else "KEEP",
                    "predicted_edit": "ADD",
                    "confidence": 0.9 if beneficial else 0.2,
                    "mean_change_probability": 0.8 if beneficial else 0.1,
                    "max_change_probability": 0.9,
                    "p95_change_probability": 0.85,
                    "mean_predictive_entropy": 0.1,
                    "prior_foreground_fraction": 0.2,
                    "predicted_change_fraction": 0.1,
                    "prior_map_iou": 0.5,
                    "committed_map_iou": 0.7 if beneficial else 0.4,
                    "map_iou_delta": 0.2 if beneficial else -0.1,
                }
            )
    return rows


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_recalibration_curve_is_finite_and_uses_target_aois(tmp_path):
    paths = {}
    for replicate in range(2):
        for name, split in (
            ("source", "train"),
            ("target_train", "train"),
            ("target_val", "val"),
        ):
            path = tmp_path / f"{name}-{replicate}.jsonl"
            _write(path, _rows(split))
            paths.setdefault(name, []).append(path)
    result = run_curve(
        paths["source"],
        paths["target_train"],
        paths["target_val"],
        source_backend="source",
        target_backend="target",
        counts=parse_counts("0,1,all"),
        folds=2,
        l2=0.01,
        selection_seed=7,
    )
    assert result["test_assets_read"] is False
    assert [row["requested_target_aois"] for row in result["aggregate"]] == [
        0,
        1,
        "all",
    ]
    assert result["aggregate"][1]["used_target_aois"]["mean"] == 1.0
    assert np.isfinite(
        result["aggregate"][1]["metrics"]["map_iou_delta"]["mean"]
    )

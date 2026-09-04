import json

from scripts.analyze_sn7_changemamba_commit_frontier import analyze


def _run(path, offset):
    path.mkdir()
    rows = []
    for aoi in range(2):
        for target, prediction, score, delta in (
            ("ADD", "ADD", 0.9 - offset, 0.4),
            ("KEEP", "ADD", 0.2 + offset, -0.4),
            ("ADD", "KEEP", 0.8, 0.0),
            ("KEEP", "KEEP", 0.8, 0.0),
        ):
            rows.append(
                {
                    "sample_id": f"{aoi}-{len(rows) % 4}",
                    "aoi_id": f"aoi-{aoi}",
                    "split": "val",
                    "target_edit": target,
                    "predicted_edit": prediction,
                    "safe_commit_score": score,
                    "prior_map_iou": 0.5,
                    "committed_map_iou": 0.5 + delta,
                    "map_iou_delta": delta,
                }
            )
    (path / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "calibration": {"selected_threshold": 0.5},
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )


def test_frontier_reports_fixed_threshold_tradeoff(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first, 0.0)
    _run(second, 0.05)

    result = analyze([first, second], thresholds=(0.0, 0.5, 1.0))

    always, selective, reject_all = result["points"]
    assert result["test_assets_read"] is False
    assert always["metrics"]["false_edit_rate"]["mean"] == 0.5
    assert selective["metrics"]["false_edit_rate"]["mean"] == 0.0
    assert (
        selective["metrics"]["map_iou_delta"]["mean"]
        > always["metrics"]["map_iou_delta"]["mean"]
    )
    assert reject_all["metrics"]["candidate_acceptance_rate"]["mean"] == 0.0

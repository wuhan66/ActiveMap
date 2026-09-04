from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.export_active_catalog_rollout_visuals import export, public_evidence_id


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_export_clean_rollout_panels(tmp_path: Path) -> None:
    image = np.linspace(0, 1, 3 * 64 * 64, dtype=np.float32).reshape(3, 64, 64)
    image_path = tmp_path / "anchor.npy"
    np.save(image_path, image)
    raw_id = "ev-test"
    public_id = public_evidence_id(raw_id)
    episodes = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes,
        [
            {
                "episode_id": "episode-1",
                "anchor_timestamp": "2020_01",
                "evidence_catalog": [
                    {
                        "evidence_id": raw_id,
                        "timestamp": "2020_01",
                        "region": [20, 18, 40, 42],
                        "scale": 1,
                        "image_path": str(image_path),
                    }
                ],
            }
        ],
    )
    traces = tmp_path / "traces.jsonl"
    _write_jsonl(
        traces,
        [
            {
                "sample_id": "sample-1",
                "source_episode": "episode-1",
                "aoi_id": "aoi-1",
                "target_edit": "ADD",
                "predicted_edit": "ADD",
                "terminal_correct": True,
                "false_edit": False,
                "missed_edit": False,
                "acquisitions": 1,
                "tool_calls": 0,
                "quality_gain": 0.2,
                "spent_cost": 0.1,
                "events": [
                    {
                        "executed_action": {"action": "ACQUIRE"},
                        "ranker_scores": {public_id: 0.25},
                        "observable_state": {
                            "belief": {"edit_probabilities": [0.1, 0.7, 0.1, 0.1]}
                        },
                    }
                ],
            }
        ],
    )
    output = tmp_path / "visuals"
    summary = export(
        traces,
        episodes,
        output,
        limit=1,
        sample_ids=set(),
        root_maps=[],
        panel_size=96,
        crop_size=48,
    )
    assert summary["sample_count"] == 1
    folder = next(path for path in output.iterdir() if path.is_dir())
    expected = {
        "01_anchor_rgb.png",
        "utility_spatial_step00.png",
        "utility_overlay_step00.png",
        "utility_catalog_step00.png",
        "belief_trajectory.png",
        "action_trajectory.png",
        "metadata.json",
    }
    assert expected <= {path.name for path in folder.iterdir()}
    assert _image_size(folder / "utility_spatial_step00.png") == (96, 96)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size

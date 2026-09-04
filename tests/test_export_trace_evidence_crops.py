import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _load_exporter():
    script = Path(__file__).parents[1] / "scripts" / "figures" / "export_trace_evidence_crops.py"
    spec = importlib.util.spec_from_file_location("trace_evidence_exporter", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exports_recorded_validation_evidence_and_refuses_test_trace(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 16, 1, 1),
    ) as dataset:
        dataset.write(np.full((3, 16, 16), 100, dtype=np.uint8))
    episode = {
        "episode_id": "episode",
        "split": "val",
        "evidence_catalog": [
            {
                "evidence_id": "initial",
                "timestamp": "2019_01",
                "region": [0, 0, 8, 8],
                "scale": 1,
                "image_path": str(image_path),
                "cost": 1.0,
            },
            {
                "evidence_id": "acquired",
                "timestamp": "2019_02",
                "region": [4, 4, 16, 16],
                "scale": 2,
                "image_path": str(image_path),
                "cost": 2.0,
            },
        ],
    }
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "source_episode": "episode",
                "task_id": "task-episode",
                "budget": 1.5,
                "split": "val",
                "test_assets_read": False,
                "selected_evidence_ids": ["initial", "acquired"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exporter = _load_exporter()
    output = tmp_path / "evidence"
    manifest = exporter.export_trace_evidence_crops(
        episodes_path=episodes,
        traces_path=trace,
        source_episode="episode",
        output_dir=output,
        image_size=32,
        task_id="task-episode",
        budget=1.5,
    )
    assert manifest["test_assets_read"] is False
    assert manifest["task_id"] == "task-episode"
    assert manifest["budget"] == 1.5
    assert [row["evidence_id"] for row in manifest["evidence"]] == ["initial", "acquired"]
    assert (output / "selected_01_2019_02_scale2.png").is_file()

    rejected = tmp_path / "test_trace.jsonl"
    rejected.write_text(
        json.dumps(
            {
                "source_episode": "episode",
                "task_id": "task-episode",
                "budget": 1.5,
                "split": "test",
                "test_assets_read": True,
                "selected_evidence_ids": ["initial", "acquired"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation-only"):
        exporter.export_trace_evidence_crops(
            episodes_path=episodes,
            traces_path=rejected,
            source_episode="episode",
            output_dir=tmp_path / "rejected",
        )


def test_requires_exact_trace_identity_when_source_episode_repeats(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 16, 1, 1),
    ) as dataset:
        dataset.write(np.full((3, 16, 16), 100, dtype=np.uint8))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_id": "episode",
                "split": "val",
                "evidence_catalog": [
                    {
                        "evidence_id": "initial",
                        "timestamp": "2019_01",
                        "region": [0, 0, 8, 8],
                        "scale": 1,
                        "image_path": str(image_path),
                        "cost": 1.0,
                    },
                    {
                        "evidence_id": "acquired",
                        "timestamp": "2019_02",
                        "region": [4, 4, 16, 16],
                        "scale": 2,
                        "image_path": str(image_path),
                        "cost": 2.0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        "".join(
            json.dumps(
                {
                    "source_episode": "episode",
                    "task_id": "task-episode",
                    "budget": budget,
                    "split": "val",
                    "test_assets_read": False,
                    "selected_evidence_ids": ["initial", "acquired"],
                }
            )
            + "\n"
            for budget in (1.5, 3.0)
        ),
        encoding="utf-8",
    )
    exporter = _load_exporter()

    with pytest.raises(ValueError, match="exactly one matching row"):
        exporter.export_trace_evidence_crops(
            episodes_path=episodes,
            traces_path=traces,
            source_episode="episode",
            output_dir=tmp_path / "ambiguous",
        )

    manifest = exporter.export_trace_evidence_crops(
        episodes_path=episodes,
        traces_path=traces,
        source_episode="episode",
        output_dir=tmp_path / "exact",
        task_id="task-episode",
        budget=3.0,
    )
    assert manifest["budget"] == 3.0

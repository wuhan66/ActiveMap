import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _load_exporter():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "figures"
        / "export_sn7_v5_qualitative_evidence.py"
    )
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("v5_qualitative_evidence_exporter", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exports_only_fixed_v5_cases_with_trace_bound_evidence(tmp_path: Path) -> None:
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

    operations = ("ADD", "DELETE", "RESHAPE")
    cases = []
    episodes = []
    rollouts = []
    for operation_index, operation in enumerate(operations):
        for sample_index in range(3):
            number = operation_index * 3 + sample_index
            source_episode = f"episode-{number}"
            task_id = f"task-{number}"
            initial = f"initial-{number}"
            acquired = f"acquired-{number}"
            case = {
                "operation": operation,
                "source_episode": source_episode,
                "task_id": task_id,
                "aoi_id": f"aoi-{operation_index}",
                "budget": 1.5,
                "state_id": f"{source_episode}__b1p5__s0",
                "initial_evidence_ids": [initial],
            }
            cases.append(case)
            episodes.append(
                {
                    "episode_id": source_episode,
                    "split": "val",
                    "evidence_catalog": [
                        {
                            "evidence_id": initial,
                            "timestamp": "2019_01",
                            "region": [0, 0, 8, 8],
                            "scale": 1,
                            "image_path": str(image_path),
                            "cost": 1.0,
                        },
                        {
                            "evidence_id": acquired,
                            "timestamp": "2019_02",
                            "region": [4, 4, 16, 16],
                            "scale": 2,
                            "image_path": str(image_path),
                            "cost": 2.0,
                        },
                    ],
                }
            )
            rollouts.append(
                {
                    "source_episode": source_episode,
                    "task_id": task_id,
                    "budget": 1.5,
                    "split": "val",
                    "test_assets_read": False,
                    "selected_extra_evidence_id": acquired,
                    "selected_evidence_ids": [initial, acquired],
                }
            )

    qualitative = tmp_path / "qualitative_manifest.json"
    qualitative.write_text(
        json.dumps(
            {
                "schema_version": "sn7-v5-predeclared-qualitative-manifest-v1",
                "split": "val",
                "test_assets_read": False,
                "controller_or_writeback_outputs_read": False,
                "cases": cases,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    rollouts_path = tmp_path / "selected_rollouts.jsonl"
    rollouts_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts), encoding="utf-8"
    )

    exporter = _load_exporter()
    output = tmp_path / "evidence"
    receipt = exporter.export_v5_qualitative_evidence(
        qualitative_manifest=qualitative,
        episodes_path=episodes_path,
        selected_rollouts_path=rollouts_path,
        output_dir=output,
        image_size=32,
    )

    assert receipt["split"] == "val"
    assert receipt["test_assets_read"] is False
    assert len(receipt["cases"]) == 9
    assert (output / "evidence_export_receipt.json").is_file()
    for case in cases:
        manifest = output / case["source_episode"] / "manifest.json"
        assert manifest.is_file()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["task_id"] == case["task_id"]
        assert payload["budget"] == case["budget"]

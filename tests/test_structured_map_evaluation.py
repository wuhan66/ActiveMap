import json
from pathlib import Path

import pytest

from activemap.data.structured_map import (
    StructuredMapObservation,
    StructuredMapSample,
    derive_structured_atomic_edits,
)
from activemap.evaluation.structured_map import evaluate_structured_map_predictions
from activemap.integrations.baselines.contracts import (
    StructuredMapPrediction,
    write_structured_prediction_jsonl,
)


def _feature(object_id: str, coordinates: list[list[float]], **properties: str) -> dict:
    return {
        "type": "Feature",
        "id": object_id,
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def _write_map(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, split: str = "val") -> tuple[Path, Path, Path]:
    prior = tmp_path / "prior.geojson"
    target = tmp_path / "target.geojson"
    _write_map(
        prior,
        [
            _feature("keep", [[0, 0], [1, 0]]),
            _feature("reshape", [[0, 1], [1, 1]], successors="delete"),
            _feature("delete", [[1, 1], [2, 1]], predecessors="reshape"),
        ],
    )
    _write_map(
        target,
        [
            _feature("keep", [[0, 0], [1, 0]]),
            _feature("reshape", [[0, 1], [2, 1]], successors="add"),
            _feature("add", [[2, 1], [3, 1]], predecessors="reshape"),
        ],
    )
    edits, _ = derive_structured_atomic_edits(prior, target, include_keep=True)
    sample = StructuredMapSample(
        schema_version="activemap-structured-map-sample-v1",
        sample_id="scene-1",
        dataset="argotweak",
        split=split,
        aoi_id="log-1",
        prior_map_path=str(prior),
        target_map_path=str(target),
        observations=[
            StructuredMapObservation(
                observation_id="camera-1",
                timestamp="0",
                modality="camera",
                path="camera.jpg",
                cost=1.0,
            )
        ],
        atomic_edits=edits,
        native_sample_id="native-1",
        test_assets_read=split == "test",
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        StructuredMapPrediction(
            schema_version="activemap-structured-map-prediction-v1",
            sample_id="scene-1",
            split=split,
            dataset="argotweak",
            baseline="oracle-fixture",
            object_id="reshape",
            operation="RESHAPE",
            geometry={"type": "LineString", "coordinates": [[0, 1], [2, 1]]},
            attributes_after={"successors": "add"},
            source_artifact="fixture",
            test_assets_read=split == "test",
        ),
        StructuredMapPrediction(
            schema_version="activemap-structured-map-prediction-v1",
            sample_id="scene-1",
            split=split,
            dataset="argotweak",
            baseline="oracle-fixture",
            object_id="delete",
            operation="DELETE",
            source_artifact="fixture",
            test_assets_read=split == "test",
        ),
        StructuredMapPrediction(
            schema_version="activemap-structured-map-prediction-v1",
            sample_id="scene-1",
            split=split,
            dataset="argotweak",
            baseline="oracle-fixture",
            object_id="add",
            operation="ADD",
            geometry={"type": "LineString", "coordinates": [[2, 1], [3, 1]]},
            attributes_after={"predecessors": "reshape"},
            source_artifact="fixture",
            test_assets_read=split == "test",
        ),
    ]
    write_structured_prediction_jsonl(rows, predictions)
    return samples, predictions, target


def test_perfect_structured_map_prediction(tmp_path: Path) -> None:
    samples, predictions, _ = _fixture(tmp_path)
    metrics = evaluate_structured_map_predictions(samples, predictions)
    assert metrics["atomic_edit"]["f1"] == 1.0
    assert metrics["unchanged_preservation"] == 1.0
    assert metrics["false_edit_rate"] == 0.0
    assert metrics["geometry_chamfer"] == 0.0
    assert metrics["topology"]["f1"] == 1.0
    assert metrics["test_assets_read"] is False


def test_omitted_change_counts_as_false_negative(tmp_path: Path) -> None:
    samples, predictions, _ = _fixture(tmp_path)
    rows = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    metrics = evaluate_structured_map_predictions(samples, predictions)
    assert metrics["atomic_edit"]["fn"] == 1
    assert metrics["atomic_edit"]["f1"] < 1.0


def test_false_change_on_keep_is_a_safety_error(tmp_path: Path) -> None:
    samples, predictions, _ = _fixture(tmp_path)
    row = StructuredMapPrediction(
        schema_version="activemap-structured-map-prediction-v1",
        sample_id="scene-1",
        split="val",
        dataset="argotweak",
        baseline="oracle-fixture",
        object_id="keep",
        operation="DELETE",
        source_artifact="fixture",
    )
    with predictions.open("a", encoding="utf-8") as handle:
        handle.write(row.model_dump_json() + "\n")
    metrics = evaluate_structured_map_predictions(samples, predictions)
    assert metrics["unchanged_preservation"] == 0.0
    assert metrics["false_edit_rate"] > 0.0


def test_duplicate_prediction_is_rejected(tmp_path: Path) -> None:
    samples, predictions, _ = _fixture(tmp_path)
    first = predictions.read_text(encoding="utf-8").splitlines()[0]
    with predictions.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    with pytest.raises(ValueError, match="duplicate sample/object"):
        evaluate_structured_map_predictions(samples, predictions)


def test_test_split_requires_explicit_unlock(tmp_path: Path) -> None:
    samples, predictions, _ = _fixture(tmp_path, split="test")
    with pytest.raises(PermissionError, match="test evaluation"):
        evaluate_structured_map_predictions(samples, predictions)
    metrics = evaluate_structured_map_predictions(samples, predictions, allow_test=True)
    assert metrics["test_assets_read"] is True

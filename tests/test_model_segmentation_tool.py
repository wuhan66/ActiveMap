from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from activemap.geo_tools.model_segmentation import (
    EditGatedRefinementPredictor,
    MapConditionedSegmentationTool,
    _prior_mask,
)
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation


class FakeUpdater:
    def predict(self, image: np.ndarray, prior_mask: np.ndarray):
        probability = np.where(image[0] > 0.5, 0.9, 0.1).astype(np.float32)
        return {
            "mask_probability": probability,
            "add_probability": probability,
            "remove_probability": 1.0 - probability,
            "edit_probabilities": np.asarray([0.1, 0.7, 0.1, 0.1]),
            "predicted_edit": "ADD",
            "gated_edit": "ADD",
            "confidence": 0.8,
        }


class FakeRefiner:
    def __init__(self) -> None:
        self.calls: list[EditOperation] = []

    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        prior_mask: np.ndarray,
        *,
        edit_type: EditOperation,
    ) -> np.ndarray:
        self.calls.append(edit_type)
        refined = coarse_mask.astype(np.float32).copy()
        refined[0, 0] = 1.0
        return refined


def _write_rgb(path: Path) -> None:
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6, 2:6] = 1.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="float32",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(image)


def test_model_segmentation_tool_emits_map_change_and_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    prior_path = tmp_path / "prior.npy"
    _write_rgb(image_path)
    np.save(prior_path, np.zeros((8, 8), dtype=np.float32))
    tool = MapConditionedSegmentationTool(FakeUpdater(), tmp_path / "artifacts")

    result = tool.run(
        GeoToolCall(
            call_id="model-segment-1",
            tool=GeoToolName.RASTER_SEGMENT,
            inputs={
                "image_path": str(image_path),
                "prior_mask_path": str(prior_path),
            },
            parameters={"threshold_grid": [0.2, 0.5, 0.8]},
        )
    )

    assert result.success
    assert result.outputs["foreground_fraction"] == 0.25
    assert result.outputs["changed_fraction"] == 0.25
    assert result.outputs["component_count"] == 1
    assert [row["threshold"] for row in result.outputs["threshold_sweep"]] == [
        0.2,
        0.5,
        0.8,
    ]
    assert result.outputs["threshold_sweep"][0]["foreground_fraction"] == 0.25
    assert result.outputs["threshold_sweep"][2]["foreground_fraction"] == 0.25
    assert result.outputs["edit_probabilities"][1] == 0.7
    assert result.outputs["learned_add_fraction"] == 0.25
    assert result.outputs["learned_remove_fraction"] == 0.75
    assert result.outputs["predicted_edit"] == "ADD"
    assert result.outputs["gated_edit"] == "ADD"
    assert len(result.artifacts) == 3
    assert all(Path(path).is_file() for path in result.artifacts)
    assert Path(result.artifacts[0]).is_file()


def test_prior_mask_uses_the_same_pixel_window_as_the_image(tmp_path: Path) -> None:
    path = tmp_path / "prior.npy"
    prior = np.zeros((8, 8), dtype=np.float32)
    prior[1:4, 2:6] = np.asarray(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=np.float32
    )
    np.save(path, prior)

    cropped = _prior_mask(path, (3, 4), pixel_window=[2, 1, 4, 3])

    assert np.array_equal(cropped, prior[1:4, 2:6])


def test_model_segmentation_tool_reads_aligned_npy_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.npy"
    prior_path = tmp_path / "prior.npy"
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6, 2:6] = 1.0
    np.save(image_path, image)
    np.save(prior_path, np.zeros((8, 8), dtype=np.float32))
    tool = MapConditionedSegmentationTool(FakeUpdater(), tmp_path / "npy-artifacts")

    result = tool.run(
        GeoToolCall(
            call_id="aligned-npy",
            tool=GeoToolName.RASTER_SEGMENT,
            inputs={
                "image_path": str(image_path),
                "prior_mask_path": str(prior_path),
            },
        )
    )

    assert result.outputs["aligned_patch"] is True
    assert result.outputs["shape"] == [3, 8, 8]
    assert result.outputs["foreground_fraction"] == 0.25


def test_edit_gated_refinement_invokes_specialist_for_add() -> None:
    refiner = FakeRefiner()
    predictor = EditGatedRefinementPredictor(FakeUpdater(), refiner)
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6, 2:6] = 1.0

    result = predictor.predict(image, np.zeros((8, 8), dtype=np.float32))

    assert refiner.calls == [EditOperation.ADD]
    assert result["base_predicted_edit"] == "ADD"
    assert result["refinement_invoked"] == 1.0
    assert result["mask_probability"][0, 0] == 1.0
    assert result["refinement_change_fraction"] == 1 / 64


class KeepUpdater(FakeUpdater):
    def predict(self, image: np.ndarray, prior_mask: np.ndarray):
        result = dict(super().predict(image, prior_mask))
        result["edit_probabilities"] = np.asarray([0.8, 0.1, 0.05, 0.05])
        result["predicted_edit"] = "KEEP"
        result["gated_edit"] = "KEEP"
        return result


def test_edit_gated_refinement_skips_specialist_for_keep() -> None:
    refiner = FakeRefiner()
    predictor = EditGatedRefinementPredictor(KeepUpdater(), refiner)

    result = predictor.predict(
        np.zeros((3, 8, 8), dtype=np.float32),
        np.zeros((8, 8), dtype=np.float32),
    )

    assert refiner.calls == []
    assert result["base_predicted_edit"] == "KEEP"
    assert result["refinement_invoked"] == 0.0
    assert result["refinement_change_fraction"] == 0.0

"""Model-backed segmentation tool for map-conditioned remote-sensing evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy import ndimage

from activemap.geo_tools.raster import _read
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.models import EditOperation


class SegmentationPredictor(Protocol):
    def predict(
        self, image: np.ndarray, prior_mask: np.ndarray
    ) -> Mapping[str, Any]: ...


class MaskRefiner(Protocol):
    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        prior_mask: np.ndarray,
        *,
        edit_type: EditOperation,
    ) -> np.ndarray: ...


class EditGatedRefinementPredictor:
    """Apply an isolated specialist refiner only to eligible typed edits."""

    def __init__(
        self,
        base_predictor: SegmentationPredictor,
        refiner: MaskRefiner,
        *,
        coarse_threshold: float = 0.5,
        eligible_edits: frozenset[EditOperation] = frozenset(
            {EditOperation.ADD, EditOperation.RESHAPE}
        ),
    ) -> None:
        if not 0.0 <= coarse_threshold <= 1.0:
            raise ValueError("coarse threshold must be in [0, 1]")
        self.base_predictor = base_predictor
        self.refiner = refiner
        self.coarse_threshold = coarse_threshold
        self.eligible_edits = eligible_edits

    @staticmethod
    def _predicted_edit(prediction: Mapping[str, Any]) -> EditOperation:
        declared = prediction.get("gated_edit", prediction.get("predicted_edit"))
        if declared is not None:
            return EditOperation(str(declared))
        probabilities = np.asarray(prediction["edit_probabilities"], dtype=np.float32)
        if probabilities.shape != (len(EditOperation),):
            raise ValueError("base predictor returned invalid edit probabilities")
        return list(EditOperation)[int(np.argmax(probabilities))]

    def predict(self, image: np.ndarray, prior_mask: np.ndarray) -> Mapping[str, Any]:
        base = dict(self.base_predictor.predict(image, prior_mask))
        probability = np.asarray(base["mask_probability"], dtype=np.float32).squeeze()
        prior = np.asarray(prior_mask, dtype=np.float32).squeeze()
        if probability.shape != prior.shape:
            raise ValueError("base segmentation and prior mask shapes do not match")
        edit = self._predicted_edit(base)
        coarse = probability >= self.coarse_threshold
        invoked = edit in self.eligible_edits
        if invoked:
            refined = np.asarray(
                self.refiner.refine(image, coarse, prior, edit_type=edit),
                dtype=np.float32,
            ).squeeze()
            if refined.shape != coarse.shape or not np.all(np.isfinite(refined)):
                raise ValueError("specialist refiner returned an invalid mask")
            refined = np.clip(refined, 0.0, 1.0)
        else:
            refined = coarse.astype(np.float32)
        changed = (refined >= 0.5) ^ coarse
        base.update(
            {
                "mask_probability": refined,
                "base_predicted_edit": edit.value,
                "refinement_invoked": float(invoked),
                "refinement_change_fraction": float(np.mean(changed)),
            }
        )
        return base


def _prior_mask(
    path: Path,
    shape: tuple[int, int],
    *,
    pixel_window: list[int] | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    prior = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if prior.ndim != 2:
        raise ValueError("prior mask must be a two-dimensional numpy array")
    if pixel_window is not None:
        if len(pixel_window) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in pixel_window
        ):
            raise ValueError("prior pixel_window must contain four integers")
        x, y, width, height = pixel_window
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > prior.shape[1]
            or y + height > prior.shape[0]
        ):
            raise ValueError(
                "prior pixel_window lies outside the prior mask: "
                f"window={pixel_window}, prior_shape={prior.shape}"
            )
        prior = prior[y : y + height, x : x + width]
    if prior.shape != shape:
        scale = (shape[0] / prior.shape[0], shape[1] / prior.shape[1])
        prior = ndimage.zoom(prior, scale, order=0, prefilter=False)
    if prior.shape != shape or not np.all(np.isfinite(prior)):
        raise ValueError("prior mask could not be aligned to the evidence crop")
    return np.clip(prior, 0.0, 1.0)


def _segmentation_statistics(
    probability: np.ndarray,
    prior_binary: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    mask = (probability >= threshold) & valid
    _, component_count = ndimage.label(mask)
    union = mask | prior_binary
    intersection = mask & prior_binary
    changed = mask ^ prior_binary
    return mask, {
        "threshold": threshold,
        "foreground_fraction": float(np.mean(mask[valid])),
        "component_count": int(component_count),
        "changed_fraction": float(np.mean(changed[valid])),
        "add_fraction": float(np.mean((mask & ~prior_binary)[valid])),
        "remove_fraction": float(np.mean((prior_binary & ~mask)[valid])),
        "prior_iou": float(np.sum(intersection) / max(np.sum(union), 1)),
        "mean_probability": float(np.mean(probability[valid])),
    }


class MapConditionedSegmentationTool:
    """Execute a frozen map updater or remote-sensing segmentation backend."""

    name = GeoToolName.RASTER_SEGMENT

    def __init__(
        self,
        predictor: SegmentationPredictor,
        output_root: Path,
        *,
        cost: float = 0.25,
    ) -> None:
        if cost < 0.0:
            raise ValueError("tool cost must be non-negative")
        self.predictor = predictor
        self.output_root = output_root
        self.cost = float(cost)

    @classmethod
    def from_updater_checkpoint(
        cls,
        checkpoint: Path,
        output_root: Path,
        *,
        device: str = "cpu",
        cost: float = 0.25,
    ) -> MapConditionedSegmentationTool:
        from activemap.inference import UpdaterPredictor

        return cls(
            UpdaterPredictor(checkpoint, device=device), output_root, cost=cost
        )

    @classmethod
    def from_rsprompter(
        cls,
        base_predictor: SegmentationPredictor,
        refinement_config: Any,
        output_root: Path,
        *,
        coarse_threshold: float = 0.5,
        cost: float = 0.75,
    ) -> MapConditionedSegmentationTool:
        """Construct the tool without importing RSPrompter into this process."""
        from activemap.integrations.rsprompter import RSPrompterRefiner

        predictor = EditGatedRefinementPredictor(
            base_predictor,
            RSPrompterRefiner(refinement_config),
            coarse_threshold=coarse_threshold,
        )
        return cls(predictor, output_root, cost=cost)

    @classmethod
    def from_sam_road(
        cls,
        repo_root: Path,
        config_path: Path,
        checkpoint: Path,
        sam_checkpoint: Path,
        output_root: Path,
        *,
        device: str = "cuda",
        road_threshold: float | None = None,
        upstream_commit: str | None = None,
        cost: float = 0.75,
    ) -> MapConditionedSegmentationTool:
        """Construct the optional official SAM-Road semantic backend."""
        from activemap.integrations.sam_road import SAMRoadPredictor

        return cls(
            SAMRoadPredictor(
                repo_root,
                config_path,
                checkpoint,
                sam_checkpoint,
                device=device,
                road_threshold=road_threshold,
                upstream_commit=upstream_commit,
            ),
            output_root,
            cost=cost,
        )

    @classmethod
    def from_prior_conditioned_sam_road(
        cls,
        repo_root: Path,
        config_path: Path,
        source_checkpoint: Path,
        sam_checkpoint: Path,
        change_checkpoint: Path,
        output_root: Path,
        *,
        device: str = "cuda",
        parameterization: str = "independent",
        operation_head: str = "global_stats",
        upstream_commit: str | None = None,
        cost: float = 0.75,
    ) -> MapConditionedSegmentationTool:
        """Construct the calibrated map-relative SAM-Road change backend."""
        from activemap.integrations.prior_conditioned_sam_road import (
            PriorConditionedSAMRoadPredictor,
        )

        return cls(
            PriorConditionedSAMRoadPredictor(
                repo_root,
                config_path,
                source_checkpoint,
                sam_checkpoint,
                change_checkpoint,
                device=device,
                parameterization=parameterization,
                operation_head=operation_head,
                upstream_commit=upstream_commit,
            ),
            output_root,
            cost=cost,
        )

    def run(self, call: GeoToolCall) -> GeoToolResult:
        image, valid, metadata = _read(
            Path(call.inputs["image_path"]),
            call.parameters,
            force_bands=[1, 2, 3],
        )
        prior = _prior_mask(
            Path(call.inputs["prior_mask_path"]),
            valid.shape,
            pixel_window=call.parameters.get("pixel_window"),
        )
        prediction = self.predictor.predict(image, prior)
        probability = np.asarray(prediction["mask_probability"], dtype=np.float32).squeeze()
        if probability.shape != valid.shape or not np.all(np.isfinite(probability)):
            raise ValueError("segmentation backend returned an invalid mask probability")
        probability = np.clip(probability, 0.0, 1.0)
        threshold = float(call.parameters.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        prior_binary = (prior >= 0.5) & valid
        mask, statistics = _segmentation_statistics(
            probability, prior_binary, valid, threshold=threshold
        )

        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}_segment.npy"
        np.save(output, mask.astype(np.uint8))
        artifacts = [str(output.resolve())]
        outputs: dict[str, Any] = {
            **metadata,
            "backend": type(self.predictor).__name__,
            **statistics,
        }
        for name in ("add", "remove"):
            value = prediction.get(f"{name}_probability")
            if value is None:
                continue
            change_probability = np.asarray(value, dtype=np.float32).squeeze()
            if change_probability.shape != valid.shape or not np.all(
                np.isfinite(change_probability)
            ):
                raise ValueError(
                    f"segmentation backend returned an invalid {name} probability"
                )
            change_probability = np.clip(change_probability, 0.0, 1.0)
            change_threshold = float(
                call.parameters.get(f"{name}_threshold", threshold)
            )
            if not 0.0 <= change_threshold <= 1.0:
                raise ValueError(f"{name}_threshold must be in [0, 1]")
            change_mask = (change_probability >= change_threshold) & valid
            change_output = self.output_root / f"{call.call_id}_{name}.npy"
            np.save(change_output, change_mask.astype(np.uint8))
            artifacts.append(str(change_output.resolve()))
            outputs.update(
                {
                    f"learned_{name}_fraction": float(np.mean(change_mask[valid])),
                    f"mean_{name}_probability": float(
                        np.mean(change_probability[valid])
                    ),
                    f"{name}_threshold": change_threshold,
                }
            )
        threshold_grid = call.parameters.get("threshold_grid")
        if threshold_grid is not None:
            if not isinstance(threshold_grid, list) or not threshold_grid:
                raise ValueError("threshold_grid must be a non-empty list")
            thresholds = sorted({float(value) for value in threshold_grid})
            if len(thresholds) > 16 or any(
                value < 0.0 or value > 1.0 for value in thresholds
            ):
                raise ValueError("threshold_grid must contain at most 16 values in [0, 1]")
            outputs["threshold_sweep"] = [
                _segmentation_statistics(
                    probability, prior_binary, valid, threshold=value
                )[1]
                for value in thresholds
            ]
        for key in (
            "edit_probabilities",
            "predicted_edit",
            "gated_edit",
            "confidence",
            "update_probability",
            "uncertainty",
            "commit_threshold",
            "base_predicted_edit",
            "refinement_invoked",
            "refinement_change_fraction",
            "backend_version",
            "model_checkpoint",
            "source_model_checkpoint",
            "road_threshold",
        ):
            value = prediction.get(key)
            if value is not None:
                outputs[key] = (
                    np.asarray(value).astype(float).tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs=outputs,
            artifacts=artifacts,
            cost=self.cost,
        )

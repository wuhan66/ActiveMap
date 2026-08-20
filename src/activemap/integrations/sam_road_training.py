"""MUNO21 supervision and metrics for SAM-Road road-head adaptation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RoadImageRecord:
    image_id: int
    sample_id: str
    image_path: Path
    aoi_id: str
    edit_type: str
    width: int
    height: int
    segmentations: tuple[dict[str, Any], ...]


def decode_uncompressed_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Decode the uncompressed, column-major RLE used by COCO."""

    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
    ):
        raise ValueError("RLE size must contain two positive integers")
    if not isinstance(counts, list) or not all(
        isinstance(value, int) and value >= 0 for value in counts
    ):
        raise ValueError("only non-negative uncompressed COCO RLE is supported")
    total = int(size[0]) * int(size[1])
    if sum(counts) != total:
        raise ValueError(f"RLE counts sum to {sum(counts)}, expected {total}")
    flat = np.zeros(total, dtype=np.uint8)
    offset = 0
    foreground = False
    for run_length in counts:
        if foreground and run_length:
            flat[offset : offset + run_length] = 1
        offset += run_length
        foreground = not foreground
    return flat.reshape((int(size[0]), int(size[1])), order="F")


def union_segmentations(
    segmentations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for segmentation in segmentations:
        decoded = decode_uncompressed_rle(segmentation)
        if decoded.shape != mask.shape:
            raise ValueError(
                f"annotation shape {decoded.shape} differs from image shape {mask.shape}"
            )
        mask |= decoded
    return mask


def load_road_records(
    annotation_path: Path,
    image_root: Path,
    *,
    include_aois: set[str] | None = None,
    exclude_aois: set[str] | None = None,
) -> list[RoadImageRecord]:
    if include_aois and exclude_aois:
        raise ValueError("include_aois and exclude_aois are mutually exclusive")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO payload must contain image and annotation lists")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, dict) or "image_id" not in annotation:
            raise ValueError("invalid COCO annotation")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, dict):
            raise ValueError("MUNO21 road adaptation requires RLE segmentations")
        by_image[int(annotation["image_id"])].append(segmentation)

    records = []
    seen_ids: set[int] = set()
    for image in images:
        image_id = int(image["id"])
        if image_id in seen_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        seen_ids.add(image_id)
        aoi_id = str(image["aoi_id"])
        if include_aois is not None and aoi_id not in include_aois:
            continue
        if exclude_aois is not None and aoi_id in exclude_aois:
            continue
        path = image_root / str(image["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            RoadImageRecord(
                image_id=image_id,
                sample_id=str(image["activemap_sample_id"]),
                image_path=path,
                aoi_id=aoi_id,
                edit_type=str(image.get("edit_type", "UNKNOWN")),
                width=int(image["width"]),
                height=int(image["height"]),
                segmentations=tuple(by_image[image_id]),
            )
        )
    if not records:
        raise ValueError(f"no images selected from {annotation_path}")
    return records


def split_support(records: list[RoadImageRecord]) -> dict[str, Any]:
    aois: dict[str, int] = defaultdict(int)
    edits: dict[str, int] = defaultdict(int)
    empty = 0
    for record in records:
        aois[record.aoi_id] += 1
        edits[record.edit_type] += 1
        empty += int(not record.segmentations)
    return {
        "images": len(records),
        "aois": dict(sorted(aois.items())),
        "edit_types": dict(sorted(edits.items())),
        "images_without_annotations": empty,
    }


class MUNO21RoadDataset:
    """Small torch-compatible dataset without importing torch at module import."""

    def __init__(
        self,
        records: list[RoadImageRecord],
        *,
        patch_size: int = 256,
        augment: bool = False,
        prior_paths: dict[str, Path] | None = None,
        valid_paths: dict[str, Path] | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.records = records
        self.patch_size = patch_size
        self.augment = augment
        self.prior_paths = prior_paths
        self.valid_paths = valid_paths
        if prior_paths is not None:
            missing = [
                record.sample_id
                for record in records
                if record.sample_id not in prior_paths
            ]
            if missing:
                raise ValueError(f"missing prior masks for {len(missing)} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torch.nn import functional as functional

        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.float32).copy()
        if image.shape[:2] != (record.height, record.width):
            raise ValueError(
                f"image shape {image.shape[:2]} differs from annotation metadata "
                f"{(record.height, record.width)}"
            )
        mask = union_segmentations(
            record.segmentations, height=record.height, width=record.width
        )
        prior = None
        valid = np.ones((record.height, record.width), dtype=np.uint8)
        if self.prior_paths is not None:
            prior = np.asarray(np.load(self.prior_paths[record.sample_id])).squeeze()
            if prior.shape != (record.height, record.width):
                raise ValueError(f"prior shape mismatch for {record.sample_id}: {prior.shape}")
            prior = (prior > 0.5).astype(np.uint8)
        if self.valid_paths is not None and record.sample_id in self.valid_paths:
            valid = np.asarray(np.load(self.valid_paths[record.sample_id])).squeeze()
            if valid.shape != (record.height, record.width):
                raise ValueError(f"valid shape mismatch for {record.sample_id}: {valid.shape}")
            valid = (valid > 0.5).astype(np.uint8)
        if self.augment:
            rotation = random.randrange(4)
            if rotation:
                image = np.rot90(image, rotation, axes=(0, 1)).copy()
                mask = np.rot90(mask, rotation, axes=(0, 1)).copy()
                valid = np.rot90(valid, rotation, axes=(0, 1)).copy()
                if prior is not None:
                    prior = np.rot90(prior, rotation, axes=(0, 1)).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                valid = np.flip(valid, axis=1).copy()
                if prior is not None:
                    prior = np.flip(prior, axis=1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
                valid = np.flip(valid, axis=0).copy()
                if prior is not None:
                    prior = np.flip(prior, axis=0).copy()
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-10.0, 10.0)
            image = np.clip((image - 127.5) * contrast + 127.5 + brightness, 0, 255)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid).float().unsqueeze(0).unsqueeze(0)
        prior_tensor = (
            torch.from_numpy(prior).float().unsqueeze(0).unsqueeze(0)
            if prior is not None
            else None
        )
        if image.shape[:2] != (self.patch_size, self.patch_size):
            image_tensor = functional.interpolate(
                image_tensor,
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_tensor = functional.interpolate(
                mask_tensor, size=(self.patch_size, self.patch_size), mode="nearest"
            )
            valid_tensor = functional.interpolate(
                valid_tensor, size=(self.patch_size, self.patch_size), mode="nearest"
            )
            if prior_tensor is not None:
                prior_tensor = functional.interpolate(
                    prior_tensor, size=(self.patch_size, self.patch_size), mode="nearest"
                )
        result = {
            "image": image_tensor[0],
            "mask": mask_tensor[0, 0],
            "valid": valid_tensor[0, 0],
            "image_id": record.image_id,
            "sample_id": record.sample_id,
            "aoi_id": record.aoi_id,
            "edit_type": record.edit_type,
            "image_path": str(record.image_path),
        }
        if prior_tensor is not None:
            result["prior"] = prior_tensor[0, 0]
        return result


def road_bce_dice_loss(
    logits: Any,
    target: Any,
    *,
    positive_weight: float,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.nn import functional as functional

    if logits.shape != target.shape:
        raise ValueError(f"logit shape {logits.shape} differs from target {target.shape}")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be non-negative with a positive sum")
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, target.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = bce_weight * bce + dice_weight * dice
    return total, {"bce": bce, "dice": dice}


class ThresholdMetrics:
    def __init__(self, thresholds: tuple[float, ...]) -> None:
        if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("thresholds must be non-empty and lie in (0, 1)")
        self.thresholds = thresholds
        self.tp = np.zeros(len(thresholds), dtype=np.int64)
        self.fp = np.zeros(len(thresholds), dtype=np.int64)
        self.fn = np.zeros(len(thresholds), dtype=np.int64)
        self.tn = np.zeros(len(thresholds), dtype=np.int64)
        self.image_iou_sum = np.zeros(len(thresholds), dtype=np.float64)
        self.image_f1_sum = np.zeros(len(thresholds), dtype=np.float64)
        self.image_count = 0

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        probability = np.asarray(probability, dtype=np.float32)
        target = np.asarray(target, dtype=bool)
        if probability.shape != target.shape:
            raise ValueError("probability and target shapes differ")
        if probability.ndim < 2:
            raise ValueError("segmentation metrics require at least two dimensions")
        if probability.ndim == 2:
            probability = probability[None, ...]
            target = target[None, ...]
        else:
            probability = probability.reshape((-1, *probability.shape[-2:]))
            target = target.reshape((-1, *target.shape[-2:]))
        for index, threshold in enumerate(self.thresholds):
            prediction = probability >= threshold
            self.tp[index] += int(np.logical_and(prediction, target).sum())
            self.fp[index] += int(np.logical_and(prediction, ~target).sum())
            self.fn[index] += int(np.logical_and(~prediction, target).sum())
            self.tn[index] += int(np.logical_and(~prediction, ~target).sum())
            for image_prediction, image_target in zip(prediction, target, strict=True):
                tp = int(np.logical_and(image_prediction, image_target).sum())
                fp = int(np.logical_and(image_prediction, ~image_target).sum())
                fn = int(np.logical_and(~image_prediction, image_target).sum())
                self.image_iou_sum[index] += tp / max(tp + fp + fn, 1)
                self.image_f1_sum[index] += 2 * tp / max(2 * tp + fp + fn, 1)
        self.image_count += int(probability.shape[0])

    def summary(self) -> list[dict[str, float | int]]:
        rows = []
        for index, threshold in enumerate(self.thresholds):
            tp, fp, fn, tn = (
                int(self.tp[index]),
                int(self.fp[index]),
                int(self.fn[index]),
                int(self.tn[index]),
            )
            rows.append(
                {
                    "threshold": threshold,
                    "iou": tp / max(tp + fp + fn, 1),
                    "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
                    "mean_image_iou": self.image_iou_sum[index]
                    / max(self.image_count, 1),
                    "mean_image_f1": self.image_f1_sum[index]
                    / max(self.image_count, 1),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )
        return rows

    def best(self) -> dict[str, float | int]:
        return max(self.summary(), key=lambda row: (float(row["f1"]), float(row["iou"])))

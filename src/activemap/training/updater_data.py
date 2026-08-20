"""Numpy-backed updater dataset for reproducible crop caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample

EDIT_TO_INDEX = {operation: index for index, operation in enumerate(EditOperation)}


@dataclass(frozen=True)
class UpdaterAugmentationConfig:
    enabled: bool = False
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    rotate90_probability: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    gaussian_noise: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> UpdaterAugmentationConfig:
        values = payload or {}
        config = cls(
            enabled=bool(values.get("enabled", False)),
            horizontal_flip_probability=float(
                values.get("horizontal_flip_probability", 0.0)
            ),
            vertical_flip_probability=float(values.get("vertical_flip_probability", 0.0)),
            rotate90_probability=float(values.get("rotate90_probability", 0.0)),
            brightness=float(values.get("brightness", 0.0)),
            contrast=float(values.get("contrast", 0.0)),
            gaussian_noise=float(values.get("gaussian_noise", 0.0)),
        )
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "rotate90_probability",
        ):
            probability = float(getattr(config, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"augmentation.{name} must be in [0, 1]")
        for name in ("brightness", "contrast", "gaussian_noise"):
            if float(getattr(config, name)) < 0.0:
                raise ValueError(f"augmentation.{name} must be non-negative")
        return config


def _image_to_channels_first(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("image array must have three dimensions")
    if array.shape[0] in {1, 3, 4}:
        output = array
    elif array.shape[-1] in {1, 3, 4}:
        output = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"cannot infer image channel axis from shape {array.shape}")
    output = output.astype(np.float32)
    if output.max(initial=0.0) > 1.0:
        output /= 255.0
    return output


def _load_mask(path: str) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"mask must have shape [H,W] or [1,H,W], got {array.shape}")
    return np.clip(array, 0.0, 1.0)


def _transform_bounds(
    bounds: Tensor,
    *,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotations: int,
) -> Tensor:
    if torch.count_nonzero(bounds).item() == 0:
        return bounds.clone()
    x1, y1, x2, y2 = bounds.unbind()
    if horizontal_flip:
        x1, x2 = 1.0 - x2, 1.0 - x1
    if vertical_flip:
        y1, y2 = 1.0 - y2, 1.0 - y1
    normalized_rotations = rotations % 4
    if normalized_rotations == 1:
        x1, y1, x2, y2 = y1, 1.0 - x2, y2, 1.0 - x1
    elif normalized_rotations == 2:
        x1, y1, x2, y2 = 1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1
    elif normalized_rotations == 3:
        x1, y1, x2, y2 = 1.0 - y2, x1, 1.0 - y1, x2
    return torch.stack((x1, y1, x2, y2))


def transform_geometry_delta(
    geometry_delta: Tensor,
    *,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotations: int,
) -> Tensor:
    """Apply paired crop transforms to target and recovered prior bounding boxes."""

    target = geometry_delta[:4]
    prior = target - geometry_delta[4:]
    transformed_target = _transform_bounds(
        target,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotations=rotations,
    )
    transformed_prior = _transform_bounds(
        prior,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotations=rotations,
    )
    return torch.cat((transformed_target, transformed_target - transformed_prior))


class UpdaterDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        samples: list[UpdaterSample],
        augmentation: UpdaterAugmentationConfig | None = None,
        input_size: int | None = None,
        temporal_pair_input: bool = False,
    ) -> None:
        self.samples = samples
        self.augmentation = augmentation or UpdaterAugmentationConfig()
        if input_size is not None and input_size < 16:
            raise ValueError("input_size must be at least 16 pixels")
        self.input_size = input_size
        self.temporal_pair_input = temporal_pair_input

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image = _image_to_channels_first(np.load(sample.image_path))
        if self.temporal_pair_input:
            if sample.prior_image_path is None:
                raise ValueError(
                    "temporal_pair_input requires prior_image_path for every sample; "
                    f"missing from {sample.sample_id}"
                )
            prior_image = _image_to_channels_first(np.load(sample.prior_image_path))
            if prior_image.shape != image.shape:
                raise ValueError(
                    "temporal image shape mismatch in sample "
                    f"{sample.sample_id}: {prior_image.shape} vs {image.shape}"
                )
            # Channel order is fixed as RGB_(t-1) || RGB_t for reproducible inference.
            image = np.concatenate((prior_image, image), axis=0)
        prior = _load_mask(sample.prior_mask_path)
        target = _load_mask(sample.target_mask_path)
        valid = (
            _load_mask(sample.valid_mask_path)
            if sample.valid_mask_path is not None
            else np.ones_like(target, dtype=np.float32)
        )
        if image.shape[-2:] != prior.shape[-2:] or prior.shape != target.shape:
            raise ValueError(f"spatial shape mismatch in sample {sample.sample_id}")
        image_tensor = torch.from_numpy(image)
        prior_tensor = torch.from_numpy(prior)
        target_tensor = torch.from_numpy(target)
        valid_tensor = torch.from_numpy(valid)
        geometry_tensor = torch.tensor(sample.geometry_delta, dtype=torch.float32)
        if self.input_size is not None and image_tensor.shape[-2:] != (
            self.input_size,
            self.input_size,
        ):
            output_size = (self.input_size, self.input_size)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            prior_tensor = F.interpolate(
                prior_tensor.unsqueeze(0), size=output_size, mode="nearest"
            ).squeeze(0)
            target_tensor = F.interpolate(
                target_tensor.unsqueeze(0), size=output_size, mode="nearest"
            ).squeeze(0)
            valid_tensor = F.interpolate(
                valid_tensor.unsqueeze(0), size=output_size, mode="nearest"
            ).squeeze(0)
        if self.augmentation.enabled:
            horizontal_flip = bool(
                torch.rand(()) < self.augmentation.horizontal_flip_probability
            )
            vertical_flip = bool(torch.rand(()) < self.augmentation.vertical_flip_probability)
            rotations = (
                int(torch.randint(1, 4, ()).item())
                if torch.rand(()) < self.augmentation.rotate90_probability
                else 0
            )
            dimensions = (-2, -1)
            if horizontal_flip:
                image_tensor = torch.flip(image_tensor, dims=(-1,))
                prior_tensor = torch.flip(prior_tensor, dims=(-1,))
                target_tensor = torch.flip(target_tensor, dims=(-1,))
                valid_tensor = torch.flip(valid_tensor, dims=(-1,))
            if vertical_flip:
                image_tensor = torch.flip(image_tensor, dims=(-2,))
                prior_tensor = torch.flip(prior_tensor, dims=(-2,))
                target_tensor = torch.flip(target_tensor, dims=(-2,))
                valid_tensor = torch.flip(valid_tensor, dims=(-2,))
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, dims=dimensions)
                prior_tensor = torch.rot90(prior_tensor, rotations, dims=dimensions)
                target_tensor = torch.rot90(target_tensor, rotations, dims=dimensions)
                valid_tensor = torch.rot90(valid_tensor, rotations, dims=dimensions)
            geometry_tensor = transform_geometry_delta(
                geometry_tensor,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rotations=rotations,
            )
            if self.augmentation.contrast > 0.0:
                contrast = 1.0 + float(
                    torch.empty(()).uniform_(
                        -self.augmentation.contrast, self.augmentation.contrast
                    )
                )
                image_tensor = (image_tensor - 0.5) * contrast + 0.5
            if self.augmentation.brightness > 0.0:
                brightness = float(
                    torch.empty(()).uniform_(
                        -self.augmentation.brightness, self.augmentation.brightness
                    )
                )
                image_tensor = image_tensor + brightness
            if self.augmentation.gaussian_noise > 0.0:
                image_tensor = image_tensor + torch.randn_like(image_tensor) * float(
                    self.augmentation.gaussian_noise
                )
            image_tensor = image_tensor.clamp(0.0, 1.0)
        return {
            "sample_id": sample.sample_id,
            "aoi_id": sample.aoi_id or sample.sample_id,
            "dataset_name": sample.dataset_name,
            "geometry_family": sample.geometry_family,
            "supervision_type": sample.supervision_type,
            "image": image_tensor,
            "prior_mask": prior_tensor,
            "target_mask": target_tensor,
            "valid_mask": valid_tensor,
            "edit_target": torch.tensor(EDIT_TO_INDEX[sample.edit_type], dtype=torch.long),
            "geometry_target": geometry_tensor,
        }

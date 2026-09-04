"""Dependency-light runtime smoke for the paired-temporal updater contract."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from activemap.data.updater_crops import build_updater_crops
from activemap.models import EditOperation
from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.oracle.updater_counterfactual import build_selector_oracle_input_cache
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import UpdaterSample


def main() -> None:
    with TemporaryDirectory(prefix="activemap-v5-temporal-") as temporary:
        root = Path(temporary)
        old_path = root / "old.npy"
        current_path = root / "current.npy"
        mask_path = root / "mask.npy"
        np.save(old_path, np.full((3, 16, 16), 0.25, dtype=np.float32))
        np.save(current_path, np.full((3, 16, 16), 0.75, dtype=np.float32))
        np.save(mask_path, np.zeros((1, 16, 16), dtype=np.float32))
        sample = UpdaterSample(
            sample_id="temporal-smoke",
            split="train",
            image_path=str(current_path),
            prior_image_path=str(old_path),
            prior_mask_path=str(mask_path),
            target_mask_path=str(mask_path),
            edit_type=EditOperation.KEEP,
            geometry_delta=[0.0] * 8,
        )
        item = UpdaterDataset([sample], temporal_pair_input=True)[0]
        image = item["image"]
        prior = item["prior_mask"]
        if not isinstance(image, torch.Tensor) or not isinstance(prior, torch.Tensor):
            raise RuntimeError("paired-temporal dataset did not return tensors")
        if image.shape != (6, 16, 16):
            raise RuntimeError(f"expected six temporal channels, got {tuple(image.shape)}")
        if not torch.allclose(image[:3], torch.full_like(image[:3], 0.25)):
            raise RuntimeError("old RGB channels are not first")
        if not torch.allclose(image[3:], torch.full_like(image[3:], 0.75)):
            raise RuntimeError("current RGB channels are not last")
        model = PriorConditionedUNet(
            UpdaterConfig(
                image_channels=6,
                temporal_pair_input=True,
                base_channels=8,
                dropout=0.0,
            )
        )
        output = model(image[None], prior[None])
        if output["segmentation_logits"].shape != (1, 1, 16, 16):
            raise RuntimeError("temporal updater forward shape is invalid")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "image_shape": list(image.shape),
                    "segmentation_shape": list(output["segmentation_logits"].shape),
                    "crop_builder_imported": build_updater_crops.__name__,
                    "oracle_cache_builder_imported": build_selector_oracle_input_cache.__name__,
                }
            )
        )


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pytest

from scripts.train_sn7_changemamba import Record
from scripts.train_sn7_opencd_ban import (
    BAN_MEAN,
    BAN_STD,
    SN7BANDataset,
    _optimizer_groups,
)

torch = pytest.importorskip("torch")


def _record(tmp_path: Path) -> Record:
    image = np.full((3, 4, 4), 0.75, dtype=np.float32)
    prior = np.zeros((4, 4), dtype=np.float32)
    prior[:, :2] = 1.0
    paths = {}
    for name, value in {
        "image": image,
        "prior": prior,
        "target": prior,
        "valid": np.ones((4, 4), dtype=np.float32),
    }.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, value)
        paths[name] = path
    return Record(
        sample_id="sample",
        aoi_id="aoi",
        split="train",
        image=paths["image"],
        prior=paths["prior"],
        target=paths["target"],
        valid=paths["valid"],
        edit_type="KEEP",
    )


def test_dataset_uses_ban_rgb_normalization(tmp_path: Path) -> None:
    item = SN7BANDataset(
        [_record(tmp_path)], augment=False, image_size=4
    )[0]
    mean = torch.tensor(BAN_MEAN)[:, None, None]
    std = torch.tensor(BAN_STD)[:, None, None]
    restored_image = item["new_rgb"] * std + mean
    restored_prior = item["old_map"] * std + mean
    assert torch.allclose(restored_image, torch.full_like(restored_image, 191.25))
    assert torch.allclose(restored_prior[:, :, :2], torch.full((3, 4, 2), 255.0))
    assert torch.allclose(
        restored_prior[:, :, 2:], torch.zeros((3, 4, 2)), atol=1e-5
    )


def test_optimizer_groups_apply_ban_multipliers() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = torch.nn.Linear(2, 2)
            self.mask_decoder = torch.nn.Linear(2, 2)
            self.norm = torch.nn.LayerNorm(2)
            self.other = torch.nn.Linear(2, 2)

    groups = _optimizer_groups(
        Tiny().named_parameters(), learning_rate=1e-4, weight_decay=1e-4
    )
    signatures = {(group["lr"], group["weight_decay"]) for group in groups}
    assert (1e-5, 1e-4) in signatures
    assert (1e-3, 1e-4) in signatures
    assert (1e-4, 1e-4) in signatures
    assert all(
        group["weight_decay"] == 0.0
        for group in groups
        if any(parameter.ndim == 1 for parameter in group["params"])
    )

from pathlib import Path

import numpy as np
import pytest

from scripts.train_sn7_changemamba import (
    Record,
    SN7ChangeDataset,
    _translate_no_wrap,
)

torch = pytest.importorskip("torch")


def _record(tmp_path: Path) -> Record:
    image = np.full((3, 4, 4), 0.75, dtype=np.float32)
    prior = np.zeros((4, 4), dtype=np.float32)
    prior[:, :2] = 1.0
    target = prior.copy()
    valid = np.ones((4, 4), dtype=np.float32)
    paths = {}
    for name, value in {
        "image": image,
        "prior": prior,
        "target": target,
        "valid": valid,
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


def _denormalize(value, dataset):
    return value * dataset.std + dataset.mean


def test_input_modes_mask_only_the_declared_modality(tmp_path: Path) -> None:
    record = _record(tmp_path)
    full = SN7ChangeDataset(
        [record], augment=False, image_size=4, input_mode="image_prior"
    )
    image_only = SN7ChangeDataset(
        [record], augment=False, image_size=4, input_mode="image_only"
    )
    prior_only = SN7ChangeDataset(
        [record], augment=False, image_size=4, input_mode="prior_only"
    )

    full_item = full[0]
    image_item = image_only[0]
    prior_item = prior_only[0]
    assert torch.allclose(full_item["new_rgb"], image_item["new_rgb"])
    assert torch.allclose(full_item["old_map"], prior_item["old_map"])
    assert torch.allclose(
        _denormalize(image_item["old_map"], image_only),
        torch.zeros_like(image_item["old_map"]),
        atol=1e-7,
    )
    assert torch.allclose(
        _denormalize(prior_item["new_rgb"], prior_only),
        torch.zeros_like(prior_item["new_rgb"]),
        atol=1e-7,
    )
    assert torch.equal(full_item["change"], image_item["change"])
    assert torch.equal(full_item["change"], prior_item["change"])


def test_invalid_input_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported input mode"):
        SN7ChangeDataset(
            [_record(tmp_path)], augment=False, image_size=4, input_mode="invalid"
        )


def test_translation_has_no_wrapped_pixels() -> None:
    value = torch.zeros((1, 4, 4))
    value[0, 0, 0] = 1
    shifted = _translate_no_wrap(value, 1, 2)
    assert shifted[0, 1, 2] == 1
    assert shifted.sum() == 1
    removed = _translate_no_wrap(value, -1, -1)
    assert removed.sum() == 0


def test_negative_translation_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SN7ChangeDataset(
            [_record(tmp_path)],
            augment=True,
            image_size=4,
            max_translation_pixels=-1,
        )


def test_prior_input_corruption_preserves_writeback_prior_and_target(
    tmp_path: Path,
) -> None:
    dataset = SN7ChangeDataset(
        [_record(tmp_path)],
        augment=False,
        image_size=4,
        input_mode="image_prior",
        prior_input_translation_pixels=2,
        corruption_seed=17,
    )

    item = dataset[0]
    rendered_prior = _denormalize(item["old_map"], dataset)[0] >= 0.5

    assert torch.equal(item["prior"], item["target"])
    assert torch.equal(item["change"], torch.zeros_like(item["change"]))
    assert not torch.equal(rendered_prior, item["prior"])
    assert tuple(item["prior_input_shift"].tolist()) != (0, 0)


def test_prior_input_corruption_is_deterministic_per_sample(
    tmp_path: Path,
) -> None:
    kwargs = {
        "augment": False,
        "image_size": 4,
        "input_mode": "image_prior",
        "prior_input_translation_pixels": 2,
        "corruption_seed": 23,
    }
    first = SN7ChangeDataset([_record(tmp_path)], **kwargs)[0]
    second = SN7ChangeDataset([_record(tmp_path)], **kwargs)[0]

    assert torch.equal(first["old_map"], second["old_map"])
    assert torch.equal(first["prior_input_shift"], second["prior_input_shift"])


def test_negative_prior_input_corruption_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="prior_input_translation_pixels must be non-negative"
    ):
        SN7ChangeDataset(
            [_record(tmp_path)],
            augment=False,
            image_size=4,
            prior_input_translation_pixels=-1,
        )

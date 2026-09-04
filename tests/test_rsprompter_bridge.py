import json
from pathlib import Path

import numpy as np

from activemap.integrations.rsprompter import (
    RSPrompterRefinementConfig,
    RSPrompterRefiner,
    audit_rsprompter_dataset,
    export_rsprompter_dataset,
    rsprompter_train_command,
)
from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample


def _write_sample(
    root: Path,
    *,
    sample_id: str,
    split: str,
    edit: EditOperation,
    geometry_family: str = "polygon",
) -> str:
    image = np.zeros((3, 16, 16), dtype=np.float32)
    image[:, 4:12, 4:12] = 1.0
    target = np.zeros((1, 16, 16), dtype=np.float32)
    if edit != EditOperation.DELETE:
        target[:, 4:12, 4:12] = 1.0
    prior = np.zeros_like(target)
    for name, array in (("image", image), ("target", target), ("prior", prior)):
        np.save(root / f"{sample_id}_{name}.npy", array)
    return UpdaterSample(
        sample_id=sample_id,
        aoi_id=f"aoi-{split}",
        split=split,
        image_path=str(root / f"{sample_id}_image.npy"),
        prior_mask_path=str(root / f"{sample_id}_prior.npy"),
        target_mask_path=str(root / f"{sample_id}_target.npy"),
        edit_type=edit,
        geometry_delta=[0.0] * 8,
        geometry_family=geometry_family,
    ).model_dump_json()


def test_export_rsprompter_coco_dataset(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.jsonl"
    records = [
        _write_sample(tmp_path, sample_id="train-delete", split="train", edit=EditOperation.DELETE),
        _write_sample(tmp_path, sample_id="val-add", split="val", edit=EditOperation.ADD),
        _write_sample(tmp_path, sample_id="test-reshape", split="test", edit=EditOperation.RESHAPE),
    ]
    manifest.write_text("\n".join(records) + "\n", encoding="utf-8")
    output = tmp_path / "coco"
    summary = export_rsprompter_dataset(
        manifest, output, splits=("train", "val", "test")
    )
    assert [split["images"] for split in summary["splits"]] == [1, 1, 1]
    assert summary["test_assets_read"] is True
    train = json.loads((output / "annotations" / "train.json").read_text(encoding="utf-8"))
    val = json.loads((output / "annotations" / "val.json").read_text(encoding="utf-8"))
    assert train["annotations"] == []
    assert len(val["annotations"]) == 1
    assert val["categories"] == [
        {"id": 1, "name": "building", "supercategory": "structure"}
    ]
    assert (output / "images" / "val" / val["images"][0]["file_name"]).is_file()


def test_export_rsprompter_uses_road_category_for_muno_polyline_data(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "road_samples.jsonl"
    records = [
        _write_sample(
            tmp_path,
            sample_id=f"road-{split}",
            split=split,
            edit=EditOperation.ADD,
            geometry_family="polyline",
        )
        for split in ("train", "val", "test")
    ]
    manifest.write_text("\n".join(records) + "\n", encoding="utf-8")

    summary = export_rsprompter_dataset(manifest, tmp_path / "road_coco")
    validation = json.loads(
        (tmp_path / "road_coco" / "annotations" / "val.json").read_text(
            encoding="utf-8"
        )
    )

    expected = [{"id": 1, "name": "road", "supercategory": "transportation"}]
    assert summary["categories"] == expected
    assert validation["categories"] == expected
    assert validation["images"][0]["geometry_family"] == "polyline"
    assert validation["annotations"][0]["category_id"] == 1
    segmentation = validation["annotations"][0]["segmentation"]
    assert segmentation["size"] == [16, 16]
    assert sum(segmentation["counts"]) == 16 * 16
    assert sum(segmentation["counts"][1::2]) == validation["annotations"][0]["area"]
    assert summary["requested_splits"] == ["train", "val"]
    assert summary["test_assets_read"] is False
    audit = audit_rsprompter_dataset(tmp_path / "road_coco")
    assert audit["valid"] is True
    assert audit["mask_encoding"] == "coco_uncompressed_rle"
    assert audit["test_assets_read"] is False
    assert [item["images"] for item in audit["splits"]] == [1, 1]


def test_rsprompter_export_rejects_unknown_or_duplicate_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        _write_sample(
            tmp_path,
            sample_id="train-add",
            split="train",
            edit=EditOperation.ADD,
        )
        + "\n",
        encoding="utf-8",
    )

    for splits in (("train", "train"), ("dev",)):
        try:
            export_rsprompter_dataset(manifest, tmp_path / "out", splits=splits)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid splits to fail: {splits}")


def test_rsprompter_command_and_keep_fallback(tmp_path: Path) -> None:
    command = rsprompter_train_command(
        python=Path("/env/bin/python"),
        repository=Path("/repo"),
        config=Path("/repo/config.py"),
        data_root=Path("/data"),
        work_dir=Path("/run"),
        resume=True,
    )
    assert command[-1] == "resume=True"
    refiner = RSPrompterRefiner(
        RSPrompterRefinementConfig(
            python=tmp_path / "missing-python",
            repository=tmp_path / "missing-repo",
            model_config=tmp_path / "missing-config",
            checkpoint=tmp_path / "missing-checkpoint",
            adapter_script=tmp_path / "missing-adapter",
            work_dir=tmp_path / "work",
        )
    )
    coarse = np.ones((8, 8), dtype=np.float32)
    result = refiner.refine(
        np.zeros((3, 8, 8), dtype=np.float32),
        coarse,
        np.zeros_like(coarse),
        edit_type=EditOperation.KEEP,
    )
    assert np.array_equal(result, coarse)

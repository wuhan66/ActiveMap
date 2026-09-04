import json
from pathlib import Path

import pytest

from scripts.stage_vlm_sft_nvme import stage_dataset


def _row(split: str, image: Path) -> dict:
    return {
        "split": split,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": "question"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"selection":"STOP"}'}],
            },
        ],
    }


def test_stage_rewrites_and_deduplicates_images(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "shared.png"
    image.write_bytes(b"fixture-image")
    for split in ("train", "val"):
        (source / f"{split}.jsonl").write_text(
            json.dumps(_row(split, image)) + "\n", encoding="utf-8"
        )
    output = tmp_path / "nvme"

    summary = stage_dataset(source, output, workers=2)

    assert summary["unique_images"] == 1
    assert summary["test_assets_read"] is False
    staged = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    staged_image = Path(staged["messages"][1]["content"][0]["image"])
    assert staged_image.is_file()
    assert staged_image.read_bytes() == b"fixture-image"
    assert (output / "stage_manifest.json").is_file()


def test_stage_refuses_test_rows(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "image.png"
    image.write_bytes(b"image")
    (source / "train.jsonl").write_text(
        json.dumps(_row("test", image)) + "\n", encoding="utf-8"
    )
    (source / "val.jsonl").write_text(
        json.dumps(_row("val", image)) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unexpected split"):
        stage_dataset(source, tmp_path / "nvme")

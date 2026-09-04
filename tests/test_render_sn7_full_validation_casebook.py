import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_sn7_full_validation_casebook import render


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    audits = []
    rows = []
    masks = []
    for index, edit in enumerate(("KEEP", "ADD", "DELETE", "RESHAPE")):
        sample_id = f"sample-{edit.lower()}"
        image = np.full((3, 12, 12), 0.2 + index * 0.1, dtype=np.float32)
        prior = np.zeros((12, 12), dtype=np.float32)
        target = np.zeros((12, 12), dtype=np.float32)
        if edit in {"KEEP", "DELETE", "RESHAPE"}:
            prior[3:9, 3:9] = 1
        if edit in {"KEEP", "ADD"}:
            target[3:9, 3:9] = 1
        if edit == "RESHAPE":
            target[2:10, 5:7] = 1
        valid = np.ones((12, 12), dtype=np.float32)
        paths = {}
        for name, value in {"image": image, "prior_image": image, "prior": prior, "target": target, "valid": valid}.items():
            path = tmp_path / f"{sample_id}-{name}.npy"
            np.save(path, value)
            paths[name] = path.name
        rows.append({
            "sample_id": sample_id,
            "aoi_id": "aoi-1",
            "split": "val",
            "image_path": paths["image"],
            "prior_image_path": paths["prior_image"],
            "prior_mask_path": paths["prior"],
            "target_mask_path": paths["target"],
            "valid_mask_path": paths["valid"],
            "edit_type": edit,
        })
        masks.append(np.packbits(np.logical_xor(prior >= 0.5, target >= 0.5).reshape(-1)))
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for method in ("frozen", "latest"):
        audit = tmp_path / method
        audit.mkdir()
        sample_rows = [
            {
                "sample_id": row["sample_id"],
                "committed_map_iou": 0.7 if method == "frozen" else 0.6,
                "change_iou": 0.5,
            }
            for row in rows
        ]
        (audit / "per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sample_rows), encoding="utf-8"
        )
        np.savez_compressed(
            audit / "predicted_change_masks.npz",
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            packed_masks=np.stack(masks),
            mask_shape=np.asarray((12, 12)),
        )
        audits.append(audit)
    return manifest, audits[0], audits[1]


def test_casebook_exports_every_shared_validation_state(tmp_path: Path) -> None:
    manifest, frozen, latest = _fixture(tmp_path)
    output = tmp_path / "casebook"
    summary = render(manifest, [("frozen", frozen), ("latest", latest)], output, tile_size=64)

    assert summary["case_count"] == 4
    assert summary["selection"] == "none; every shared validation state is rendered"
    index = [json.loads(line) for line in (output / "casebook_index.jsonl").read_text().splitlines()]
    assert {row["target_edit"] for row in index} == {"KEEP", "ADD", "DELETE", "RESHAPE"}
    for row in index:
        folder = output / row["folder"]
        assert (folder / "manifest.json").is_file()
        assert Image.open(folder / "overview.png").size == (256, 128)
        assert Image.open(folder / "zoom.png").size == (256, 128)


def test_casebook_individual_mode_omits_composite_boards(tmp_path: Path) -> None:
    manifest, frozen, latest = _fixture(tmp_path)
    output = tmp_path / "individual"
    summary = render(
        manifest,
        [("frozen", frozen), ("latest", latest)],
        output,
        tile_size=64,
        asset_mode="individual",
    )

    assert summary["asset_mode"] == "individual"
    row = json.loads((output / "casebook_index.jsonl").read_text().splitlines()[0])
    folder = output / row["folder"]
    assert not (folder / "overview.png").exists()
    assert not (folder / "zoom.png").exists()
    assert (folder / row["assets"]["Current image"]["layer"]).is_file()
    assert (folder / row["assets"]["Current image"]["crop"]).is_file()

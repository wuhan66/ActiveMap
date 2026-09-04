import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_sn7_full_validation_atlas import render_atlas


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit"
    audit.mkdir()
    manifest_rows = []
    audit_rows = []
    masks = []
    for index, edit in enumerate(("KEEP", "ADD", "DELETE", "RESHAPE")):
        sample_id = f"sample-{edit.lower()}"
        image = np.full((3, 8, 8), 0.2 + index * 0.1, dtype=np.float32)
        prior = np.zeros((8, 8), dtype=np.float32)
        target = np.zeros((8, 8), dtype=np.float32)
        if edit in {"KEEP", "DELETE", "RESHAPE"}:
            prior[2:6, 2:6] = 1
        if edit in {"KEEP", "ADD"}:
            target[2:6, 2:6] = 1
        if edit == "RESHAPE":
            target[1:7, 3:5] = 1
        paths = {}
        for name, value in {
            "image": image,
            "prior": prior,
            "target": target,
        }.items():
            path = tmp_path / f"{sample_id}-{name}.npy"
            np.save(path, value)
            paths[name] = path.name
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "aoi_id": "aoi",
                "split": "val",
                "image_path": paths["image"],
                "prior_mask_path": paths["prior"],
                "target_mask_path": paths["target"],
                "valid_mask_path": None,
                "edit_type": edit,
            }
        )
        change = np.logical_xor(prior >= 0.5, target >= 0.5)
        masks.append(np.packbits(change.reshape(-1)))
        audit_rows.append(
            {
                "sample_id": sample_id,
                "target_edit": edit,
                "committed_map_iou": 0.8,
                "map_iou_delta": 0.2,
                "change_iou": 0.7,
                "target_change_fraction": float(change.mean()),
                "predicted_change_fraction": float(change.mean()),
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    (audit / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audit_rows), encoding="utf-8"
    )
    np.savez_compressed(
        audit / "predicted_change_masks.npz",
        sample_ids=np.asarray([row["sample_id"] for row in audit_rows]),
        packed_masks=np.stack(masks),
        mask_shape=np.asarray((8, 8)),
    )
    return manifest, audit


def test_full_atlas_renders_every_shared_validation_sample(tmp_path: Path) -> None:
    manifest, audit = _write_fixture(tmp_path)
    output = tmp_path / "atlas"
    summary = render_atlas(
        manifest,
        "primary",
        audit,
        "baseline",
        audit,
        output,
        columns=2,
        rows=2,
        tile_size=8,
    )

    assert summary["sample_count"] == 4
    assert summary["selection"] == "none; every shared validation sample is rendered"
    pages = sorted(output.glob("*/*.png"))
    assert len(pages) == 4
    assert all(Image.open(path).size == (32, 16) for path in pages)

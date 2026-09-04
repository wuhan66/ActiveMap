import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_sn7_updater_qualitative import render


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit"
    audit.mkdir()
    rows = []
    masks = []
    manifest_rows = []
    for index, edit in enumerate(("KEEP", "ADD", "DELETE", "RESHAPE")):
        sample_id = f"sample-{edit.lower()}"
        image = np.full((3, 8, 8), 0.2 + index * 0.15, dtype=np.float32)
        prior = np.zeros((8, 8), dtype=np.float32)
        target = np.zeros((8, 8), dtype=np.float32)
        if edit in {"KEEP", "DELETE", "RESHAPE"}:
            prior[2:6, 2:6] = 1
        if edit in {"KEEP", "ADD"}:
            target[2:6, 2:6] = 1
        if edit == "RESHAPE":
            target[1:7, 3:5] = 1
        valid = np.ones((8, 8), dtype=np.float32)
        paths = {}
        for name, value in {
            "image": image,
            "prior_image": image * 0.8,
            "prior": prior,
            "target": target,
            "valid": valid,
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
                "prior_image_path": paths["prior_image"],
                "prior_mask_path": paths["prior"],
                "target_mask_path": paths["target"],
                "valid_mask_path": paths["valid"],
                "edit_type": edit,
            }
        )
        change = np.logical_xor(prior >= 0.5, target >= 0.5)
        masks.append(np.packbits(change.reshape(-1)))
        rows.append(
            {
                "sample_id": sample_id,
                "target_edit": edit,
                "map_iou_delta": 0.2,
                "committed_map_iou": 0.8,
                "change_iou": 0.7,
                "operation_correct": True,
                "prior_foreground_fraction": float((prior >= 0.5).mean()),
                "target_change_fraction": float(change.mean()),
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (audit / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    np.savez_compressed(
        audit / "predicted_change_masks.npz",
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        packed_masks=np.stack(masks),
        mask_shape=np.asarray((8, 8)),
    )
    return manifest, audit


def test_render_exports_unlabeled_individual_panels(tmp_path: Path) -> None:
    manifest, audit = _write_fixture(tmp_path)
    output = tmp_path / "visuals"
    summary = render(
        manifest,
        [("method", audit)],
        output,
        per_edit=1,
        failures_per_edit=0,
        disagreements_per_edit=0,
        min_visual_fraction=0.01,
        output_size=8,
    )

    assert summary["sample_count"] == 4
    folders = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(folders) == 4
    for folder in folders:
        expected = {
            "00_previous_rgb.png",
            "01_current_rgb.png",
            "02_prior_mask.png",
            "03_reference_final_mask.png",
            "04_truth_change_mask.png",
            "05_method_change_mask.png",
            "06_method_committed_mask.png",
            "07_method_overlay.png",
            "08_method_tp_fp_fn.png",
        }
        assert {path.name for path in folder.glob("*.png")} == expected
        for path in folder.glob("*.png"):
            assert Image.open(path).size == (8, 8)
        zoom = folder / "zoom"
        assert zoom.is_dir()
        assert {path.name for path in zoom.glob("*.png")} == expected
        for path in zoom.glob("*.png"):
            assert Image.open(path).size == (8, 8)

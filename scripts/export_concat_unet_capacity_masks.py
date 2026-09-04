#!/usr/bin/env python3
"""Export clean same-sample masks for the concat U-Net capacity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.training.selector import resolve_device
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import load_updater_samples


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_samples(
    prediction_paths: list[tuple[str, Path]],
    *,
    per_edit: int,
) -> list[dict[str, Any]]:
    predictions = {
        name: {str(row["sample_id"]): row for row in _read_jsonl(path)}
        for name, path in prediction_paths
    }
    shared = set.intersection(*(set(rows) for rows in predictions.values()))
    selected: list[dict[str, Any]] = []
    for edit in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        candidates = []
        for sample_id in shared:
            rows = [predictions[name][sample_id] for name, _ in prediction_paths]
            if str(rows[0]["target_edit"]) != edit:
                continue
            ious = [float(row["raster_iou"]) for row in rows]
            candidates.append(
                {
                    "sample_id": sample_id,
                    "target_edit": edit,
                    "disagreement": max(ious) - min(ious),
                    "worst_iou": min(ious),
                    "per_method": {
                        name: {
                            "predicted_edit": predictions[name][sample_id][
                                "predicted_edit"
                            ],
                            "raster_iou": float(
                                predictions[name][sample_id]["raster_iou"]
                            ),
                        }
                        for name, _ in prediction_paths
                    },
                }
            )
        candidates.sort(
            key=lambda row: (row["disagreement"], -row["worst_iou"]), reverse=True
        )
        selected.extend(candidates[:per_edit])
    return selected


def _save_rgb(path: Path, tensor: torch.Tensor, size: int) -> None:
    value = tensor.detach().cpu().numpy()
    value = np.moveaxis(value[:3], 0, -1)
    image = Image.fromarray((np.clip(value, 0.0, 1.0) * 255).astype(np.uint8))
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(path)


def _save_mask(path: Path, tensor: torch.Tensor, size: int) -> None:
    value = tensor.detach().cpu().numpy().squeeze() > 0.5
    image = Image.fromarray(np.where(value, 255, 0).astype(np.uint8), mode="L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    image.save(path)


def export_masks(
    samples_path: Path,
    checkpoints: list[tuple[str, Path]],
    prediction_paths: list[tuple[str, Path]],
    output_dir: Path,
    *,
    split: str,
    device_name: str,
    per_edit: int,
    output_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    checkpoint_names = [name for name, _ in checkpoints]
    prediction_names = [name for name, _ in prediction_paths]
    if checkpoint_names != prediction_names:
        raise ValueError("checkpoint and prediction names/order must match")
    selected = select_samples(prediction_paths, per_edit=per_edit)
    selected_ids = {row["sample_id"] for row in selected}
    samples = [
        sample
        for sample in load_updater_samples(samples_path, split=split)
        if sample.sample_id in selected_ids
    ]
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if set(sample_by_id) != selected_ids:
        raise ValueError("selected predictions do not match dataset samples")

    device = resolve_device(device_name)
    models: dict[str, PriorConditionedUNet] = {}
    checkpoint_meta = {}
    for name, path in checkpoints:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = PriorConditionedUNet(UpdaterConfig(**payload["model_config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models[name] = model
        checkpoint_meta[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    output_dir.mkdir(parents=True)
    index = []
    with torch.no_grad():
        for rank, row in enumerate(selected, start=1):
            sample = sample_by_id[row["sample_id"]]
            item = UpdaterDataset([sample])[0]
            folder = output_dir / f"{rank:03d}_{row['target_edit'].lower()}"
            folder.mkdir()
            _save_rgb(folder / "image.png", item["image"], output_size)
            _save_mask(folder / "prior.png", item["prior_mask"], output_size)
            _save_mask(folder / "target.png", item["target_mask"], output_size)
            image = item["image"].unsqueeze(0).to(device)
            prior = item["prior_mask"].unsqueeze(0).to(device)
            for name in checkpoint_names:
                outputs = models[name](image, prior)
                predicted = torch.sigmoid(outputs["segmentation_logits"]) >= 0.5
                _save_mask(folder / f"{name}.png", predicted[0], output_size)
            metadata = {**row, "folder": folder.name}
            (folder / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            index.append(metadata)

    manifest = {
        "schema_version": "sn7-concat-unet-capacity-mask-export-v1",
        "split": split,
        "test_assets_read": split == "test",
        "selection": "largest same-sample raster-IoU disagreement per edit",
        "per_edit": per_edit,
        "sample_count": len(index),
        "output_contract": (
            "one folder per sample; image/prior/target and method masks are "
            "independent PNG files without labels or frames"
        ),
        "checkpoints": checkpoint_meta,
        "predictions": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in prediction_paths
        },
        "samples": index,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=_named_path, action="append", required=True)
    parser.add_argument("--predictions", type=_named_path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per-edit", type=int, default=4)
    parser.add_argument("--output-size", type=int, default=512)
    args = parser.parse_args()
    if args.split == "test":
        parser.error("this paper-visual exporter is validation-only")
    payload = export_masks(
        args.samples,
        args.checkpoint,
        args.predictions,
        args.output_dir,
        split=args.split,
        device_name=args.device,
        per_edit=args.per_edit,
        output_size=args.output_size,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

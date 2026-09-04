#!/usr/bin/env python3
"""Export auditable SN7 predictions from an Open-CD BAN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_sn7_changemamba import (
    aggregate_rows,
    paper_prediction,
    sample_metrics,
)
from scripts.train_sn7_changemamba import _predicted_edit, _read_records
from scripts.train_sn7_opencd_ban import BANPairModel, SN7BANDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("opencd_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--no-save-masks", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("schema_version") != "sn7-opencd-ban-v1":
        raise ValueError("checkpoint is not an SN7 Open-CD BAN run")
    records = _read_records(args.manifest, args.split, None)
    loader = DataLoader(
        SN7BANDataset(records, augment=False, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = BANPairModel(
        opencd_repo=args.opencd_repo,
        config_path=args.config,
        clip_checkpoint=None,
        side_checkpoint=None,
        initialize=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()

    rows: list[dict[str, Any]] = []
    packed_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {args.split}", unit="batch"):
            logits = model(
                batch["old_map"].to(args.device, non_blocking=True),
                batch["new_rgb"].to(args.device, non_blocking=True),
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1).bool().cpu()
            pixel_confidence = probabilities.max(dim=1).values.cpu()
            change_probability = probabilities[:, 1].cpu()
            predictive_entropy = (
                -(probabilities.clamp_min(1e-8).log() * probabilities)
                .sum(dim=1)
                .cpu()
            )
            for index in range(predictions.shape[0]):
                prediction = predictions[index]
                prior = batch["prior"][index].bool()
                target = batch["target"][index].bool()
                valid = batch["valid"][index].bool()
                predicted_edit = _predicted_edit(prior, prediction, valid)
                metrics = sample_metrics(
                    prior.numpy(),
                    target.numpy(),
                    prediction.numpy(),
                    valid.numpy(),
                    target_edit=batch["edit_type"][index],
                    predicted_edit=predicted_edit,
                )
                valid_probability = change_probability[index][valid]
                valid_count = max(int(valid.sum().item()), 1)
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "aoi_id": batch["aoi_id"][index],
                        "split": args.split,
                        "target_edit": batch["edit_type"][index],
                        "predicted_edit": predicted_edit,
                        "confidence": float(
                            pixel_confidence[index][valid].mean().item()
                        ),
                        "mean_change_probability": float(
                            valid_probability.mean().item()
                        ),
                        "max_change_probability": float(valid_probability.max().item()),
                        "p95_change_probability": float(
                            torch.quantile(valid_probability, 0.95).item()
                        ),
                        "mean_predictive_entropy": float(
                            predictive_entropy[index][valid].mean().item()
                        ),
                        "prior_foreground_fraction": float(
                            (prior & valid).sum().item() / valid_count
                        ),
                        **metrics,
                    }
                )
                if not args.no_save_masks:
                    packed_predictions.append(
                        np.packbits(prediction.numpy().reshape(-1))
                    )

    if not args.no_save_masks:
        np.savez_compressed(
            args.output_dir / "predicted_change_masks.npz",
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            packed_masks=np.stack(packed_predictions),
            mask_shape=np.asarray((args.image_size, args.image_size)),
        )
    with (args.output_dir / "per_sample.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(paper_prediction(row)) + "\n")
    aggregate: dict[str, Any] = {
        "schema_version": "sn7-opencd-ban-audit-v1",
        "sample_count": len(rows),
        "split": args.split,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest_sha256": _sha256(args.manifest),
        "masks_saved": not args.no_save_masks,
        "test_assets_read": args.split == "test",
        **aggregate_rows(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

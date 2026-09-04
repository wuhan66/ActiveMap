#!/usr/bin/env python3
"""Export auditable per-sample predictions for a trained SN7 ChangeMamba run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_sn7_changemamba import (
    SN7ChangeDataset,
    _predicted_edit,
    _read_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sample_metrics(
    prior: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    *,
    target_edit: str,
    predicted_edit: str,
) -> dict[str, float | bool]:
    prior = prior.astype(bool)
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    valid = valid.astype(bool)
    committed = np.logical_xor(prior, prediction)
    truth_change = np.logical_xor(prior, target)

    def iou(left: np.ndarray, right: np.ndarray) -> float:
        intersection = np.logical_and(np.logical_and(left, right), valid).sum()
        union = np.logical_and(np.logical_or(left, right), valid).sum()
        return float(intersection / union) if union else 1.0

    valid_count = max(int(valid.sum()), 1)
    prior_iou = iou(prior, target)
    committed_iou = iou(committed, target)
    return {
        "prior_map_iou": prior_iou,
        "committed_map_iou": committed_iou,
        "map_iou_delta": committed_iou - prior_iou,
        "change_iou": iou(prediction, truth_change),
        "predicted_change_fraction": float(
            np.logical_and(prediction, valid).sum() / valid_count
        ),
        "target_change_fraction": float(
            np.logical_and(truth_change, valid).sum() / valid_count
        ),
        "operation_correct": predicted_edit == target_edit,
        "false_edit": target_edit == "KEEP" and predicted_edit != "KEEP",
        "missed_edit": target_edit != "KEEP" and predicted_edit == "KEEP",
        "wrong_edit": (
            target_edit != "KEEP"
            and predicted_edit not in {"KEEP", target_edit}
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty predictions")
    stable = [row for row in rows if row["target_edit"] == "KEEP"]
    updates = [row for row in rows if row["target_edit"] != "KEEP"]
    if not stable or not updates:
        raise ValueError("evaluation requires stable and update samples")
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "prior_map_iou",
            "committed_map_iou",
            "map_iou_delta",
            "change_iou",
            "operation_correct",
        )
    }
    result.update(
        {
            "false_edit_rate": float(
                np.mean([row["false_edit"] for row in stable])
            ),
            "missed_edit_rate": float(
                np.mean([row["missed_edit"] for row in updates])
            ),
            "wrong_edit_rate": float(
                np.mean([row["wrong_edit"] for row in updates])
            ),
        }
    )
    return result


def paper_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "aoi_id": row["aoi_id"],
        "target_edit": row["target_edit"],
        "predicted_edit": row["predicted_edit"],
        "confidence": row["confidence"],
        "committed": True,
        "raster_iou": row["committed_map_iou"],
        "polygon_iou": None,
        "topology_valid": None,
        "metadata": {
            "split": row["split"],
            "prior_map_iou": row["prior_map_iou"],
            "map_iou_delta": row["map_iou_delta"],
            "change_iou": row["change_iou"],
            "confidence_source": "mean_valid_pixel_max_softmax",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("change_mamba_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--input-mode",
        choices=("image_prior", "image_only", "prior_only"),
        help="Defaults to the checkpoint training mode.",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Skip packed mask export for calibration-only inference.",
    )
    parser.add_argument(
        "--prior-input-translation-pixels",
        type=int,
        default=0,
        help=(
            "Deterministically translate only the rendered prior-map model "
            "input. Ground-truth geometry and executable writeback stay fixed."
        ),
    )
    parser.add_argument(
        "--corruption-seed",
        type=int,
        default=0,
        help="Seed for deterministic per-sample prior-input translations.",
    )
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

    sys.path.insert(0, str(args.change_mamba_repo))
    from changedetection.configs.config import _C
    from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
    from changedetection.script.script_utils import get_vssm_kwargs

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_mode = checkpoint.get("arguments", {}).get("input_mode", "image_prior")
    input_mode = args.input_mode or checkpoint_mode
    if args.input_mode is not None and args.input_mode != checkpoint_mode:
        raise ValueError(
            f"input mode mismatch: checkpoint={checkpoint_mode}, requested={args.input_mode}"
        )
    records = _read_records(args.manifest, args.split, None)
    loader = DataLoader(
        SN7ChangeDataset(
            records,
            augment=False,
            image_size=args.image_size,
            input_mode=input_mode,
            prior_input_translation_pixels=args.prior_input_translation_pixels,
            corruption_seed=args.corruption_seed,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    config = _C.clone()
    config.defrost()
    config.merge_from_file(str(args.config))
    config.freeze()
    model = ChangeMambaBCD(pretrained=None, **get_vssm_kwargs(config))
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
                confidence = float(
                    pixel_confidence[index][valid].mean().item()
                )
                valid_count = max(int(valid.sum().item()), 1)
                valid_change_probability = change_probability[index][valid]
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "aoi_id": batch["aoi_id"][index],
                        "split": args.split,
                        "target_edit": batch["edit_type"][index],
                        "predicted_edit": predicted_edit,
                        "confidence": confidence,
                        "mean_change_probability": float(
                            valid_change_probability.mean().item()
                        ),
                        "max_change_probability": float(
                            valid_change_probability.max().item()
                        ),
                        "p95_change_probability": float(
                            torch.quantile(valid_change_probability, 0.95).item()
                        ),
                        "mean_predictive_entropy": float(
                            predictive_entropy[index][valid].mean().item()
                        ),
                        "prior_foreground_fraction": float(
                            (prior & valid).sum().item() / valid_count
                        ),
                        "prior_input_shift_y": int(
                            batch["prior_input_shift"][index, 0].item()
                        ),
                        "prior_input_shift_x": int(
                            batch["prior_input_shift"][index, 1].item()
                        ),
                        **metrics,
                    }
                )
                if not args.no_save_masks:
                    packed_predictions.append(
                        np.packbits(prediction.numpy().reshape(-1))
                    )

    if not args.no_save_masks:
        prediction_matrix = np.stack(packed_predictions)
        np.savez_compressed(
            args.output_dir / "predicted_change_masks.npz",
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            packed_masks=prediction_matrix,
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
        "schema_version": "sn7-changemamba-audit-v1",
        "sample_count": len(rows),
        "split": args.split,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest_sha256": _sha256(args.manifest),
        "masks_saved": not args.no_save_masks,
        "test_assets_read": args.split == "test",
        "prior_input_corruption": {
            "kind": "deterministic_translation_no_wrap",
            "max_pixels": args.prior_input_translation_pixels,
            "seed": args.corruption_seed,
            "writeback_prior_corrupted": False,
            "target_geometry_corrupted": False,
        },
        **aggregate_rows(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a prior-conditioned SN7 updater and export full visual-audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_sn7_changemamba import paper_prediction, sample_metrics


EDIT_NAMES = ("KEEP", "ADD", "DELETE", "RESHAPE")


def predicted_edit_from_change(
    prior: np.ndarray,
    change: np.ndarray,
    valid: np.ndarray,
) -> str:
    """Recover the executable edit type from a binary change mask."""

    changed = change.astype(bool) & valid.astype(bool)
    prior = prior.astype(bool)
    added = bool(np.any(changed & ~prior))
    removed = bool(np.any(changed & prior))
    if added and removed:
        return "RESHAPE"
    if added:
        return "ADD"
    if removed:
        return "DELETE"
    return "KEEP"


def binary_iou(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    left = left.astype(bool) & valid.astype(bool)
    right = right.astype(bool) & valid.astype(bool)
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def operation_slices(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for operation in EDIT_NAMES:
        subset = [row for row in rows if row["target_edit"] == operation]
        if not subset:
            continue
        output[operation] = {
            "sample_count": len(subset),
            "committed_map_iou": float(
                np.mean([row["committed_map_iou"] for row in subset])
            ),
            "map_iou_delta": float(np.mean([row["map_iou_delta"] for row in subset])),
            "change_iou": float(np.mean([row["change_iou"] for row in subset])),
            "operation_accuracy": float(
                np.mean([row["operation_correct"] for row in subset])
            ),
        }
    return output


def aggregate_visual_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        raise ValueError("cannot aggregate empty predictions")
    stable = [row for row in rows if row["target_edit"] == "KEEP"]
    updates = [row for row in rows if row["target_edit"] != "KEEP"]
    result: dict[str, float | None] = {
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
            "false_edit_rate": (
                float(np.mean([row["false_edit"] for row in stable])) if stable else None
            ),
            "missed_edit_rate": (
                float(np.mean([row["missed_edit"] for row in updates])) if updates else None
            ),
            "wrong_edit_rate": (
                float(np.mean([row["wrong_edit"] for row in updates])) if updates else None
            ),
        }
    )
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from activemap.nn.updater import (
        PriorConditionedUNet,
        UpdaterConfig,
        hierarchical_edit_predictions,
        operation_probabilities,
    )
    from activemap.training.selector import resolve_device
    from activemap.training.updater_data import EDIT_TO_INDEX, UpdaterDataset
    from activemap.updater_records import load_updater_samples

    if args.split == "test":
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = UpdaterConfig(**checkpoint["model_config"])
    model = PriorConditionedUNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    samples = load_updater_samples(args.manifest, split=args.split)
    if args.limit is not None:
        samples = samples[: args.limit]
    sample_by_id = {sample.sample_id: sample for sample in samples}
    loader = DataLoader(
        UpdaterDataset(
            samples,
            input_size=args.input_size,
            temporal_pair_input=config.temporal_pair_input,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    index_to_edit = {index: operation.value for operation, index in EDIT_TO_INDEX.items()}
    rows: list[dict[str, Any]] = []
    packed_change_masks: list[np.ndarray] = []
    packed_final_masks: list[np.ndarray] = []
    mask_shape: tuple[int, int] | None = None

    with torch.no_grad():
        for raw_batch in tqdm(loader, desc=f"visual audit {args.split}", unit="batch"):
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model(batch["image"], batch["prior_mask"])

            prior = batch["prior_mask"] >= 0.5
            if "temporal_change_logits" in outputs:
                temporal = torch.sigmoid(outputs["temporal_change_logits"])
                add_probability = temporal[:, 0:1]
                remove_probability = temporal[:, 1:2]
                final_probability = torch.where(prior, 1.0 - remove_probability, add_probability)
                change_probability = torch.where(prior, remove_probability, add_probability)
                predicted_final = torch.where(
                    prior,
                    remove_probability < args.remove_threshold,
                    add_probability >= args.add_threshold,
                )
            else:
                final_probability = torch.sigmoid(outputs["segmentation_logits"])
                predicted_final = final_probability >= args.segmentation_threshold
                change_probability = torch.where(prior, 1.0 - final_probability, final_probability)

            predicted_change = torch.logical_xor(prior, predicted_final)
            target = batch["target_mask"] >= 0.5
            valid = batch["valid_mask"] >= 0.5
            pixel_confidence = torch.maximum(final_probability, 1.0 - final_probability)
            entropy_probability = final_probability.float().clamp(1e-6, 1.0 - 1e-6)
            entropy = -(
                entropy_probability * entropy_probability.log()
                + (1.0 - entropy_probability) * (1.0 - entropy_probability).log()
            )
            head_edits = hierarchical_edit_predictions(
                outputs,
                batch["prior_mask"],
                presence_threshold=args.presence_threshold,
                change_threshold=args.change_threshold,
            )
            auxiliary_edits = torch.argmax(
                operation_probabilities(outputs, batch["prior_mask"]), dim=-1
            )
            full_scene = torch.tensor(
                [value == "full_scene_temporal" for value in raw_batch["supervision_type"]],
                device=device,
            )
            decoded_head_edits = torch.where(full_scene, auxiliary_edits, head_edits)

            for index, sample_id_value in enumerate(raw_batch["sample_id"]):
                sample_id = str(sample_id_value)
                sample = sample_by_id[sample_id]
                prior_np = prior[index, 0].detach().cpu().numpy()
                target_np = target[index, 0].detach().cpu().numpy()
                valid_np = valid[index, 0].detach().cpu().numpy()
                final_np = predicted_final[index, 0].detach().cpu().numpy() & valid_np
                change_np = predicted_change[index, 0].detach().cpu().numpy() & valid_np
                if mask_shape is None:
                    mask_shape = tuple(int(value) for value in change_np.shape)
                elif mask_shape != change_np.shape:
                    raise ValueError("all visual-audit masks must share one shape")

                target_edit = sample.edit_type.value
                predicted_edit = predicted_edit_from_change(prior_np, change_np, valid_np)
                metrics = sample_metrics(
                    prior_np,
                    target_np,
                    change_np,
                    valid_np,
                    target_edit=target_edit,
                    predicted_edit=predicted_edit,
                )
                truth_change = np.logical_xor(prior_np, target_np) & valid_np
                target_added = truth_change & ~prior_np
                target_removed = truth_change & prior_np
                predicted_added = change_np & ~prior_np
                predicted_removed = change_np & prior_np
                valid_count = max(int(np.count_nonzero(valid_np)), 1)
                valid_tensor = valid[index]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "aoi_id": sample.aoi_id or sample_id,
                        "split": args.split,
                        "target_edit": target_edit,
                        "predicted_edit": predicted_edit,
                        "head_predicted_edit": index_to_edit[
                            int(decoded_head_edits[index].detach().cpu())
                        ],
                        "confidence": float(pixel_confidence[index][valid_tensor].mean().detach().cpu()),
                        "head_confidence": float(torch.sigmoid(outputs["confidence_logits"])[index].detach().cpu()),
                        "mean_change_probability": float(change_probability[index][valid_tensor].mean().detach().cpu()),
                        "max_change_probability": float(change_probability[index][valid_tensor].max().detach().cpu()),
                        "p95_change_probability": float(
                            torch.quantile(
                                change_probability[index][valid_tensor].float(), 0.95
                            )
                            .detach()
                            .cpu()
                        ),
                        "mean_predictive_entropy": float(entropy[index][valid_tensor].mean().detach().cpu()),
                        "prior_foreground_fraction": float(np.count_nonzero(prior_np & valid_np) / valid_count),
                        "target_added_iou": binary_iou(predicted_added, target_added, valid_np),
                        "target_removed_iou": binary_iou(predicted_removed, target_removed, valid_np),
                        **metrics,
                    }
                )
                packed_change_masks.append(np.packbits(change_np.reshape(-1)))
                packed_final_masks.append(np.packbits(final_np.reshape(-1)))

    if mask_shape is None:
        raise ValueError("visual audit produced no predictions")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    np.savez_compressed(
        args.output_dir / "predicted_change_masks.npz",
        sample_ids=sample_ids,
        packed_masks=np.stack(packed_change_masks),
        packed_change_masks=np.stack(packed_change_masks),
        packed_final_masks=np.stack(packed_final_masks),
        mask_shape=np.asarray(mask_shape),
        mask_semantics=np.asarray("change=prior_xor_predicted_final"),
    )
    with (args.output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(paper_prediction(row)) + "\n")

    summary = {
        "schema_version": "sn7-prior-conditioned-visual-audit-v1",
        "sample_count": len(rows),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "model_config": checkpoint["model_config"],
        "thresholds": {
            "segmentation": args.segmentation_threshold,
            "add": args.add_threshold,
            "remove": args.remove_threshold,
            "presence": args.presence_threshold,
            "change": args.change_threshold,
        },
        "mask_semantics": "change = prior XOR predicted_final",
        "inference_input_size": args.input_size,
        "masks_saved": True,
        "test_assets_read": args.split == "test",
        "operation_slices": operation_slices(rows),
        **aggregate_visual_rows(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--add-threshold", type=float, default=0.5)
    parser.add_argument("--remove-threshold", type=float, default=0.5)
    parser.add_argument("--presence-threshold", type=float)
    parser.add_argument("--change-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(
        args.segmentation_threshold,
        args.add_threshold,
        args.remove_threshold,
        args.change_threshold,
    ) < 0.0 or max(
        args.segmentation_threshold,
        args.add_threshold,
        args.remove_threshold,
        args.change_threshold,
    ) > 1.0:
        raise ValueError("all thresholds must be in [0, 1]")
    if args.presence_threshold is not None and not 0.0 <= args.presence_threshold <= 1.0:
        raise ValueError("presence threshold must be in [0, 1]")
    if args.batch_size < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid loader configuration")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.input_size is not None and args.input_size < 16:
        raise ValueError("input size must be at least 16")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()

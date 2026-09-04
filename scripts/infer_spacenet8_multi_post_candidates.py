#!/usr/bin/env python3
"""Export frozen change-detector outcomes for every POST candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_sn7_opencd_generic import OpenCDPairModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("opencd_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    import torch

    args.output_root.mkdir(parents=True)
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    device = torch.device(args.device)
    model = OpenCDPairModel(
        opencd_repo=args.opencd_repo,
        config_path=args.config,
        pretrained_checkpoint=None,
        initialize=False,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
    outputs: list[dict] = []
    mask_root = args.output_root / "masks"

    with torch.no_grad():
        for index, record in enumerate(records):
            payload = np.load(str(record["array_path"]))
            pre = torch.from_numpy(payload["pre"].astype(np.float32)).to(device)
            post = torch.from_numpy(payload["post"].astype(np.float32)).to(device)
            target = torch.from_numpy(payload["target"].astype(bool)).to(device)
            valid = torch.from_numpy(payload["valid"].astype(bool)).to(device)
            logits = model(((pre - mean) / std)[None], ((post - mean) / std)[None])[0]
            probability = torch.softmax(logits, dim=0)[1]
            prediction = logits.argmax(dim=0).bool() & valid
            target = target & valid
            tp = int((prediction & target).sum())
            fp = int((prediction & ~target & valid).sum())
            fn = int((~prediction & target & valid).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            clipped = probability.clamp(1e-7, 1.0 - 1e-7)
            entropy = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
            row = {
                **record,
                "iou": tp / max(tp + fp + fn, 1),
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "precision": precision,
                "recall": recall,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "mean_probability": float(probability[valid].mean().cpu()) if valid.any() else 0.0,
                "mean_entropy": float(entropy[valid].mean().cpu()) if valid.any() else 0.0,
                "predicted_fraction": float(prediction.sum().cpu()) / max(int(valid.sum()), 1),
                "valid_fraction": float(valid.float().mean().cpu()),
                "test_assets_read": False,
            }
            if args.save_masks:
                mask_path = mask_root / f"candidate_{index:04d}.npz"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    mask_path,
                    probability=probability.cpu().numpy().astype(np.float16),
                    prediction=prediction.cpu().numpy().astype(np.uint8),
                )
                row["mask_path"] = str(mask_path)
            outputs.append(row)

    with (args.output_root / "per_candidate.jsonl").open("x", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "activemap-spacenet8-multi-post-inference-v1",
        "checkpoint": str(args.checkpoint),
        "candidates": len(outputs),
        "episodes": len({row["sample_id"] for row in outputs}),
        "mean_candidate_f1": float(np.mean([row["f1"] for row in outputs])),
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

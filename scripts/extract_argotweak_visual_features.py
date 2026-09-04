#!/usr/bin/env python3
"""Extract frozen seven-camera ResNet50 features for ArgoTweak evidence selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid feature shard")

    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import resnet50
    from torchvision.transforms import v2
    from tqdm.auto import tqdm

    episodes = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][args.shard_index :: args.num_shards]
    if not episodes:
        raise ValueError("feature shard has no episodes")
    frames: list[dict[str, Any]] = []
    images: list[tuple[int, str, str]] = []
    for episode in episodes:
        for evidence in episode["evidence"]:
            bundle = json.loads(Path(evidence["camera_bundle_path"]).read_text(encoding="utf-8"))
            frame_index = len(frames)
            frames.append(
                {
                    "episode_id": episode["episode_id"],
                    "segment_id": episode["segment_id"],
                    "split": episode["split"],
                    "evidence_id": evidence["evidence_id"],
                    "timestamp": evidence["timestamp"],
                    "cost": evidence["cost"],
                    "utility": evidence["utility"],
                    "target_operation_counts": evidence["target_operation_counts"],
                }
            )
            for camera, path in sorted(bundle["images"].items()):
                images.append((frame_index, camera, path))

    transform = v2.Compose(
        [
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    class ImageDataset(Dataset[tuple[int, torch.Tensor]]):
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
            frame_index, _, path = images[index]
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB"))
            return frame_index, tensor

    model = resnet50(weights=None)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = nn.Identity()
    model = model.to(args.device).eval()
    loader = DataLoader(
        ImageDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    camera_features: list[list[np.ndarray]] = [[] for _ in frames]
    with torch.inference_mode():
        for frame_indices, batch in tqdm(loader, desc="ArgoTweak ResNet50"):
            encoded = model(batch.to(args.device, non_blocking=True)).float().cpu().numpy()
            for frame_index, feature in zip(frame_indices.tolist(), encoded, strict=True):
                camera_features[frame_index].append(feature)
    if any(len(rows) != 7 for rows in camera_features):
        raise RuntimeError("every ArgoTweak evidence bundle must contain seven camera features")
    features = np.stack(
        [
            np.concatenate([np.mean(rows, axis=0), np.max(rows, axis=0)])
            for rows in camera_features
        ]
    ).astype(np.float16)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "activemap-argotweak-resnet50-features-v1",
        "episodes": len(episodes),
        "evidence_frames": len(frames),
        "camera_images": len(images),
        "feature_dim": int(features.shape[1]),
        "aggregation": "camera_mean_plus_max",
        "weights": {"path": str(args.weights.resolve()), "sha256": _sha256(args.weights)},
        "source": {"path": str(args.episodes.resolve()), "sha256": _sha256(args.episodes)},
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

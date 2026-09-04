#!/usr/bin/env python3
"""Extract frozen Qwen3-VL camera features for ArgoTweak evidence selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max-evidence", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    import numpy as np
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    all_episodes = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes = all_episodes[args.shard_index :: args.num_shards]
    records: list[dict[str, Any]] = []
    image_rows: list[tuple[int, str]] = []
    for episode in episodes:
        for evidence in episode["evidence"]:
            if args.max_evidence is not None and len(records) >= args.max_evidence:
                break
            bundle = json.loads(Path(evidence["camera_bundle_path"]).read_text(encoding="utf-8"))
            frame_index = len(records)
            records.append(
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
            for _, path in sorted(bundle["images"].items()):
                image_rows.append((frame_index, path))
        if args.max_evidence is not None and len(records) >= args.max_evidence:
            break
    if not records:
        raise ValueError("empty feature shard")

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    camera_features: list[list[np.ndarray]] = [[] for _ in records]
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_rows), args.batch_size), desc="Qwen3-VL cameras"):
            batch_rows = image_rows[start : start + args.batch_size]
            images = []
            for _, path in batch_rows:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(args.device)
            image_grid_thw = inputs["image_grid_thw"].to(args.device)
            output = model.get_image_features(pixel_values, image_grid_thw)
            for (frame_index, _), tokens in zip(batch_rows, output.pooler_output, strict=True):
                camera_features[frame_index].append(tokens.float().mean(dim=0).cpu().numpy())
    if any(len(rows) != 7 for rows in camera_features):
        raise RuntimeError("every evidence frame must have seven camera features")
    features = np.stack(
        [np.concatenate([np.mean(rows, axis=0), np.max(rows, axis=0)]) for rows in camera_features]
    ).astype(np.float16)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-qwen3vl-features-v1",
        "episodes": len(episodes),
        "evidence_frames": len(records),
        "camera_images": len(image_rows),
        "feature_dim": int(features.shape[1]),
        "aggregation": "token_mean_then_camera_mean_plus_max",
        "model": str(args.model.resolve()),
        "source_sha256": hashlib.sha256(args.episodes.read_bytes()).hexdigest(),
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure frozen updater latency on a declared validation input and GPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from activemap.inference import UpdaterPredictor
from activemap.oracle.updater_counterfactual import _geometry, _read_candidate, load_episodes


def _remap_path(value: str, mappings: list[str]) -> str:
    original = Path(value)
    for mapping in mappings:
        if "=" not in mapping:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source_text, target_text = mapping.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        try:
            return str(target / original.relative_to(source))
        except ValueError:
            continue
    return value


def gpu_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--asset-root-map", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("latency benchmark requires CUDA")
    if args.batch_size < 1 or args.warmup < 1 or args.repetitions < 5:
        raise ValueError("batch size, warmup, and repetitions are too small")
    episode = next(item for item in load_episodes(args.episodes) if item.split == "val")
    item = next(
        evidence for evidence in episode.evidence_catalog
        if evidence.timestamp == episode.anchor_timestamp
    )
    item = item.model_copy(
        update={
            "image_path": _remap_path(item.image_path, args.asset_root_map),
            "udm_path": (
                _remap_path(item.udm_path, args.asset_root_map)
                if item.udm_path is not None
                else None
            ),
        }
    )
    predictor = UpdaterPredictor(args.checkpoint, device=args.device)
    image, prior, _, _, _ = _read_candidate(
        item,
        prior_geometry=_geometry(episode.prior_geometry),
        target_geometry=_geometry(episode.target_geometry),
        image_size=args.image_size,
        image_channels=predictor.model.config.image_channels,
        road_width_source_pixels=(
            float(episode.metadata["road_width_source_pixels"])
            if episode.metadata.get("road_width_source_pixels") is not None
            else None
        ),
    )
    image_tensor = torch.from_numpy(np.repeat(image[None], args.batch_size, axis=0)).to(args.device)
    prior_tensor = torch.from_numpy(np.repeat(prior[None, None], args.batch_size, axis=0)).to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            predictor.model(image_tensor, prior_tensor)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        with torch.no_grad():
            predictor.model(image_tensor, prior_tensor)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / args.batch_size)
    payload = {
        "schema_version": "activemap-updater-latency-benchmark-v1",
        "split": "val",
        "test_assets_read": False,
        "hardware": {
            "gpu": gpu_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup,
            "timed_batches": args.repetitions,
            "synchronization": "torch.cuda.synchronize after each forward",
            "input": "one validation anchor crop replicated within a batch",
        },
        "per_candidate_latency_ms": {
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
            "p95": float(np.quantile(samples, 0.95)),
            "min": min(samples),
            "max": max(samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

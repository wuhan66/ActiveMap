#!/usr/bin/env python3
"""Export real selected-evidence crops for a fixed validation rollout trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(
    path: Path,
    source_episode: str,
    *,
    task_id: str | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source_episode") != source_episode:
            continue
        if task_id is not None and str(row.get("task_id", "")) != task_id:
            continue
        if budget is not None and float(row.get("budget", -1.0)) != float(budget):
            continue
        matches.append(row)
    if len(matches) != 1:
        scope = f"source_episode={source_episode!r}"
        if task_id is not None:
            scope += f", task_id={task_id!r}"
        if budget is not None:
            scope += f", budget={budget}"
        raise ValueError(f"trace must contain exactly one matching row for {scope}")
    return matches[0]


def load_episode(path: Path, source_episode: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == source_episode:
            return row
    raise KeyError(f"episode is absent from manifest: {source_episode}")


def remap_path(path: Path, root_map: tuple[Path, Path] | None) -> Path:
    if root_map is None:
        return path
    source, destination = root_map
    try:
        return destination / path.relative_to(source)
    except ValueError:
        return path


def raster_crop(path: Path, region: tuple[int, int, int, int], size: int) -> np.ndarray:
    x_min, y_min, x_max, y_max = region
    window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
    with rasterio.open(path) as dataset:
        indexes = list(range(1, min(dataset.count, 3) + 1))
        data = dataset.read(
            indexes,
            window=window,
            out_shape=(len(indexes), size, size),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    while data.shape[0] < 3:
        data = np.concatenate([data, data[-1:]], axis=0)
    scale = 255.0 if float(data.max(initial=0.0)) <= 255.0 else 10000.0
    data = np.clip(data / scale, 0.0, 1.0)
    return (np.moveaxis(data, 0, -1) * 255.0).round().astype(np.uint8)


def export_trace_evidence_crops(
    *,
    episodes_path: Path,
    traces_path: Path,
    source_episode: str,
    output_dir: Path,
    image_size: int = 256,
    asset_root_map: tuple[Path, Path] | None = None,
    task_id: str | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence output: {output_dir}")
    trace = load_trace(traces_path, source_episode, task_id=task_id, budget=budget)
    if trace.get("split") != "val" or trace.get("test_assets_read") is not False:
        raise ValueError("evidence export requires a validation-only trace")
    episode = load_episode(episodes_path, source_episode)
    if episode.get("split") != "val":
        raise ValueError("evidence export requires a validation episode")
    selected_ids = [str(value) for value in trace.get("selected_evidence_ids", [])]
    if len(selected_ids) < 2:
        raise ValueError("trace does not contain an acquired selected evidence item")
    evidence_by_id = {str(item["evidence_id"]): item for item in episode["evidence_catalog"]}
    selected = []
    for index, evidence_id in enumerate(selected_ids):
        item = evidence_by_id.get(evidence_id)
        if item is None:
            raise KeyError(f"selected evidence is absent from episode: {evidence_id}")
        image_path = remap_path(Path(str(item["image_path"])), asset_root_map)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        selected.append((index, item, image_path))

    output_dir.mkdir(parents=True)
    records = []
    for index, item, image_path in selected:
        timestamp = str(item["timestamp"])
        scale = int(item["scale"])
        region = tuple(int(value) for value in item["region"])
        filename = f"selected_{index:02d}_{timestamp}_scale{scale}.png"
        output_path = output_dir / filename
        Image.fromarray(raster_crop(image_path, region, image_size), mode="RGB").save(
            output_path
        )
        records.append(
            {
                "selected_index": index,
                "evidence_id": str(item["evidence_id"]),
                "timestamp": timestamp,
                "scale": scale,
                "cost": float(item["cost"]),
                "region": list(region),
                "source_image": str(image_path),
                "source_image_sha256": sha256(image_path),
                "output": filename,
                "output_sha256": sha256(output_path),
            }
        )
    manifest = {
        "schema_version": "fixed-validation-trace-evidence-crops-v1",
        "source_episode": source_episode,
        "task_id": trace.get("task_id"),
        "budget": trace.get("budget"),
        "split": "val",
        "test_assets_read": False,
        "selection_source": "recorded selected_evidence_ids in fixed rollout trace",
        "trace_path": str(traces_path),
        "trace_sha256": sha256(traces_path),
        "episodes_path": str(episodes_path),
        "episodes_sha256": sha256(episodes_path),
        "image_size": image_size,
        "asset_root_map": (
            {"source": str(asset_root_map[0]), "destination": str(asset_root_map[1])}
            if asset_root_map is not None
            else None
        ),
        "evidence": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("source_episode")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--task-id")
    parser.add_argument("--budget", type=float)
    parser.add_argument(
        "--asset-root-map",
        help="Rewrite absolute source root as SOURCE=DESTINATION before reading evidence.",
    )
    args = parser.parse_args()
    root_map = None
    if args.asset_root_map is not None:
        if "=" not in args.asset_root_map:
            raise ValueError("asset root map must be SOURCE=DESTINATION")
        source, destination = args.asset_root_map.split("=", 1)
        root_map = (Path(source), Path(destination))
    print(
        json.dumps(
            export_trace_evidence_crops(
                episodes_path=args.episodes,
                traces_path=args.traces,
                source_episode=args.source_episode,
                output_dir=args.output_dir,
                image_size=args.image_size,
                asset_root_map=root_map,
                task_id=args.task_id,
                budget=args.budget,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build leakage-separated MUNO21 RGB/prior visual inputs for Direct VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FORBIDDEN_INPUT_KEYS = ("gt", "target", "oracle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remap_path(value: str, mappings: tuple[tuple[Path, Path], ...]) -> str:
    path = Path(value)
    for source, destination in mappings:
        try:
            return str(destination / path.relative_to(source))
        except ValueError:
            continue
    return value


def _parse_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source, destination = (Path(item) for item in value.split("=", 1))
        if not source.is_absolute() or not destination.is_absolute():
            raise ValueError("asset root maps must use absolute paths")
        mappings.append((source, destination))
    return tuple(mappings)


def _initial_evidence(episode: Any) -> Any:
    exact = [
        item
        for item in episode.evidence_catalog
        if item.timestamp == episode.anchor_timestamp
    ]
    return exact[0] if exact else max(
        episode.evidence_catalog, key=lambda item: item.timestamp
    )


def _visuals(image: np.ndarray, prior: np.ndarray) -> tuple[Image.Image, Image.Image, Image.Image]:
    rgb = np.moveaxis(image[:3], 0, -1)
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    mask = prior >= 0.5
    mask_u8 = (mask.astype(np.uint8) * 255)
    overlay = rgb_u8.copy()
    overlay[mask] = np.rint(
        0.25 * overlay[mask].astype(np.float32)
        + 0.75 * np.asarray([255.0, 196.0, 0.0], dtype=np.float32)
    ).astype(np.uint8)
    return Image.fromarray(rgb_u8), Image.fromarray(mask_u8), Image.fromarray(overlay)


def _composite(
    rgb: Image.Image,
    mask: Image.Image,
    overlay: Image.Image,
    *,
    panel_size: int,
) -> Image.Image:
    panels = [
        rgb.resize((panel_size, panel_size), Image.Resampling.BILINEAR),
        mask.convert("RGB").resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        ),
        overlay.resize((panel_size, panel_size), Image.Resampling.BILINEAR),
    ]
    result = Image.new("RGB", (panel_size * 3, panel_size))
    for index, panel in enumerate(panels):
        result.paste(panel, (index * panel_size, 0))
    return result


def _assert_no_leak(record: dict[str, Any]) -> None:
    serialized = json.dumps(record, sort_keys=True).lower()
    leaked = [key for key in FORBIDDEN_INPUT_KEYS if f'"{key}' in serialized]
    if leaked:
        raise ValueError(f"visual input contains forbidden target keys: {leaked}")


def _one_per_operation(episodes: list[Any]) -> list[Any]:
    selected = {}
    for episode in episodes:
        operation = episode.gt_edit.op.value
        selected.setdefault(operation, episode)
    order = ("KEEP", "ADD", "DELETE", "RESHAPE")
    missing = [operation for operation in order if operation not in selected]
    if missing:
        raise ValueError(f"missing operations for balanced examples: {missing}")
    return [selected[operation] for operation in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--composite-panel-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--one-per-operation", action="store_true")
    parser.add_argument("--asset-root-map", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from shapely.geometry import shape

    from activemap.models import EpisodeRecord
    from activemap.oracle.updater_counterfactual import _read_candidate

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    mappings = _parse_root_maps(args.asset_root_map)
    episodes = []
    with args.episodes.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = EpisodeRecord.model_validate_json(line)
            if episode.split == "test":
                raise ValueError("visual preparation refuses files containing test episodes")
            if episode.split == args.split:
                episodes.append(episode)
    if args.one_per_operation and args.limit is not None:
        raise ValueError("--one-per-operation and --limit are mutually exclusive")
    if args.one_per_operation:
        episodes = _one_per_operation(episodes)
    elif args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError(f"no {args.split} episodes")

    image_root = args.output_dir / "images"
    image_root.mkdir(parents=True)
    inputs_path = args.output_dir / "inputs.jsonl"
    labels_path = args.output_dir / "labels.jsonl"
    source_records = []
    with inputs_path.open("w", encoding="utf-8") as inputs, labels_path.open(
        "w", encoding="utf-8"
    ) as labels:
        for episode in episodes:
            evidence = _initial_evidence(episode)
            source_image = Path(_remap_path(evidence.image_path, mappings))
            remapped = evidence.model_copy(update={"image_path": str(source_image)})
            prior_geometry = (
                shape(episode.prior_geometry.model_dump(mode="json"))
                if episode.prior_geometry is not None
                else None
            )
            image, prior, _, _, _ = _read_candidate(
                remapped,
                prior_geometry=prior_geometry,
                target_geometry=None,
                image_size=args.image_size,
                image_channels=3,
                road_width_source_pixels=float(
                    episode.metadata["road_width_source_pixels"]
                ),
            )
            rgb, mask, overlay = _visuals(image, prior)
            example_root = image_root / episode.episode_id
            example_root.mkdir()
            paths = {
                "rgb": example_root / "rgb.png",
                "prior_mask": example_root / "prior_mask.png",
                "overlay": example_root / "overlay.png",
                "composite": example_root / "composite.png",
            }
            rgb.save(paths["rgb"], optimize=True)
            mask.save(paths["prior_mask"], optimize=True)
            overlay.save(paths["overlay"], optimize=True)
            composite = _composite(
                rgb,
                mask,
                overlay,
                panel_size=args.composite_panel_size,
            )
            composite.save(paths["composite"], optimize=True)
            composite.close()
            record = {
                "schema_version": "muno21-direct-vlm-visual-input-v1",
                "example_id": episode.episode_id,
                "split": episode.split,
                "images": {name: str(path.resolve()) for name, path in paths.items()},
                "observation": {
                    "aoi_id": episode.aoi_id,
                    "anchor_timestamp": episode.anchor_timestamp,
                    "anchor_evidence_id": evidence.evidence_id,
                    "available_operations": ["KEEP", "ADD", "DELETE", "RESHAPE"],
                    "geometry_family": episode.metadata.get("geometry_family"),
                },
            }
            _assert_no_leak(record)
            inputs.write(json.dumps(record, separators=(",", ":")) + "\n")
            labels.write(
                json.dumps(
                    {
                        "schema_version": "muno21-direct-vlm-label-v1",
                        "example_id": episode.episode_id,
                        "split": episode.split,
                        "edit": episode.gt_edit.op.value,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            source_records.append(
                {
                    "example_id": episode.episode_id,
                    "source_image": str(source_image),
                    "source_image_sha256": _sha256(source_image),
                }
            )

    manifest = {
        "schema_version": "muno21-direct-vlm-visual-manifest-v1",
        "split": args.split,
        "sample_count": len(episodes),
        "image_size": args.image_size,
        "composite_panel_size": args.composite_panel_size,
        "selection": "one_per_operation" if args.one_per_operation else "ordered",
        "episodes": str(args.episodes.resolve()),
        "episodes_sha256": _sha256(args.episodes),
        "inputs": str(inputs_path.resolve()),
        "inputs_sha256": _sha256(inputs_path),
        "labels": str(labels_path.resolve()),
        "labels_sha256": _sha256(labels_path),
        "input_target_separation": True,
        "forbidden_input_keys": list(FORBIDDEN_INPUT_KEYS),
        "source_records": source_records,
        "test_assets_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("split", "sample_count", "image_size")}))


if __name__ == "__main__":
    main()

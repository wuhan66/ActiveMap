#!/usr/bin/env python3
"""Export validation-only, map-native MUNO21 paired writeback panels.

This renderer deliberately consumes completed rollout/writeback receipts. It
never loads a checkpoint, changes a policy, or evaluates a test episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


Key = tuple[str, float]
ORANGE = np.array([235, 142, 52], dtype=np.uint8)
GREEN = np.array([53, 160, 97], dtype=np.uint8)
BLUE = np.array([52, 116, 201], dtype=np.uint8)
RED = np.array([217, 86, 72], dtype=np.uint8)
LIGHT_BLUE = np.array([112, 184, 222], dtype=np.uint8)
WHITE = np.array([250, 250, 248], dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _assert_validation(row: dict[str, Any], path: Path) -> None:
    if row.get("split") != "val" or row.get("test_assets_read") is not False:
        raise ValueError(f"not a validation-only writeback: {path}")


def _index_writebacks(path: Path) -> dict[Key, dict[str, Any]]:
    result: dict[Key, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        _assert_validation(row, path)
        key = (str(row["task_id"]), float(row["budget"]))
        if key in result:
            raise ValueError(f"duplicate task/budget record in {path}: {key}")
        result[key] = row
    return result


def _raw_id(row: dict[str, Any]) -> str:
    selected = row.get("selected_evidence_ids") or []
    if selected:
        return str(selected[0]).split("__", 1)[0]
    raise ValueError(f"cannot recover public source id for {row['task_id']}")


def _load_mask(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} lacks {key}")
        return np.asarray(data[key], dtype=np.float32) >= 0.5


def _load_rgb(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected HWC or CHW RGB array: {path}")
    if array.shape[0] in {3, 4}:
        array = np.moveaxis(array[:3], 0, -1)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"cannot infer RGB channels from {path}")
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    low, high = np.quantile(finite, (0.01, 0.99))
    scaled = (array - low) / max(float(high - low), 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    for y_offset, x_offset in ((0, 1), (2, 1), (1, 0), (1, 2)):
        interior &= padded[
            y_offset : y_offset + mask.shape[0],
            x_offset : x_offset + mask.shape[1],
        ]
    return mask & ~interior


def _draw_boundary(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    image = np.broadcast_to(WHITE, (*mask.shape, 3)).copy()
    image[_edge(mask)] = color
    return image


def _residual(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    image = np.broadcast_to(WHITE, (*prediction.shape, 3)).copy()
    correct = _edge(prediction & reference)
    spurious = _edge(prediction & ~reference)
    missed = _edge(reference & ~prediction)
    image[correct] = GREEN
    image[spurious] = RED
    image[missed] = LIGHT_BLUE
    return image


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def _expand_bounds(mask: np.ndarray, size: int) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        center_x, center_y = width // 2, height // 2
    else:
        center_x = int(round(float(xs.min() + xs.max()) / 2.0))
        center_y = int(round(float(ys.min() + ys.max()) / 2.0))
    side = min(max(size, 1), height, width)
    left = max(0, min(center_x - side // 2, width - side))
    top = max(0, min(center_y - side // 2, height - side))
    return left, top, left + side, top + side


def _zoom_masks(
    direct: np.ndarray, active: np.ndarray, reference: np.ndarray
) -> list[np.ndarray]:
    first = np.logical_xor(active, reference)
    second = np.logical_xor(direct, active)
    if not second.any() or np.array_equal(first, second):
        second = np.logical_xor(reference, direct)
    return [first, second]


def _rank_cases(
    direct: dict[Key, dict[str, Any]],
    active: dict[Key, dict[str, Any]],
    budget: float,
) -> list[tuple[str, Key]]:
    common = set(direct) & set(active)
    choices = [key for key in common if abs(key[1] - budget) < 1e-9]
    if not choices:
        raise ValueError(f"no paired cases at budget {budget}")
    improved = sorted(
        choices,
        key=lambda key: float(active[key]["raster_iou_gain"])
        - float(direct[key]["raster_iou_gain"]),
        reverse=True,
    )
    safe = sorted(
        (
            key
            for key in choices
            if bool(direct[key].get("false_edit"))
            and not bool(active[key].get("false_edit"))
        ),
        key=lambda key: float(active[key]["raster_iou_gain"])
        - float(direct[key]["raster_iou_gain"]),
        reverse=True,
    )
    selected: list[tuple[str, Key]] = []
    used: set[Key] = set()
    for category, values in (("improved_writeback", improved), ("safe_rejection", safe)):
        for key in values:
            if key not in used:
                selected.append((category, key))
                used.add(key)
                break
    if len(selected) < 2:
        for key in improved:
            if key not in used:
                selected.append(("paired_comparison", key))
                used.add(key)
            if len(selected) == 2:
                break
    return selected


def render(
    direct_path: Path,
    active_path: Path,
    arrays_dir: Path,
    output_dir: Path,
    *,
    budget: float,
    zoom_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    direct = _index_writebacks(direct_path)
    active = _index_writebacks(active_path)
    if set(direct) != set(active):
        raise ValueError("direct and ActiveMap writeback supports differ")
    output_dir.mkdir(parents=True)
    cases = []
    for rank, (category, key) in enumerate(_rank_cases(direct, active, budget), start=1):
        direct_row = direct[key]
        active_row = active[key]
        raw_id = _raw_id(active_row)
        image_path = arrays_dir / f"{raw_id}-image.npy"
        prior_path = arrays_dir / f"{raw_id}-prior.npy"
        target_path = arrays_dir / f"{raw_id}-target.npy"
        if not all(path.is_file() for path in (image_path, prior_path, target_path)):
            raise FileNotFoundError(f"missing processed arrays for {raw_id}")
        rgb = _load_rgb(image_path)
        prior = np.asarray(np.load(prior_path), dtype=np.float32) >= 0.5
        reference = np.asarray(np.load(target_path), dtype=np.float32) >= 0.5
        direct_mask = _load_mask(Path(str(direct_row["mask_artifact"])), "committed_mask")
        active_mask = _load_mask(Path(str(active_row["mask_artifact"])), "committed_mask")
        if not all(mask.shape == prior.shape for mask in (reference, direct_mask, active_mask)):
            raise ValueError(f"mask shape mismatch for {raw_id}")
        folder = output_dir / f"{rank:02d}_{category}_{key[0]}"
        folder.mkdir()
        _save(rgb, folder / "current_rgb.png")
        _save(_draw_boundary(prior, ORANGE), folder / "prior_map.png")
        _save(_draw_boundary(direct_mask, BLUE), folder / "direct_writeback.png")
        _save(_draw_boundary(active_mask, BLUE), folder / "activemap_writeback.png")
        _save(_draw_boundary(reference, GREEN), folder / "reference_map.png")
        _save(_residual(direct_mask, reference), folder / "residual_direct.png")
        _save(_residual(active_mask, reference), folder / "residual_activemap.png")
        zooms = []
        for zoom_index, zoom_mask in enumerate(_zoom_masks(direct_mask, active_mask, reference), start=1):
            bounds = _expand_bounds(zoom_mask, zoom_size)
            left, top, right, bottom = bounds
            zoom_dir = folder / f"zoom{zoom_index:02d}"
            zoom_dir.mkdir()
            for filename in (
                "current_rgb.png",
                "prior_map.png",
                "direct_writeback.png",
                "activemap_writeback.png",
                "reference_map.png",
                "residual_direct.png",
                "residual_activemap.png",
            ):
                source = np.asarray(Image.open(folder / filename))
                _save(source[top:bottom, left:right], zoom_dir / filename)
            zooms.append({"bounds_xyxy": bounds, "directory": zoom_dir.name})
        evidence_ids = list(active_row.get("selected_evidence_ids") or [])
        tool_calls = int(active_row.get("semantic_tool_called") or 0)
        manifest = {
            "schema_version": "activemap-map-native-qualitative-v1",
            "dataset": "MUNO21",
            "split": "val",
            "test_assets_read": False,
            "case_id": key[0],
            "source_example_id": raw_id,
            "category": category,
            "budget": budget,
            "coordinate_reference": "shared processed updater array grid",
            "direct_method": "generic_selector_safe_delta",
            "activemap_method": "p50_edit_conditioned_selector_safe_delta",
            "direct_metrics": {
                "false_edit": bool(direct_row["false_edit"]),
                "missed_edit": bool(direct_row["missed_edit"]),
                "raster_iou_gain": float(direct_row["raster_iou_gain"]),
            },
            "activemap_metrics": {
                "false_edit": bool(active_row["false_edit"]),
                "missed_edit": bool(active_row["missed_edit"]),
                "raster_iou_gain": float(active_row["raster_iou_gain"]),
            },
            "tool_call_count": tool_calls,
            "selected_evidence_ids": evidence_ids,
            "evidence_exported": False,
            "selection_rule": "validation paired residual ranking at fixed budget",
            "sources": {
                "direct_writeback_jsonl": {"path": str(direct_path), "sha256": _sha256(direct_path)},
                "activemap_writeback_jsonl": {"path": str(active_path), "sha256": _sha256(active_path)},
                "direct_mask": str(direct_row["mask_artifact"]),
                "activemap_mask": str(active_row["mask_artifact"]),
                "image": str(image_path),
                "prior": str(prior_path),
                "reference": str(target_path),
            },
            "zooms": zooms,
        }
        (folder / "trace.json").write_text(
            json.dumps({"direct": direct_row, "activemap": active_row}, indent=2) + "\n",
            encoding="utf-8",
        )
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        cases.append({"folder": folder.name, **manifest})
    summary = {
        "schema_version": "muno21-map-native-paired-qualitative-v1",
        "split": "val",
        "test_assets_read": False,
        "budget": budget,
        "case_count": len(cases),
        "selection_rule": "validation paired residual ranking at fixed budget",
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direct_writeback", type=Path)
    parser.add_argument("activemap_writeback", type=Path)
    parser.add_argument("arrays_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--zoom-size", type=int, default=256)
    args = parser.parse_args()
    if args.zoom_size <= 0:
        raise ValueError("zoom-size must be positive")
    print(
        json.dumps(
            render(
                args.direct_writeback,
                args.activemap_writeback,
                args.arrays_dir,
                args.output_dir,
                budget=args.budget,
                zoom_size=args.zoom_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

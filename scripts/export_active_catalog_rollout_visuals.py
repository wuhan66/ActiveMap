#!/usr/bin/env python3
"""Export clean, independently composable rollout panels for paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
ACTION_COLORS = {
    "ACQUIRE": (31, 119, 180),
    "USE_TOOL": (148, 103, 189),
    "COMMIT": (44, 160, 44),
    "REJECT": (127, 127, 127),
}


def public_evidence_id(raw_id: str) -> str:
    digest = hashlib.sha256(f"activemap:evidence:{raw_id}".encode()).hexdigest()[:16]
    return f"evidence-{digest}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def _parse_root_map(value: str) -> tuple[str, str]:
    source, separator, target = value.partition("=")
    if not separator or not source or not target:
        raise argparse.ArgumentTypeError("root maps must use SOURCE=TARGET")
    return source.rstrip("/"), target.rstrip("/")


def _remap(path: str, root_maps: list[tuple[str, str]]) -> Path:
    for source, target in root_maps:
        if path == source or path.startswith(source + "/"):
            return Path(target + path[len(source) :])
    return Path(path)


def _load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.asarray(np.load(path), dtype=np.float32)
    else:
        try:
            import rasterio
        except ImportError:
            value = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        else:
            with rasterio.open(path) as dataset:
                value = dataset.read([1, 2, 3]).astype(np.float32)
    if value.ndim != 3:
        raise ValueError(f"expected three-dimensional image: {path}")
    if value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value[:3], 0, -1)
    elif value.shape[-1] >= 3:
        value = value[..., :3]
    else:
        raise ValueError(f"cannot infer RGB channels: {path}")
    finite = value[np.isfinite(value)]
    if not finite.size:
        return np.zeros((*value.shape[:2], 3), dtype=np.uint8)
    low, high = np.quantile(finite, (0.01, 0.99))
    scaled = (value - low) / max(float(high - low), 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _crop_bounds(
    regions: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    minimum_size: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    x0 = min(region[0] for region in regions)
    y0 = min(region[1] for region in regions)
    x1 = max(region[2] for region in regions)
    y1 = max(region[3] for region in regions)
    size = max(x1 - x0, y1 - y0, minimum_size)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = max(0, min(int(round(center_x - size / 2)), width - size))
    top = max(0, min(int(round(center_y - size / 2)), height - size))
    right = min(width, left + size)
    bottom = min(height, top + size)
    return left, top, right, bottom


def _colorize(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    from matplotlib import colormaps

    normalized = (values - low) / max(high - low, 1e-9)
    rgba = (colormaps["viridis"](np.clip(normalized, 0.0, 1.0)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _resize_save(array: np.ndarray, path: Path, size: int, *, nearest: bool = False) -> None:
    image = Image.fromarray(array)
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    image.resize((size, size), resampling).save(path)


def _event_action(event: dict[str, Any]) -> str:
    action = event.get("executed_action") or {}
    return str(action.get("action", "REJECT"))


def _select_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    categories = (
        ("tool_positive", lambda row: row.get("tool_calls", 0) > 0 and row.get("terminal_correct")),
        (
            "acquire_positive",
            lambda row: row.get("acquisitions", 0) > 0 and row.get("terminal_correct"),
        ),
        ("false_edit", lambda row: row.get("false_edit")),
        ("missed_edit", lambda row: row.get("missed_edit")),
        ("correct_stop", lambda row: row.get("terminal_correct") and not row.get("acquisitions")),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    per_category = max(1, limit // len(categories))
    for _, predicate in categories:
        candidates = sorted(
            (row for row in rows if predicate(row)),
            key=lambda row: (
                float(row.get("quality_gain", 0.0)),
                -float(row.get("spent_cost", 0.0)),
            ),
            reverse=True,
        )
        for row in candidates:
            if str(row["sample_id"]) in used:
                continue
            selected.append(row)
            used.add(str(row["sample_id"]))
            if sum(predicate(item) for item in selected) >= per_category:
                break
    for row in rows:
        if len(selected) >= limit:
            break
        if str(row["sample_id"]) not in used:
            selected.append(row)
            used.add(str(row["sample_id"]))
    return selected[:limit]


def _belief_matrix(events: list[dict[str, Any]]) -> np.ndarray:
    columns = []
    for event in events:
        state = event.get("observable_state") or {}
        belief = state.get("belief") or {}
        probabilities = belief.get("edit_probabilities")
        if isinstance(probabilities, list) and len(probabilities) == len(OPERATIONS):
            columns.append([float(value) for value in probabilities])
    if not columns:
        return np.zeros((len(OPERATIONS), 1), dtype=np.float32)
    return np.asarray(columns, dtype=np.float32).T


def _catalog_matrix(
    event: dict[str, Any],
    public_to_item: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[str], list[int]]:
    scores = event.get("ranker_scores") or {}
    matched = [
        (public_to_item[evidence_id], float(score))
        for evidence_id, score in scores.items()
        if evidence_id in public_to_item
    ]
    timestamps = sorted({str(item["timestamp"]) for item, _ in matched})
    scales = sorted({int(item["scale"]) for item, _ in matched})
    matrix = np.full((len(scales), len(timestamps)), np.nan, dtype=np.float32)
    for item, score in matched:
        matrix[scales.index(int(item["scale"])), timestamps.index(str(item["timestamp"]))] = score
    return matrix, timestamps, scales


def export(
    traces_path: Path,
    episodes_path: Path,
    output_dir: Path,
    *,
    limit: int,
    sample_ids: set[str],
    root_maps: list[tuple[str, str]],
    panel_size: int,
    crop_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    traces = _read_jsonl(traces_path)
    episodes = {
        str(row["episode_id"]): row for row in _read_jsonl(episodes_path)
    }
    selected = (
        [row for row in traces if str(row["sample_id"]) in sample_ids]
        if sample_ids
        else _select_rows(traces, limit)
    )
    if not selected:
        raise ValueError("no requested rollout samples were found")
    output_dir.mkdir(parents=True)
    index = []
    for rank, trace in enumerate(selected, start=1):
        episode = episodes[str(trace["source_episode"])]
        items = list(episode["evidence_catalog"])
        public_to_item = {
            public_evidence_id(str(item["evidence_id"])): item for item in items
        }
        anchor_items = [
            item
            for item in items
            if item["timestamp"] == episode.get("anchor_timestamp")
        ]
        anchor = min(anchor_items or items, key=lambda item: int(item["scale"]))
        rgb = _load_rgb(_remap(str(anchor["image_path"]), root_maps))
        regions = [tuple(map(int, item["region"])) for item in items]
        bounds = _crop_bounds(regions, rgb.shape[:2], crop_size)
        left, top, right, bottom = bounds
        folder = output_dir / f"{rank:03d}_{trace['target_edit'].lower()}_{trace['sample_id']}"
        folder.mkdir()
        _resize_save(rgb[top:bottom, left:right], folder / "01_anchor_rgb.png", panel_size)

        all_scores = [
            float(score)
            for event in trace.get("events", [])
            for score in (event.get("ranker_scores") or {}).values()
        ]
        score_low = min(all_scores, default=-1.0)
        score_high = max(all_scores, default=1.0)
        event_metadata = []
        for step, event in enumerate(trace.get("events", [])):
            scores = event.get("ranker_scores") or {}
            spatial_sum = np.zeros(rgb.shape[:2], dtype=np.float32)
            spatial_count = np.zeros(rgb.shape[:2], dtype=np.float32)
            for evidence_id, score in scores.items():
                item = public_to_item.get(str(evidence_id))
                if item is None:
                    continue
                x0, y0, x1, y1 = map(int, item["region"])
                spatial_sum[y0:y1, x0:x1] += float(score)
                spatial_count[y0:y1, x0:x1] += 1.0
            spatial = np.divide(
                spatial_sum,
                spatial_count,
                out=np.zeros_like(spatial_sum),
                where=spatial_count > 0,
            )
            valid = spatial_count > 0
            spatial_rgba = _colorize(spatial, valid, score_low, score_high)
            cropped_heatmap = spatial_rgba[top:bottom, left:right]
            _resize_save(
                cropped_heatmap,
                folder / f"utility_spatial_step{step:02d}.png",
                panel_size,
                nearest=True,
            )
            overlay = rgb[top:bottom, left:right].copy()
            alpha = cropped_heatmap[..., 3:4].astype(np.float32) / 255.0 * 0.62
            overlay = (
                overlay.astype(np.float32) * (1.0 - alpha)
                + cropped_heatmap[..., :3].astype(np.float32) * alpha
            ).astype(np.uint8)
            _resize_save(overlay, folder / f"utility_overlay_step{step:02d}.png", panel_size)

            catalog, timestamps, scales = _catalog_matrix(event, public_to_item)
            catalog_valid = np.isfinite(catalog)
            catalog_values = np.nan_to_num(catalog, nan=score_low)
            catalog_rgba = _colorize(catalog_values, catalog_valid, score_low, score_high)
            _resize_save(
                catalog_rgba,
                folder / f"utility_catalog_step{step:02d}.png",
                panel_size,
                nearest=True,
            )
            event_metadata.append(
                {
                    "step": step,
                    "action": _event_action(event),
                    "timestamps": timestamps,
                    "scales": scales,
                    "utility_min": score_low,
                    "utility_max": score_high,
                }
            )

        belief = _belief_matrix(trace.get("events", []))
        belief_rgba = _colorize(belief, np.ones_like(belief, dtype=bool), 0.0, 1.0)
        _resize_save(belief_rgba, folder / "belief_trajectory.png", panel_size, nearest=True)
        actions = [_event_action(event) for event in trace.get("events", [])]
        action_strip = np.asarray(
            [[ACTION_COLORS.get(action, (80, 80, 80)) for action in actions or ["REJECT"]]],
            dtype=np.uint8,
        )
        _resize_save(action_strip, folder / "action_trajectory.png", panel_size, nearest=True)

        metadata = {
            "sample_id": trace["sample_id"],
            "source_episode": trace["source_episode"],
            "aoi_id": trace.get("aoi_id"),
            "target_edit": trace["target_edit"],
            "predicted_edit": trace["predicted_edit"],
            "terminal_correct": trace["terminal_correct"],
            "false_edit": trace["false_edit"],
            "missed_edit": trace["missed_edit"],
            "quality_gain": trace.get("quality_gain"),
            "spent_cost": trace.get("spent_cost"),
            "belief_rows": list(OPERATIONS),
            "action_colors": ACTION_COLORS,
            "crop_bounds_xyxy": bounds,
            "events": event_metadata,
        }
        (folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        index.append({"folder": folder.name, **metadata})
    summary = {
        "schema_version": "active-catalog-rollout-visuals-v1",
        "trace_path": str(traces_path),
        "episode_path": str(episodes_path),
        "sample_count": len(index),
        "individual_unlabeled_panels": True,
        "test_assets_read": False,
        "samples": index,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--asset-root-map", action="append", type=_parse_root_map, default=[])
    parser.add_argument("--panel-size", type=int, default=768)
    parser.add_argument("--crop-size", type=int, default=128)
    args = parser.parse_args()
    if args.limit <= 0 or args.panel_size <= 0 or args.crop_size <= 0:
        raise ValueError("limit and image sizes must be positive")
    summary = export(
        args.traces,
        args.episodes,
        args.output_dir,
        limit=args.limit,
        sample_ids=set(args.sample_id),
        root_maps=args.asset_root_map,
        panel_size=args.panel_size,
        crop_size=args.crop_size,
    )
    print(json.dumps({"output": str(args.output_dir), "samples": summary["sample_count"]}))


if __name__ == "__main__":
    main()

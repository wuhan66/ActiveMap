#!/usr/bin/env python3
"""Export clean per-panel SN7 updater visuals from saved validation audits."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.train_sn7_changemamba import Record, _read_records


@dataclass(frozen=True)
class Audit:
    name: str
    rows: dict[str, dict[str, Any]]
    masks: dict[str, np.ndarray]
    mask_shape: tuple[int, int]


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError(f"invalid method name: {name}")
    return name, Path(raw_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_audit(name: str, directory: Path) -> Audit:
    rows = _read_jsonl(directory / "per_sample.jsonl")
    archive = np.load(directory / "predicted_change_masks.npz")
    sample_ids = [str(value) for value in archive["sample_ids"]]
    packed_key = (
        "packed_change_masks"
        if "packed_change_masks" in archive.files
        else "packed_masks"
    )
    packed = np.asarray(archive[packed_key], dtype=np.uint8)
    shape = tuple(int(value) for value in archive["mask_shape"])
    if len(shape) != 2:
        raise ValueError(f"invalid mask shape in {directory}: {shape}")
    if len(rows) != len(sample_ids) or len(sample_ids) != len(packed):
        raise ValueError(f"audit row/mask count mismatch: {directory}")
    row_ids = [str(row["sample_id"]) for row in rows]
    if row_ids != sample_ids:
        raise ValueError(f"audit row/mask identity mismatch: {directory}")
    pixel_count = int(np.prod(shape))
    masks = {
        sample_id: np.unpackbits(values)[:pixel_count].reshape(shape).astype(bool)
        for sample_id, values in zip(sample_ids, packed, strict=True)
    }
    return Audit(
        name=name,
        rows={str(row["sample_id"]): row for row in rows},
        masks=masks,
        mask_shape=shape,
    )


def _load_mask(path: Path | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if path is None:
        if shape is None:
            raise ValueError("shape is required for a missing mask")
        return np.ones(shape, dtype=bool)
    value = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if value.ndim != 2:
        raise ValueError(f"expected 2D mask: {path}")
    mask = value >= 0.5
    if shape is not None and mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ).astype(bool)
    return mask


def _load_rgb(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"expected RGB array: {path}")
    if value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value[:3], 0, -1)
    elif value.shape[-1] >= 3:
        value = value[..., :3]
    else:
        raise ValueError(f"cannot infer RGB channels: {path}")
    finite = value[np.isfinite(value)]
    if not finite.size:
        return np.zeros((*value.shape[:2], 3), dtype=np.uint8)
    if finite.max() <= 1.0 and finite.min() >= 0.0:
        scaled = value
    else:
        low, high = np.quantile(finite, (0.01, 0.99))
        scaled = (value - low) / max(float(high - low), 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _resize_rgb(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == shape:
        return rgb
    return np.asarray(
        Image.fromarray(rgb).resize((shape[1], shape[0]), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ).astype(bool)


def _save_mask(
    path: Path,
    mask: np.ndarray,
    valid: np.ndarray,
    output_size: int,
) -> None:
    value = np.zeros((*mask.shape, 4), dtype=np.uint8)
    value[..., :3] = np.where(mask[..., None], 255, 0)
    value[..., 3] = np.where(valid, 255, 0)
    image = Image.fromarray(value, mode="RGBA")
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    image.save(path)


def _overlay(
    rgb: np.ndarray,
    prior: np.ndarray,
    target: np.ndarray,
    committed: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    output = rgb.astype(np.float32)
    layers = (
        (prior, np.asarray((255, 188, 0), dtype=np.float32)),
        (target, np.asarray((45, 215, 90), dtype=np.float32)),
        (committed, np.asarray((0, 205, 205), dtype=np.float32)),
    )
    for mask, color in layers:
        active = mask & valid
        output[active] = output[active] * 0.58 + color * 0.42
    output[~valid] *= 0.35
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def _residual(
    rgb: np.ndarray,
    truth_change: np.ndarray,
    predicted_change: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    output = rgb.astype(np.float32) * 0.30 + 34.0
    output[~valid] *= 0.25
    true_positive = truth_change & predicted_change & valid
    false_positive = predicted_change & ~truth_change & valid
    false_negative = truth_change & ~predicted_change & valid
    for region, color in (
        (true_positive, np.asarray((0, 158, 115), dtype=np.float32)),
        (false_positive, np.asarray((213, 94, 0), dtype=np.float32)),
        (false_negative, np.asarray((86, 180, 233), dtype=np.float32)),
    ):
        output[region] = output[region] * 0.12 + color * 0.88
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def _prior_image_paths(manifest: Path, split: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in _read_jsonl(manifest):
        if row.get("split") != split or not row.get("prior_image_path"):
            continue
        value = Path(str(row["prior_image_path"]))
        paths[str(row["sample_id"])] = (
            value if value.is_absolute() else (manifest.parent / value).resolve()
        )
    return paths


def _zoom_box(
    masks: list[np.ndarray],
    *,
    fallback: np.ndarray,
    minimum_side: int,
) -> tuple[slice, slice]:
    region = np.zeros_like(fallback, dtype=bool)
    for mask in masks:
        region |= mask.astype(bool)
    if not region.any():
        region = fallback.astype(bool)
    if not region.any():
        return slice(0, fallback.shape[0]), slice(0, fallback.shape[1])
    rows, columns = np.where(region)
    center_y = 0.5 * (float(rows.min()) + float(rows.max()) + 1.0)
    center_x = 0.5 * (float(columns.min()) + float(columns.max()) + 1.0)
    extent = max(float(rows.max() - rows.min() + 1), float(columns.max() - columns.min() + 1))
    side = min(
        float(min(fallback.shape)),
        max(float(minimum_side), extent * 1.8),
    )
    top = int(round(center_y - side / 2.0))
    left = int(round(center_x - side / 2.0))
    top = min(max(top, 0), fallback.shape[0] - int(round(side)))
    left = min(max(left, 0), fallback.shape[1] - int(round(side)))
    side_int = int(round(side))
    return slice(top, top + side_int), slice(left, left + side_int)


def _save_rgb_crop(
    path: Path,
    rgb: np.ndarray,
    box: tuple[slice, slice],
    output_size: int,
) -> None:
    crop = rgb[box]
    Image.fromarray(crop).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    ).save(path)


def _shared_ids(
    records: dict[str, Record], audits: list[Audit]
) -> list[str]:
    shared = set(records)
    for audit in audits:
        shared &= set(audit.rows)
        shared &= set(audit.masks)
    return sorted(shared)


def _select(
    records: dict[str, Record],
    audits: list[Audit],
    *,
    per_edit: int,
    failures_per_edit: int,
    disagreements_per_edit: int,
    min_visual_fraction: float,
) -> list[tuple[str, str]]:
    shared = _shared_ids(records, audits)
    selected: list[tuple[str, str]] = []
    for edit in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        ids = [sample_id for sample_id in shared if records[sample_id].edit_type == edit]
        visible = [
            sample_id
            for sample_id in ids
            if max(
                float(audits[0].rows[sample_id]["prior_foreground_fraction"]),
                float(audits[0].rows[sample_id]["target_change_fraction"]),
            )
            >= min_visual_fraction
        ]
        candidates = visible or ids
        primary = audits[0]
        baseline = audits[1] if len(audits) > 1 else None
        positive = sorted(
            candidates,
            key=lambda sample_id: (
                bool(primary.rows[sample_id].get("operation_correct", False)),
                float(primary.rows[sample_id]["committed_map_iou"]),
                float(primary.rows[sample_id].get("change_iou", 0.0)),
                float(primary.rows[sample_id]["map_iou_delta"]),
                max(
                    float(audits[0].rows[sample_id]["prior_foreground_fraction"]),
                    float(audits[0].rows[sample_id]["target_change_fraction"]),
                ),
            ),
            reverse=True,
        )
        failures = sorted(
            candidates,
            key=lambda sample_id: (
                bool(primary.rows[sample_id].get("operation_correct", False)),
                float(primary.rows[sample_id]["committed_map_iou"]),
                float(primary.rows[sample_id].get("change_iou", 0.0)),
            ),
        )
        disagreement = sorted(
            candidates,
            key=lambda sample_id: (
                float(primary.rows[sample_id]["committed_map_iou"])
                - (
                    float(baseline.rows[sample_id]["committed_map_iou"])
                    if baseline is not None
                    else 0.0
                ),
                float(primary.rows[sample_id].get("change_iou", 0.0))
                - (
                    float(baseline.rows[sample_id].get("change_iou", 0.0))
                    if baseline is not None
                    else 0.0
                ),
            ),
            reverse=True,
        )
        used: set[str] = set()
        for sample_id in positive:
            if len([item for item in selected if item[1] == f"{edit.lower()}_success"]) >= per_edit:
                break
            if sample_id not in used:
                selected.append((sample_id, f"{edit.lower()}_success"))
                used.add(sample_id)
        for sample_id in disagreement:
            if len([item for item in selected if item[1] == f"{edit.lower()}_improvement"]) >= disagreements_per_edit:
                break
            if sample_id not in used:
                selected.append((sample_id, f"{edit.lower()}_improvement"))
                used.add(sample_id)
        for sample_id in failures:
            if len([item for item in selected if item[1] == f"{edit.lower()}_failure"]) >= failures_per_edit:
                break
            if sample_id not in used:
                selected.append((sample_id, f"{edit.lower()}_failure"))
                used.add(sample_id)
    return selected


def render(
    manifest: Path,
    audit_paths: list[tuple[str, Path]],
    output_dir: Path,
    *,
    per_edit: int,
    failures_per_edit: int,
    disagreements_per_edit: int,
    min_visual_fraction: float,
    output_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    records = {
        record.sample_id: record for record in _read_records(manifest, "val", None)
    }
    prior_images = _prior_image_paths(manifest, "val")
    audits = [_load_audit(name, path) for name, path in audit_paths]
    if not audits:
        raise ValueError("at least one audit is required")
    selected = _select(
        records,
        audits,
        per_edit=per_edit,
        failures_per_edit=failures_per_edit,
        disagreements_per_edit=disagreements_per_edit,
        min_visual_fraction=min_visual_fraction,
    )
    index_rows = []
    for rank, (sample_id, category) in enumerate(selected, start=1):
        record = records[sample_id]
        short_category = category.removeprefix(f"{record.edit_type.lower()}_")
        folder = output_dir / f"{rank:03d}_{record.edit_type.lower()}_{short_category}"
        folder.mkdir()
        audit_shape = audits[0].masks[sample_id].shape
        rgb = _resize_rgb(_load_rgb(record.image), audit_shape)
        prior = _load_mask(record.prior, audit_shape)
        target = _load_mask(record.target, audit_shape)
        valid = _load_mask(record.valid, audit_shape)
        truth_change = np.logical_xor(prior, target)
        rgb_image = Image.fromarray(rgb)
        if rgb_image.size != (output_size, output_size):
            rgb_image = rgb_image.resize(
                (output_size, output_size), Image.Resampling.LANCZOS
            )
        if sample_id in prior_images:
            prior_rgb = Image.fromarray(_load_rgb(prior_images[sample_id]))
            if prior_rgb.size != (output_size, output_size):
                prior_rgb = prior_rgb.resize(
                    (output_size, output_size), Image.Resampling.LANCZOS
                )
            prior_rgb.save(folder / "00_previous_rgb.png")
        rgb_image.save(folder / "01_current_rgb.png")
        _save_mask(folder / "02_prior_mask.png", prior, valid, output_size)
        _save_mask(folder / "03_reference_final_mask.png", target, valid, output_size)
        _save_mask(
            folder / "04_truth_change_mask.png",
            truth_change,
            valid,
            output_size,
        )
        method_rows: dict[str, Any] = {}
        method_predictions = {
            audit.name: _resize_binary(audit.masks[sample_id], audit_shape) & valid
            for audit in audits
        }
        for offset, audit in enumerate(audits):
            prediction = method_predictions[audit.name]
            committed = np.logical_xor(prior, prediction) & valid
            prefix = 5 + offset * 4
            _save_mask(
                folder / f"{prefix:02d}_{audit.name}_change_mask.png",
                prediction,
                valid,
                output_size,
            )
            _save_mask(
                folder / f"{prefix + 1:02d}_{audit.name}_committed_mask.png",
                committed,
                valid,
                output_size,
            )
            overlay = _overlay(rgb, prior, target, committed, valid)
            if overlay.size != (output_size, output_size):
                overlay = overlay.resize(
                    (output_size, output_size), Image.Resampling.LANCZOS
                )
            overlay.save(folder / f"{prefix + 2:02d}_{audit.name}_overlay.png")
            residual = _residual(rgb, truth_change, prediction, valid)
            if residual.size != (output_size, output_size):
                residual = residual.resize(
                    (output_size, output_size), Image.Resampling.NEAREST
                )
            residual.save(folder / f"{prefix + 3:02d}_{audit.name}_tp_fp_fn.png")
            method_rows[audit.name] = audit.rows[sample_id]

        zoom_masks = (
            [truth_change]
            if truth_change.any()
            else list(method_predictions.values())
        )
        zoom = _zoom_box(
            zoom_masks,
            fallback=prior | target,
            minimum_side=max(32, int(round(min(audit_shape) * 0.25))),
        )
        zoom_dir = folder / "zoom"
        zoom_dir.mkdir()
        if sample_id in prior_images:
            previous_rgb = _resize_rgb(_load_rgb(prior_images[sample_id]), audit_shape)
            _save_rgb_crop(zoom_dir / "00_previous_rgb.png", previous_rgb, zoom, output_size)
        _save_rgb_crop(zoom_dir / "01_current_rgb.png", rgb, zoom, output_size)
        _save_mask(zoom_dir / "02_prior_mask.png", prior[zoom], valid[zoom], output_size)
        _save_mask(
            zoom_dir / "03_reference_final_mask.png",
            target[zoom],
            valid[zoom],
            output_size,
        )
        _save_mask(
            zoom_dir / "04_truth_change_mask.png",
            truth_change[zoom],
            valid[zoom],
            output_size,
        )
        for offset, audit in enumerate(audits):
            prediction = method_predictions[audit.name]
            committed = np.logical_xor(prior, prediction) & valid
            prefix = 5 + offset * 4
            _save_mask(
                zoom_dir / f"{prefix:02d}_{audit.name}_change_mask.png",
                prediction[zoom],
                valid[zoom],
                output_size,
            )
            _save_mask(
                zoom_dir / f"{prefix + 1:02d}_{audit.name}_committed_mask.png",
                committed[zoom],
                valid[zoom],
                output_size,
            )
            overlay = _overlay(rgb, prior, target, committed, valid)
            overlay.crop(
                (zoom[1].start, zoom[0].start, zoom[1].stop, zoom[0].stop)
            ).resize((output_size, output_size), Image.Resampling.LANCZOS).save(
                zoom_dir / f"{prefix + 2:02d}_{audit.name}_overlay.png"
            )
            residual = _residual(rgb, truth_change, prediction, valid)
            residual.crop(
                (zoom[1].start, zoom[0].start, zoom[1].stop, zoom[0].stop)
            ).resize((output_size, output_size), Image.Resampling.NEAREST).save(
                zoom_dir / f"{prefix + 3:02d}_{audit.name}_tp_fp_fn.png"
            )
        comparison = None
        if len(audits) > 1:
            comparison = {
                "primary": audits[0].name,
                "baseline": audits[1].name,
                "committed_map_iou_delta": float(
                    audits[0].rows[sample_id]["committed_map_iou"]
                    - audits[1].rows[sample_id]["committed_map_iou"]
                ),
                "change_iou_delta": float(
                    audits[0].rows[sample_id].get("change_iou", 0.0)
                    - audits[1].rows[sample_id].get("change_iou", 0.0)
                ),
            }
        index_rows.append(
            {
                "rank": rank,
                "folder": folder.name,
                "sample_id": sample_id,
                "aoi_id": record.aoi_id,
                "target_edit": record.edit_type,
                "category": short_category,
                "methods": method_rows,
                "comparison": comparison,
                "zoom_box_yxyx": [
                    zoom[0].start,
                    zoom[1].start,
                    zoom[0].stop,
                    zoom[1].stop,
                ],
            }
        )
    with (output_dir / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "sn7-updater-qualitative-v1",
        "manifest": str(manifest),
        "audits": {name: str(path) for name, path in audit_paths},
        "sample_count": len(index_rows),
        "per_edit_success": per_edit,
        "per_edit_failure": failures_per_edit,
        "per_edit_disagreement": disagreements_per_edit,
        "minimum_visual_fraction": min_visual_fraction,
        "render_mask_size": list(prior.shape),
        "output_size": output_size,
        "mask_resampling": "nearest",
        "selection_rule": (
            "primary method first: operation correctness, final-map IoU, change IoU, "
            "map-IoU gain; paired improvement uses primary-minus-second audit"
        ),
        "panels_are_unlabeled_individual_files": True,
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--audit",
        action="append",
        type=_parse_named_path,
        required=True,
        help="Repeatable NAME=PATH validation audit.",
    )
    parser.add_argument("--per-edit", type=int, default=2)
    parser.add_argument("--failures-per-edit", type=int, default=1)
    parser.add_argument("--disagreements-per-edit", type=int, default=1)
    parser.add_argument("--min-visual-fraction", type=float, default=0.01)
    parser.add_argument("--output-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.per_edit, args.failures_per_edit, args.disagreements_per_edit) < 0:
        raise ValueError("selection counts must be non-negative")
    if not 0.0 <= args.min_visual_fraction < 1.0:
        raise ValueError("minimum visual fraction must be in [0, 1)")
    if args.output_size <= 0:
        raise ValueError("output size must be positive")
    summary = render(
        args.manifest,
        args.audit,
        args.output_dir,
        per_edit=args.per_edit,
        failures_per_edit=args.failures_per_edit,
        disagreements_per_edit=args.disagreements_per_edit,
        min_visual_fraction=args.min_visual_fraction,
        output_size=args.output_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

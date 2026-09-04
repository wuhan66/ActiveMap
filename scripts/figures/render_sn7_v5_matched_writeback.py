#!/usr/bin/env python3
"""Render the registered SN7 V5 matched-writeback qualitative plate.

The renderer is intentionally fail-closed.  It accepts only the pre-registered
validation cases and reconstructs every map panel from the stored executable
writeback artifacts.  It never selects cases by outcome and never substitutes
reference geometry for a prediction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)
FORCED_ACQUISITION_POLICY = "forced_safe_commit"
OPERATIONS = ("ADD", "DELETE", "RESHAPE")
RENDERER_VERSION = "sn7-v5-matched-writeback-qualitative-v4"
PRIOR_COLOR = np.asarray((243, 156, 18), dtype=np.float32)
DIRECT_COLOR = np.asarray((0, 114, 178), dtype=np.float32)
SELECTED_COLOR = np.asarray((0, 158, 115), dtype=np.float32)
SAFE_COLOR = np.asarray((110, 64, 170), dtype=np.float32)
REFERENCE_COLOR = np.asarray((20, 150, 80), dtype=np.float32)
TP_COLOR = np.asarray((0, 158, 115), dtype=np.float32)
FP_COLOR = np.asarray((213, 94, 0), dtype=np.float32)
FN_COLOR = np.asarray((86, 180, 233), dtype=np.float32)


@dataclass(frozen=True)
class CaseAssets:
    case: dict[str, Any]
    episode: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    selected_rollout: dict[str, Any]
    direct_rollout: dict[str, Any]
    artifacts: dict[str, dict[str, np.ndarray]]
    rgb: np.ndarray
    evidence_strip: Image.Image | None
    sources: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def parse_root_map(value: str | None) -> tuple[Path, Path] | None:
    if value is None:
        return None
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise ValueError("asset root map must be SOURCE=DESTINATION")
    return Path(source), Path(destination)


def remap_path(path: Path, root_map: tuple[Path, Path] | None) -> Path:
    if root_map is None:
        return path
    source, destination = root_map
    try:
        return destination / path.relative_to(source)
    except ValueError:
        return path


def require_validation(value: dict[str, Any], *, name: str) -> None:
    if value.get("split") != "val" or value.get("test_assets_read") is not False:
        raise ValueError(f"{name} must be validation-only and test-free")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != "sn7-v5-predeclared-qualitative-manifest-v1":
        raise ValueError("unexpected V5 qualitative manifest schema")
    require_validation(manifest, name="qualitative manifest")
    if manifest.get("controller_or_writeback_outputs_read") is not False:
        raise ValueError("qualitative manifest was not frozen before controller outputs")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 9:
        raise ValueError("V5 qualitative manifest must contain exactly nine fixed cases")
    counts = {operation: 0 for operation in OPERATIONS}
    identities: set[tuple[str, float]] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("V5 qualitative case must be an object")
        operation = str(case.get("operation"))
        if operation not in counts:
            raise ValueError(f"unsupported V5 qualitative operation: {operation!r}")
        task_id = str(case.get("task_id", ""))
        source_episode = str(case.get("source_episode", ""))
        budget = float(case.get("budget", 0.0))
        if not task_id or not source_episode or budget <= 0.0:
            raise ValueError("V5 qualitative case lacks task, source episode, or budget")
        key = (task_id, budget)
        if key in identities:
            raise ValueError(f"duplicate V5 qualitative task-budget: {key}")
        identities.add(key)
        counts[operation] += 1
    if counts != {operation: 3 for operation in OPERATIONS}:
        raise ValueError(f"V5 qualitative manifest has invalid operation support: {counts}")
    return manifest


def load_episodes(path: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = {str(case["source_episode"]) for case in cases}
    episodes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        episode_id = str(row.get("episode_id", ""))
        if episode_id not in required:
            continue
        require_validation(row, name=f"episode {episode_id}")
        if episode_id in episodes:
            raise ValueError(f"duplicate source episode in manifest: {episode_id}")
        episodes[episode_id] = row
    missing = sorted(required - set(episodes))
    if missing:
        raise ValueError(f"V5 qualitative source episodes are missing: {missing}")
    return episodes


def indexed_rows(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for row in read_jsonl(path):
        require_validation(row, name=f"writeback row in {path}")
        key = (str(row.get("task_id", "")), float(row.get("budget", 0.0)))
        if not key[0] or key[1] <= 0.0 or key in rows:
            raise ValueError(f"invalid or duplicate writeback key in {path}: {key}")
        if not isinstance(row.get("mask_artifact"), str) or not row["mask_artifact"]:
            raise ValueError(f"writeback row lacks mask artifact in {path}: {key}")
        rows[key] = row
    return rows


def indexed_rollouts(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for row in read_jsonl(path):
        require_validation(row, name=f"rollout row in {path}")
        key = (str(row.get("task_id", "")), float(row.get("budget", 0.0)))
        if not key[0] or key[1] <= 0.0 or key in rows:
            raise ValueError(f"invalid or duplicate rollout key in {path}: {key}")
        rows[key] = row
    return rows


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = np.load(path)
    expected = ("committed_mask", "prior_mask", "target_mask", "valid_mask")
    missing = [name for name in expected if name not in archive]
    if missing:
        raise ValueError(f"mask artifact lacks {missing}: {path}")
    result = {name: np.asarray(archive[name], dtype=bool) for name in expected}
    shapes = {value.shape for value in result.values()}
    if len(shapes) != 1 or next(iter(shapes)).__len__() != 2:
        raise ValueError(f"mask artifact has incompatible 2D masks: {path}")
    return result


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    encoded = Image.fromarray(image, mode="RGB").resize(
        (shape[1], shape[0]), Image.Resampling.LANCZOS
    )
    return np.asarray(encoded, dtype=np.uint8)


def _normalise_bands(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError("raster crop must be [channel, height, width]")
    data = data[:3].astype(np.float32)
    while data.shape[0] < 3:
        data = np.concatenate((data, data[-1:]), axis=0)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return np.zeros((*data.shape[1:], 3), dtype=np.uint8)
    lower, upper = np.quantile(finite, (0.01, 0.99))
    if upper <= lower:
        upper = lower + 1.0
    scaled = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    return (np.moveaxis(scaled, 0, -1) * 255.0).round().astype(np.uint8)


def crop_rgb(path: Path, region: list[int], shape: tuple[int, int]) -> np.ndarray:
    if len(region) != 4:
        raise ValueError("evidence region must have four coordinates")
    top, left, bottom, right = (int(value) for value in region)
    if bottom <= top or right <= left:
        raise ValueError(f"invalid evidence region: {region}")
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.windows import Window
        except ImportError as exc:
            raise RuntimeError("GeoTIFF rendering requires rasterio") from exc
        with rasterio.open(path) as dataset:
            indexes = list(range(1, min(dataset.count, 3) + 1))
            raw = dataset.read(
                indexes,
                window=Window(left, top, right - left, bottom - top),
                out_shape=(len(indexes), shape[0], shape[1]),
                boundless=True,
                fill_value=0,
                resampling=Resampling.bilinear,
            )
        return _normalise_bands(raw)
    with Image.open(path) as image:
        rgb = image.convert("RGB").crop((left, top, right, bottom))
        rgb = rgb.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
    return result


def mask_outline(mask: np.ndarray, *, width: int = 2) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    interior = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return _dilate(mask & ~interior, max(0, width - 1))


def blend_layer(
    canvas: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    color: np.ndarray,
    *,
    fill_alpha: float = 0.17,
    line_alpha: float = 0.96,
) -> None:
    mask = mask.astype(bool) & valid
    interior = mask & ~mask_outline(mask, width=1)
    if interior.any():
        canvas[interior] = canvas[interior] * (1.0 - fill_alpha) + color * fill_alpha
    outline = mask_outline(mask, width=2) & valid
    if outline.any():
        canvas[outline] = canvas[outline] * (1.0 - line_alpha) + color * line_alpha


def map_panel(
    rgb: np.ndarray,
    *,
    prior: np.ndarray,
    final: np.ndarray,
    valid: np.ndarray,
    final_color: np.ndarray,
) -> Image.Image:
    canvas = rgb.astype(np.float32) * 0.83 + 18.0
    canvas[~valid] *= 0.30
    blend_layer(canvas, prior, valid, PRIOR_COLOR, fill_alpha=0.08, line_alpha=0.92)
    if not np.array_equal(prior, final):
        blend_layer(canvas, final, valid, final_color, fill_alpha=0.18, line_alpha=0.98)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def residual_panel(
    rgb: np.ndarray,
    *,
    prior: np.ndarray,
    final: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    canvas = rgb.astype(np.float32) * 0.65 + 28.0
    canvas[~valid] *= 0.30
    truth_delta = np.logical_xor(prior, target) & valid
    final_delta = np.logical_xor(prior, final) & valid
    true_positive = truth_delta & final_delta
    false_positive = final_delta & ~truth_delta
    false_negative = truth_delta & ~final_delta
    for region, color in (
        (true_positive, TP_COLOR),
        (false_positive, FP_COLOR),
        (false_negative, FN_COLOR),
    ):
        if region.any():
            canvas[region] = canvas[region] * 0.18 + color * 0.82
            outline = mask_outline(region, width=2) & valid
            canvas[outline] = color
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def zoom_box(mask: np.ndarray, *, fallback: np.ndarray) -> tuple[slice, slice]:
    if mask.shape != fallback.shape:
        raise ValueError("V5 zoom mask and fallback must share a shape")
    if not mask.any():
        mask = fallback
    if not mask.any():
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    rows, cols = np.where(mask)
    margin = max(4, int(np.ceil(max(mask.shape) * 0.10)))
    top, bottom = (
        max(0, int(rows.min()) - margin),
        min(mask.shape[0], int(rows.max()) + margin + 1),
    )
    left, right = (
        max(0, int(cols.min()) - margin),
        min(mask.shape[1], int(cols.max()) + margin + 1),
    )
    return slice(top, bottom), slice(left, right)


def case_zoom_boxes(
    *,
    prior: np.ndarray,
    direct: np.ndarray,
    selected: np.ndarray,
    safe: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, tuple[slice, slice]]:
    change = np.zeros_like(prior, dtype=bool)
    for candidate in (direct, selected, safe, target):
        change |= np.logical_xor(prior, candidate)
    change &= valid
    residual = np.logical_xor(safe, target) & valid
    return {
        "change": zoom_box(change, fallback=residual),
        "residual": zoom_box(residual, fallback=change),
    }


def safe_commit_delta_panel(
    rgb: np.ndarray,
    *,
    prior: np.ndarray,
    proposed: np.ndarray,
    accepted: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    canvas = rgb.astype(np.float32) * 0.58 + 36.0
    canvas[~valid] *= 0.30
    proposal_delta = np.logical_xor(prior, proposed) & valid
    accepted_delta = np.logical_xor(prior, accepted) & valid
    rejected_delta = proposal_delta & ~accepted_delta
    retained_delta = proposal_delta & accepted_delta
    for region, color in (
        (retained_delta, SAFE_COLOR),
        (rejected_delta, FP_COLOR),
    ):
        if region.any():
            canvas[region] = canvas[region] * 0.18 + color * 0.82
            canvas[mask_outline(region, width=2) & valid] = color
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def save_image(image: Image.Image, path: Path, *, size: int) -> None:
    image.resize((size, size), Image.Resampling.NEAREST).save(path)


def paste_tiled(images: list[Image.Image], *, panel_size: int, rows: int) -> Image.Image:
    if len(images) % rows:
        raise ValueError("tile count must divide the requested rows")
    columns = len(images) // rows
    plate = Image.new("RGB", (columns * panel_size, rows * panel_size), "white")
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        plate.paste(
            image.resize((panel_size, panel_size), Image.Resampling.NEAREST),
            (col * panel_size, row * panel_size),
        )
    return plate


def write_svg_from_png(png_path: Path, output_path: Path) -> None:
    image = Image.open(png_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    output_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">'
        f'<image width="{image.width}" height="{image.height}" '
        f'href="data:image/png;base64,{encoded}"/>'
        "</svg>\n",
        encoding="utf-8",
    )


def output_bundle(plate: Image.Image, output_dir: Path, name: str) -> dict[str, str]:
    png = output_dir / f"{name}.png"
    tiff = output_dir / f"{name}.tiff"
    pdf = output_dir / f"{name}.pdf"
    svg = output_dir / f"{name}.svg"
    plate.save(png)
    plate.save(tiff, compression="tiff_lzw")
    plate.save(pdf, "PDF", resolution=300.0)
    write_svg_from_png(png, svg)
    return {
        suffix: sha256(path)
        for suffix, path in {"png": png, "tiff": tiff, "pdf": pdf, "svg": svg}.items()
    }


def evidence_strip(
    evidence_root: Path | None,
    *,
    source_episode: str,
    selected_extra_id: str | None,
    task_id: str,
    budget: float,
) -> tuple[Image.Image | None, dict[str, Any] | None]:
    if selected_extra_id is None:
        return None, None
    if evidence_root is None:
        raise ValueError(
            f"{source_episode} acquired evidence but no --evidence-root was supplied; "
            "export the recorded crop first"
        )
    manifest_path = evidence_root / source_episode / "manifest.json"
    manifest = read_json(manifest_path)
    require_validation(manifest, name=f"evidence manifest for {source_episode}")
    if manifest.get("source_episode") != source_episode:
        raise ValueError(f"evidence manifest source episode mismatch: {manifest_path}")
    if manifest.get("task_id") != task_id or float(manifest.get("budget", -1.0)) != budget:
        raise ValueError(f"evidence manifest trace identity mismatch: {manifest_path}")
    records = [
        item
        for item in manifest.get("evidence", [])
        if item.get("evidence_id") == selected_extra_id
    ]
    if len(records) != 1:
        raise ValueError(f"recorded acquired evidence crop is unavailable for {source_episode}")
    record = records[0]
    if "timestamp" not in record or "scale" not in record:
        raise ValueError(f"evidence crop lacks timestamp or scale: {manifest_path}")
    image_path = evidence_root / source_episode / str(record["output"])
    if not image_path.is_file() or sha256(image_path) != record.get("output_sha256"):
        raise ValueError(f"evidence crop hash mismatch: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image, {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "record": record,
    }


def select_cases(manifest: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    cases = list(manifest["cases"])
    if scope == "supplement":
        return cases
    chosen: list[dict[str, Any]] = []
    for operation in OPERATIONS:
        chosen.append(next(case for case in cases if case["operation"] == operation))
    return chosen


def source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def require_aggregate(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    if summary.get("schema_version") != "sn7-v5-matched-nonkeep-factorial-v2":
        raise ValueError("unexpected V5 factorial summary schema")
    require_validation(summary, name="V5 factorial summary")
    if summary.get("model_seeds") != [20260817, 20260818, 20260819]:
        raise ValueError("V5 factorial summary lacks the three registered seeds")
    if set(summary.get("policies", [])) != set(POLICIES + (FORCED_ACQUISITION_POLICY,)):
        raise ValueError("V5 factorial summary lacks the registered 2x2 matrix and forced control")
    promotion = summary.get("promotion")
    if not isinstance(promotion, dict) or "eligible_for_extension_claim" not in promotion:
        raise ValueError("V5 factorial summary lacks its registered promotion decision")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("V5 factorial summary lacks its registered writeback inputs")
    for seed in summary["model_seeds"]:
        by_policy = inputs.get(str(seed))
        if not isinstance(by_policy, dict) or set(by_policy) != set(
            POLICIES + (FORCED_ACQUISITION_POLICY,)
        ):
            raise ValueError("V5 factorial summary has incomplete registered writeback inputs")
    return summary


def render(
    *,
    qualitative_manifest: Path,
    episodes_path: Path,
    run_root: Path,
    output_dir: Path,
    scope: str,
    render_seed: int | None,
    evidence_root: Path | None,
    aggregate_summary: Path | None,
    asset_root_map: tuple[Path, Path] | None,
    panel_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V5 qualitative output: {output_dir}")
    if scope == "main" and aggregate_summary is None:
        raise ValueError("main-paper V5 rendering requires the completed factorial aggregate")
    manifest = load_manifest(qualitative_manifest)
    aggregate: dict[str, Any] | None = None
    if aggregate_summary is not None:
        aggregate = require_aggregate(aggregate_summary)
    if scope == "main" and aggregate["promotion"].get("eligible_for_extension_claim") is not True:
        raise ValueError("main-paper V5 rendering requires a passed registered promotion gate")
    seed = int(render_seed if render_seed is not None else manifest["seed"])
    if seed not in (20260817, 20260818, 20260819):
        raise ValueError("V5 rendering seed must be one of the registered three seeds")
    cases = select_cases(manifest, scope)
    episodes = load_episodes(episodes_path, cases)
    writeback_paths = {
        policy: run_root / "writebacks" / f"seed{seed}" / "val" / policy / "writeback.jsonl"
        for policy in POLICIES
    }
    if aggregate is not None:
        registered_inputs = aggregate["inputs"][str(seed)]
        for policy, writeback_path in writeback_paths.items():
            registered_path = Path(str(registered_inputs[policy])).resolve()
            if registered_path != writeback_path.resolve():
                raise ValueError(
                    f"V5 aggregate input does not match renderer writeback for {seed}:{policy}"
                )
    writebacks = {policy: indexed_rows(path) for policy, path in writeback_paths.items()}
    rollouts_dir = run_root / "rollouts" / f"seed{seed}_val"
    selected_rollouts_path = rollouts_dir / "selected_rollouts.jsonl"
    direct_rollouts_path = rollouts_dir / "direct_rollouts.jsonl"
    selected_rollouts = indexed_rollouts(selected_rollouts_path)
    direct_rollouts = indexed_rollouts(direct_rollouts_path)
    checkpoint_receipt = run_root / "authorization" / "checkpoint_receipts" / f"seed{seed}.json"
    calibration_receipt = run_root / "train_calibration" / f"seed{seed}.json"
    receipt_payload = read_json(checkpoint_receipt)
    calibration_payload = read_json(calibration_receipt)
    if receipt_payload.get("test_assets_read") is not False:
        raise ValueError("updater checkpoint receipt is not test-free")
    if (
        calibration_payload.get("split") != "train"
        or calibration_payload.get("test_assets_read") is not False
    ):
        raise ValueError("Safe Commit receipt is not train-only")

    output_dir.mkdir(parents=True)
    case_entries = []
    main_full_images: list[Image.Image] = []
    main_change_zoom_images: list[Image.Image] = []
    main_residual_zoom_images: list[Image.Image] = []
    for index, case in enumerate(cases, start=1):
        key = (str(case["task_id"]), float(case["budget"]))
        if key not in selected_rollouts or key not in direct_rollouts:
            raise ValueError(f"rollout support is missing for fixed case {key}")
        selected_rollout, direct_rollout = selected_rollouts[key], direct_rollouts[key]
        source_episode = str(case["source_episode"])
        if (
            selected_rollout.get("source_episode") != source_episode
            or direct_rollout.get("source_episode") != source_episode
        ):
            raise ValueError(f"rollout source episode mismatch for fixed case {key}")
        rows: dict[str, dict[str, Any]] = {}
        artifacts: dict[str, dict[str, np.ndarray]] = {}
        for policy in POLICIES:
            if key not in writebacks[policy]:
                raise ValueError(f"{policy} lacks fixed V5 case {key}")
            row = writebacks[policy][key]
            if str(row.get("target")) != f"COMMIT:{case['operation']}":
                raise ValueError(f"{policy} has an operation mismatch for fixed V5 case {key}")
            rows[policy] = row
            artifact_path = remap_path(Path(str(row["mask_artifact"])), asset_root_map)
            artifacts[policy] = load_artifact(artifact_path)
        prior = artifacts["direct_commit"]["prior_mask"]
        target = artifacts["direct_commit"]["target_mask"]
        valid = artifacts["direct_commit"]["valid_mask"]
        for policy, artifact in artifacts.items():
            for name, reference in (
                ("prior_mask", prior),
                ("target_mask", target),
                ("valid_mask", valid),
            ):
                if not np.array_equal(artifact[name], reference):
                    raise ValueError(
                        f"{policy} does not share the geographic frame for fixed case {key}"
                    )
        initial_id = str(selected_rollout.get("initial_evidence_id", ""))
        episode = episodes[source_episode]
        catalog = {
            str(item.get("evidence_id")): item for item in episode.get("evidence_catalog", [])
        }
        initial = catalog.get(initial_id)
        if initial is None:
            raise ValueError(f"initial evidence is absent from source episode {source_episode}")
        image_path = remap_path(Path(str(initial.get("image_path", ""))), asset_root_map)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rgb = crop_rgb(image_path, list(initial["region"]), prior.shape)
        extra_id = selected_rollout.get("selected_extra_evidence_id")
        strip, strip_source = evidence_strip(
            evidence_root,
            source_episode=source_episode,
            selected_extra_id=str(extra_id) if extra_id else None,
            task_id=str(case["task_id"]),
            budget=float(case["budget"]),
        )
        assets = CaseAssets(
            case=case,
            episode=episode,
            rows=rows,
            selected_rollout=selected_rollout,
            direct_rollout=direct_rollout,
            artifacts=artifacts,
            rgb=rgb,
            evidence_strip=strip,
            sources={
                "source_image": {
                    "path": str(image_path),
                    "sha256": sha256(image_path),
                    "region": list(initial["region"]),
                },
                "evidence": strip_source,
            },
        )
        direct = (
            assets.artifacts["direct_commit"]["committed_mask"]
            if bool(assets.rows["direct_commit"].get("writeback_changed"))
            else prior
        )
        selected = (
            assets.artifacts["selected_commit"]["committed_mask"]
            if bool(assets.rows["selected_commit"].get("writeback_changed"))
            else prior
        )
        safe = (
            assets.artifacts["selected_safe_commit"]["committed_mask"]
            if bool(assets.rows["selected_safe_commit"].get("writeback_changed"))
            else prior
        )
        panels = [
            (
                "prior",
                map_panel(
                    assets.rgb, prior=prior, final=prior, valid=valid, final_color=PRIOR_COLOR
                ),
            ),
            (
                "direct_commit",
                map_panel(
                    assets.rgb, prior=prior, final=direct, valid=valid, final_color=DIRECT_COLOR
                ),
            ),
            (
                "selected_commit",
                map_panel(
                    assets.rgb, prior=prior, final=selected, valid=valid, final_color=SELECTED_COLOR
                ),
            ),
            (
                "selected_safe_commit",
                map_panel(assets.rgb, prior=prior, final=safe, valid=valid, final_color=SAFE_COLOR),
            ),
            (
                "reference",
                map_panel(
                    assets.rgb, prior=prior, final=target, valid=valid, final_color=REFERENCE_COLOR
                ),
            ),
            (
                "residual",
                residual_panel(assets.rgb, prior=prior, final=safe, target=target, valid=valid),
            ),
        ]
        zoom_boxes = case_zoom_boxes(
            prior=prior,
            direct=direct,
            selected=selected,
            safe=safe,
            target=target,
            valid=valid,
        )
        case_dir = output_dir / f"{index:02d}_{str(case['operation']).lower()}_{source_episode}"
        case_dir.mkdir()
        panel_sources = {}
        full_images: list[Image.Image] = []
        change_zoom_images: list[Image.Image] = []
        residual_zoom_images: list[Image.Image] = []
        for name, image in panels:
            full_path = case_dir / f"{name}.png"
            save_image(image, full_path, size=panel_size)
            full_images.append(Image.open(full_path).convert("RGB"))
            zoom_sources: dict[str, dict[str, str]] = {}
            for zoom_name, (row_slice, col_slice) in zoom_boxes.items():
                zoom_path = case_dir / f"{name}_{zoom_name}_zoom.png"
                zoom = image.crop((col_slice.start, row_slice.start, col_slice.stop, row_slice.stop))
                save_image(zoom, zoom_path, size=panel_size)
                if zoom_name == "change":
                    change_zoom_images.append(Image.open(zoom_path).convert("RGB"))
                else:
                    residual_zoom_images.append(Image.open(zoom_path).convert("RGB"))
                zoom_sources[zoom_name] = {
                    "file": zoom_path.name,
                    "sha256": sha256(zoom_path),
                }
            panel_sources[name] = {
                "full": full_path.name,
                "full_sha256": sha256(full_path),
                "zooms": zoom_sources,
            }
        safe_delta_path = case_dir / "safe_commit_delta.png"
        save_image(
            safe_commit_delta_panel(
                assets.rgb,
                prior=prior,
                proposed=selected,
                accepted=safe,
                valid=valid,
            ),
            safe_delta_path,
            size=panel_size,
        )
        panel_sources["safe_commit_delta"] = {
            "file": safe_delta_path.name,
            "sha256": sha256(safe_delta_path),
        }
        if assets.evidence_strip is not None:
            evidence_path = case_dir / "selected_evidence.png"
            assets.evidence_strip.resize((panel_size, panel_size), Image.Resampling.LANCZOS).save(
                evidence_path
            )
            panel_sources["selected_evidence"] = {
                "file": evidence_path.name,
                "sha256": sha256(evidence_path),
            }
        plate = paste_tiled(
            full_images + change_zoom_images + residual_zoom_images,
            panel_size=panel_size,
            rows=3,
        )
        plate_hashes = output_bundle(plate, case_dir, "plate")
        if scope == "main":
            main_full_images.extend(full_images)
            main_change_zoom_images.extend(change_zoom_images)
            main_residual_zoom_images.extend(residual_zoom_images)
        case_entries.append(
            {
                "operation": case["operation"],
                "source_episode": source_episode,
                "task_id": case["task_id"],
                "budget": case["budget"],
                "aoi_id": case["aoi_id"],
                "directory": case_dir.name,
                "zoom_bounds": {
                    name: [row_slice.start, col_slice.start, row_slice.stop, col_slice.stop]
                    for name, (row_slice, col_slice) in zoom_boxes.items()
                },
                "selected_extra_evidence": bool(extra_id),
                "selected_extra_evidence_id": extra_id,
                "panels": panel_sources,
                "plate_sha256": plate_hashes,
                "writebacks": {
                    policy: source_record(writeback_paths[policy]) for policy in POLICIES
                },
                "rollouts": {
                    "direct": source_record(direct_rollouts_path),
                    "selected": source_record(selected_rollouts_path),
                },
                "source": assets.sources,
            }
        )
    if scope == "main":
        main_assets = {
            "full_context": output_bundle(
                paste_tiled(main_full_images, panel_size=panel_size, rows=3),
                output_dir,
                "v5_main_context",
            ),
            "change_zooms": output_bundle(
                paste_tiled(main_change_zoom_images, panel_size=panel_size, rows=3),
                output_dir,
                "v5_main_change_zooms",
            ),
            "residual_zooms": output_bundle(
                paste_tiled(main_residual_zoom_images, panel_size=panel_size, rows=3),
                output_dir,
                "v5_main_residual_zooms",
            ),
            "contact_sheet": output_bundle(
                paste_tiled(
                    main_full_images + main_change_zoom_images + main_residual_zoom_images,
                    panel_size=panel_size,
                    rows=9,
                ),
                output_dir,
                "v5_main_contact_sheet",
            ),
        }
    else:
        main_assets = {}
    manifest_output = {
        "schema_version": RENDERER_VERSION,
        "scope": scope,
        "render_seed": seed,
        "split": "val",
        "test_assets_read": False,
        "qualitative_manifest": source_record(qualitative_manifest),
        "episodes": source_record(episodes_path),
        "factorial_summary": source_record(aggregate_summary) if aggregate_summary else None,
        "updater_checkpoint_receipt": source_record(checkpoint_receipt),
        "updater_checkpoint_sha256": receipt_payload.get("checkpoint_sha256"),
        "safe_commit_calibration": source_record(calibration_receipt),
        "panel_size": panel_size,
        "renderer": {
            "version": RENDERER_VERSION,
            "python": sys.version,
            "argv": sys.argv,
        },
        "cases": case_entries,
        "main_assets": main_assets,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_output, indent=2) + "\n", encoding="utf-8")
    return manifest_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualitative_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scope", choices=("main", "supplement"), default="supplement")
    parser.add_argument("--render-seed", type=int)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--aggregate-summary", type=Path)
    parser.add_argument("--asset-root-map")
    parser.add_argument("--panel-size", type=int, default=384)
    args = parser.parse_args()
    if args.panel_size < 64:
        raise ValueError("panel size must be at least 64 pixels")
    result = render(
        qualitative_manifest=args.qualitative_manifest,
        episodes_path=args.episodes,
        run_root=args.run_root,
        output_dir=args.output_dir,
        scope=args.scope,
        render_seed=args.render_seed,
        evidence_root=args.evidence_root,
        aggregate_summary=args.aggregate_summary,
        asset_root_map=parse_root_map(args.asset_root_map),
        panel_size=args.panel_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render every validation ArgoTweak native evidence bundle as a map-native board.

The exporter is deliberately exhaustive: it orders all validation episode/evidence
pairs deterministically and makes no result- or score-based example selection.
Reference maps are rendered only as offline labels for visual review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from activemap.data.argotweak_native import ArgoTweakNativeEpisode, proposal_city_geometry
try:
    from scripts.visual_asset_utils import save_panel_layers
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from visual_asset_utils import save_panel_layers


CAMERA_ORDER = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)
OPERATION_COLORS = {
    "ADD": (56, 118, 216),
    "DELETE": (202, 68, 73),
    "RESHAPE": (139, 92, 183),
    "KEEP": (122, 128, 137),
}
PRIOR_COLOR = (212, 137, 56)
TARGET_COLOR = (41, 158, 111)
BLOCKED_COLOR = (136, 142, 151)
BACKGROUND = (19, 26, 34)
GRID_COLOR = (52, 61, 72)
TEXT_COLOR = (24, 31, 40)


def _read_episodes(path: Path) -> list[ArgoTweakNativeEpisode]:
    episodes = [
        ArgoTweakNativeEpisode.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes:
        raise ValueError(f"no episodes in {path}")
    if any(episode.split != "val" or episode.test_assets_read for episode in episodes):
        raise PermissionError("casebook rendering accepts validation-only native episodes")
    return sorted(episodes, key=lambda episode: episode.segment_id)


def _load_feature_collection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"invalid GeoJSON feature collection: {path}")
    return [feature for feature in features if isinstance(feature, dict)]


def _rings(geometry: dict[str, Any] | None) -> Iterable[list[tuple[float, float]]]:
    if not isinstance(geometry, dict):
        return []
    coordinates = geometry.get("coordinates")
    kind = geometry.get("type")
    if kind == "Polygon" and isinstance(coordinates, list):
        return [
            [(float(point[0]), float(point[1])) for point in ring if len(point) >= 2]
            for ring in coordinates
            if isinstance(ring, list)
        ]
    if kind == "MultiPolygon" and isinstance(coordinates, list):
        return [
            [(float(point[0]), float(point[1])) for point in ring if len(point) >= 2]
            for polygon in coordinates
            if isinstance(polygon, list)
            for ring in polygon
            if isinstance(ring, list)
        ]
    if kind == "LineString" and isinstance(coordinates, list):
        return [[(float(point[0]), float(point[1])) for point in coordinates if len(point) >= 2]]
    return []


def _visible_rings(
    features: Iterable[dict[str, Any]], center: tuple[float, float], radius: float
) -> list[list[tuple[float, float]]]:
    visible: list[list[tuple[float, float]]] = []
    left, bottom = center[0] - radius, center[1] - radius
    right, top = center[0] + radius, center[1] + radius
    for feature in features:
        for ring in _rings(feature.get("geometry")):
            if len(ring) < 2:
                continue
            xs, ys = zip(*ring)
            if max(xs) < left or min(xs) > right or max(ys) < bottom or min(ys) > top:
                continue
            visible.append(ring)
    return visible


def _to_pixels(
    ring: list[tuple[float, float]], center: tuple[float, float], radius: float, size: int
) -> list[tuple[int, int]]:
    scale = size / (2.0 * radius)
    return [
        (
            int(round((x - (center[0] - radius)) * scale)),
            int(round((center[1] + radius - y) * scale)),
        )
        for x, y in ring
    ]


def _draw_rings(
    draw: ImageDraw.ImageDraw,
    rings: Iterable[list[tuple[float, float]]],
    center: tuple[float, float],
    radius: float,
    size: int,
    color: tuple[int, int, int],
    width: int,
    *,
    alpha: int = 255,
    dashed: bool = False,
) -> None:
    rgba = (*color, alpha)
    for ring in rings:
        points = _to_pixels(ring, center, radius, size)
        if len(points) < 2:
            continue
        if dashed:
            for index in range(0, len(points) - 1, 2):
                draw.line([points[index], points[index + 1]], fill=rgba, width=width)
        else:
            draw.line(points, fill=rgba, width=width, joint="curve")


def _proposal_rings(evidence: Any) -> list[tuple[dict[str, Any], list[tuple[float, float]]]]:
    result: list[tuple[dict[str, Any], list[tuple[float, float]]]] = []
    for proposal in evidence.proposals:
        try:
            geometry = proposal_city_geometry(proposal, evidence.city_se3_egovehicle).model_dump()
        except (KeyError, TypeError, ValueError):
            continue
        for ring in _rings(geometry):
            if len(ring) >= 2:
                result.append((proposal, ring))
    return result


def _render_bev(
    *,
    prior: list[dict[str, Any]],
    target: list[dict[str, Any]],
    evidence: Any,
    center: tuple[float, float],
    radius: float,
    size: int,
    include_prior: bool = True,
    include_target: bool = True,
    include_proposals: bool = True,
) -> Image.Image:
    image = Image.new("RGBA", (size, size), (*BACKGROUND, 255))
    draw = ImageDraw.Draw(image)
    for tick in range(0, size + 1, max(1, size // 6)):
        draw.line([(tick, 0), (tick, size)], fill=(*GRID_COLOR, 220), width=1)
        draw.line([(0, tick), (size, tick)], fill=(*GRID_COLOR, 220), width=1)

    if include_prior:
        _draw_rings(
            draw,
            _visible_rings(prior, center, radius),
            center,
            radius,
            size,
            PRIOR_COLOR,
            2,
            alpha=210,
        )
    if include_target:
        _draw_rings(draw, _visible_rings(target, center, radius), center, radius, size, TARGET_COLOR, 2, alpha=220)

    if include_proposals:
        ready = set(evidence.commit_ready_proposal_ids)
        for proposal, ring in _proposal_rings(evidence):
            proposal_id = str(proposal.get("proposal_id", ""))
            operation = str(proposal.get("operation", "KEEP")).upper()
            color = OPERATION_COLORS.get(operation, OPERATION_COLORS["KEEP"])
            allowed = proposal_id in ready
            _draw_rings(
                draw,
                [ring],
                center,
                radius,
                size,
                color if allowed else BLOCKED_COLOR,
                4 if allowed else 2,
                dashed=not allowed,
            )

    ego_x, ego_y = _to_pixels([center], center, radius, size)[0]
    draw.ellipse((ego_x - 7, ego_y - 7, ego_x + 7, ego_y + 7), fill=(250, 250, 250), outline=(0, 0, 0), width=2)
    draw.polygon([(ego_x, ego_y - 14), (ego_x - 7, ego_y + 8), (ego_x + 7, ego_y + 8)], fill=(250, 250, 250))
    return image.convert("RGB")


def _labeled_tile(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    tile = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size[0], 24), fill=(255, 255, 255))
    draw.text((6, 5), label, fill=TEXT_COLOR, font=ImageFont.load_default())
    return tile


def _camera_contact_sheet(bundle_path: Path, tile_size: tuple[int, int]) -> Image.Image:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    images = bundle.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"camera bundle lacks images: {bundle_path}")
    width, height = tile_size
    sheet = Image.new("RGB", (4 * width, 2 * height), "white")
    for index, camera in enumerate(CAMERA_ORDER):
        image_path = Path(str(images.get(camera, "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing camera image for {camera}: {image_path}")
        with Image.open(image_path) as source:
            tile = _labeled_tile(source, camera.replace("ring_", ""), tile_size)
        row, column = divmod(index, 4)
        sheet.paste(tile, (column * width, row * height))
    note = Image.new("RGB", tile_size, (243, 245, 247))
    note_draw = ImageDraw.Draw(note)
    note_draw.text((12, 16), "Synchronized", fill=TEXT_COLOR, font=ImageFont.load_default())
    note_draw.text((12, 35), "seven-camera", fill=TEXT_COLOR, font=ImageFont.load_default())
    note_draw.text((12, 54), "evidence bundle", fill=TEXT_COLOR, font=ImageFont.load_default())
    sheet.paste(note, (3 * width, height))
    return sheet


def _receipt(evidence: Any, *, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    operation_counts = Counter(str(row.get("operation", "KEEP")).upper() for row in evidence.proposals)
    blocked_counts = Counter(evidence.blocked_proposals.values())
    belief = list(evidence.operation_belief)
    lines = [
        f"Evidence {evidence.evidence_id}   timestamp {evidence.timestamp}   cost {evidence.cost:g}",
        "Frozen typed proposals: " + ", ".join(f"{name} {operation_counts.get(name, 0)}" for name in ("ADD", "DELETE", "RESHAPE", "KEEP")),
        "Authorization: "
        + f"commit-ready {len(evidence.commit_ready_proposal_ids)} / {len(evidence.proposals)}"
        + ("; blocked " + ", ".join(f"{reason} {count}" for reason, count in sorted(blocked_counts.items())) if blocked_counts else ""),
        "Operation belief: " + ", ".join(f"{name.lower()} {belief[index]:.2f}" for index, name in enumerate(("ADD", "DELETE", "RESHAPE", "KEEP"))),
        "Legend: prior amber | reference green | ready typed proposal color | blocked gray dashed",
    ]
    for row, line in enumerate(lines):
        draw.text((12, 10 + row * 16), line, fill=TEXT_COLOR, font=font)
    return image


def _focus_center(evidence: Any, ego_center: tuple[float, float]) -> tuple[float, float]:
    points = [point for _, ring in _proposal_rings(evidence) for point in ring]
    if not points:
        return ego_center
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _preserve_source(source: Path, destination: Path) -> str:
    """Expose a raw asset at the case path without silently transforming it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _individual_assets(
    case_dir: Path,
    *,
    evidence: Any,
    prior: list[dict[str, Any]],
    target: list[dict[str, Any]],
    map_size: int,
    overview_radius: float,
    crop_radius: float,
) -> dict[str, Any]:
    """Emit independent camera and vector layers for author-side composition."""

    bundle = json.loads(Path(evidence.camera_bundle_path).read_text(encoding="utf-8"))
    images = bundle.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"camera bundle lacks images: {evidence.camera_bundle_path}")
    camera_assets = {}
    for camera in CAMERA_ORDER:
        source = Path(str(images.get(camera, "")))
        if not source.is_file():
            raise FileNotFoundError(f"missing camera image for {camera}: {source}")
        destination = case_dir / "camera" / f"{camera}.jpg"
        camera_assets[camera] = {
            "asset": str(destination.relative_to(case_dir)),
            "preservation": _preserve_source(source, destination),
            "source": str(source),
        }
    ego = tuple(float(value) for value in evidence.city_se3_egovehicle["translation"][:2])
    focus = _focus_center(evidence, ego)
    panels: list[tuple[str, np.ndarray]] = []
    for view_name, center, radius in (
        ("ego", ego, overview_radius),
        ("proposal_crop", focus, crop_radius),
    ):
        for layer_name, flags in (
            ("prior", {"include_prior": True, "include_target": False, "include_proposals": False}),
            ("reference_offline", {"include_prior": False, "include_target": True, "include_proposals": False}),
            ("typed_proposals", {"include_prior": False, "include_target": False, "include_proposals": True}),
        ):
            panels.append(
                (
                    f"map_{view_name}_{layer_name}",
                    np.asarray(
                        _render_bev(
                            prior=prior,
                            target=target,
                            evidence=evidence,
                            center=center,
                            radius=radius,
                            size=map_size,
                            **flags,
                        )
                    ),
                )
            )
    return {"camera": camera_assets, "map_layers": save_panel_layers(case_dir, panels)}


def _board(
    episode: ArgoTweakNativeEpisode,
    evidence: Any,
    prior: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    camera_size: tuple[int, int],
    map_size: int,
    overview_radius: float,
    crop_radius: float,
) -> Image.Image:
    cameras = _camera_contact_sheet(Path(evidence.camera_bundle_path), camera_size)
    ego = tuple(float(value) for value in evidence.city_se3_egovehicle["translation"][:2])
    focus = _focus_center(evidence, ego)
    overview = _render_bev(
        prior=prior,
        target=target,
        evidence=evidence,
        center=ego,
        radius=overview_radius,
        size=map_size,
        include_prior=True,
        include_target=True,
        include_proposals=True,
    )
    crop = _render_bev(
        prior=prior,
        target=target,
        evidence=evidence,
        center=focus,
        radius=crop_radius,
        size=map_size,
        include_prior=True,
        include_target=True,
        include_proposals=True,
    )
    width = cameras.width + 2 * map_size
    header_height, receipt_height = 30, 94
    board = Image.new("RGB", (width, header_height + cameras.height + receipt_height), "white")
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, width, header_height), fill=(24, 40, 56))
    draw.text(
        (10, 9),
        f"ArgoTweak validation | {episode.segment_id} | frozen native typed-edit adapter",
        fill="white",
        font=ImageFont.load_default(),
    )
    board.paste(cameras, (0, header_height))
    board.paste(_labeled_tile(overview, "Ego-centered vector map", (map_size, map_size)), (cameras.width, header_height))
    board.paste(_labeled_tile(crop, "Proposal-centered local map", (map_size, map_size)), (cameras.width + map_size, header_height))
    board.paste(_receipt(evidence, width=width, height=receipt_height), (0, header_height + cameras.height))
    return board


def render(
    episodes_path: Path,
    output_dir: Path,
    *,
    camera_width: int,
    camera_height: int,
    map_size: int,
    overview_radius: float,
    crop_radius: float,
    asset_mode: str = "board",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if asset_mode not in {"board", "individual", "both"}:
        raise ValueError("asset_mode must be board, individual, or both")
    episodes = _read_episodes(episodes_path)
    cases_root = output_dir / "cases"
    cases_root.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    operation_totals: Counter[str] = Counter()
    for number, episode in enumerate(episodes, start=1):
        prior = _load_feature_collection(Path(episode.prior_map_path))
        target = _load_feature_collection(Path(episode.target_map_path))
        for evidence_index, evidence in enumerate(sorted(episode.evidence, key=lambda row: row.evidence_id), start=1):
            case_id = f"{number:03d}_{episode.segment_id}_{evidence_index:04d}_{evidence.timestamp}"
            case_dir = cases_root / case_id
            case_dir.mkdir()
            assets = {}
            if asset_mode in {"board", "both"}:
                _board(
                    episode,
                    evidence,
                    prior,
                    target,
                    camera_size=(camera_width, camera_height),
                    map_size=map_size,
                    overview_radius=overview_radius,
                    crop_radius=crop_radius,
                ).save(case_dir / "overview.png", optimize=True)
            if asset_mode in {"individual", "both"}:
                assets = _individual_assets(
                    case_dir,
                    evidence=evidence,
                    prior=prior,
                    target=target,
                    map_size=map_size,
                    overview_radius=overview_radius,
                    crop_radius=crop_radius,
                )
            proposal_counts = Counter(str(row.get("operation", "KEEP")).upper() for row in evidence.proposals)
            operation_totals.update(proposal_counts)
            bundle = json.loads(Path(evidence.camera_bundle_path).read_text(encoding="utf-8"))
            row = {
                "schema_version": "argotweak-full-validation-casebook-v1",
                "dataset": "ArgoTweak / TbV",
                "split": "val",
                "test_assets_read": False,
                "case_id": case_id,
                "segment_id": episode.segment_id,
                "episode_id": episode.episode_id,
                "timestamp": evidence.timestamp,
                "evidence_id": evidence.evidence_id,
                "folder": str(case_dir.relative_to(output_dir)),
                "operation_counts": dict(sorted(proposal_counts.items())),
                "commit_ready_count": len(evidence.commit_ready_proposal_ids),
                "blocked_count": len(evidence.blocked_proposals),
                "blocked_reason_counts": dict(sorted(Counter(evidence.blocked_proposals.values()).items())),
                "operation_belief": list(evidence.operation_belief),
                "target_operation_counts": dict(evidence.target_operation_counts),
                "assets": assets,
                "sources": {
                    "native_episode_catalog": str(episodes_path),
                    "camera_bundle": str(evidence.camera_bundle_path),
                    "camera_images": dict(bundle.get("images") or {}),
                    "prior_map": episode.prior_map_path,
                    "target_map_offline_reference": episode.target_map_path,
                },
                "selection": "none; exhaustive validation evidence export",
                "reference_map_role": "offline visual evaluation only; never an online controller input",
            }
            (case_dir / "manifest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            index_rows.append(row)
    with (output_dir / "casebook_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "argotweak-full-validation-casebook-v1",
        "dataset": "ArgoTweak / TbV",
        "split": "val",
        "test_assets_read": False,
        "episode_count": len(episodes),
        "evidence_count": len(index_rows),
        "proposal_operation_counts": dict(sorted(operation_totals.items())),
        "artifacts_per_case": (
            ["overview.png", "manifest.json"]
            if asset_mode == "board"
            else ["camera/*.jpg", "layers/*.png", "manifest.json"]
            if asset_mode == "individual"
            else ["overview.png", "camera/*.jpg", "layers/*.png", "manifest.json"]
        ),
        "asset_mode": asset_mode,
        "selection": "none; every validation evidence bundle is rendered",
        "ordering": "segment id then evidence id",
        "source_episodes": str(episodes_path),
        "frozen_perception": True,
        "scope": "native HD-map typed-edit interface visualization; not a perception-adaptation claim",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=144)
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--overview-radius", type=float, default=90.0)
    parser.add_argument("--crop-radius", type=float, default=35.0)
    parser.add_argument("--asset-mode", choices=("board", "individual", "both"), default="board")
    args = parser.parse_args()
    if min(args.camera_width, args.camera_height, args.map_size) < 64:
        raise ValueError("render tile sizes must be at least 64 pixels")
    if min(args.overview_radius, args.crop_radius) <= 0.0:
        raise ValueError("map radii must be positive")
    print(
        json.dumps(
            render(
                args.episodes,
                args.output_dir,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                map_size=args.map_size,
                overview_radius=args.overview_radius,
                crop_radius=args.crop_radius,
                asset_mode=args.asset_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

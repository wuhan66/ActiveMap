#!/usr/bin/env python3
"""Render validation-only SpaceNet8 multi-POST selection and Safe Commit cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

GREEN = np.array([53, 160, 97], dtype=np.uint8)
BLUE = np.array([52, 116, 201], dtype=np.uint8)
RED = np.array([217, 86, 72], dtype=np.uint8)
LIGHT_BLUE = np.array([112, 184, 222], dtype=np.uint8)
WHITE = np.array([250, 250, 248], dtype=np.uint8)
FEATURES = (
    "bounds_iou",
    "valid_fraction",
    "mean_probability",
    "mean_entropy",
    "predicted_fraction",
)


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row[key]) for key in FEATURES] for row in rows], dtype=np.float64
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _assert_validation(row: dict[str, Any], path: Path) -> None:
    if row.get("split") != "val" or row.get("test_assets_read") is not False:
        raise ValueError(f"not a validation-only record: {path}")


def _assert_non_test(row: dict[str, Any], path: Path) -> None:
    if row.get("split") not in {"train", "val"} or row.get("test_assets_read") is not False:
        raise ValueError(f"record is not a train/validation artifact: {path}")


def _rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[0] in {3, 4}:
        array = np.moveaxis(array[:3], 0, -1)
    finite = array[np.isfinite(array)]
    low, high = np.quantile(finite, (0.01, 0.99)) if finite.size else (0.0, 1.0)
    scaled = (array - low) / max(float(high - low), 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    inner = mask.copy()
    for y, x in ((0, 1), (2, 1), (1, 0), (1, 2)):
        inner &= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
    return mask & ~inner


def _boundary(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    image = np.broadcast_to(WHITE, (*mask.shape, 3)).copy()
    image[_edge(mask)] = color
    return image


def _residual(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    image = np.broadcast_to(WHITE, (*prediction.shape, 3)).copy()
    image[_edge(prediction & reference)] = GREEN
    image[_edge(prediction & ~reference)] = RED
    image[_edge(reference & ~prediction)] = LIGHT_BLUE
    return image


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def _bounds(mask: np.ndarray, side: int) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    center_x = int(round(float(xs.mean()))) if len(xs) else width // 2
    center_y = int(round(float(ys.mean()))) if len(ys) else height // 2
    side = min(max(1, side), height, width)
    left = max(0, min(center_x - side // 2, width - side))
    top = max(0, min(center_y - side // 2, height - side))
    return left, top, left + side, top + side


def _load_seed_rows(
    candidate_root: Path,
    seed_pattern: str = "rank100_changer_seed*/per_candidate.jsonl",
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    matched: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(candidate_root.glob(seed_pattern))
    if len(paths) != 3:
        raise ValueError(f"expected exactly 3 seed candidate files beneath {candidate_root}")
    for path in paths:
        for row in _read_jsonl(path):
            _assert_non_test(row, path)
            matched[(str(row["sample_id"]), str(row["candidate_id"]))].append(row)
    if any(len(rows) != 3 for rows in matched.values()):
        raise ValueError("candidate seed support is incomplete")
    return matched


def _safe_gate(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [row for row in rows if row["split"] == "train"]
    ranker = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03, random_state=20260802
    ).fit(feature_matrix(train), [row["map_iou"] for row in train])
    for row, score in zip(rows, ranker.predict(feature_matrix(rows))):
        row["recomputed_ranker_score"] = float(score)
    beneficial = np.asarray(
        [row["map_iou"] > (0.0 if row["target_positive"] else 1.0) + 1e-9 for row in train],
        dtype=np.int64,
    )
    gate = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=20260802))
    gate.fit(feature_matrix(train), beneficial)
    return {
        str(id(row)): float(probability)
        for row, probability in zip(rows, gate.predict_proba(feature_matrix(rows))[:, 1])
    }


def render(candidate_root: Path, selector_root: Path, output_dir: Path, *, zoom_size: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    aggregate_path = selector_root / "aggregated_candidates.jsonl"
    summary_path = selector_root / "summary.json"
    aggregate = _read_jsonl(aggregate_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["protocol"].get("test_assets_read") is not False:
        raise ValueError("selector summary is not validation-only")
    threshold = float(summary["protocol"]["safe_commit_threshold"])
    seed_rows = _load_seed_rows(candidate_root)
    all_rows = [dict(row) for row in aggregate]
    for row in all_rows:
        _assert_non_test(row, aggregate_path)
    probabilities = _safe_gate(all_rows, threshold)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[str(row["sample_id"])].append(row)
    candidates: list[tuple[float, str, list[dict[str, Any]], dict[str, Any], bool]] = []
    for sample_id, group in groups.items():
        group.sort(key=lambda row: int(row["candidate_index"]))
        if len(group) < 2:
            continue
        selected = max(group, key=lambda row: float(row["ranker_score"]))
        safe_commit = probabilities[str(id(selected))] >= threshold
        first = group[0]
        improvement = float(selected["map_iou"]) - float(first["map_iou"])
        if bool(selected["target_positive"]):
            candidates.append((improvement, sample_id, group, selected, safe_commit))
    if not candidates:
        raise ValueError("no positive validation multi-POST case exists")
    # The qualitative selection must demonstrate the active-selection mechanism,
    # not merely a correct Safe Commit on the first available POST observation.
    candidates = [item for item in candidates if item[0] > 1e-12]
    if not candidates:
        raise ValueError("no validation case has positive learned-minus-first gain")
    candidates.sort(key=lambda item: (item[0], item[4]), reverse=True)
    chosen = candidates[0]
    _, sample_id, group, selected, safe_commit = chosen
    first = group[0]
    selected_key = (sample_id, str(selected["candidate_id"]))
    first_key = (sample_id, str(first["candidate_id"]))
    selected_seed_rows = seed_rows[selected_key]
    first_seed_rows = seed_rows[first_key]
    with np.load(str(selected["array_path"])) as payload:
        pre = _rgb(payload["pre"])
        post = _rgb(payload["post"])
        target = np.asarray(payload["target"], dtype=bool)
        valid = np.asarray(payload["valid"], dtype=bool)
    with np.load(str(first["array_path"])) as payload:
        first_post = _rgb(payload["post"])
    def ensemble_mask(rows: list[dict[str, Any]]) -> np.ndarray:
        probability = np.mean(
            [np.asarray(np.load(str(row["mask_path"]))["probability"], dtype=np.float32) for row in rows],
            axis=0,
        )
        return (probability >= 0.5) & valid
    direct_mask = ensemble_mask(first_seed_rows)
    selected_mask = ensemble_mask(selected_seed_rows)
    safe_mask = selected_mask if safe_commit else np.zeros_like(selected_mask)
    output_dir.mkdir(parents=True)
    case_dir = output_dir / "01_rank100_changer_validation_case"
    case_dir.mkdir()
    panels = {
        "pre_event_rgb.png": pre,
        "first_post_rgb.png": first_post,
        "selected_post_rgb.png": post,
        "direct_change_draft.png": _boundary(direct_mask, BLUE),
        "selected_change_draft.png": _boundary(selected_mask, BLUE),
        "safe_commit_writeback.png": _boundary(safe_mask, BLUE),
        "reference_change.png": _boundary(target, GREEN),
        "residual_direct.png": _residual(direct_mask, target),
        "residual_safe_commit.png": _residual(safe_mask, target),
    }
    for name, image in panels.items():
        _save(image, case_dir / name)
    residual_masks = [np.logical_xor(direct_mask, target), np.logical_xor(safe_mask, target)]
    zooms = []
    for index, mask in enumerate(residual_masks, start=1):
        left, top, right, bottom = _bounds(mask, zoom_size)
        zoom_dir = case_dir / f"zoom{index:02d}"
        zoom_dir.mkdir()
        for name, image in panels.items():
            _save(image[top:bottom, left:right], zoom_dir / name)
        zooms.append({"directory": zoom_dir.name, "bounds_xyxy": [left, top, right, bottom]})
    manifest = {
        "schema_version": "activemap-spacenet8-map-native-qualitative-v1",
        "dataset": "SpaceNet8",
        "split": "val",
        "test_assets_read": False,
        "case_id": sample_id,
        "backend": "Changer rank-100, three-seed mean probability",
        "coordinate_reference": "shared frozen 512x512 candidate array grid",
        "selection": {
            "first_candidate_id": first["candidate_id"],
            "learned_selected_candidate_id": selected["candidate_id"],
            "first_map_iou": float(first["map_iou"]),
            "selected_map_iou": float(selected["map_iou"]),
            "learned_minus_first": float(selected["map_iou"]) - float(first["map_iou"]),
        },
        "safe_commit": {"threshold": threshold, "commit": bool(safe_commit)},
        "tool_call_count": 1,
        "evidence_exported": True,
        "selection_rule": "frozen rank-100 Changer validation ranking; highest learned-minus-first map-IoU validation case, retaining its frozen Safe Commit outcome",
        "sources": {
            "aggregate": {"path": str(aggregate_path), "sha256": _sha256(aggregate_path)},
            "selector_summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "candidate_root": str(candidate_root),
            "selected_array": str(selected["array_path"]),
            "first_array": str(first["array_path"]),
            "selected_seed_masks": [str(row["mask_path"]) for row in selected_seed_rows],
            "first_seed_masks": [str(row["mask_path"]) for row in first_seed_rows],
        },
        "zooms": zooms,
    }
    (case_dir / "trace.json").write_text(
        json.dumps({"first": first, "selected": selected, "candidate_group": group}, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {"schema_version": "spacenet8-map-native-qualitative-v1", "split": "val", "test_assets_read": False, "case_count": 1, "case": {"folder": case_dir.name, **manifest}}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("selector_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--zoom-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(render(args.candidate_root, args.selector_root, args.output_dir, zoom_size=args.zoom_size), indent=2))


if __name__ == "__main__":
    main()

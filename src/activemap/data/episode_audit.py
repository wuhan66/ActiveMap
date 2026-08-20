"""Streaming structural and provenance audit for evidence episodes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EditOperation, EpisodeRecord


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def audit_episode_dataset(
    episodes_path: Path,
    *,
    expected_derivation_version: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    episode_ids: set[str] = set()
    duplicate_ids: list[str] = []
    aoi_splits: defaultdict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    evidence_counts: list[float] = []
    path_exists: dict[str, bool] = {}
    episode_count = 0

    def exists(path_value: str) -> bool:
        if path_value not in path_exists:
            path_exists[path_value] = Path(path_value).is_file()
        return path_exists[path_value]

    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = EpisodeRecord.model_validate_json(line)
            except Exception as exc:
                errors.append(f"line {line_number}: invalid episode: {exc}")
                continue
            episode_count += 1
            if episode.episode_id in episode_ids:
                duplicate_ids.append(episode.episode_id)
            episode_ids.add(episode.episode_id)
            aoi_key = episode.aoi_id or episode.episode_id
            aoi_splits[aoi_key].add(episode.split)
            split_counts[episode.split] += 1
            operation_counts[episode.gt_edit.op.value] += 1
            version_counts[episode.derivation_version] += 1
            evidence_counts.append(float(len(episode.evidence_catalog)))

            prefix = episode.episode_id
            if episode.aoi_id is None or episode.anchor_timestamp is None:
                errors.append(f"{prefix}: SN7 episode is missing AOI or anchor timestamp")
            if not episode.evidence_catalog:
                errors.append(f"{prefix}: evidence catalog is empty")
            elif episode.anchor_timestamp not in {
                item.timestamp for item in episode.evidence_catalog
            }:
                errors.append(f"{prefix}: anchor timestamp is absent from evidence catalog")
            if not exists(episode.map_before) or not exists(episode.target_map):
                errors.append(f"{prefix}: source or target map path is missing")
            for item in episode.evidence_catalog:
                if not exists(item.image_path):
                    errors.append(f"{prefix}: missing evidence image {item.image_path}")
                    break
                if item.udm_path is not None and not exists(item.udm_path):
                    errors.append(f"{prefix}: missing evidence UDM {item.udm_path}")
                    break

            if episode.hypothesis.op != episode.gt_edit.op:
                errors.append(f"{prefix}: hypothesis and ground-truth edit types differ")
            requires_prior = episode.gt_edit.op in {
                EditOperation.KEEP,
                EditOperation.DELETE,
                EditOperation.RESHAPE,
            }
            requires_target = episode.gt_edit.op in {
                EditOperation.KEEP,
                EditOperation.ADD,
                EditOperation.RESHAPE,
            }
            if requires_prior != (episode.prior_geometry is not None):
                errors.append(f"{prefix}: prior geometry contract is violated")
            if requires_target != (episode.target_geometry is not None):
                errors.append(f"{prefix}: target geometry contract is violated")
            if len(errors) >= 100:
                break

    if duplicate_ids:
        errors.append(f"duplicate episode IDs: {sorted(set(duplicate_ids))[:10]}")
    leaked = {aoi: sorted(splits) for aoi, splits in aoi_splits.items() if len(splits) > 1}
    if leaked:
        errors.append(f"AOIs occur in multiple splits: {leaked}")
    if expected_derivation_version is not None and set(version_counts) != {
        expected_derivation_version
    }:
        errors.append(
            "unexpected derivation versions: "
            f"expected {expected_derivation_version!r}, got {dict(version_counts)}"
        )

    return {
        "source": str(episodes_path.resolve()),
        "episode_count": episode_count,
        "aoi_count": len(aoi_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "derivation_versions": dict(sorted(version_counts.items())),
        "evidence_per_episode": _quantiles(evidence_counts) if evidence_counts else None,
        "unique_paths_checked": len(path_exists),
        "duplicate_episode_count": len(duplicate_ids),
        "error_count": len(errors),
        "errors": errors[:100],
        "passed": not errors,
    }


def write_episode_audit(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

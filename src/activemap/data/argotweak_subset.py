"""Deterministic, label-support-aware ArgoTweak pilot subset selection."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SECTION_NAMES = {
    "laneSegments": "lane_segment",
    "pedCrossings": "pedestrian_crossing",
    "drivableAreas": "drivable_area",
}
OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
PILOT_SUPPORT_KEYS = tuple(
    f"{feature_class}:{operation}"
    for feature_class in ("lane_segment", "pedestrian_crossing")
    for operation in ("ADD", "DELETE", "RESHAPE")
)


def _operation(change: dict[str, Any]) -> str:
    unchanged = change.get("changes") == [0]
    current = change.get("old")
    stale = current if unchanged else change.get("new")
    if stale is None and current is not None:
        return "ADD"
    if stale is not None and current is None:
        return "DELETE"
    if stale is not None and current is not None:
        return "KEEP" if unchanged else "RESHAPE"
    raise ValueError("ArgoTweak change contains neither stale nor current object")


def audit_argotweak_annotation(path: Path, *, split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    native_codes: Counter[str] = Counter()
    for section, feature_class in SECTION_NAMES.items():
        objects = payload.get(section, {})
        if not isinstance(objects, dict):
            raise ValueError(f"ArgoTweak {section} must contain an object: {path}")
        for change in objects.values():
            if not isinstance(change, dict):
                raise ValueError(f"invalid ArgoTweak change in {path}")
            counts[f"{feature_class}:{_operation(change)}"] += 1
            for code in change.get("changes") or []:
                native_codes[str(code)] += 1
    return {
        "split": split,
        "segment_id": path.stem,
        "annotation_path": str(path.resolve()),
        "counts": dict(sorted(counts.items())),
        "native_change_codes": dict(sorted(native_codes.items())),
    }


def _stable_rank(split: str, segment_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{segment_id}".encode()).hexdigest()


def _select_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    count: int,
    minimum_logs_per_key: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count > len(rows):
        raise ValueError(f"requested {count} {split} logs from only {len(rows)}")
    availability = {
        key: sum(int(row["counts"].get(key, 0) > 0) for row in rows)
        for key in PILOT_SUPPORT_KEYS
    }
    targets = {
        key: min(minimum_logs_per_key, available)
        for key, available in availability.items()
        if available > 0
    }
    selected: list[dict[str, Any]] = []
    covered: Counter[str] = Counter()
    remaining = list(rows)
    while len(selected) < count and any(covered[key] < target for key, target in targets.items()):
        def score(row: dict[str, Any]) -> tuple[int, str]:
            gain = sum(
                int(covered[key] < target and row["counts"].get(key, 0) > 0)
                for key, target in targets.items()
            )
            return gain, _stable_rank(split, row["segment_id"], seed)

        best = min(remaining, key=lambda row: (-score(row)[0], score(row)[1]))
        if score(best)[0] == 0:
            break
        selected.append(best)
        remaining.remove(best)
        for key in targets:
            covered[key] += int(best["counts"].get(key, 0) > 0)
    remaining.sort(key=lambda row: _stable_rank(split, row["segment_id"], seed))
    selected.extend(remaining[: count - len(selected)])
    selected.sort(key=lambda row: row["segment_id"])
    return selected, targets


def build_argotweak_balanced_subset(
    splits_path: Path,
    annotation_root: Path,
    output_dir: Path,
    *,
    train_count: int = 24,
    val_count: int = 8,
    train_minimum_logs_per_key: int = 3,
    val_minimum_logs_per_key: int = 2,
    seed: str = "activemap-argotweak-pilot-v1",
) -> dict[str, Any]:
    """Audit all train/val labels and freeze a support-aware pilot subset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "annotation_support.csv",
        output_dir / "selected_train24_val8.json",
        output_dir / "selected_dependencies.jsonl",
        output_dir / "selection_summary.json",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite ArgoTweak subset audit: {output_dir}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_rows = splits.get(split)
        if not isinstance(split_rows, dict):
            raise ValueError(f"official ArgoTweak split {split!r} is missing")
        for segment_id in split_rows.values():
            path = annotation_root / f"{segment_id}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(audit_argotweak_annotation(path, split=split))

    fieldnames = ["split", "segment_id", *(
        f"{feature_class}:{operation}"
        for feature_class in SECTION_NAMES.values()
        for operation in OPERATIONS
    )]
    with outputs[0].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "split": row["split"],
                "segment_id": row["segment_id"],
                **{key: row["counts"].get(key, 0) for key in fieldnames[2:]},
            })

    selected: dict[str, list[dict[str, Any]]] = {}
    targets: dict[str, dict[str, int]] = {}
    for split, count, minimum in (
        ("train", train_count, train_minimum_logs_per_key),
        ("val", val_count, val_minimum_logs_per_key),
    ):
        selected[split], targets[split] = _select_split(
            [row for row in rows if row["split"] == split],
            split=split,
            count=count,
            minimum_logs_per_key=minimum,
            seed=seed,
        )

    frozen = {
        "schema_version": "activemap-argotweak-balanced-pilot-v1",
        "selection_policy": "label-support coverage then fixed SHA256 rank; no model outputs",
        "seed": seed,
        "test_assets_read": False,
        "splits": {
            split: {f"{index:05d}": row["segment_id"] for index, row in enumerate(split_rows)}
            for split, split_rows in selected.items()
        },
    }
    outputs[1].write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    dependency_rows = []
    for split, split_rows in selected.items():
        for row in split_rows:
            segment_id = row["segment_id"]
            dependency_rows.append({
                "schema_version": "activemap-argotweak-selected-dependency-v1",
                "split": split,
                "segment_id": segment_id,
                "annotation_path": row["annotation_path"],
                "s3_prefix": f"s3://argoverse/datasets/av2/tbv/{segment_id}/",
                "support": row["counts"],
                "test_assets_read": False,
            })
    outputs[2].write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in dependency_rows),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "schema_version": "activemap-argotweak-balanced-pilot-summary-v1",
        "audited_logs": len(rows),
        "selected_logs": {split: len(split_rows) for split, split_rows in selected.items()},
        "support_targets_in_logs": targets,
        "selected_support": {},
        "test_assets_read": False,
    }
    for split, split_rows in selected.items():
        object_counts = Counter()
        log_counts = Counter()
        for row in split_rows:
            object_counts.update(row["counts"])
            log_counts.update(key for key, value in row["counts"].items() if value > 0)
        summary["selected_support"][split] = {
            "object_counts": dict(sorted(object_counts.items())),
            "log_counts": dict(sorted(log_counts.items())),
        }
    outputs[3].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

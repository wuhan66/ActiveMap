#!/usr/bin/env python3
"""Export real evidence thumbnails for the fixed SN7 V5 qualitative cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from export_trace_evidence_crops import export_trace_evidence_crops


OPERATIONS = ("ADD", "DELETE", "RESHAPE")
QUALITATIVE_SCHEMA = "sn7-v5-predeclared-qualitative-manifest-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validation_only(payload: dict[str, Any], *, name: str) -> None:
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"{name} must be validation-only and test-free")


def rollout_index(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        validation_only(row, name=f"selected rollout in {path}")
        task_id = str(row.get("task_id", ""))
        budget = float(row.get("budget", 0.0))
        key = (task_id, budget)
        if not task_id or budget <= 0.0 or key in rows:
            raise ValueError(f"invalid or duplicate selected rollout in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"empty selected rollout file: {path}")
    return rows


def parse_root_map(value: str | None) -> tuple[Path, Path] | None:
    if value is None:
        return None
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise ValueError("asset root map must be SOURCE=DESTINATION")
    return Path(source), Path(destination)


def export_v5_qualitative_evidence(
    *,
    qualitative_manifest: Path,
    episodes_path: Path,
    selected_rollouts_path: Path,
    output_dir: Path,
    image_size: int = 256,
    asset_root_map: tuple[Path, Path] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V5 evidence output: {output_dir}")
    manifest = read_json(qualitative_manifest)
    if manifest.get("schema_version") != QUALITATIVE_SCHEMA:
        raise ValueError("unexpected V5 qualitative manifest schema")
    validation_only(manifest, name="V5 qualitative manifest")
    if manifest.get("controller_or_writeback_outputs_read") is not False:
        raise ValueError("V5 qualitative cases were not predeclared before controller output")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 9:
        raise ValueError("V5 qualitative manifest must contain nine fixed cases")
    rollouts = rollout_index(selected_rollouts_path)
    output_dir.mkdir(parents=True)
    seen_sources: set[str] = set()
    entries = []
    operation_counts = {operation: 0 for operation in OPERATIONS}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("V5 qualitative case is not an object")
        operation = str(case.get("operation", ""))
        task_id = str(case.get("task_id", ""))
        source_episode = str(case.get("source_episode", ""))
        budget = float(case.get("budget", 0.0))
        if operation not in operation_counts or not task_id or not source_episode or budget <= 0.0:
            raise ValueError("V5 qualitative case lacks registered identity")
        if source_episode in seen_sources:
            raise ValueError("V5 evidence export requires unique source episodes")
        seen_sources.add(source_episode)
        operation_counts[operation] += 1
        key = (task_id, budget)
        rollout = rollouts.get(key)
        if rollout is None:
            raise ValueError(f"selected rollout lacks fixed V5 case {key}")
        if rollout.get("source_episode") != source_episode:
            raise ValueError(f"selected rollout source mismatch for fixed V5 case {key}")
        extra_id = rollout.get("selected_extra_evidence_id")
        entry: dict[str, Any] = {
            "operation": operation,
            "task_id": task_id,
            "source_episode": source_episode,
            "budget": budget,
            "selected_extra_evidence": bool(extra_id),
            "selected_extra_evidence_id": extra_id,
        }
        if extra_id:
            case_output = output_dir / source_episode
            crop_manifest = export_trace_evidence_crops(
                episodes_path=episodes_path,
                traces_path=selected_rollouts_path,
                source_episode=source_episode,
                output_dir=case_output,
                image_size=image_size,
                asset_root_map=asset_root_map,
                task_id=task_id,
                budget=budget,
            )
            selected_ids = {str(item["evidence_id"]) for item in crop_manifest["evidence"]}
            if str(extra_id) not in selected_ids:
                raise ValueError(f"exported evidence does not contain recorded acquisition for {key}")
            entry["evidence_manifest"] = {
                "path": str((case_output / "manifest.json").resolve()),
                "sha256": sha256(case_output / "manifest.json"),
            }
        entries.append(entry)
    if any(count != 3 for count in operation_counts.values()):
        raise ValueError("V5 evidence export lost operation-balanced support")
    receipt = {
        "schema_version": "sn7-v5-fixed-qualitative-evidence-export-v1",
        "split": "val",
        "test_assets_read": False,
        "qualitative_manifest": {
            "path": str(qualitative_manifest.resolve()),
            "sha256": sha256(qualitative_manifest),
        },
        "episodes": {"path": str(episodes_path.resolve()), "sha256": sha256(episodes_path)},
        "selected_rollouts": {
            "path": str(selected_rollouts_path.resolve()),
            "sha256": sha256(selected_rollouts_path),
        },
        "image_size": image_size,
        "asset_root_map": (
            {"source": str(asset_root_map[0]), "destination": str(asset_root_map[1])}
            if asset_root_map is not None
            else None
        ),
        "cases": entries,
    }
    (output_dir / "evidence_export_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualitative_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("selected_rollouts", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--asset-root-map")
    args = parser.parse_args()
    print(
        json.dumps(
            export_v5_qualitative_evidence(
                qualitative_manifest=args.qualitative_manifest,
                episodes_path=args.episodes,
                selected_rollouts_path=args.selected_rollouts,
                output_dir=args.output_dir,
                image_size=args.image_size,
                asset_root_map=parse_root_map(args.asset_root_map),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

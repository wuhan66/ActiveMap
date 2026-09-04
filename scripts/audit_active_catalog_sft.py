#!/usr/bin/env python3
"""Audit portable active-catalog visual SFT before model processing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

FORBIDDEN_PROMPT_KEYS = {
    "gt_edit",
    "oracle_utilities",
    "target_operation",
    "policy_relative_advantage",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_keys(item))
        return result
    return set()


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("message content must be a multimodal list")
    values = [part.get("text") for part in content if part.get("type") == "text"]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("message must contain exactly one text block")
    return values[0]


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("schema_version") != "active-catalog-sequential-selector-sft-v2":
        raise ValueError("active-catalog SFT must include the model-hidden evaluation index")
    if source_summary.get("test_assets_read") is not False:
        raise ValueError("source summary does not prove test isolation")
    if source_summary.get("prompt_target_metadata_exposed") is not False:
        raise ValueError("source summary reports prompt target leakage")
    if source_summary.get("portable_relative_image_paths") is not True:
        raise ValueError("source summary does not declare portable image paths")

    seen_examples: set[str] = set()
    tasks_by_split: dict[str, set[str]] = {"train": set(), "val": set()}
    referenced_images: set[Path] = set()
    counts: Counter[str] = Counter()
    split_files: dict[str, Any] = {}
    evaluation_files: dict[str, Any] = {}
    for split in ("train", "val"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        records = 0
        contracts: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != split or row.get("stage") != "SELECT":
                    raise ValueError(f"split/stage mismatch at {path}:{line_number}")
                messages = row.get("messages")
                if not isinstance(messages, list) or [item.get("role") for item in messages] != [
                    "system",
                    "user",
                    "assistant",
                ]:
                    raise ValueError(f"invalid messages at {path}:{line_number}")
                example_id = str(row.get("example_id", ""))
                if not example_id or example_id in seen_examples:
                    raise ValueError(f"missing or duplicate example_id: {example_id}")
                seen_examples.add(example_id)
                task_id = str(row.get("task_id", ""))
                if not task_id:
                    raise ValueError("missing task_id")
                tasks_by_split[split].add(task_id)

                image_parts = [
                    part
                    for part in messages[1].get("content", [])
                    if part.get("type") == "image"
                ]
                if len(image_parts) != 1:
                    raise ValueError("each prompt must contain exactly one image")
                image_reference = Path(str(image_parts[0].get("image", "")))
                if image_reference.is_absolute():
                    raise ValueError(f"absolute image path is not portable: {image_reference}")
                image_path = (root / image_reference).resolve()
                try:
                    image_path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"image path escapes SFT root: {image_reference}") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                referenced_images.add(image_path)

                state = json.loads(_text(messages[1]))
                leaked = FORBIDDEN_PROMPT_KEYS.intersection(_keys(state))
                if leaked:
                    raise ValueError(f"prompt target leakage at {example_id}: {sorted(leaked)}")
                candidates = state.get("candidate_evidence")
                if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
                    raise ValueError(f"invalid candidate catalog at {example_id}")
                candidate_ids = [str(item["evidence_id"]) for item in candidates]
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError(f"duplicate candidate ids at {example_id}")
                selected_ids = {str(value) for value in state.get("selected_evidence_ids", [])}
                if selected_ids.intersection(candidate_ids):
                    raise ValueError(f"selected evidence remains available at {example_id}")
                remaining = float(state["budget"]["remaining"])
                if any(float(item["cost"]) > remaining + 1e-6 for item in candidates):
                    raise ValueError(f"unaffordable candidate at {example_id}")

                action = json.loads(_text(messages[2]))
                if action.get("stage") != "SELECT" or action.get("selection") not in {
                    "STOP",
                    "ACQUIRE",
                }:
                    raise ValueError(f"invalid SELECT action at {example_id}")
                selection = str(action["selection"])
                if selection == "ACQUIRE":
                    if str(action.get("evidence_id")) not in candidate_ids:
                        raise ValueError(f"ACQUIRE target is not available at {example_id}")
                elif action.get("evidence_id") is not None:
                    raise ValueError(f"STOP carries evidence_id at {example_id}")
                contracts[example_id] = {
                    "target_selection": selection,
                    "target_evidence_id": action.get("evidence_id"),
                    "candidate_ids": set(candidate_ids),
                }
                counts[f"split:{split}"] += 1
                counts[f"split:{split}:action:{selection}"] += 1
                records += 1
        split_files[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "records": records,
            "tasks": len(tasks_by_split[split]),
        }
        evaluation_path = root / f"{split}_evaluation_index.jsonl"
        if not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        evaluation_ids: set[str] = set()
        with evaluation_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                example_id = str(item.get("example_id", ""))
                if example_id in evaluation_ids or example_id not in contracts:
                    raise ValueError(
                        f"evaluation index example mismatch at {evaluation_path}:{line_number}"
                    )
                if (
                    item.get("split") != split
                    or item.get("model_visible") is not False
                    or item.get("test_assets_read") is not False
                ):
                    raise ValueError(f"invalid evaluation boundary at {example_id}")
                contract = contracts[example_id]
                index_candidates = {
                    str(candidate["evidence_id"]) for candidate in item["candidates"]
                }
                if index_candidates != contract["candidate_ids"]:
                    raise ValueError(f"evaluation candidate mismatch at {example_id}")
                if (
                    item.get("target_selection") != contract["target_selection"]
                    or item.get("target_evidence_id")
                    != contract["target_evidence_id"]
                ):
                    raise ValueError(f"evaluation target mismatch at {example_id}")
                evaluation_ids.add(example_id)
        if evaluation_ids != set(contracts):
            raise ValueError(f"incomplete evaluation index for {split}")
        evaluation_digest = _sha256(evaluation_path)
        declared = source_summary.get("evaluation_indices", {}).get(split, {})
        if (
            declared.get("sha256") != evaluation_digest
            or declared.get("model_visible") is not False
        ):
            raise ValueError(f"evaluation index hash/boundary mismatch for {split}")
        evaluation_files[split] = {
            "path": str(evaluation_path),
            "sha256": evaluation_digest,
            "records": len(evaluation_ids),
            "model_visible": False,
        }

    overlap = tasks_by_split["train"].intersection(tasks_by_split["val"])
    if overlap:
        raise ValueError(f"train/validation task overlap: {len(overlap)}")
    disk_images = {path.resolve() for path in (root / "images").glob("*.jpg")}
    if referenced_images != disk_images:
        raise ValueError(
            "image/reference mismatch: "
            f"referenced={len(referenced_images)} disk={len(disk_images)}"
        )
    return {
        "schema_version": "active-catalog-sft-audit-v1",
        "root": str(root),
        "states": len(seen_examples),
        "images": len(referenced_images),
        "counts": dict(sorted(counts.items())),
        "task_overlap": 0,
        "prompt_target_metadata_exposed": False,
        "all_actions_executable": True,
        "all_candidates_affordable": True,
        "portable_relative_image_paths": True,
        "files": split_files,
        "evaluation_indices": evaluation_files,
        "source_summary_sha256": _sha256(summary_path),
        "test_assets_read": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.sft_root)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

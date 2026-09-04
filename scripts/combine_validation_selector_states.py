#!/usr/bin/env python3
"""Validate and aggregate parallel validation-only selector-oracle state shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                split = str(payload["split"])
                episode_id = str(payload["episode_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid episode record in {path} at line {line_number}") from exc
            if split != "val":
                raise ValueError(f"non-validation episode in {path} at line {line_number}")
            if not episode_id or episode_id in identifiers:
                raise ValueError(f"invalid or duplicate episode_id in {path} at line {line_number}")
            identifiers.add(episode_id)
    return identifiers


def combine_validation_states(
    shard_dir: Path,
    output: Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    receipt_path = shard_dir / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "activemap-validation-selector-episode-shards-v1":
        raise ValueError("unexpected episode shard receipt schema")
    if int(receipt.get("shard_count", -1)) != shard_count:
        raise ValueError("shard count does not match episode shard receipt")
    if receipt.get("test_assets_read") is not False:
        raise PermissionError("refusing aggregation without a test-free shard receipt")
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"refusing to overwrite aggregated states: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial aggregated states: {partial}")

    expected_episode_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    seen_source_episode_ids: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as destination:
        for index in range(shard_count):
            episodes_path = shard_dir / f"episodes_val_shard_{index:02d}.jsonl"
            states_path = shard_dir / f"selector_states_val_shard_{index:02d}.jsonl"
            if not episodes_path.is_file() or not states_path.is_file():
                raise FileNotFoundError(f"missing paired shard artifact for index {index}")
            shard_episode_ids = _episode_ids(episodes_path)
            if expected_episode_ids.intersection(shard_episode_ids):
                raise ValueError(f"episode overlap across shard {index}")
            expected_episode_ids.update(shard_episode_ids)
            state_rows = 0
            shard_source_episode_ids: set[str] = set()
            with states_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        sample_id = str(payload["sample_id"])
                        split = str(payload["split"])
                        source_episode = str(payload["metadata"]["source_episode"])
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"invalid selector state in {states_path} at line {line_number}"
                        ) from exc
                    if split != "val":
                        raise ValueError(
                            f"non-validation state in {states_path} at line {line_number}"
                        )
                    if not sample_id or sample_id in seen_sample_ids:
                        raise ValueError(f"duplicate sample_id {sample_id!r}")
                    if source_episode not in shard_episode_ids:
                        raise ValueError(
                            f"state source episode {source_episode!r} is outside shard {index}"
                        )
                    seen_sample_ids.add(sample_id)
                    seen_source_episode_ids.add(source_episode)
                    shard_source_episode_ids.add(source_episode)
                    destination.write(line if line.endswith("\n") else line + "\n")
                    state_rows += 1
            shard_summaries.append(
                {
                    "index": index,
                    "episode_rows": len(shard_episode_ids),
                    "state_rows": state_rows,
                    "state_source_episodes": len(shard_source_episode_ids),
                    "states_path": str(states_path.resolve()),
                    "states_sha256": sha256_file(states_path),
                }
            )
    if not seen_sample_ids:
        partial.unlink(missing_ok=True)
        raise ValueError("no selector states were emitted by any shard")

    os.replace(partial, output)
    summary = {
        "schema": "activemap-validation-selector-state-aggregation-v1",
        "episode_shard_receipt": str(receipt_path.resolve()),
        "episode_shard_receipt_sha256": sha256_file(receipt_path),
        "source_episodes_sha256": receipt["source_episodes_sha256"],
        "shard_count": shard_count,
        "validation_episodes": len(expected_episode_ids),
        "state_source_episodes": len(seen_source_episode_ids),
        "episodes_without_states": len(expected_episode_ids - seen_source_episode_ids),
        "state_rows": len(seen_sample_ids),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "shards": shard_summaries,
        "test_assets_read": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    result = combine_validation_states(args.shard_dir, args.output, shard_count=args.shards)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge selector-state JSONL files with split and identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.selector_records import SelectorSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_state_files(inputs: list[Path], output: Path) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two selector-state inputs are required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite {partial}")
    seen: set[str] = set()
    source_episodes: set[str] = set()
    counts: Counter[str] = Counter()
    sources = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as target:
        for path in inputs:
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = SelectorSample.model_validate_json(line)
                    except Exception as exc:
                        raise ValueError(f"invalid state at {path}:{line_number}") from exc
                    if row.split not in {"train", "val"}:
                        raise ValueError(f"forbidden split: {row.split}")
                    if row.sample_id in seen:
                        raise ValueError(f"duplicate selector state: {row.sample_id}")
                    seen.add(row.sample_id)
                    source_episodes.add(str(row.metadata["source_episode"]))
                    counts[f"split:{row.split}"] += 1
                    counts[
                        "target:STOP"
                        if row.target_index() == len(row.evidence_ids)
                        else "target:ACQUIRE"
                    ] += 1
                    target.write(row.model_dump_json() + "\n")
    partial.replace(output)
    summary = {
        "schema_version": "selector-state-file-merge-v1",
        "state_count": len(seen),
        "source_episode_count": len(source_episodes),
        "counts": dict(sorted(counts.items())),
        "sources": sources,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "test_assets_read": False,
    }
    output.with_suffix(".merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_state_files(args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build validation-only oracle graph files for evaluator self-consistency checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def prepare_self_check(
    annotations_path: Path,
    regions_path: Path,
    graph_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    regions = set(json.loads(regions_path.read_text(encoding="utf-8")))
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = []
    changed_count = 0
    nochange_count = 0
    for index, annotation in enumerate(annotations):
        cluster = annotation["Cluster"]
        if cluster["Region"] not in regions:
            continue
        is_nochange = "nochange" in annotation.get("Tags", [])
        tile_x, tile_y = cluster["Tile"]
        source = graph_dir / (
            f"{cluster['Region']}_{tile_x}_{tile_y}_2020-07-01.graph"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / f"{index}.graph"
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
            materialization = "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            materialization = "copy"
        mappings.append(
            {
                "annotation_index": index,
                "category": "nochange" if is_nochange else "changed",
                "source": str(source.resolve()),
                "destination": str(destination.resolve()),
                "materialization": materialization,
            }
        )
        if is_nochange:
            nochange_count += 1
        else:
            changed_count += 1
    if not mappings:
        raise ValueError("validation region selection produced no graph mappings")
    summary = {
        "schema_version": "muno21-graph-metric-self-check-v1",
        "regions": sorted(regions),
        "graph_count": len(mappings),
        "changed_graph_count": changed_count,
        "nochange_graph_count": nochange_count,
        "purpose": "evaluator_self_consistency_only",
        "model_result": False,
        "test_assets_read": False,
        "mappings": mappings,
    }
    (output_dir / "self_check_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("regions", type=Path)
    parser.add_argument("graph_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = prepare_self_check(
        args.annotations, args.regions, args.graph_dir, args.output_dir
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

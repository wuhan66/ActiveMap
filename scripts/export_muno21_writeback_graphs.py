#!/usr/bin/env python3
"""Export retained ActiveMap writeback masks to official MUNO21 .graph files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from activemap.evaluation.road_graph import MunoRoadGraph

EPISODE_PATTERN = re.compile(r"^task-[0-9a-f]{16}$")
RAW_EPISODE_PATTERN = re.compile(r"muno21-(?P<index>\d+)-p(?P<patch>\d+)-")


def _episode_index_lookup(
    episodes_path: Path, *, expected_split: str
) -> dict[str, tuple[int, int]]:
    from activemap.agent.identifiers import public_task_id
    from activemap.models import EpisodeRecord

    output = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        episode = EpisodeRecord.model_validate_json(line)
        if episode.split not in {"train", "val", "test"}:
            raise ValueError(f"unknown episode split: {episode.split!r}")
        if episode.split != expected_split:
            continue
        match = RAW_EPISODE_PATTERN.search(episode.episode_id)
        if match is None:
            raise ValueError(f"cannot recover MUNO21 annotation index: {episode.episode_id}")
        output[public_task_id(episode.episode_id)] = (
            int(match.group("index")),
            int(match.group("patch")),
        )
    if not output:
        raise ValueError(f"no MUNO21 episodes found for split {expected_split!r}")
    return output


def export_graphs(
    rows: list[dict[str, Any]],
    episode_lookup: dict[str, tuple[int, int]],
    output_dir: Path,
    *,
    simplify_tolerance: float,
    test_assets_read: bool = False,
) -> dict[str, Any]:
    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError("graph export requires the graph-eval optional dependency") from exc

    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row["task_id"])
        if not EPISODE_PATTERN.fullmatch(task_id) or task_id not in episode_lookup:
            raise ValueError(f"unknown opaque MUNO21 task id: {task_id}")
        annotation_index, patch_index = episode_lookup[task_id]
        grouped[(float(row["budget"]), annotation_index)].append(
            {**row, "patch_index": patch_index}
        )

    graphs = 0
    patch_counts = []
    for (budget, annotation_index), selected in sorted(grouped.items()):
        graph = MunoRoadGraph()
        for row in sorted(selected, key=lambda item: item["patch_index"]):
            artifact = Path(str(row["mask_artifact"]))
            if not artifact.is_file():
                raise FileNotFoundError(f"missing writeback mask artifact: {artifact}")
            with np.load(artifact) as payload:
                committed = np.asarray(payload["committed_mask"]) >= 0.5
                transform = Affine(*np.asarray(payload["transform"]).tolist())
            graph.add_skeleton(
                skeletonize(committed),
                transform,
                simplify_tolerance=simplify_tolerance,
            )
        suffix = str(budget).replace(".", "p")
        graph.save(output_dir / f"budget-{suffix}" / f"{annotation_index}.graph")
        graphs += 1
        patch_counts.append(len(selected))
    return {
        "protocol": "clean-room-mask-to-muno21-graph-v1",
        "simplify_tolerance": simplify_tolerance,
        "graph_count": graphs,
        "annotation_count": len({key[1] for key in grouped}),
        "budgets": sorted({key[0] for key in grouped}),
        "maximum_patches_per_graph": max(patch_counts, default=0),
        "bidirectional_edges": True,
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback_jsonl", type=Path)
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--simplify-tolerance", type=float, default=1.0)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        if not args.frozen_test:
            raise PermissionError("test graph export requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen test graph export: {args.output_dir}"
            )
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    rows = [
        json.loads(line)
        for line in args.writeback_jsonl.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = export_graphs(
        rows,
        _episode_index_lookup(args.episodes_jsonl, expected_split=args.split),
        args.output_dir,
        simplify_tolerance=args.simplify_tolerance,
        test_assets_read=args.split == "test",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

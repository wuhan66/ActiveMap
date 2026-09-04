from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.data.argotweak import (
    build_argotweak_segment_episode,
    convert_argotweak_annotation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a frozen ArgoTweak subset into portable map scenes."
    )
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    subset = json.loads(args.subset_file.read_text(encoding="utf-8"))
    if subset.get("test_assets_read") is not False:
        raise ValueError("frozen ArgoTweak subset must explicitly lock test assets")
    combined: dict[str, list[str]] = {"train": [], "val": []}
    summaries = []
    for split in ("train", "val"):
        for segment_id in subset["splits"][split].values():
            maps = args.output_root / "maps" / split / segment_id
            prior = maps / "prior.geojson"
            target = maps / "target.geojson"
            if not (prior.is_file() and target.is_file()):
                convert_argotweak_annotation(
                    args.annotation_root / f"{segment_id}.json",
                    maps,
                )
            scene, summary = build_argotweak_segment_episode(
                args.segment_root / segment_id,
                prior,
                target,
                args.output_root / "bundles" / split / segment_id,
                split=split,
                stride=args.stride,
                bundle_cost=1.0,
            )
            combined[split].append(scene.model_dump_json())
            summaries.append(summary)

    for split, lines in combined.items():
        destination = args.output_root / f"{split}_scenes.jsonl"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite combined scenes: {destination}")
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "schema_version": "activemap-argotweak-balanced-scenes-v1",
        "segments": len(summaries),
        "split_episodes": {
            split: len(lines) for split, lines in combined.items()
        },
        "total_evidence_bundles": sum(
            summary["evidence_bundle_count"] for summary in summaries
        ),
        "cameras_per_bundle": 7,
        "evidence_unit": "synchronized_seven_camera_frame_bundle",
        "stride": args.stride,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()

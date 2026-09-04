from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ArgoTweak proposal-to-episode alignment.")
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposals: dict[tuple[str, str], set[str]] = defaultdict(set)
    proposal_files = sorted(args.proposal_root.glob("*/atomic_edit_proposals.jsonl"))
    for path in proposal_files:
        for row in _jsonl(path):
            proposals[(row["split"], row["segment_id"])].add(str(row["timestamp"]))

    episodes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "val"):
        for scene in _jsonl(args.episode_root / f"{split}_scenes.jsonl"):
            segment_id = scene["aoi_id"]
            episodes[(split, segment_id)].update(
                str(observation["timestamp"]) for observation in scene["observations"]
            )

    keys = sorted(set(proposals) | set(episodes))
    rows = []
    for split, segment_id in keys:
        proposal_timestamps = proposals[(split, segment_id)]
        episode_timestamps = episodes[(split, segment_id)]
        rows.append(
            {
                "split": split,
                "segment_id": segment_id,
                "proposal_frames": len(proposal_timestamps),
                "evidence_bundles": len(episode_timestamps),
                "missing_episode_timestamps": sorted(proposal_timestamps - episode_timestamps),
                "extra_episode_timestamps": sorted(episode_timestamps - proposal_timestamps),
            }
        )

    mismatches = [
        row
        for row in rows
        if row["missing_episode_timestamps"] or row["extra_episode_timestamps"]
    ]
    summary = {
        "schema_version": "activemap-argotweak-episode-alignment-v1",
        "segments": len(rows),
        "proposal_files": len(proposal_files),
        "proposal_frames": sum(row["proposal_frames"] for row in rows),
        "evidence_bundles": sum(row["evidence_bundles"] for row in rows),
        "mismatched_segments": len(mismatches),
        "rows": rows,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: summary[key] for key in (
                "segments", "proposal_frames", "evidence_bundles", "mismatched_segments"
            )}
        )
    )
    for row in mismatches:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

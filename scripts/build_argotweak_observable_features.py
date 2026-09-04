#!/usr/bin/env python3
"""Build label-free proposal/belief features for ArgoTweak evidence selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--visual-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _confidence(proposal: dict[str, object]) -> float:
    confidence = proposal.get("confidence") or {}
    assert isinstance(confidence, dict)
    return float(confidence.get("joint", 0.0))


def _observable(evidence: dict[str, object], temporal_position: float) -> list[float]:
    proposals = list(evidence["proposals"])
    total = max(len(proposals), 1)
    belief = [float(value) for value in evidence["operation_belief"]]
    features = [*belief, math.log1p(len(proposals)), temporal_position]
    for operation in OPERATIONS:
        rows = [row for row in proposals if row["operation"] == operation]
        scores = [_confidence(row) for row in rows]
        features.extend(
            [
                len(rows) / total,
                sum(scores) / len(scores) if scores else 0.0,
                max(scores, default=0.0),
            ]
        )
    matched = sum(bool(row.get("object_id")) for row in proposals)
    ready = len(evidence["commit_ready_proposal_ids"])
    blocked = evidence["blocked_proposals"]
    features.extend(
        [
            matched / total,
            ready / total,
            sum(reason == "BELOW_COMMIT_CONFIDENCE" for reason in blocked.values()) / total,
            sum(reason == "OBJECT_ASSIGNMENT_REQUIRED" for reason in blocked.values()) / total,
        ]
    )
    return features


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    import numpy as np

    episodes = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visual_index: dict[tuple[str, str], np.ndarray] = {}
    if args.visual_features is not None:
        visual_rows = [
            json.loads(line)
            for line in (args.visual_features / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        visual_values = np.load(args.visual_features / "features.npy").astype(np.float32)
        visual_index = {
            (row["episode_id"], row["evidence_id"]): value
            for row, value in zip(visual_rows, visual_values, strict=True)
        }
    records = []
    features = []
    for episode in episodes:
        evidence_rows = episode["evidence"]
        denominator = max(len(evidence_rows) - 1, 1)
        for index, evidence in enumerate(evidence_rows):
            row = {
                "episode_id": episode["episode_id"],
                "segment_id": episode["segment_id"],
                "split": episode["split"],
                "evidence_id": evidence["evidence_id"],
                "timestamp": evidence["timestamp"],
                "cost": evidence["cost"],
                "utility": evidence["utility"],
                "target_operation_counts": evidence["target_operation_counts"],
            }
            observable = np.asarray(
                _observable(evidence, index / denominator), dtype=np.float32
            )
            if args.visual_features is not None:
                visual = visual_index[(episode["episode_id"], evidence["evidence_id"])]
                observable = np.concatenate([visual, observable])
            records.append(row)
            features.append(observable)
    values = np.stack(features).astype(np.float16)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", values)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-observable-features-v1",
        "episodes": len(episodes),
        "evidence_frames": len(records),
        "feature_dim": int(values.shape[1]),
        "visual_features": str(args.visual_features) if args.visual_features else None,
        "label_free_inputs": True,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

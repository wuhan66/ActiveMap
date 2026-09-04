#!/usr/bin/env python3
"""Extract a split-safe minimal bundle for recurrent active-catalog rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    states_path: Path,
    episodes_path: Path,
    output_root: Path,
    *,
    split: str = "val",
    frozen_test: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("closed-loop bundle split must be train, val, or test")
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test bundle preparation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    state_output = output_root / f"states_{split}_step0.jsonl"
    identities = set()
    episode_ids = set()
    aois = set()
    budgets: Counter[str] = Counter()
    state_count = 0
    with (
        states_path.open(encoding="utf-8") as source,
        state_output.open("x", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            state = SelectorSample.model_validate_json(line)
            if test_assets_read and state.split != "test":
                raise ValueError(f"frozen selector states must be test-only at line {line_number}")
            if not test_assets_read and state.split not in {"train", "val"}:
                raise ValueError(f"forbidden selector split at line {line_number}")
            if state.split != split or int(state.metadata.get("oracle_step", -1)) != 0:
                continue
            episode_id = str(state.metadata["source_episode"])
            budget = float(state.metadata["budget"])
            identity = (episode_id, budget)
            if identity in identities:
                raise ValueError(f"duplicate {split} initial state: {identity}")
            identities.add(identity)
            episode_ids.add(episode_id)
            aois.add(str(state.metadata["aoi_id"]))
            budgets[f"{budget:g}"] += 1
            if not isinstance(state.metadata.get("evidence_predictions"), dict):
                raise ValueError(f"state lacks frozen evidence predictions: {state.sample_id}")
            destination.write(state.model_dump_json() + "\n")
            state_count += 1
    if not state_count:
        raise ValueError(f"no {split} step-0 states")

    episode_output = output_root / f"episodes_{split}.jsonl"
    found = set()
    episode_aois = set()
    with (
        episodes_path.open(encoding="utf-8") as source,
        episode_output.open("x", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            episode = EpisodeRecord.model_validate_json(line)
            if test_assets_read and episode.split != "test":
                raise ValueError(f"frozen episodes must be test-only at line {line_number}")
            if not test_assets_read and episode.split not in {"train", "val"}:
                raise ValueError(f"forbidden episode split at line {line_number}")
            if episode.split != split:
                continue
            if episode.episode_id in found:
                raise ValueError(f"duplicate {split} episode: {episode.episode_id}")
            found.add(episode.episode_id)
            episode_aois.add(episode.aoi_id)
            destination.write(episode.model_dump_json() + "\n")
    if found != episode_ids:
        missing = sorted(episode_ids - found)
        extra = sorted(found - episode_ids)
        raise ValueError(
            f"state/episode coverage mismatch: missing={missing[:1]}, extra={extra[:1]}"
        )
    if episode_aois != aois:
        raise ValueError("state/episode AOI sets disagree")
    summary = {
        "schema_version": "active-catalog-closed-loop-bundle-v1",
        "states": state_count,
        "episodes": len(found),
        "aois": len(aois),
        "budgets": dict(sorted(budgets.items())),
        "step": 0,
        "split": split,
        "state_source_sha256": sha256(states_path),
        "episode_source_sha256": sha256(episodes_path),
        "states_sha256": sha256(state_output),
        "episodes_sha256": sha256(episode_output),
        "test_assets_read": test_assets_read,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.states,
                args.episodes,
                args.output_root,
                split=args.split,
                frozen_test=args.frozen_test,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

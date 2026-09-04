#!/usr/bin/env python3
"""Render label-free visual indices for one-shot Active-Catalog test rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_task_id
from activemap.frozen_test import assert_frozen_test_access
from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample
from scripts.build_episode_sequential_selector_sft import render_initial_state


def build(states: Path, episodes: Path, output_root: Path) -> dict[str, Any]:
    assert_frozen_test_access()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    state_rows = [
        SelectorSample.model_validate_json(line)
        for line in states.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_rows = [
        EpisodeRecord.model_validate_json(line)
        for line in episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not state_rows or not episode_rows:
        raise ValueError("frozen visual index inputs are empty")
    if any(row.split != "test" for row in [*state_rows, *episode_rows]):
        raise ValueError("frozen visual index inputs must be test-only")
    by_episode = {row.episode_id: row for row in episode_rows}
    initial_evidence: dict[str, str] = {}
    for state in state_rows:
        if int(state.metadata.get("oracle_step", -1)) != 0:
            raise ValueError("frozen visual index accepts step-0 states only")
        episode_id = str(state.metadata["source_episode"])
        evidence_id = str(state.metadata["initial_evidence_id"])
        previous = initial_evidence.setdefault(episode_id, evidence_id)
        if previous != evidence_id:
            raise ValueError(f"inconsistent initial evidence for {episode_id}")
    if set(initial_evidence) != set(by_episode):
        raise ValueError("test state/episode coverage mismatch")

    output_root.mkdir(parents=True)
    image_root = output_root / "images"
    prompts = output_root / "test_visual_prompts.jsonl"
    index = output_root / "test_visual_index.jsonl"
    with prompts.open("x", encoding="utf-8") as prompt_file, index.open(
        "x", encoding="utf-8"
    ) as index_file:
        for episode_id in sorted(by_episode):
            example_id = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
            task_id = public_task_id(episode_id)
            image_path = image_root / f"{task_id}.jpg"
            render_initial_state(by_episode[episode_id], initial_evidence[episode_id], image_path)
            prompt_row = {
                "example_id": example_id,
                "task_id": task_id,
                "split": "test",
                "protocol": "active-catalog-frozen-visual-index-v1",
                "messages": [
                    {"role": "system", "content": "Frozen map-update evaluation."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"images/{image_path.name}"},
                            {"type": "text", "text": "Inspect the current map evidence."},
                        ],
                    },
                ],
            }
            prompt_file.write(json.dumps(prompt_row, separators=(",", ":")) + "\n")
            index_file.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "task_id": task_id,
                        "split": "test",
                        "source_episode": episode_id,
                        "model_visible": False,
                        "test_assets_read": True,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary = {
        "schema_version": "active-catalog-frozen-visual-index-v1",
        "episodes": len(by_episode),
        "prompts": str(prompts.resolve()),
        "evaluation_index": str(index.resolve()),
        "assistant_targets_present": False,
        "test_assets_read": True,
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
    args = parser.parse_args()
    print(json.dumps(build(args.states, args.episodes, args.output_root), indent=2))


if __name__ == "__main__":
    main()

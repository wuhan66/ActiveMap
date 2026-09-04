#!/usr/bin/env python3
"""Build a label-free validation visual index for recurrent policy transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_task_id
from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows = [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _remap_episodes(
    episodes: list[EpisodeRecord],
    mappings: tuple[tuple[Path, Path], ...],
) -> list[EpisodeRecord]:
    from activemap.oracle.updater_counterfactual import remap_episode_assets

    return remap_episode_assets(episodes, mappings)


def _render_initial_state(
    episode: EpisodeRecord, evidence_id: str, output: Path
) -> None:
    from scripts.build_episode_sequential_selector_sft import render_initial_state

    render_initial_state(episode, evidence_id, output)


def build(
    states_path: Path,
    episodes_path: Path,
    output_root: Path,
    *,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    states = [
        row
        for row in _read_jsonl(states_path, SelectorSample)
        if row.split == "val" and int(row.metadata.get("oracle_step", -1)) == 0
    ]
    episodes = {
        row.episode_id: row
        for row in _remap_episodes(
            _read_jsonl(episodes_path, EpisodeRecord), asset_root_maps
        )
        if row.split == "val"
    }
    if not states or not episodes:
        raise ValueError("validation visual-index inputs are empty")
    initial_evidence: dict[str, str] = {}
    for state in states:
        episode_id = str(state.metadata["source_episode"])
        evidence_id = str(state.metadata["initial_evidence_id"])
        if episode_id not in episodes:
            raise ValueError(f"missing validation episode: {episode_id}")
        previous = initial_evidence.setdefault(episode_id, evidence_id)
        if previous != evidence_id:
            raise ValueError(f"inconsistent initial evidence for {episode_id}")
    if set(initial_evidence) != set(episodes):
        raise ValueError("validation state/episode coverage mismatch")

    output_root.mkdir(parents=True)
    image_root = output_root / "images"
    prompts = output_root / "val_visual_prompts.jsonl"
    index = output_root / "val_visual_index.jsonl"
    with prompts.open("x", encoding="utf-8") as prompt_file, index.open(
        "x", encoding="utf-8"
    ) as index_file:
        for episode_id in sorted(episodes):
            example_id = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
            task_id = public_task_id(episode_id)
            image_path = image_root / f"{task_id}.jpg"
            _render_initial_state(
                episodes[episode_id], initial_evidence[episode_id], image_path
            )
            prompt_file.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "task_id": task_id,
                        "split": "val",
                        "protocol": "active-catalog-validation-visual-index-v1",
                        "messages": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Validation map-update evaluation.",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": f"images/{image_path.name}",
                                    },
                                    {
                                        "type": "text",
                                        "text": "Inspect the current map evidence.",
                                    },
                                ],
                            },
                        ],
                        "test_assets_read": False,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            index_file.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "task_id": task_id,
                        "split": "val",
                        "source_episode": episode_id,
                        "model_visible": False,
                        "test_assets_read": False,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary = {
        "schema_version": "active-catalog-validation-visual-index-v1",
        "episodes": len(episodes),
        "step0_states": len(states),
        "prompts": str(prompts.resolve()),
        "prompt_sha256": _sha256(prompts),
        "evaluation_index": str(index.resolve()),
        "evaluation_index_sha256": _sha256(index),
        "assistant_targets_present": False,
        "split": "val",
        "sources": {
            "states": {
                "path": str(states_path.resolve()),
                "sha256": _sha256(states_path),
            },
            "episodes": {
                "path": str(episodes_path.resolve()),
                "sha256": _sha256(episodes_path),
            },
        },
        "asset_root_maps": [
            {"source": str(source), "target": str(target)}
            for source, target in asset_root_maps
        ],
        "test_assets_read": False,
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
    parser.add_argument("--asset-root-map", action="append", default=[])
    args = parser.parse_args()
    mappings = []
    for value in args.asset_root_map:
        if "=" not in value:
            parser.error("--asset-root-map must use SOURCE=TARGET")
        source, target = value.split("=", 1)
        mappings.append((Path(source), Path(target)))
    print(
        json.dumps(
            build(
                args.states,
                args.episodes,
                args.output_root,
                asset_root_maps=tuple(mappings),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

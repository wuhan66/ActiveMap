#!/usr/bin/env python3
"""Report shared pre-acquisition perception and incremental evidence costs.

ActiveMap's current selector consumes frozen features for every candidate.
This audit prevents incremental evidence-processing cost from being described
as end-to-end visual inference savings. It reports both quantities explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EpisodeRecord


def parse_record(value: str) -> tuple[str, str, Path]:
    head, separator, path = value.partition("=")
    model_seed, divider, policy = head.partition(":")
    if not separator or not divider or not model_seed or not policy:
        raise argparse.ArgumentTypeError("record must be SEED:POLICY=/path/to/writeback.jsonl")
    return model_seed, policy, Path(path)


def load_episodes(path: Path) -> dict[str, EpisodeRecord]:
    episodes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = EpisodeRecord.model_validate_json(line)
        if episode.split != "val":
            continue
        if episode.episode_id in episodes:
            raise ValueError(f"duplicate episode id: {episode.episode_id}")
        episodes[episode.episode_id] = episode
    if not episodes:
        raise ValueError(f"no validation episodes in {path}")
    return episodes


def load_writeback(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty writeback: {path}")
    for row in rows:
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError("cost audit accepts validation-only writebacks")
    return rows


def _task_id_to_episode(episodes: dict[str, EpisodeRecord]) -> dict[str, EpisodeRecord]:
    from activemap.agent.identifiers import public_task_id

    mapped = {public_task_id(episode_id): episode for episode_id, episode in episodes.items()}
    if len(mapped) != len(episodes):
        raise ValueError("public task-id collision in episode manifest")
    return mapped


def analyze(
    episodes: dict[str, EpisodeRecord],
    records: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    updater_latency_ms: float,
) -> dict[str, Any]:
    if updater_latency_ms <= 0:
        raise ValueError("updater latency must be positive")
    by_task = _task_id_to_episode(episodes)
    policies = sorted({policy for by_policy in records.values() for policy in by_policy})
    seeds = sorted(records)
    if len(seeds) < 2:
        raise ValueError("cost audit requires at least two model seeds")
    cells = {}
    per_seed = {}
    canonical_support = None
    for seed in seeds:
        if set(records[seed]) != set(policies):
            raise ValueError("each seed must include identical policies")
        per_seed[seed] = {}
        seed_support = None
        for policy in policies:
            rows = records[seed][policy]
            identities = {(str(row["task_id"]), float(row["budget"])) for row in rows}
            if len(identities) != len(rows):
                raise ValueError(f"duplicate task-budget rows for {seed}:{policy}")
            if seed_support is None:
                seed_support = identities
            elif seed_support != identities:
                raise ValueError(f"policy support mismatch for seed {seed}")
            task_metrics = []
            for row in rows:
                task_id = str(row["task_id"])
                episode = by_task.get(task_id)
                if episode is None:
                    raise ValueError(f"writeback task absent from episode manifest: {task_id}")
                candidates = len(episode.evidence_catalog)
                selected = len(row.get("selected_evidence_ids", ()))
                if candidates < 1 or selected < 1 or selected > candidates:
                    raise ValueError(f"invalid candidate/selected count for {task_id}")
                pre_calls = candidates
                incremental_calls = selected
                pre_ms = pre_calls * updater_latency_ms
                # The policy's recorded cost is the selective downstream evidence cost.
                incremental_cost = float(row["spent_cost"])
                task_metrics.append(
                    {
                        "candidate_count": candidates,
                        "selected_count": selected,
                        "shared_preacquisition_updater_calls": pre_calls,
                        "incremental_selected_evidence_calls": incremental_calls,
                        "shared_preacquisition_perception_ms": pre_ms,
                        "incremental_selected_evidence_cost": incremental_cost,
                    }
                )
            per_seed[seed][policy] = {
                "task_budget_count": len(task_metrics),
                **{
                    key: float(np.mean([row[key] for row in task_metrics]))
                    for key in task_metrics[0]
                },
            }
        if canonical_support is None:
            canonical_support = seed_support
        elif canonical_support != seed_support:
            raise ValueError("model seed support mismatch")
    for policy in policies:
        cells[policy] = {
            key: {
                "mean": float(np.mean([per_seed[seed][policy][key] for seed in seeds])),
                "seed_std": float(np.std([per_seed[seed][policy][key] for seed in seeds], ddof=1)),
            }
            for key in next(iter(per_seed.values()))[policy]
        }
    return {
        "schema_version": "activemap-preacquisition-cost-audit-v1",
        "split": "val",
        "test_assets_read": False,
        "protocol": {
            "candidate_perception": "all candidates are passed through the frozen updater before selection",
            "shared_preacquisition_cost": "candidate_count x fixed-hardware updater latency",
            "incremental_cost": "recorded selected-evidence processing cost from the rollout/writeback",
            "end_to_end_cost": "reported as the two-component vector (shared perception ms, incremental evidence-budget units); no scalar sum is emitted without a predeclared unit conversion",
            "claim_boundary": "only incremental selected-evidence cost may be used to claim a sparse post-perception cost advantage",
        },
        "updater_latency_ms": updater_latency_ms,
        "seed_count": len(seeds),
        "task_budget_count": len(canonical_support or ()),
        "per_seed": per_seed,
        "policies": cells,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Pre-acquisition Cost Audit",
        "",
        "All candidate images are already processed by the frozen updater before selection.",
        "Therefore, lower selected-evidence cost is an incremental post-perception saving, not a reduction in total visual-backbone calls.",
        "",
        "| Policy | Candidate updater calls | Selected evidence calls | Shared updater time (ms) | Incremental evidence budget |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, metrics in result["policies"].items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.6f} |".format(
                policy,
                metrics["shared_preacquisition_updater_calls"]["mean"],
                metrics["incremental_selected_evidence_calls"]["mean"],
                metrics["shared_preacquisition_perception_ms"]["mean"],
                metrics["incremental_selected_evidence_cost"]["mean"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--updater-latency-ms", type=float, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for seed, policy, path in args.record:
        if policy in records[seed]:
            raise ValueError(f"duplicate seed-policy input {seed}:{policy}")
        records[seed][policy] = load_writeback(path)
    result = analyze(
        load_episodes(args.episodes),
        dict(records),
        updater_latency_ms=args.updater_latency_ms,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_markdown(result), encoding="utf-8")
    with (args.output_dir / "policies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", *result["policies"][next(iter(result["policies"]))]])
        for policy, metrics in result["policies"].items():
            writer.writerow([policy, *[metrics[key]["mean"] for key in metrics]])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

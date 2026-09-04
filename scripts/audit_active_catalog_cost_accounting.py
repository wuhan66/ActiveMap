#!/usr/bin/env python3
"""Audit full versus incremental costs from raw active-catalog rollouts.

The frozen selector in this protocol is constructed after every candidate has
already been processed by the updater. Consequently, sparse selection can only
claim savings in *incremental* evidence/tool processing. This audit reports
that quantity separately from shared all-candidate perception work.
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
    head, separator, location = value.partition("=")
    seed, divider, policy = head.partition(":")
    if not separator or not divider or not seed or not policy:
        raise argparse.ArgumentTypeError("record must be SEED:POLICY=/path/to/rollout.jsonl")
    return seed, policy, Path(location)


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


def load_rollouts(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty rollout trace: {path}")
    required = {
        "source_episode", "budget", "split", "test_assets_read", "acquisitions",
        "tool_calls", "spent_cost", "tool_cost",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"rollout schema missing {missing}: {path}")
    result = []
    identities = set()
    for row in rows:
        if row["split"] != "val" or row["test_assets_read"] is not False:
            raise ValueError(f"cost audit is validation-only: {path}")
        key = (str(row["source_episode"]), float(row["budget"]))
        if key in identities:
            raise ValueError(f"duplicate episode-budget rollout: {path}: {key}")
        identities.add(key)
        result.append(row)
    return result


def analyze(
    episodes: dict[str, EpisodeRecord],
    records: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    updater_latency_ms: float,
) -> dict[str, Any]:
    if updater_latency_ms <= 0.0:
        raise ValueError("updater latency must be positive")
    seeds = sorted(records)
    policies = sorted({policy for by_policy in records.values() for policy in by_policy})
    if len(seeds) < 2:
        raise ValueError("cost audit requires at least two independently trained selector seeds")
    if len(policies) < 2:
        raise ValueError("cost audit requires at least two policies")

    support: set[tuple[str, float]] | None = None
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        if set(records[seed]) != set(policies):
            raise ValueError("each seed must cover the same policy set")
        per_seed[seed] = {}
        seed_support: set[tuple[str, float]] | None = None
        for policy in policies:
            rows = records[seed][policy]
            identities = {(str(row["source_episode"]), float(row["budget"])) for row in rows}
            if seed_support is None:
                seed_support = identities
            elif identities != seed_support:
                raise ValueError(f"policy rollout support mismatch for seed={seed}")
            values = []
            for row in rows:
                episode_id = str(row["source_episode"])
                episode = episodes.get(episode_id)
                if episode is None:
                    raise ValueError(f"rollout episode absent from manifest: {episode_id}")
                candidate_count = len(episode.evidence_catalog)
                values.append(
                    {
                        "candidate_count": float(candidate_count),
                        "shared_preacquisition_updater_calls": float(candidate_count),
                        "shared_preacquisition_perception_ms": float(candidate_count * updater_latency_ms),
                        "incremental_acquisitions": float(row["acquisitions"]),
                        "incremental_tool_calls": float(row["tool_calls"]),
                        "incremental_evidence_budget": float(row["spent_cost"]),
                        "incremental_tool_budget": float(row["tool_cost"]),
                        "incremental_total_budget": float(row["spent_cost"] + row["tool_cost"]),
                    }
                )
            per_seed[seed][policy] = {
                "episode_budget_count": float(len(values)),
                **{key: float(np.mean([value[key] for value in values])) for key in values[0]},
            }
        if support is None:
            support = seed_support
        elif support != seed_support:
            raise ValueError("cross-seed rollout support mismatch")
    assert support is not None

    aggregate = {
        policy: {
            name: {
                "mean": float(np.mean([per_seed[seed][policy][name] for seed in seeds])),
                "seed_std": float(np.std([per_seed[seed][policy][name] for seed in seeds], ddof=1)),
            }
            for name in next(iter(per_seed.values()))[policy]
        }
        for policy in policies
    }
    return {
        "schema_version": "activemap-active-catalog-cost-accounting-v1",
        "split": "val",
        "test_assets_read": False,
        "updater_latency_ms": updater_latency_ms,
        "protocol": {
            "raw_source": "policy rollout trace, not writeback rendering input",
            "shared_preacquisition_perception": "all episode candidates are processed by the frozen updater before selection",
            "incremental_policy_work": "actual rollout acquisitions, tool calls, and recorded budget",
            "reporting_rule": "shared updater latency and abstract evidence budget remain separate axes; no scalar total is reported without a predeclared conversion",
            "claim_boundary": "a lower incremental budget is not a claim of lower end-to-end visual-backbone inference",
        },
        "seed_count": len(seeds),
        "episode_budget_count": len(support),
        "per_seed": per_seed,
        "policies": aggregate,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Active-Catalog Cost Accounting",
        "",
        "All-candidate updater work is shared and already incurred before policy selection.",
        "Only acquisitions, tool calls, and evidence budget are policy-variable incremental costs.",
        "",
        "| Policy | Candidates / episode | Shared updater time (ms) | Acquisitions | Tool calls | Incremental evidence budget | Tool budget |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, values in result["policies"].items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                policy,
                values["candidate_count"]["mean"],
                values["shared_preacquisition_perception_ms"]["mean"],
                values["incremental_acquisitions"]["mean"],
                values["incremental_tool_calls"]["mean"],
                values["incremental_evidence_budget"]["mean"],
                values["incremental_tool_budget"]["mean"],
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
            raise ValueError(f"duplicate seed-policy record: {seed}:{policy}")
        records[seed][policy] = load_rollouts(path)
    result = analyze(
        load_episodes(args.episodes),
        dict(records),
        updater_latency_ms=args.updater_latency_ms,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(markdown(result), encoding="utf-8")
    with (args.output_dir / "policies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(next(iter(result["policies"].values())))
        writer.writerow(["policy", *names])
        for policy, values in result["policies"].items():
            writer.writerow([policy, *[values[name]["mean"] for name in names]])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

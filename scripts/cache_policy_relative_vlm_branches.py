#!/usr/bin/env python3
"""Cache direct and post-tool VLM branches for policy-relative acquisition labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_relative_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy-relative cache is empty")
    old = [bool(row["static_use_tool"]) for row in rows]
    new = [bool(row["policy_relative_use_tool"]) for row in rows]
    old_positive = sum(old)
    new_positive = sum(new)
    overlap = sum(left and right for left, right in zip(old, new, strict=True))
    union = sum(left or right for left, right in zip(old, new, strict=True))
    direct = [float(row["direct_utility"]) for row in rows]
    post = [float(row["post_tool_utility"]) for row in rows]
    oracle = [max(left, right) for left, right in zip(direct, post, strict=True)]
    static = [
        right if call else left
        for left, right, call in zip(direct, post, old, strict=True)
    ]
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "sample_count": len(rows),
        "static_positive_count": old_positive,
        "policy_relative_positive_count": new_positive,
        "static_positive_rate": old_positive / len(rows),
        "policy_relative_positive_rate": new_positive / len(rows),
        "label_agreement": sum(left == right for left, right in zip(old, new, strict=True))
        / len(rows),
        "positive_jaccard": overlap / max(union, 1),
        "static_label_precision": overlap / max(old_positive, 1),
        "static_label_recall": overlap / max(new_positive, 1),
        "static_positive_to_stop": sum(
            left and not right for left, right in zip(old, new, strict=True)
        ),
        "static_stop_to_positive": sum(
            not left and right for left, right in zip(old, new, strict=True)
        ),
        "mean_direct_utility": mean(direct),
        "mean_always_post_tool_utility": mean(post),
        "mean_static_label_utility": mean(static),
        "mean_policy_relative_oracle_utility": mean(oracle),
        "policy_relative_oracle_gain_over_direct": mean(oracle) - mean(direct),
        "static_label_gain_over_direct": mean(static) - mean(direct),
        "static_label_acquisition_regret": mean(oracle) - mean(static),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_shard(items: list[Any], num_shards: int, shard_index: int) -> list[Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.records import AgentAction
    from activemap.agent.tool_sft import terminal_reward
    from activemap.agent.vlm_evaluation import message_text, operation_from_action
    from activemap.agent.vlm_sft import load_vlm_sft_rows
    from activemap.models import EditOperation
    from scripts.evaluate_semantic_vlm_rollouts import (
        _generate,
        _parse_action,
        _terminal_operation,
        index_pairs,
    )

    pairs = index_pairs(load_vlm_sft_rows(args.rollout_jsonl))
    source_pair_count = len(pairs)
    if args.expected_pairs is not None and source_pair_count != args.expected_pairs:
        raise ValueError(f"expected {args.expected_pairs} pairs, found {source_pair_count}")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    limited_pair_count = len(pairs)
    pairs = select_shard(pairs, args.num_shards, args.shard_index)
    input_hash = _sha256(args.rollout_jsonl)
    manifest = {
        "schema_version": "policy-relative-vlm-branch-cache-launch-v1",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "rollout_jsonl": str(args.rollout_jsonl.resolve()),
        "rollout_sha256": input_hash,
        "seed": args.seed,
        "pair_count": len(pairs),
        "source_pair_count": source_pair_count,
        "limited_pair_count": limited_pair_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_new_tokens": args.max_new_tokens,
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "launch_manifest.json"
    traces_path = args.output_dir / "traces.jsonl"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requested without launch_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("resume manifest differs from the requested protocol")
    else:
        _write_json(manifest_path, manifest)

    completed_rows = []
    completed_ids = set()
    if args.resume and traces_path.is_file():
        completed_rows = [
            json.loads(line)
            for line in traces_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed_ids = {str(row["example_id"]) for row in completed_rows}
        if len(completed_ids) != len(completed_rows):
            raise ValueError("resume cache contains duplicate example IDs")
    pair_ids = [str(pre["example_id"]) for pre, _ in pairs]
    if not completed_ids.issubset(pair_ids):
        raise ValueError("resume cache contains examples outside the rollout input")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device).eval()

    pending = [(pre, post) for pre, post in pairs if str(pre["example_id"]) not in completed_ids]
    mode = "a" if completed_rows else "w"
    with traces_path.open(mode, encoding="utf-8") as handle:
        for index, (pre, post) in enumerate(
            tqdm(pending, desc="Policy-relative direct/post cache"), start=1
        ):
            example_id = str(pre["example_id"])
            target_action = AgentAction.model_validate_json(message_text(post["messages"][2]))
            target = operation_from_action(target_action)
            if target is None:
                raise ValueError(f"non-terminal target action: {example_id}")
            observation = json.loads(message_text(pre["messages"][1]))
            belief = EditOperation(pre["consensus"]["baseline_operation"])
            cost = float(observation["semantic_tool_cost"])

            raw_direct = _generate(model, processor, pre, args)
            direct_action, direct_error = _parse_action(raw_direct)
            direct_parsed = _terminal_operation(direct_action)
            direct = direct_parsed if direct_parsed is not None else belief

            raw_post = _generate(model, processor, post, args)
            post_action, post_error = _parse_action(raw_post)
            post_parsed = _terminal_operation(post_action)
            post_operation = post_parsed if post_parsed is not None else belief

            direct_utility = terminal_reward(target, direct)
            post_utility = terminal_reward(target, post_operation) - cost
            advantage = post_utility - direct_utility
            row = {
                "example_id": example_id,
                "task_id": str(pre["task_id"]),
                "split": str(pre["split"]),
                "target_operation": target.value,
                "belief_operation": belief.value,
                "direct_operation": direct.value,
                "post_tool_operation": post_operation.value,
                "static_use_tool": bool(pre["oracle_use_tool"]),
                "policy_relative_use_tool": advantage > 0.0,
                "policy_relative_advantage": advantage,
                "direct_utility": direct_utility,
                "post_tool_utility": post_utility,
                "tool_cost": cost,
                "direct_terminal_valid": direct_parsed is not None,
                "post_terminal_valid": post_parsed is not None,
                "raw_direct": raw_direct,
                "raw_post": raw_post,
                "direct_error": direct_error,
                "post_error": post_error,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            completed_rows.append(row)
            if index % args.checkpoint_every == 0:
                _write_json(
                    args.output_dir / "progress.json",
                    {
                        "schema_version": "policy-relative-vlm-branch-cache-progress-v1",
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "completed": len(completed_rows),
                        "total": len(pairs),
                        "test_assets_read": False,
                    },
                )

    by_id = {str(row["example_id"]): row for row in completed_rows}
    ordered = [by_id[example_id] for example_id in pair_ids]
    if len(ordered) != len(pairs):
        raise RuntimeError("branch cache did not cover every requested pair")
    if ordered != completed_rows:
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "policy-relative-vlm-branch-cache-v1",
        **manifest,
        "metrics": policy_relative_metrics(ordered),
        "direct_terminal_valid_rate": sum(row["direct_terminal_valid"] for row in ordered)
        / len(ordered),
        "post_terminal_valid_rate": sum(row["post_terminal_valid"] for row in ordered)
        / len(ordered),
        "trace_sha256": _sha256(traces_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(
        args.output_dir / "progress.json",
        {
            "schema_version": "policy-relative-vlm-branch-cache-progress-v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": len(ordered),
            "total": len(ordered),
            "status": "complete",
            "test_assets_read": False,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

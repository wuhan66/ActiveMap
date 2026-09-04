#!/usr/bin/env python3
"""Build multimodal active-catalog preferences from frozen shortlist utilities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    message = row["messages"][-1]
    if message.get("role") != "assistant":
        raise ValueError("SFT row does not end with assistant")
    content = message["content"]
    text = next(part["text"] for part in content if part.get("type") == "text")
    value = json.loads(text)
    if value.get("stage") != "SELECT" or value.get("selection") not in {
        "STOP",
        "ACQUIRE",
    }:
        raise ValueError(f"invalid selector target: {value}")
    return value


def action_key(selection: str, evidence_id: str | None = None) -> str:
    return "STOP" if selection == "STOP" else f"ACQUIRE:{evidence_id}"


def action_message(key: str) -> dict[str, Any]:
    if key == "STOP":
        payload = {"stage": "SELECT", "selection": "STOP"}
    elif key.startswith("ACQUIRE:"):
        payload = {
            "stage": "SELECT",
            "selection": "ACQUIRE",
            "evidence_id": key.split(":", 1)[1],
        }
    else:
        raise ValueError(f"unsupported action key: {key}")
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
    }


def preference_family(chosen: str, rejected: str) -> str:
    if chosen == "STOP":
        return "stop_over_acquire"
    if rejected == "STOP":
        return "acquire_over_stop"
    return "evidence_ranking"


def resolved_prompt(messages: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    prompt = copy.deepcopy(messages[:-1])
    image_parts = [
        part
        for message in prompt
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    if len(image_parts) != 1:
        raise ValueError("preference prompt must contain exactly one image")
    image = Path(str(image_parts[0].get("image")))
    if not image.is_absolute():
        image = (root / image).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    image_parts[0]["image"] = str(image)
    return prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_preferences(
    sft_path: Path,
    evaluation_index_path: Path,
    output_path: Path,
    *,
    expected_split: str,
    pairs_per_state: int = 2,
    minimum_margin: float = 1e-6,
) -> dict[str, Any]:
    if expected_split not in {"train", "val"}:
        raise ValueError("preferences may only use train or val")
    if pairs_per_state <= 0 or minimum_margin < 0:
        raise ValueError("invalid preference settings")
    sft_rows = read_jsonl(sft_path)
    index_rows = read_jsonl(evaluation_index_path)
    index = {str(row["example_id"]): row for row in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("duplicate example IDs in evaluation index")
    if len(sft_rows) != len(index_rows):
        raise ValueError(f"SFT/index size mismatch: {len(sft_rows)} vs {len(index_rows)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_count = 0
    skipped_ties = 0
    pair_count = 0
    families: Counter[str] = Counter()
    margins: list[float] = []
    seen: set[str] = set()
    with output_path.open("x", encoding="utf-8") as destination:
        for row in sft_rows:
            example_id = str(row["example_id"])
            if example_id in seen:
                raise ValueError(f"duplicate SFT example ID: {example_id}")
            seen.add(example_id)
            hidden = index.get(example_id)
            if hidden is None:
                raise ValueError(f"missing evaluation row: {example_id}")
            if row.get("split") != expected_split or hidden.get("split") != expected_split:
                raise ValueError(f"split mismatch for {example_id}")
            if hidden.get("model_visible") is not False or hidden.get("test_assets_read") is not False:
                raise ValueError(f"invalid hidden sidecar contract for {example_id}")
            target = assistant_payload(row)
            target_key = action_key(target["selection"], target.get("evidence_id"))
            hidden_key = action_key(
                str(hidden["target_selection"]), hidden.get("target_evidence_id")
            )
            if target_key != hidden_key:
                raise ValueError(f"SFT/sidecar target mismatch for {example_id}")
            utilities = {"STOP": float(hidden["stop_utility"])}
            costs = {"STOP": 0.0}
            for candidate in hidden["candidates"]:
                key = action_key("ACQUIRE", str(candidate["evidence_id"]))
                if key in utilities:
                    raise ValueError(f"duplicate candidate for {example_id}: {key}")
                utilities[key] = float(candidate["utility"])
                costs[key] = float(candidate["cost"])
            target_utility = float(hidden["target_utility"])
            if target_key not in utilities or abs(utilities[target_key] - target_utility) > 1e-8:
                raise ValueError(f"target utility mismatch for {example_id}")
            if target_utility + 1e-8 < max(utilities.values()):
                raise ValueError(f"target is not shortlist-optimal for {example_id}")
            alternatives = sorted(
                (
                    (key, utility)
                    for key, utility in utilities.items()
                    if key != target_key and target_utility - utility >= minimum_margin
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if not alternatives:
                skipped_ties += 1
                continue
            state_count += 1
            for rank, (rejected_key, rejected_utility) in enumerate(
                alternatives[:pairs_per_state], 1
            ):
                margin = target_utility - rejected_utility
                family = preference_family(target_key, rejected_key)
                preference = {
                    "schema_version": "active-catalog-vlm-preference-v1",
                    "example_id": example_id,
                    "task_id": row["task_id"],
                    "split": expected_split,
                    "prompt": resolved_prompt(row["messages"], sft_path.parent),
                    "chosen": action_message(target_key),
                    "rejected": action_message(rejected_key),
                    "chosen_action_key": target_key,
                    "rejected_action_key": rejected_key,
                    "chosen_utility": target_utility,
                    "rejected_utility": rejected_utility,
                    "utility_margin": margin,
                    "chosen_cost": costs[target_key],
                    "rejected_cost": costs[rejected_key],
                    "preference_family": family,
                    "negative_rank": rank,
                    "model_visible_utility": False,
                    "test_assets_read": False,
                }
                destination.write(json.dumps(preference, separators=(",", ":")) + "\n")
                pair_count += 1
                families[family] += 1
                margins.append(margin)
    if seen != set(index):
        raise ValueError("SFT and evaluation index IDs do not match exactly")
    summary = {
        "schema_version": "active-catalog-vlm-preference-summary-v1",
        "split": expected_split,
        "sft_records": len(sft_rows),
        "preference_states": state_count,
        "preference_pairs": pair_count,
        "skipped_tied_states": skipped_ties,
        "pairs_per_state": pairs_per_state,
        "minimum_margin": minimum_margin,
        "family_counts": dict(sorted(families.items())),
        "margin_min": min(margins) if margins else None,
        "margin_mean": sum(margins) / len(margins) if margins else None,
        "sft_sha256": sha256(sft_path),
        "evaluation_index_sha256": sha256(evaluation_index_path),
        "output_sha256": sha256(output_path),
        "model_visible_utility": False,
        "test_assets_read": False,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    parser.add_argument("--pairs-per-state", type=int, default=2)
    parser.add_argument("--minimum-margin", type=float, default=1e-6)
    args = parser.parse_args()
    print(
        json.dumps(
            build_preferences(
                args.sft,
                args.evaluation_index,
                args.output,
                expected_split=args.expected_split,
                pairs_per_state=args.pairs_per_state,
                minimum_margin=args.minimum_margin,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

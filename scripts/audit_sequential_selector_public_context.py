#!/usr/bin/env python3
"""Audit whether grouped SELECT candidates are public at inference time.

This report is read-only.  It prevents an invalid listwise controller that
would join sibling offline records while hiding those sibling candidates from
the model's actual SELECT prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FORBIDDEN_PROMPT_KEYS = {
    "gt_edit",
    "oracle_utilities",
    "policy_relative_advantage",
    "selected_tool",
    "target_selection",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("missing system/user messages")
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise ValueError("user content is invalid")
    texts = [part["text"] for part in content if part.get("type") == "text"]
    images = [str(part["image"]) for part in content if part.get("type") == "image"]
    if len(texts) != 1 or len(images) != 1:
        raise ValueError("SELECT prompt must have exactly one text state and one image")
    state = json.loads(str(texts[0]))
    if state.get("controller_stage") != "SELECT":
        raise ValueError("prompt is not a SELECT state")
    leaked = FORBIDDEN_PROMPT_KEYS.intersection(state)
    if leaked:
        raise ValueError(f"SELECT prompt leaks target metadata: {sorted(leaked)}")
    return state, images[0]


def public_candidate_ids(state: dict[str, Any]) -> set[str]:
    """Return only IDs explicitly exposed by the user state."""
    ids: set[str] = set()
    if isinstance(state.get("evidence_id"), str):
        ids.add(str(state["evidence_id"]))
    for key in ("candidate_evidence_ids", "selected_evidence_ids"):
        values = state.get(key)
        if isinstance(values, list):
            ids.update(str(value) for value in values)
    candidates = state.get("candidate_evidence")
    if isinstance(candidates, list):
        ids.update(
            str(candidate["evidence_id"])
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("evidence_id"), str)
        )
    if not ids:
        raise ValueError("SELECT prompt exposes no public evidence identifier")
    return ids


def _public_context(state: dict[str, Any], image: str) -> dict[str, str]:
    return {
        "image": image,
        "policy_snapshot": json.dumps(state.get("policy_snapshot"), sort_keys=True),
        "available_tools": json.dumps(state.get("available_tools"), sort_keys=True),
    }


def _candidate_descriptor(state: dict[str, Any]) -> dict[str, str]:
    """Public per-candidate attributes that may vary in a redesigned prompt."""
    return {
        "belief": json.dumps(state.get("belief"), sort_keys=True),
        "budget": json.dumps(state.get("budget"), sort_keys=True),
        "direct_draft": json.dumps(state.get("direct_draft"), sort_keys=True),
    }


def audit_rows(
    rows: list[dict[str, Any]], expected_split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one split and return summary plus one record for each task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectories: set[str] = set()
    for row in rows:
        if str(row.get("split")) != expected_split:
            raise ValueError(f"expected only split={expected_split}")
        if str(row.get("stage")) != "SELECT":
            raise ValueError("manifest contains a non-SELECT row")
        trajectory_id = str(row.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id in trajectories:
            raise ValueError("missing or duplicate SELECT trajectory")
        trajectories.add(trajectory_id)
        state, image = _observation(row)
        by_task[str(row["task_id"])].append(
            {
                "trajectory_id": trajectory_id,
                "public_ids": public_candidate_ids(state),
                "context": _public_context(state, image),
                "candidate_descriptor": _candidate_descriptor(state),
            }
        )

    task_rows = []
    for task_id, states in sorted(by_task.items()):
        candidate_universe = set().union(*(state["public_ids"] for state in states))
        all_visible = all(candidate_universe <= state["public_ids"] for state in states)
        shared_context = {
            field: len({state["context"][field] for state in states}) == 1
            for field in ("image", "policy_snapshot", "available_tools")
        }
        descriptor_varies = {
            field: len({state["candidate_descriptor"][field] for state in states}) > 1
            for field in ("belief", "budget", "direct_draft")
        }
        reformulation_eligible = (
            len(states) > 1
            and len(candidate_universe) == len(states)
            and shared_context["image"]
            and shared_context["policy_snapshot"]
        )
        task_rows.append(
            {
                "task_id": task_id,
                "state_count": len(states),
                "candidate_count_from_group": len(candidate_universe),
                "public_candidate_counts": sorted(len(state["public_ids"]) for state in states),
                "all_group_candidates_visible_in_each_prompt": all_visible,
                "shared_context": shared_context,
                "candidate_descriptor_varies": descriptor_varies,
                "explicit_multicandidate_reformulation_eligible": reformulation_eligible,
            }
        )

    task_count = len(task_rows)
    state_count = sum(row["state_count"] for row in task_rows)
    single_candidate_states = sum(
        count == 1 for row in task_rows for count in row["public_candidate_counts"]
    )
    all_visible_tasks = sum(
        row["all_group_candidates_visible_in_each_prompt"] for row in task_rows
    )
    reformulation_tasks = sum(
        row["explicit_multicandidate_reformulation_eligible"] for row in task_rows
    )
    shared_context_task_rates = {
        field: sum(row["shared_context"][field] for row in task_rows) / max(task_count, 1)
        for field in ("image", "policy_snapshot", "available_tools")
    }
    result = {
        "schema_version": "sequential-selector-public-context-audit-v1",
        "role": "read-only-causal-eligibility-audit",
        "split": expected_split,
        "task_count": task_count,
        "state_count": state_count,
        "state_count_histogram": dict(
            sorted(Counter(row["state_count"] for row in task_rows).items())
        ),
        "single_public_candidate_state_rate": single_candidate_states / max(state_count, 1),
        "joint_candidate_visibility_task_rate": all_visible_tasks / max(task_count, 1),
        "current_prompt_listwise_eligible": all_visible_tasks == task_count,
        "shared_context_task_rates": shared_context_task_rates,
        "explicit_multicandidate_reformulation_task_rate": reformulation_tasks
        / max(task_count, 1),
        "prompt_target_metadata_exposed": False,
        "test_assets_read": False,
    }
    return result, task_rows


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    result, task_rows = audit_rows(_rows(args.manifest), args.split)
    args.output.mkdir(parents=True)
    task_path = args.output / "per_task.jsonl"
    with task_path.open("w", encoding="utf-8") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result.update(
        {
            "manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": _sha256(args.manifest),
            },
            "per_task_sha256": _sha256(task_path),
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build staged VLM SFT from frozen direct/post policy branches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentAction, AgentActionType
from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
    SequentialControllerTrajectory,
)
from activemap.agent.tool_sft import terminal_action
from activemap.agent.vlm_sft import load_vlm_sft_rows
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation
from scripts.evaluate_semantic_vlm_rollouts import index_pairs

SYSTEM_PROMPT = (
    "You are the ActiveMap sequential visual controller. Follow the requested controller_stage and "
    "return exactly one JSON action. First draft a typed map edit, then decide whether evidence is "
    "worth its cost, execute only an available tool, revise belief from its result, and finally "
    "COMMIT or REJECT. Minimize false edits and unnecessary evidence cost. Return JSON only."
)

FORBIDDEN_PROMPT_KEYS = {
    "target_operation",
    "policy_relative_advantage",
    "oracle_use_tool",
    "gt_edit",
    "ground_truth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    content = row["messages"][1]["content"]
    text = [part["text"] for part in content if part.get("type") == "text"]
    if len(text) != 1:
        raise ValueError("rollout user message must contain exactly one text observation")
    return json.loads(text[0])


def _messages(
    source: dict[str, Any], state: dict[str, Any], action: SequentialControllerAction
) -> list[dict[str, Any]]:
    images = [
        copy.deepcopy(part)
        for part in source["messages"][1]["content"]
        if part.get("type") == "image"
    ]
    if len(images) != 1:
        raise ValueError("rollout user message must contain exactly one image")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                images[0],
                {"type": "text", "text": json.dumps(state, separators=(",", ":"))},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": action.model_dump_json(exclude_none=True),
                }
            ],
        },
    ]


def _confidence(prediction: EditOperation, target: EditOperation) -> float:
    return 0.9 if prediction == target else 0.1


def _state(
    observation: dict[str, Any],
    *,
    stage: ControllerStage,
    policy_snapshot: str,
    direct_operation: EditOperation | None = None,
    direct_confidence: float | None = None,
    selected_tool: bool | None = None,
    tool_result: dict[str, Any] | None = None,
    updated_operation: EditOperation | None = None,
    updated_confidence: float | None = None,
) -> dict[str, Any]:
    cost = float(observation["semantic_tool_cost"])
    spent = cost if tool_result is not None else 0.0
    result: dict[str, Any] = {
        "controller_stage": stage.value,
        "policy_snapshot": policy_snapshot,
        "belief": observation["belief"],
        "evidence_id": observation["evidence_id"],
        "available_tools": observation["available_tools"],
        "budget": {
            "initial": cost,
            "spent": spent,
            "remaining": max(cost - spent, 0.0),
        },
        "safety": {"false_edit_risk_limit": 0.02},
    }
    if direct_operation is not None:
        result["direct_draft"] = {
            "edit": direct_operation.value,
            "confidence": direct_confidence,
        }
    if selected_tool is not None:
        result["selection"] = "ACQUIRE" if selected_tool else "STOP"
    if tool_result is not None:
        result["tool_result"] = tool_result
    if updated_operation is not None:
        result["belief_update"] = {
            "edit": updated_operation.value,
            "confidence": updated_confidence,
        }
    return result


def _row(
    source: dict[str, Any],
    state: dict[str, Any],
    action: SequentialControllerAction,
    *,
    trajectory_id: str,
    task_id: str,
    split: str,
    step: int,
    selected_tool: bool,
    advantage: float,
) -> dict[str, Any]:
    return {
        "messages": _messages(source, state, action),
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "split": split,
        "stage": action.stage.value,
        "step": step,
        "selected_tool": selected_tool,
        "policy_relative_advantage": advantage,
        "protocol": "draft-conditioned-sequential-controller-v1",
    }


def audit_prompt_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError("sequential SFT only permits train or validation")
        messages = row["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("sequential SFT roles are invalid")
        prompt = next(
            part["text"] for part in messages[1]["content"] if part.get("type") == "text"
        )
        state = json.loads(prompt)
        forbidden = FORBIDDEN_PROMPT_KEYS.intersection(state)
        if forbidden:
            raise ValueError(f"sequential prompt leaks target metadata: {sorted(forbidden)}")
        action_text = next(
            part["text"] for part in messages[2]["content"] if part.get("type") == "text"
        )
        action = SequentialControllerAction.model_validate_json(action_text)
        if action.stage.value != row["stage"] or state["controller_stage"] != row["stage"]:
            raise ValueError("row, prompt, and action stages disagree")
        if row["stage"] in {"DRAFT", "SELECT", "TOOL"} and "tool_result" in state:
            raise ValueError("tool result appears before belief update")
        stage_counts[row["stage"]] += 1
        action_counts[
            action.selection.value
            if action.selection is not None
            else action.executable_action.key
            if action.executable_action is not None
            else action.stage.value
        ] += 1
    return {
        "record_count": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "forbidden_prompt_keys": sorted(FORBIDDEN_PROMPT_KEYS),
        "target_metadata_exposed_to_model": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return _sha256(path)


def build_sequential_sft(
    rollout_jsonl: Path,
    branch_root: Path,
    output_dir: Path,
    *,
    selector_positive_multiplier: int,
    tool_stage_multiplier: int,
    additional_rollout_jsonls: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if selector_positive_multiplier < 1 or tool_stage_multiplier < 1:
        raise ValueError("SFT multipliers must be positive")
    branch_summary = json.loads((branch_root / "summary.json").read_text(encoding="utf-8"))
    if branch_summary.get("test_assets_read") is not False:
        raise ValueError("branch cache violates the frozen-test protocol")
    rollout_inputs = (rollout_jsonl, *additional_rollout_jsonls)
    rollout_rows = []
    for rollout_input in rollout_inputs:
        rollout_rows.extend(load_vlm_sft_rows(rollout_input))
    pairs = index_pairs(rollout_rows)
    input_splits = {
        str(pre["split"])
        for pre, post in pairs
        if str(pre["split"]) == str(post["split"])
    }
    if len(input_splits) != 1:
        raise ValueError("sequential SFT output must be constructed from exactly one split")
    branches = {str(row["example_id"]): row for row in _jsonl(branch_root / "traces.jsonl")}
    if len(branches) != len(pairs):
        raise ValueError("rollout and branch cache counts differ")
    adapter = str(branch_summary["adapter"])
    policy_snapshot = hashlib.sha256(adapter.encode("utf-8")).hexdigest()[:16]
    joint: list[dict[str, Any]] = []
    components = {"draft_terminal": [], "selector": [], "tool_belief": []}
    trajectories: list[SequentialControllerTrajectory] = []
    for pre, post in pairs:
        example_id = str(pre["example_id"])
        branch = branches.get(example_id)
        if branch is None:
            raise ValueError(f"missing branch cache row: {example_id}")
        split = str(pre["split"])
        if split != post["split"] or split != branch["split"] or split not in {"train", "val"}:
            raise ValueError("rollout pair and branch split disagree")
        task_id = str(pre["task_id"])
        if task_id != str(post["task_id"]) or task_id != str(branch["task_id"]):
            raise ValueError("rollout pair and branch task IDs disagree")
        pre_observation = _observation(pre)
        post_observation = _observation(post)
        direct = EditOperation(branch["direct_operation"])
        post_tool = EditOperation(branch["post_tool_operation"])
        target = EditOperation(branch["target_operation"])
        selected = bool(branch["policy_relative_use_tool"])
        advantage = float(branch["policy_relative_advantage"])
        if selected != (advantage > 0.0):
            raise ValueError("policy-relative selection and advantage disagree")
        direct_confidence = _confidence(direct, target)
        post_confidence = _confidence(post_tool, target)
        trajectory_id = f"seq-{example_id}"
        evidence_id = str(pre_observation["evidence_id"])
        actions = [
            SequentialControllerAction(
                stage=ControllerStage.DRAFT,
                draft_edit=direct,
                confidence=direct_confidence,
            ),
            SequentialControllerAction(
                stage=ControllerStage.SELECT,
                selection=(
                    SelectionDecision.ACQUIRE if selected else SelectionDecision.STOP
                ),
                evidence_id=evidence_id if selected else None,
            ),
        ]
        if selected:
            actions.extend(
                [
                    SequentialControllerAction(
                        stage=ControllerStage.TOOL,
                        executable_action=AgentAction(
                            action=AgentActionType.USE_TOOL,
                            tool_call=GeoToolCall(
                                call_id=f"seq-{example_id[:12]}",
                                tool=GeoToolName.RASTER_SEGMENT,
                                inputs={"evidence_id": evidence_id},
                            ),
                        ),
                    ),
                    SequentialControllerAction(
                        stage=ControllerStage.BELIEF_UPDATE,
                        updated_edit=post_tool,
                        confidence=post_confidence,
                    ),
                ]
            )
        chosen = post_tool if selected else direct
        actions.append(
            SequentialControllerAction(
                stage=ControllerStage.TERMINAL,
                executable_action=terminal_action(chosen),
            )
        )
        states = [
            _state(
                pre_observation,
                stage=ControllerStage.DRAFT,
                policy_snapshot=policy_snapshot,
            ),
            _state(
                pre_observation,
                stage=ControllerStage.SELECT,
                policy_snapshot=policy_snapshot,
                direct_operation=direct,
                direct_confidence=direct_confidence,
            ),
        ]
        sources = [pre, pre]
        if selected:
            states.extend(
                [
                    _state(
                        pre_observation,
                        stage=ControllerStage.TOOL,
                        policy_snapshot=policy_snapshot,
                        direct_operation=direct,
                        direct_confidence=direct_confidence,
                        selected_tool=True,
                    ),
                    _state(
                        post_observation,
                        stage=ControllerStage.BELIEF_UPDATE,
                        policy_snapshot=policy_snapshot,
                        direct_operation=direct,
                        direct_confidence=direct_confidence,
                        selected_tool=True,
                        tool_result=post_observation["tool_result"],
                    ),
                ]
            )
            sources.extend([pre, post])
        states.append(
            _state(
                post_observation if selected else pre_observation,
                stage=ControllerStage.TERMINAL,
                policy_snapshot=policy_snapshot,
                direct_operation=direct,
                direct_confidence=direct_confidence,
                selected_tool=selected,
                tool_result=post_observation["tool_result"] if selected else None,
                updated_operation=post_tool if selected else None,
                updated_confidence=post_confidence if selected else None,
            )
        )
        sources.append(post if selected else pre)
        trajectory_rows = [
            _row(
                source,
                state,
                action,
                trajectory_id=trajectory_id,
                task_id=task_id,
                split=split,
                step=step,
                selected_tool=selected,
                advantage=advantage,
            )
            for step, (source, state, action) in enumerate(
                zip(sources, states, actions, strict=True)
            )
        ]
        joint.extend(trajectory_rows)
        components["draft_terminal"].extend(
            row for row in trajectory_rows if row["stage"] in {"DRAFT", "TERMINAL"}
        )
        components["selector"].append(
            next(row for row in trajectory_rows if row["stage"] == "SELECT")
        )
        components["tool_belief"].extend(
            row for row in trajectory_rows if row["stage"] in {"TOOL", "BELIEF_UPDATE"}
        )
        trajectories.append(
            SequentialControllerTrajectory(
                trajectory_id=trajectory_id,
                task_id=task_id,
                split=split,
                policy_snapshot=policy_snapshot,
                selected_tool=selected,
                policy_relative_advantage=advantage,
                direct_operation=direct,
                post_tool_operation=post_tool,
                chosen_operation=chosen,
                target_operation=target,
                direct_utility=float(branch["direct_utility"]),
                post_tool_utility=float(branch["post_tool_utility"]),
                total_cost=float(branch["tool_cost"]) if selected else 0.0,
                actions=actions,
            )
        )

    selector_balanced = []
    for row in components["selector"]:
        copies = selector_positive_multiplier if row["selected_tool"] else 1
        for copy_index in range(copies):
            selector_balanced.append({**copy.deepcopy(row), "augmentation_copy": copy_index})
    joint_curriculum = list(joint)
    for row in components["selector"]:
        if row["selected_tool"]:
            for copy_index in range(1, selector_positive_multiplier):
                joint_curriculum.append(
                    {**copy.deepcopy(row), "augmentation_copy": copy_index}
                )
    for row in components["tool_belief"]:
        for copy_index in range(1, tool_stage_multiplier):
            joint_curriculum.append({**copy.deepcopy(row), "augmentation_copy": copy_index})

    audit = audit_prompt_contract(joint)
    output_dir.mkdir(parents=True)
    files = {
        "joint_sft": joint,
        "joint_curriculum_sft": joint_curriculum,
        "draft_terminal_sft": components["draft_terminal"],
        "selector_sft": components["selector"],
        "selector_balanced_sft": selector_balanced,
        "tool_belief_sft": components["tool_belief"],
    }
    file_summaries = {}
    for name, rows in files.items():
        path = output_dir / f"{name}.jsonl"
        file_summaries[name] = {
            "path": str(path.resolve()),
            "records": len(rows),
            "sha256": _write_jsonl(path, rows),
            "audit": audit_prompt_contract(rows),
        }
    trajectory_path = output_dir / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.model_dump_json(exclude_none=True) + "\n")
    summary = {
        "schema_version": "draft-conditioned-sequential-controller-sft-v1",
        "split": trajectories[0].split,
        "trajectory_count": len(trajectories),
        "task_count": len({trajectory.task_id for trajectory in trajectories}),
        "selected_tool_count": sum(trajectory.selected_tool for trajectory in trajectories),
        "selected_tool_rate": sum(trajectory.selected_tool for trajectory in trajectories)
        / len(trajectories),
        "policy_snapshot": policy_snapshot,
        "decision_order": "DRAFT-SELECT-TOOL-BELIEF_UPDATE-TERMINAL",
        "confidence_supervision": "smoothed-terminal-correctness-0.9-or-0.1",
        "selector_positive_multiplier": selector_positive_multiplier,
        "tool_stage_multiplier": tool_stage_multiplier,
        "joint_audit": audit,
        "files": file_summaries,
        "trajectories": {
            "path": str(trajectory_path.resolve()),
            "records": len(trajectories),
            "sha256": _sha256(trajectory_path),
        },
        "sources": {
            "rollouts": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in rollout_inputs
            ],
            "branch_summary": {
                "path": str((branch_root / "summary.json").resolve()),
                "sha256": _sha256(branch_root / "summary.json"),
            },
            "branch_trace_sha256": branch_summary["trace_sha256"],
        },
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("branch_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--additional-rollout-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional split manifest(s) to combine with the primary rollout input.",
    )
    parser.add_argument("--selector-positive-multiplier", type=int, default=3)
    parser.add_argument("--tool-stage-multiplier", type=int, default=3)
    args = parser.parse_args()
    result = build_sequential_sft(
        args.rollout_jsonl,
        args.branch_root,
        args.output_dir,
        selector_positive_multiplier=args.selector_positive_multiplier,
        tool_stage_multiplier=args.tool_stage_multiplier,
        additional_rollout_jsonls=tuple(args.additional_rollout_jsonl),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

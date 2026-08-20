"""Build sparse, grounded tool-control supervision from frozen validation protocol data."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.tool_belief_data import ToolBeliefSequenceExample
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.models import EditOperation

TOOL_SYSTEM_PROMPT = (
    "You are the ActiveMap sparse tool controller. Choose exactly one JSON action. "
    "Use a tool only when its expected map-quality or safety gain exceeds its cost. "
    "USE_TOOL may inspect only an item listed in selected_evidence_ids and a tool "
    "listed in available_tools. For USE_TOOL, emit evidence_index as the zero-based "
    "index into selected_evidence_ids. The runtime resolves that index to the opaque "
    "evidence handle and creates the call id. Valid forms are "
    '{"action":"USE_TOOL","tool_call":{"tool":'
    '"IMAGE_QUALITY|TEMPORAL_CHANGE","inputs":{"evidence_index":0},'
    '"parameters":{}}}; {"action":"COMMIT","edit":"ADD|DELETE|RESHAPE"}; or '
    '{"action":"REJECT"}. Return JSON only.'
)


def policy_action_payload(action: AgentAction, observation: AgentObservation) -> dict[str, Any]:
    """Serialize a trainable action without sample-specific tool identifiers.

    Evidence handles and tool call IDs are executor details, not semantic action
    choices.  Asking a language model to reproduce both makes a correct tool
    decision look invalid on a new episode.  Tool actions therefore point into
    the observation's selected-evidence list; the rollout executor resolves the
    pointer back to an opaque handle before validation and execution.
    """

    payload = action.model_dump(mode="json", exclude_none=True)
    if action.action != AgentActionType.USE_TOOL:
        return payload
    if action.tool_call is None:
        raise ValueError("USE_TOOL action lacks tool_call")
    try:
        evidence_id = str(action.tool_call.inputs["evidence_id"])
        evidence_index = list(observation.selected_evidence_ids).index(evidence_id)
    except (KeyError, ValueError) as exc:
        raise ValueError("tool action evidence is absent from selected_evidence_ids") from exc
    tool_call = dict(payload["tool_call"])
    tool_call.pop("call_id", None)
    inputs = dict(tool_call.get("inputs", {}))
    inputs.pop("evidence_id", None)
    inputs["evidence_index"] = evidence_index
    tool_call["inputs"] = inputs
    payload["tool_call"] = tool_call
    return payload


def policy_action_json(action: AgentAction, observation: AgentObservation) -> str:
    return json.dumps(policy_action_payload(action, observation), separators=(",", ":"))


def terminal_reward(target: EditOperation, prediction: EditOperation) -> float:
    if target == prediction:
        return 1.0
    if target == EditOperation.KEEP:
        return -1.0
    if prediction == EditOperation.KEEP:
        return -0.75
    return -0.5


def terminal_action(operation: EditOperation) -> AgentAction:
    if operation == EditOperation.KEEP:
        return AgentAction(action=AgentActionType.REJECT)
    return AgentAction(action=AgentActionType.COMMIT, edit=operation)


def select_sparse_tool_stage(
    rows: list[dict[str, Any]], *, minimum_gain: float = 0.0
) -> tuple[int, list[float], list[EditOperation]]:
    """Select the cheapest utility-maximizing tool prefix, gated against stage zero."""

    if len(rows) != 3:
        raise ValueError("a tool sequence requires exactly three detail rows")
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in ordered] != [1, 2, 3]:
        raise ValueError("tool detail rows must contain steps 1, 2, 3")
    sequence_ids = {str(row["sequence_id"]) for row in ordered}
    targets = {str(row["target"]) for row in ordered}
    if len(sequence_ids) != 1 or len(targets) != 1:
        raise ValueError("tool detail rows disagree on sequence or target")
    target = EditOperation(next(iter(targets)))
    predictions = [
        EditOperation(str(ordered[0]["baseline"])),
        *[EditOperation(str(row["paired"])) for row in ordered],
    ]
    costs = [0.0, *[float(row["spent_cost"]) for row in ordered]]
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in costs[1:]):
        raise ValueError("tool prefix costs must be finite and positive")
    if any(right <= left for left, right in zip(costs[:-1], costs[1:], strict=True)):
        raise ValueError("tool prefix costs must increase strictly")
    utilities = [
        terminal_reward(target, prediction) - cost
        for prediction, cost in zip(predictions, costs, strict=True)
    ]
    selected = max(range(4), key=lambda stage: (utilities[stage], -costs[stage]))
    if utilities[selected] < utilities[0] + minimum_gain:
        selected = 0
    return selected, utilities, predictions


def _compact_result(result: GeoToolResult) -> GeoToolResult:
    fields = {
        GeoToolName.IMAGE_QUALITY: (
            "valid_fraction",
            "saturation_fraction",
            "sharpness",
            "source_dtype",
        ),
        GeoToolName.TEMPORAL_CHANGE: (
            "threshold",
            "changed_fraction",
            "source_dtype",
        ),
    }[result.tool]
    outputs = {key: result.outputs[key] for key in fields if key in result.outputs}
    stats = result.outputs.get("stats")
    if isinstance(stats, dict):
        outputs["stats"] = {
            key: stats[key] for key in ("mean", "std", "p95") if key in stats
        }
    return result.model_copy(update={"outputs": outputs, "artifacts": []})


def _tool_action(
    sequence_id: str, evidence_id: str, tool: GeoToolName, ordinal: int
) -> AgentAction:
    return AgentAction(
        action=AgentActionType.USE_TOOL,
        tool_call=GeoToolCall(
            call_id=f"sft-{sequence_id[:12]}-{ordinal}",
            tool=tool,
            inputs={"evidence_id": evidence_id},
        ),
    )


def _observation(
    sequence: ToolBeliefSequenceExample,
    *,
    belief: Any,
    history: list[GeoToolResult],
    spent_cost: float,
    initial_budget: float,
) -> AgentObservation:
    return AgentObservation(
        task_id=sequence.episode_id,
        split=sequence.split,
        step=len(history),
        initial_budget=initial_budget,
        remaining_budget=max(initial_budget - spent_cost, 0.0),
        spent_cost=spent_cost,
        selected_evidence_ids=[step.evidence_id for step in sequence.steps],
        belief=belief,
        candidates=[],
        available_tools=[GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE],
        tool_history=history,
    )


def build_sparse_tool_examples(
    sequences: list[ToolBeliefSequenceExample],
    detail_rows: list[dict[str, Any]],
    *,
    minimum_gain: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    splits = {sequence.split for sequence in sequences}
    if not splits or not splits <= {"train", "val", "test"}:
        raise ValueError(f"invalid sparse-tool splits: {sorted(splits)}")
    test_assets_read = "test" in splits
    if test_assets_read:
        if splits != {"test"}:
            raise ValueError("sparse-tool labels must not mix test with train/val")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["sequence_id"])].append(row)
    if set(grouped) != {sequence.sequence_id for sequence in sequences}:
        missing = sorted({sequence.sequence_id for sequence in sequences} - set(grouped))
        extra = sorted(set(grouped) - {sequence.sequence_id for sequence in sequences})
        raise ValueError(f"sequence/detail mismatch: missing={missing[:3]}, extra={extra[:3]}")

    sft_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    stage_counts: Counter[int] = Counter()
    gains: list[float] = []
    for sequence in sequences:
        selected_stage, utilities, predictions = select_sparse_tool_stage(
            grouped[sequence.sequence_id], minimum_gain=minimum_gain
        )
        stage_counts[selected_stage] += 1
        gains.append(utilities[selected_stage] - utilities[0])
        total_budget = sum(
            step.quality_result.cost + step.temporal_result.cost
            for step in sequence.steps
        )
        belief = sequence.initial_belief
        history: list[GeoToolResult] = []
        spent_cost = 0.0
        ordinal = 0

        first_observation = _observation(
            sequence,
            belief=belief,
            history=history,
            spent_cost=spent_cost,
            initial_budget=total_budget,
        )
        if selected_stage:
            first_chosen = _tool_action(
                sequence.sequence_id,
                sequence.steps[0].evidence_id,
                GeoToolName.IMAGE_QUALITY,
                ordinal,
            )
            first_rejected = terminal_action(predictions[0])
            first_chosen_utility = utilities[selected_stage]
            first_rejected_utility = utilities[0]
        else:
            first_chosen = terminal_action(sequence.gt_edit)
            first_rejected = _tool_action(
                sequence.sequence_id,
                sequence.steps[0].evidence_id,
                GeoToolName.IMAGE_QUALITY,
                ordinal,
            )
            first_chosen_utility = utilities[0]
            first_rejected_utility = utilities[1]
        if first_chosen_utility > first_rejected_utility:
            preference_rows.append(
                {
                    "system": TOOL_SYSTEM_PROMPT,
                    "prompt": (
                        f"{TOOL_SYSTEM_PROMPT}\n"
                        f"{first_observation.model_dump_json(exclude_none=True)}\nAction:"
                    ),
                    "chosen": policy_action_json(first_chosen, first_observation),
                    "rejected": policy_action_json(first_rejected, first_observation),
                    "chosen_utility": first_chosen_utility,
                    "rejected_utility": first_rejected_utility,
                    "trajectory_id": sequence.sequence_id,
                    "split": sequence.split,
                    "protocol": "post-acquisition-sparse-tool-controller-v1",
                }
            )

        for stage in range(selected_stage):
            step = sequence.steps[stage]
            for tool, result in (
                (GeoToolName.IMAGE_QUALITY, step.quality_result),
                (GeoToolName.TEMPORAL_CHANGE, step.temporal_result),
            ):
                observation = _observation(
                    sequence,
                    belief=belief,
                    history=history,
                    spent_cost=spent_cost,
                    initial_budget=total_budget,
                )
                action = _tool_action(
                    sequence.sequence_id, step.evidence_id, tool, ordinal
                )
                if ordinal > 0:
                    rejected = terminal_action(predictions[stage])
                    preference_rows.append(
                        {
                            "system": TOOL_SYSTEM_PROMPT,
                            "prompt": (
                                f"{TOOL_SYSTEM_PROMPT}\n"
                                f"{observation.model_dump_json(exclude_none=True)}\nAction:"
                            ),
                            "chosen": policy_action_json(action, observation),
                            "rejected": policy_action_json(rejected, observation),
                            "chosen_utility": utilities[selected_stage],
                            "rejected_utility": utilities[stage],
                            "trajectory_id": sequence.sequence_id,
                            "split": sequence.split,
                            "protocol": "post-acquisition-sparse-tool-controller-v1",
                        }
                    )
                sft_rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": observation.model_dump_json(exclude_none=True),
                            },
                            {
                                "role": "assistant",
                                "content": policy_action_json(action, observation),
                            },
                        ],
                        "trajectory_id": sequence.sequence_id,
                        "step": observation.step,
                        "oracle_tool_stage": selected_stage,
                        "utility_gain": gains[-1],
                        "protocol": "post-acquisition-sparse-tool-controller-v1",
                    }
                )
                compact = _compact_result(result)
                history = [*history, compact]
                spent_cost += compact.cost
                ordinal += 1
                if tool == GeoToolName.TEMPORAL_CHANGE:
                    belief = step.cumulative_target_belief

        terminal_observation = _observation(
            sequence,
            belief=belief,
            history=history,
            spent_cost=spent_cost,
            initial_budget=total_budget,
        )
        terminal = terminal_action(sequence.gt_edit)
        sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": terminal_observation.model_dump_json(exclude_none=True),
                    },
                    {
                        "role": "assistant",
                        "content": policy_action_json(terminal, terminal_observation),
                    },
                ],
                "trajectory_id": sequence.sequence_id,
                "step": terminal_observation.step,
                "oracle_tool_stage": selected_stage,
                "utility_gain": gains[-1],
                "protocol": "post-acquisition-sparse-tool-controller-v1",
            }
        )
        if selected_stage > 0:
            if selected_stage < len(sequence.steps):
                rejected = _tool_action(
                    sequence.sequence_id,
                    sequence.steps[selected_stage].evidence_id,
                    GeoToolName.IMAGE_QUALITY,
                    ordinal,
                )
                rejected_utility = utilities[selected_stage + 1]
            else:
                wrong_edit = (
                    EditOperation.ADD
                    if sequence.gt_edit == EditOperation.KEEP
                    else EditOperation.KEEP
                )
                rejected = terminal_action(wrong_edit)
                rejected_utility = terminal_reward(sequence.gt_edit, wrong_edit) - spent_cost
            if utilities[selected_stage] > rejected_utility:
                preference_rows.append(
                    {
                        "system": TOOL_SYSTEM_PROMPT,
                        "prompt": (
                            f"{TOOL_SYSTEM_PROMPT}\n"
                            f"{terminal_observation.model_dump_json(exclude_none=True)}\n"
                            "Action:"
                        ),
                        "chosen": policy_action_json(terminal, terminal_observation),
                        "rejected": policy_action_json(rejected, terminal_observation),
                        "chosen_utility": utilities[selected_stage],
                        "rejected_utility": rejected_utility,
                        "trajectory_id": sequence.sequence_id,
                        "split": sequence.split,
                        "protocol": "post-acquisition-sparse-tool-controller-v1",
                    }
                )

    positive = sum(count for stage, count in stage_counts.items() if stage > 0)
    summary = {
        "schema_version": "sparse-grounded-tool-sft-v1",
        "controller_scope": "post_acquisition",
        "sequence_count": len(sequences),
        "sft_count": len(sft_rows),
        "preference_count": len(preference_rows),
        "oracle_stage_counts": {
            str(stage): stage_counts[stage] for stage in range(4)
        },
        "tool_positive_sequence_count": positive,
        "tool_positive_rate": positive / max(len(sequences), 1),
        "mean_oracle_utility_gain": math.fsum(gains) / max(len(gains), 1),
        "minimum_gain": minimum_gain,
        "test_assets_read": test_assets_read,
    }
    return sft_rows, preference_rows, summary


def write_sparse_tool_dataset(
    sequences: list[ToolBeliefSequenceExample],
    detail_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    minimum_gain: float = 1e-6,
) -> dict[str, Any]:
    sft_rows, preference_rows, summary = build_sparse_tool_examples(
        sequences, detail_rows, minimum_gain=minimum_gain
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("sft.jsonl", sft_rows), ("preferences.jsonl", preference_rows)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary

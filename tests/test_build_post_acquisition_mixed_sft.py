import json

from activemap.agent.records import AgentAction, AgentActionType, AgentBelief, AgentObservation
from activemap.agent.tool_sft import TOOL_SYSTEM_PROMPT, policy_action_json
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from scripts.build_post_acquisition_mixed_sft import (
    build_split,
    parse_action_weights,
    rebalance_train_rows,
)


def _row(action: AgentAction, *, split: str = "train") -> dict[str, object]:
    observation = AgentObservation(
        task_id=f"task-{action.key}",
        split=split,
        step=1,
        initial_budget=1.5,
        remaining_budget=1.0,
        spent_cost=0.5,
        selected_evidence_ids=["evidence-a"],
        belief=AgentBelief(
            edit_probabilities=[0.8, 0.1, 0.05, 0.05],
            confidence=0.8,
            geometry_delta=[0.0] * 8,
            uncertainty=0.2,
        ),
        candidates=[],
        available_tools=[GeoToolName.IMAGE_QUALITY],
    )
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": observation.model_dump_json(exclude_none=True)},
            {"role": "assistant", "content": policy_action_json(action, observation)},
        ],
        "trajectory_id": f"traj-{action.key}",
        "step": 1,
    }


def test_post_acquisition_rebalance_keeps_all_action_support():
    rows = [
        *[_row(AgentAction(action=AgentActionType.REJECT)) for _ in range(4)],
        _row(AgentAction(action=AgentActionType.COMMIT, edit="ADD")),
        _row(
            AgentAction(
                action=AgentActionType.USE_TOOL,
                tool_call=GeoToolCall(
                    call_id="call-a",
                    tool=GeoToolName.IMAGE_QUALITY,
                    inputs={"evidence_id": "evidence-a"},
                ),
            )
        ),
    ]
    weights = parse_action_weights("REJECT=1,COMMIT=1,USE_TOOL=1")
    balanced, summary = rebalance_train_rows(rows, action_weights=weights, seed=7)

    assert len(balanced) == 12
    assert summary["output_action_counts"] == {"COMMIT": 4, "REJECT": 4, "USE_TOOL": 4}
    assert all(json.loads(row["messages"][1]["content"])["split"] == "train" for row in balanced)


def test_post_acquisition_validation_is_never_rebalanced():
    terminal = [_row(AgentAction(action=AgentActionType.REJECT), split="val")]
    tools = [
        _row(
            AgentAction(
                action=AgentActionType.USE_TOOL,
                tool_call=GeoToolCall(
                    call_id="call-a",
                    tool=GeoToolName.IMAGE_QUALITY,
                    inputs={"evidence_id": "evidence-a"},
                ),
            ),
            split="val",
        )
    ]
    rows, summary = build_split(
        terminal, tools, split="val", action_weights=None, seed=7
    )

    assert len(rows) == 2
    assert summary["balance"]["enabled"] is False

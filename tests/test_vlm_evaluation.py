from activemap.agent.records import AgentAction
from activemap.agent.vlm_evaluation import (
    majority_operation,
    message_text,
    multiclass_metrics,
    operation_from_action,
    realized_tool_utility,
)
from activemap.models import EditOperation


def test_message_text_reads_multimodal_text_part():
    assert message_text(
        {"content": [{"type": "image", "image": "x"}, {"type": "text", "text": "state"}]}
    ) == "state"


def test_terminal_action_maps_reject_to_keep():
    action = AgentAction.model_validate({"action": "REJECT"})
    assert operation_from_action(action) == EditOperation.KEEP


def test_majority_operation_is_deterministic():
    assert majority_operation(["ADD", "DELETE", "ADD"]) == EditOperation.ADD
    assert majority_operation(["DELETE", "ADD"]) == EditOperation.ADD


def test_multiclass_metrics_includes_supported_zero_recall_class():
    metrics = multiclass_metrics(["A", "B"], ["A", "A"], ["A", "B"])
    assert metrics["accuracy"] == 0.5
    assert metrics["per_class"]["B"]["recall"] == 0.0


def test_realized_tool_utility_scores_executed_operation_and_cost():
    assert realized_tool_utility(EditOperation.ADD, EditOperation.ADD, 0.25) == 0.75
    assert realized_tool_utility(EditOperation.ADD, EditOperation.DELETE, 0.25) == -0.75

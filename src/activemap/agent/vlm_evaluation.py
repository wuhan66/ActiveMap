"""Metrics and parsing helpers for the visual semantic-tool policy."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from activemap.agent.records import AgentAction, AgentActionType
from activemap.models import EditOperation


def realized_tool_utility(
    target: EditOperation, prediction: EditOperation, cost: float
) -> float:
    """Score the operation actually executed after paying the realized tool cost."""

    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("tool cost must be finite and non-negative")
    from activemap.agent.tool_sft import terminal_reward

    return terminal_reward(target, prediction) - cost


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or a multimodal list")
    texts = [part.get("text") for part in content if part.get("type") == "text"]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("message must contain exactly one text part")
    return texts[0]


def extract_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generated action is not a JSON object")
    return value


def operation_from_action(action: AgentAction) -> EditOperation | None:
    if action.action == AgentActionType.REJECT:
        return EditOperation.KEEP
    if action.action == AgentActionType.COMMIT:
        return action.edit
    return None


def action_class(action: AgentAction) -> str:
    if action.action == AgentActionType.COMMIT:
        assert action.edit is not None
        return f"COMMIT:{action.edit.value}"
    return action.action.value


def majority_operation(values: list[str | EditOperation]) -> EditOperation:
    operations = [
        value if isinstance(value, EditOperation) else EditOperation(value)
        for value in values
    ]
    if not operations:
        raise ValueError("semantic operation list is empty")
    counts = Counter(operations)
    order = list(EditOperation)
    return max(order, key=lambda operation: (counts[operation], -order.index(operation)))


def multiclass_metrics(
    targets: list[str], predictions: list[str], labels: list[str]
) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction counts differ")
    per_class = {}
    f1_values = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions, strict=True))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions, strict=True))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions, strict=True))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(target == label for target in targets),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": sum(t == p for t, p in zip(targets, predictions, strict=True))
        / max(len(targets), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_class": per_class,
    }

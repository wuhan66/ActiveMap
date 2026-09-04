#!/usr/bin/env python3
"""Convert selector traces into a terminal-evidence-only writeback diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_task_id
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def _terminal_action(operation: EditOperation) -> str:
    return "REJECT" if operation == EditOperation.KEEP else f"COMMIT:{operation.value}"


def load_initial_states(path: Path) -> dict[tuple[str, float], SelectorSample]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sample = SelectorSample.model_validate_json(line)
        if sample.split != "val" or int(sample.metadata.get("oracle_step", -1)) != 0:
            continue
        key = (str(sample.metadata["source_episode"]), float(sample.metadata["budget"]))
        if key in result:
            raise ValueError(f"duplicate initial state: {key}")
        result[key] = sample
    if not result:
        raise ValueError("no validation step-0 states")
    return result


def convert(trace: dict[str, Any], sample: SelectorSample) -> dict[str, Any]:
    selected = [str(item) for item in trace["selected_evidence_ids"]]
    if not selected:
        raise ValueError("trace has no selected evidence")
    terminal_evidence = selected[-1]
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict) or terminal_evidence not in outcomes:
        raise ValueError(f"missing executable outcome for {terminal_evidence}")
    predicted = EditOperation(str(outcomes[terminal_evidence]["predicted_operation"]))
    target = EditOperation(str(sample.metadata["gt_edit"]))
    return {
        "task_id": public_task_id(str(trace["source_episode"])),
        "aoi_id": str(trace["aoi_id"]),
        "budget": float(trace["budget"]),
        "target": _terminal_action(target),
        "prediction": _terminal_action(predicted),
        "selected_evidence_ids": [terminal_evidence],
        "semantic_tool_called": False,
        "semantic_tool_cost": 0.0,
        "policy_utility": float(trace["quality_cost_utility"]),
        "spent_cost": float(trace["spent_cost"]),
        "source_example_id": str(trace["sample_id"]),
        "source_policy": f"{trace['policy']}_terminal_evidence_only",
        "split": "val",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    states = load_initial_states(args.states)
    traces = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for trace in traces:
        key = (str(trace["source_episode"]), float(trace["budget"]))
        if key not in states:
            raise ValueError(f"trace has no matching state: {key}")
        rows.append(convert(trace, states[key]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()

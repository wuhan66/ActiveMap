#!/usr/bin/env python3
"""Create validation-only candidate-aligned writeback rows from frozen SN7 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_key(source_episode: Any, budget: Any) -> tuple[str, float]:
    return str(source_episode), round(float(budget), 6)


def terminal_action(operation: EditOperation) -> str:
    return "REJECT" if operation == EditOperation.KEEP else f"COMMIT:{operation.value}"


def target_operation(row: dict[str, Any]) -> EditOperation:
    target = str(row["target"])
    if target == "REJECT":
        return EditOperation.KEEP
    if not target.startswith("COMMIT:"):
        raise ValueError(f"unsupported target action: {target!r}")
    return EditOperation(target.removeprefix("COMMIT:"))


def align_rollout(sample: SelectorSample, row: dict[str, Any]) -> dict[str, Any]:
    if sample.split != "val" or row.get("split") != "val":
        raise ValueError("candidate-alignment conversion is validation-only")
    if row.get("test_assets_read") is not False:
        raise ValueError("candidate-alignment conversion forbids test provenance")
    source_selected = [str(value) for value in row["selected_evidence_ids"]]
    initial_id = str(
        row.get("initial_evidence_id") or sample.metadata.get("initial_evidence_id")
    )
    acquired_id = row.get("selected_extra_evidence_id")
    writeback_evidence_id = str(acquired_id) if acquired_id is not None else initial_id
    if writeback_evidence_id not in source_selected:
        raise ValueError("writeback evidence is absent from the source rollout")
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict) or writeback_evidence_id not in outcomes:
        raise ValueError("writeback evidence is absent from executable outcomes")
    outcome = outcomes[writeback_evidence_id]
    if not isinstance(outcome, dict) or "predicted_operation" not in outcome:
        raise ValueError("writeback evidence has no predicted operation")
    prediction = terminal_action(EditOperation(str(outcome["predicted_operation"])))
    standalone_belief = CounterfactualBeliefUpdater(sample).fuse([writeback_evidence_id])
    standalone_prediction = terminal_action(standalone_belief.predicted_edit)
    return {
        **row,
        "schema_version": "sn7-counterfactual-aligned-rollout-v1",
        "prediction": prediction,
        "selected_evidence_ids": [writeback_evidence_id],
        "source_selected_evidence_ids": source_selected,
        "writeback_evidence_mode": "counterfactual_aligned",
        "source_fused_prediction": row["prediction"],
        "standalone_belief_prediction": standalone_prediction,
        "counterfactual_action_override": prediction != standalone_prediction,
        "test_assets_read": False,
    }


def convert(
    states_path: Path,
    rollouts_path: Path,
    output_dir: Path,
    *,
    operation: EditOperation | None,
    require_acquisition: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    states: dict[tuple[str, float], tuple[SelectorSample, str]] = {}
    for line_number, line in enumerate(states_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        sample = SelectorSample.model_validate_json(line)
        if sample.split != "val" or int(sample.metadata.get("oracle_step", -1)) != 0:
            continue
        key = state_key(sample.metadata.get("source_episode"), sample.metadata.get("budget"))
        if key in states:
            raise ValueError(f"duplicate validation state at line {line_number}: {key}")
        states[key] = sample, line
    if not states:
        raise ValueError("no validation step-0 states")

    selected: list[tuple[SelectorSample, str, dict[str, Any]]] = []
    for line_number, line in enumerate(rollouts_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError(f"invalid validation provenance at rollout line {line_number}")
        key = state_key(row.get("source_episode"), row.get("budget"))
        if key not in states:
            raise ValueError(f"rollout has no matching step-0 state at line {line_number}")
        if require_acquisition and not bool(row.get("selected_extra_evidence")):
            continue
        if operation is not None and target_operation(row) != operation:
            continue
        sample, state_line = states[key]
        state_target = EditOperation(str(sample.metadata["gt_edit"]))
        if state_target != target_operation(row):
            raise ValueError(f"state/rollout target mismatch at line {line_number}")
        selected.append((sample, state_line, align_rollout(sample, row)))
    if not selected:
        raise ValueError("no rollout rows matched the requested alignment slice")

    output_dir.mkdir(parents=True)
    output_states = output_dir / "states.jsonl"
    output_rollouts = output_dir / "rollouts.jsonl"
    output_states.write_text("".join(line + "\n" for _, line, _ in selected), encoding="utf-8")
    output_rollouts.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for _, _, row in selected),
        encoding="utf-8",
    )
    rows = [row for _, _, row in selected]
    receipt = {
        "schema_version": "sn7-counterfactual-aligned-conversion-receipt-v1",
        "source_states": str(states_path.resolve()),
        "source_states_sha256": sha256(states_path),
        "source_rollouts": str(rollouts_path.resolve()),
        "source_rollouts_sha256": sha256(rollouts_path),
        "output_states": str(output_states.resolve()),
        "output_states_sha256": sha256(output_states),
        "output_rollouts": str(output_rollouts.resolve()),
        "output_rollouts_sha256": sha256(output_rollouts),
        "operation": operation.value if operation is not None else None,
        "require_acquisition": require_acquisition,
        "row_count": len(rows),
        "source_episode_count": len({str(row["source_episode"]) for row in rows}),
        "action_override_count": sum(bool(row["counterfactual_action_override"]) for row in rows),
        "split": "val",
        "test_assets_read": False,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--operation", choices=tuple(item.value for item in EditOperation))
    parser.add_argument("--require-acquisition", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.states,
                args.rollouts,
                args.output_dir,
                operation=EditOperation(args.operation) if args.operation else None,
                require_acquisition=args.require_acquisition,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

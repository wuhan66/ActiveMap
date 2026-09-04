#!/usr/bin/env python3
"""Build paired V5 direct, selected, and forced-acquisition rollouts.

The V5 matched evaluation keeps the initial minimum-cost evidence fixed for
all policies. The learned policy may select one additional candidate only
when its frozen STOP-aware selector scores it above STOP. The forced control
always acquires that same highest-scoring candidate, even when STOP would win.
It is a preregistered cost control, not a fifth factorial factor. This script
writes decisions, not map metrics: each output must subsequently pass through
``evaluate_agent_map_writeback.py`` for real typed vector writeback.

It is deliberately fail-closed.  A V5 rollout cannot be built before the
three-seed candidate-headroom authorization, and a validation state file must
match the authorization receipt for its updater seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from activemap.agent.identifiers import public_task_id
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

AUTHORIZATION_SCHEMA = "sn7-v5b-three-seed-headroom-authorization-v1"


class ActionScorer(Protocol):
    def action_scores(self, sample: SelectorSample) -> np.ndarray: ...


class EvidenceValueActionScorer:
    """Adapt a Value Head's candidate-plus-STOP scores to the V5 interface."""

    def __init__(self, predictor: Callable[[SelectorSample], np.ndarray]) -> None:
        self.predictor = predictor

    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        return self.predictor(sample)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_action(operation: EditOperation) -> str:
    return "REJECT" if operation == EditOperation.KEEP else f"COMMIT:{operation.value}"


def read_initial_states(path: Path, *, split: str) -> list[SelectorSample]:
    if split not in {"train", "val"}:
        raise ValueError("V5 matched rollouts are train/validation-only")
    states: list[SelectorSample] = []
    identities: set[tuple[str, float]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            state = SelectorSample.model_validate_json(line)
            if state.split != split or int(state.metadata.get("oracle_step", -1)) != 0:
                continue
            source_episode = str(state.metadata.get("source_episode", ""))
            budget = float(state.metadata.get("budget", 0.0))
            if not source_episode or budget <= 0.0:
                raise ValueError(f"state lacks source episode or budget at {path}:{line_number}")
            identity = (source_episode, budget)
            if identity in identities:
                raise ValueError(f"duplicate V5 initial state {identity}")
            selected = state.metadata.get("selected_evidence_ids")
            if not isinstance(selected, list) or len(selected) != 1:
                raise ValueError(
                    "V5 matched protocol requires exactly one fixed initial evidence item"
                )
            if str(selected[0]) == "":
                raise ValueError("initial evidence identifier cannot be empty")
            identities.add(identity)
            states.append(state)
    if not states:
        raise ValueError(f"no {split} initial selector states in {path}")
    return states


def _target_action(sample: SelectorSample) -> str:
    target = EditOperation(str(sample.metadata["gt_edit"]))
    return terminal_action(target)


def _fused_terminal_action(sample: SelectorSample, evidence_ids: list[str]) -> tuple[str, float]:
    belief = CounterfactualBeliefUpdater(sample).fuse(evidence_ids)
    return terminal_action(belief.predicted_edit), float(belief.confidence)


def _counterfactual_terminal_action(sample: SelectorSample, evidence_id: str) -> str:
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict) or evidence_id not in outcomes:
        raise ValueError(f"selector sample lacks executable outcome for {evidence_id!r}")
    outcome = outcomes[evidence_id]
    if not isinstance(outcome, dict) or "predicted_operation" not in outcome:
        raise ValueError(f"executable outcome lacks predicted operation for {evidence_id!r}")
    return terminal_action(EditOperation(str(outcome["predicted_operation"])))


def build_policy_rollout(
    sample: SelectorSample,
    *,
    policy: str,
    scorer: ActionScorer | None,
    writeback_evidence_mode: str = "all",
) -> dict[str, Any]:
    if policy not in {"direct", "selected", "forced"}:
        raise ValueError(f"unsupported V5 policy {policy!r}")
    if writeback_evidence_mode not in {"all", "counterfactual_aligned"}:
        raise ValueError(f"unsupported V5 writeback evidence mode {writeback_evidence_mode!r}")
    initial_evidence = [str(value) for value in sample.metadata["selected_evidence_ids"]]
    if len(initial_evidence) != 1:
        raise ValueError("V5 matched protocol requires exactly one initial evidence item")

    selected_evidence = list(initial_evidence)
    selector_scores: list[float] | None = None
    selected_extra_evidence_id: str | None = None
    extra_evidence_cost = 0.0
    if policy in {"selected", "forced"}:
        if scorer is None:
            raise ValueError(f"{policy} policy requires a frozen selector scorer")
        raw_scores = np.asarray(scorer.action_scores(sample), dtype=np.float64)
        expected_shape = (len(sample.evidence_ids) + 1,)
        if raw_scores.shape != expected_shape or not np.all(np.isfinite(raw_scores)):
            raise ValueError(
                f"selector score shape must be {expected_shape}; received {raw_scores.shape}"
            )
        candidate_index = int(np.argmax(raw_scores[:-1]))
        stop_score = float(raw_scores[-1])
        selector_scores = raw_scores.tolist()
        # The forced control differs only at the STOP decision. Whenever the
        # selected policy acquires, both policies therefore use the same item.
        if policy == "forced" or float(raw_scores[candidate_index]) > stop_score:
            selected_extra_evidence_id = str(sample.evidence_ids[candidate_index])
            selected_evidence.append(selected_extra_evidence_id)
            extra_evidence_cost = float(sample.evidence_costs[candidate_index])

    source_selected_evidence = list(selected_evidence)
    source_fused_prediction, source_fused_confidence = _fused_terminal_action(
        sample, source_selected_evidence
    )
    if writeback_evidence_mode == "counterfactual_aligned":
        writeback_evidence_id = selected_extra_evidence_id or initial_evidence[0]
        selected_evidence = [writeback_evidence_id]
        prediction = _counterfactual_terminal_action(sample, writeback_evidence_id)
        standalone_belief_prediction, fused_confidence = _fused_terminal_action(
            sample, selected_evidence
        )
    else:
        prediction = source_fused_prediction
        fused_confidence = source_fused_confidence
        standalone_belief_prediction = None
    source_episode = str(sample.metadata["source_episode"])
    return {
        "schema_version": (
            "sn7-v5-matched-rollout-v2"
            if writeback_evidence_mode == "counterfactual_aligned"
            else "sn7-v5-matched-rollout-v1"
        ),
        "task_id": public_task_id(source_episode),
        "source_episode": source_episode,
        "aoi_id": str(sample.metadata["aoi_id"]),
        "split": sample.split,
        "budget": float(sample.metadata["budget"]),
        "policy": policy,
        "target": _target_action(sample),
        "prediction": prediction,
        "selected_evidence_ids": selected_evidence,
        "source_selected_evidence_ids": source_selected_evidence,
        "writeback_evidence_mode": writeback_evidence_mode,
        "source_fused_prediction": source_fused_prediction,
        "source_fused_confidence": source_fused_confidence,
        "standalone_belief_prediction": standalone_belief_prediction,
        "counterfactual_action_override": (
            standalone_belief_prediction is not None
            and prediction != standalone_belief_prediction
        ),
        "initial_evidence_id": initial_evidence[0],
        "selected_extra_evidence_id": selected_extra_evidence_id,
        "selected_extra_evidence": selected_extra_evidence_id is not None,
        "shared_initial_evidence_cost_excluded": True,
        "spent_cost": extra_evidence_cost,
        # This legacy field is retained because the shared writeback evaluator
        # uses it as an intervention-rate column.  It means candidate evidence,
        # not a semantic segmentation tool, in this V5 protocol.
        "semantic_tool_called": selected_extra_evidence_id is not None,
        "fused_confidence_from_selector_state": fused_confidence,
        "selector_scores": selector_scores,
        "selector_stop_score": (selector_scores[-1] if selector_scores is not None else None),
        "test_assets_read": False,
    }


def _authorized_record(
    authorization_path: Path,
    *,
    updater_seed: int,
    updater_checkpoint: Path,
) -> dict[str, Any]:
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unexpected V5 authorization schema")
    if payload.get("authorization") != "matched_nonkeep_factorial_writeback":
        raise PermissionError("V5 candidate-headroom authorization has not passed")
    if payload.get("test_assets_read") is not False:
        raise PermissionError("V5 authorization is not validation-only")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("V5 authorization has no seed records")
    record = next((item for item in records if int(item.get("seed", -1)) == updater_seed), None)
    if not isinstance(record, dict) or record.get("passed") is not True:
        raise PermissionError(f"V5 updater seed {updater_seed} is not authorized")
    if record.get("test_assets_read") is not False:
        raise PermissionError("authorized seed is not validation-only")
    actual_checkpoint_hash = sha256(updater_checkpoint)
    if record.get("checkpoint_sha256") != actual_checkpoint_hash:
        raise ValueError("updater checkpoint hash does not match authorization")
    return record


def validate_validation_state(
    state_path: Path,
    record: dict[str, Any],
) -> None:
    expected_hash = record.get("state_file_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("authorization record lacks a validation state SHA-256")
    actual_hash = sha256(state_path)
    if actual_hash != expected_hash:
        raise ValueError("validation selector-state hash does not match authorization")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--updater-checkpoint", required=True, type=Path)
    parser.add_argument("--updater-seed", required=True, type=int)
    scorer_group = parser.add_mutually_exclusive_group(required=True)
    scorer_group.add_argument("--selector-checkpoint", type=Path)
    scorer_group.add_argument("--evidence-value-checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--writeback-evidence-mode",
        choices=("all", "counterfactual_aligned"),
        default="all",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rollout directory: {args.output_dir}")
    record = _authorized_record(
        args.authorization,
        updater_seed=args.updater_seed,
        updater_checkpoint=args.updater_checkpoint,
    )
    if args.split == "val":
        validate_validation_state(args.states, record)
    states = read_initial_states(args.states, split=args.split)

    if args.selector_checkpoint is not None:
        from activemap.inference import SelectorPredictor

        scorer: ActionScorer = SelectorPredictor(
            args.selector_checkpoint, device=args.device
        )
        scorer_kind = "selector"
        scorer_checkpoint = args.selector_checkpoint
    else:
        from activemap.agent.evidence_value_head import EvidenceValuePredictor

        scorer = EvidenceValueActionScorer(
            EvidenceValuePredictor(str(args.evidence_value_checkpoint), device=args.device)
        )
        scorer_kind = "evidence_value"
        scorer_checkpoint = args.evidence_value_checkpoint
    if scorer_checkpoint is None:
        raise RuntimeError("matched rollout scorer checkpoint was not provided")
    direct_rows = [
        build_policy_rollout(
            state,
            policy="direct",
            scorer=None,
            writeback_evidence_mode=args.writeback_evidence_mode,
        )
        for state in states
    ]
    selected_rows = [
        build_policy_rollout(
            state,
            policy="selected",
            scorer=scorer,
            writeback_evidence_mode=args.writeback_evidence_mode,
        )
        for state in states
    ]
    forced_rows = [
        build_policy_rollout(
            state,
            policy="forced",
            scorer=scorer,
            writeback_evidence_mode=args.writeback_evidence_mode,
        )
        for state in states
    ]
    direct_keys = {(row["task_id"], row["budget"]) for row in direct_rows}
    selected_keys = {(row["task_id"], row["budget"]) for row in selected_rows}
    forced_keys = {(row["task_id"], row["budget"]) for row in forced_rows}
    if direct_keys != selected_keys or direct_keys != forced_keys:
        raise RuntimeError("V5 rollout policies lost paired support")

    args.output_dir.mkdir(parents=True)
    direct_path = args.output_dir / "direct_rollouts.jsonl"
    selected_path = args.output_dir / "selected_rollouts.jsonl"
    forced_path = args.output_dir / "forced_rollouts.jsonl"
    write_jsonl(direct_path, direct_rows)
    write_jsonl(selected_path, selected_rows)
    write_jsonl(forced_path, forced_rows)
    receipt = {
        "schema_version": (
            "sn7-v5-matched-rollout-receipt-v2"
            if args.writeback_evidence_mode == "counterfactual_aligned"
            else "sn7-v5-matched-rollout-receipt-v1"
        ),
        "split": args.split,
        "writeback_evidence_mode": args.writeback_evidence_mode,
        "updater_seed": args.updater_seed,
        "authorization": str(args.authorization.resolve()),
        "authorization_sha256": sha256(args.authorization),
        "updater_checkpoint": str(args.updater_checkpoint.resolve()),
        "updater_checkpoint_sha256": sha256(args.updater_checkpoint),
        "scorer_kind": scorer_kind,
        "scorer_checkpoint": str(scorer_checkpoint.resolve()),
        "scorer_checkpoint_sha256": sha256(scorer_checkpoint),
        "selector_checkpoint": (
            str(args.selector_checkpoint.resolve())
            if args.selector_checkpoint is not None
            else None
        ),
        "selector_checkpoint_sha256": (
            sha256(args.selector_checkpoint)
            if args.selector_checkpoint is not None
            else None
        ),
        "evidence_value_checkpoint": (
            str(args.evidence_value_checkpoint.resolve())
            if args.evidence_value_checkpoint is not None
            else None
        ),
        "evidence_value_checkpoint_sha256": (
            sha256(args.evidence_value_checkpoint)
            if args.evidence_value_checkpoint is not None
            else None
        ),
        "states": str(args.states.resolve()),
        "states_sha256": sha256(args.states),
        "state_count": len(states),
        "direct_rollouts": str(direct_path.resolve()),
        "direct_rollouts_sha256": sha256(direct_path),
        "selected_rollouts": str(selected_path.resolve()),
        "selected_rollouts_sha256": sha256(selected_path),
        "forced_rollouts": str(forced_path.resolve()),
        "forced_rollouts_sha256": sha256(forced_path),
        "selected_extra_evidence_rate": float(
            np.mean([row["selected_extra_evidence"] for row in selected_rows])
        ),
        "forced_extra_evidence_rate": float(
            np.mean([row["selected_extra_evidence"] for row in forced_rows])
        ),
        "test_assets_read": False,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

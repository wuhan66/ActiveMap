#!/usr/bin/env python3
"""Align post-acquisition tool supervision with reachable selector states."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import (
    public_evidence_id,
    public_task_id,
    resolve_evidence_id,
)
from activemap.agent.records import AgentAction, AgentCandidate, AgentObservation
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_sft import TOOL_SYSTEM_PROMPT, policy_action_json, terminal_action
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult


PROTOCOL = "post-acquisition-reachable-tool-controller-v3-pointer-actions"


def _read_pairs(path: Path, split: str) -> list[PostAcquisitionToolPairExample]:
    rows: list[PostAcquisitionToolPairExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = PostAcquisitionToolPairExample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid pair at {path}:{line_number}") from exc
            if row.split == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"no {split} pair rows in {path}")
    return rows


def _selector_task_id(sample: Any) -> str:
    metadata = sample.metadata
    direct = metadata.get("task_id")
    if direct not in (None, ""):
        return str(direct)
    source = metadata.get("source_episode")
    return public_task_id(str(source if source not in (None, "") else sample.sample_id))


def _selector_evidence_catalog(sample: Any) -> list[str]:
    """Return candidate and already-selected raw IDs for one selector state."""

    values = [str(value) for value in sample.evidence_ids]
    metadata = sample.metadata
    initial = metadata.get("initial_evidence_id")
    if initial not in (None, "") and not str(initial).startswith("evidence-"):
        values.append(str(initial))
    selected = metadata.get("selected_evidence_ids", [])
    if isinstance(selected, list):
        values.extend(
            str(value)
            for value in selected
            if value not in (None, "") and not str(value).startswith("evidence-")
        )
    return list(dict.fromkeys(values))


def _public_evidence_handle(value: Any, catalog: list[str]) -> str:
    text = str(value)
    if text in catalog:
        return public_evidence_id(text)
    if text.startswith("evidence-"):
        return text
    return public_evidence_id(text)


def _public_selected_ids(sample: Any) -> set[str]:
    selected = sample.metadata.get("selected_evidence_ids", [])
    if not isinstance(selected, list):
        return set()
    catalog = _selector_evidence_catalog(sample)
    return {_public_evidence_handle(value, catalog) for value in selected}


def _public_catalog_evidence(sample: Any, exposed_id: Any) -> str:
    """Resolve a raw/public evidence handle against one selector catalog."""

    raw_id = resolve_evidence_id(str(exposed_id), _selector_evidence_catalog(sample))
    return public_evidence_id(raw_id)


def _index_selector_states(
    selector_states: Path, split: str
) -> dict[str, list[Any]]:
    from activemap.training.data import load_selector_samples

    indexed: dict[str, list[Any]] = defaultdict(list)
    for sample in load_selector_samples(selector_states, split=split):
        if int(sample.metadata.get("oracle_step", -1)) != 1:
            continue
        indexed[_selector_task_id(sample)].append(sample)
    for values in indexed.values():
        values.sort(key=lambda sample: str(sample.sample_id))
    if not indexed:
        raise ValueError(f"no oracle_step=1 selector states for split={split}")
    return dict(indexed)


def _match_pair(
    pair: PostAcquisitionToolPairExample,
    states_by_task: dict[str, list[Any]],
) -> tuple[Any, str] | None:
    candidates = states_by_task.get(pair.task_id, [])
    if not candidates:
        candidates = states_by_task.get(public_task_id(pair.task_id), [])
    for sample in candidates:
        selected = _public_selected_ids(sample)
        try:
            initial = _public_catalog_evidence(
                sample, pair.metadata.get("initial_evidence_id", "")
            )
            evidence_id = _public_catalog_evidence(sample, pair.evidence_id)
        except (TypeError, ValueError):
            continue
        if initial in selected and evidence_id in selected and len(selected) >= 2:
            return sample, evidence_id
    return None


def _result_with_evidence(result: GeoToolResult, evidence_id: str) -> GeoToolResult:
    return result.model_copy(
        update={"outputs": {**result.outputs, "evidence_id": evidence_id}}
    )


def _observation(
    pair: PostAcquisitionToolPairExample,
    sample: Any,
    *,
    evidence_id: str,
    step: int,
    spent_cost: float,
    belief: Any,
    history: list[GeoToolResult],
) -> AgentObservation:
    catalog = _selector_evidence_catalog(sample)
    raw_to_public = {raw: public_evidence_id(raw) for raw in sample.evidence_ids}
    selected = [
        _public_evidence_handle(value, catalog)
        for value in sample.metadata.get("selected_evidence_ids", [])
    ]
    candidate_ids = set(selected)
    candidates = []
    for index, raw_id in enumerate(sample.evidence_ids):
        public_id = raw_to_public[raw_id]
        if public_id in candidate_ids:
            continue
        candidates.append(
            AgentCandidate(
                evidence_id=public_id,
                cost=float(sample.evidence_costs[index]),
                selector_score=float(sample.oracle_utilities[index]),
                features=sample.evidence_features[index],
            )
        )
    metadata_budget = sample.metadata.get("budget")
    budget = float(metadata_budget) if metadata_budget is not None else 1.5
    minimum_budget = float(pair.tool_cost + spent_cost + 1e-3)
    budget = max(budget, minimum_budget)
    return AgentObservation(
        task_id=pair.task_id,
        split=pair.split,
        step=step,
        initial_budget=budget,
        remaining_budget=max(budget - spent_cost, 0.0),
        spent_cost=spent_cost,
        selected_evidence_ids=selected,
        belief=belief,
        candidates=candidates,
        terminal_score=float(sample.stop_utility),
        available_tools=[GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE],
        tool_history=history,
    )


def _action(tool: GeoToolName, evidence_id: str, call_id: str) -> AgentAction:
    return AgentAction(
        action="USE_TOOL",
        tool_call=GeoToolCall(
            call_id=call_id,
            tool=tool,
            inputs={"evidence_id": evidence_id},
        ),
    )


def _record(
    pair: PostAcquisitionToolPairExample,
    observation: AgentObservation,
    action: AgentAction,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": observation.model_dump_json(exclude_none=True),
            },
            {"role": "assistant", "content": policy_action_json(action, observation)},
        ],
        "trajectory_id": f"reachable-{pair.example_id}",
        "step": observation.step,
        "oracle_tool_stage": 1,
        "utility_gain": 0.0,
        "protocol": PROTOCOL,
        "action_encoding": "selected_evidence_index_v1",
        "source_example_id": pair.example_id,
    }


def build_dataset(
    selector_states: Path,
    pairs: Path,
    output: Path,
    *,
    split: str,
    max_pairs_per_task: int,
    audit_only: bool = False,
) -> dict[str, Any]:
    if max_pairs_per_task <= 0:
        raise ValueError("max_pairs_per_task must be positive")
    states_by_task = _index_selector_states(selector_states, split)
    pair_rows = _read_pairs(pairs, split)
    pair_task_ids = {row.task_id for row in pair_rows}
    selector_task_ids = set(states_by_task)
    selected_by_task: dict[str, int] = Counter()
    records: list[dict[str, Any]] = []
    matched_pair_ids: set[str] = set()
    unmatched_pair_ids: list[str] = []
    overlap_examples = []
    for task_id in sorted(pair_task_ids & selector_task_ids)[:3]:
        sample = states_by_task[task_id][0]
        pair = next(row for row in pair_rows if row.task_id == task_id)
        try:
            pair_public_evidence = _public_catalog_evidence(sample, pair.evidence_id)
        except (TypeError, ValueError):
            pair_public_evidence = None
        try:
            pair_public_initial = _public_catalog_evidence(
                sample, pair.metadata.get("initial_evidence_id", "")
            )
        except (TypeError, ValueError):
            pair_public_initial = None
        overlap_examples.append(
            {
                "task_id": task_id,
                "selector_sample_id": str(sample.sample_id),
                "selector_oracle_step": sample.metadata.get("oracle_step"),
                "selector_selected_evidence_ids": sample.metadata.get(
                    "selected_evidence_ids"
                ),
                "selector_selected_public_evidence_ids": sorted(
                    _public_selected_ids(sample)
                ),
                "pair_initial_evidence_id": pair.metadata.get("initial_evidence_id"),
                "pair_public_initial_evidence_id": pair_public_initial,
                "pair_evidence_id": pair.evidence_id,
                "pair_public_evidence_id": pair_public_evidence,
            }
        )
    for pair in sorted(pair_rows, key=lambda row: row.example_id):
        if selected_by_task[pair.task_id] >= max_pairs_per_task:
            continue
        matched = _match_pair(pair, states_by_task)
        if matched is None:
            unmatched_pair_ids.append(pair.example_id)
            continue
        sample, evidence_id = matched
        selected_by_task[pair.task_id] += 1
        matched_pair_ids.add(pair.example_id)
        quality = _result_with_evidence(pair.quality_result, evidence_id)
        temporal = _result_with_evidence(pair.temporal_result, evidence_id)
        pre = _observation(
            pair,
            sample,
            evidence_id=evidence_id,
            step=0,
            spent_cost=0.0,
            belief=pair.post_acquisition_belief,
            history=[],
        )
        records.append(
            _record(
                pair,
                pre,
                _action(
                    GeoToolName.IMAGE_QUALITY,
                    evidence_id,
                    f"reachable-{pair.example_id[:12]}-quality",
                ),
            )
        )
        after_quality = _observation(
            pair,
            sample,
            evidence_id=evidence_id,
            step=1,
            spent_cost=float(quality.cost),
            belief=pair.post_acquisition_belief,
            history=[quality],
        )
        records.append(
            _record(
                pair,
                after_quality,
                _action(
                    GeoToolName.TEMPORAL_CHANGE,
                    evidence_id,
                    f"reachable-{pair.example_id[:12]}-temporal",
                ),
            )
        )
        after_tools = _observation(
            pair,
            sample,
            evidence_id=evidence_id,
            step=2,
            spent_cost=float(quality.cost + temporal.cost),
            belief=pair.target_belief,
            history=[quality, temporal],
        )
        records.append(_record(pair, after_tools, terminal_action(pair.gt_edit)))

    if not records and not audit_only:
        raise ValueError("no selector-reachable post-acquisition pairs")
    summary = {
        "schema_version": "activemap-reachable-post-acquisition-sft-v2",
        "split": split,
        "selector_states": str(selector_states.resolve()),
        "pair_source": str(pairs.resolve()),
        "input_pair_count": len(pair_rows),
        "pair_task_count": len(pair_task_ids),
        "selector_task_count": len(selector_task_ids),
        "selector_state_count": sum(len(values) for values in states_by_task.values()),
        "pair_selector_task_overlap_count": len(pair_task_ids & selector_task_ids),
        "pair_selector_overlap_examples": overlap_examples,
        "matched_pair_count": len(matched_pair_ids),
        "unmatched_pair_count": len(unmatched_pair_ids),
        "unmatched_pair_ids": unmatched_pair_ids[:50],
        "matched_task_count": len(selected_by_task),
        "max_pairs_per_task": max_pairs_per_task,
        "sft_record_count": len(records),
        "action_counts": dict(Counter(
            json.loads(row["messages"][2]["content"])["action"] for row in records
        )),
        "action_encoding": "selected_evidence_index_v1",
        "protocol": PROTOCOL,
        "reachable_by_selector_state_match": True,
        "test_assets_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if not audit_only:
        with output.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--max-pairs-per-task", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.audit_only:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    print(
        json.dumps(
            build_dataset(
                args.selector_states,
                args.pairs,
                args.output,
                split=args.split,
                max_pairs_per_task=args.max_pairs_per_task,
                audit_only=args.audit_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

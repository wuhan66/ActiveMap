from pathlib import Path

import pytest

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.audit_active_catalog_grounded_tool_data import audit


def _belief():
    return AgentBelief(
        edit_probabilities=[0.2, 0.7, 0.05, 0.05],
        confidence=0.8,
        uncertainty=0.3,
    )


def _write(path: Path, *, split: str, task: str, transition: str) -> None:
    evidence = f"evidence-{task}"
    quality = GeoToolResult(
        call_id=f"q-{task}",
        tool=GeoToolName.IMAGE_QUALITY,
        success=True,
        outputs={"evidence_id": evidence, "valid_fraction": 0.8, "sharpness": 0.4},
        cost=0.03,
    )
    temporal = GeoToolResult(
        call_id=f"t-{task}",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=True,
        outputs={"evidence_id": evidence, "changed_fraction": 0.2},
        cost=0.15,
    )
    row = PostAcquisitionToolPairExample(
        example_id=f"example-{task}",
        task_id=task,
        split=split,
        evidence_id=evidence,
        post_acquisition_belief=_belief(),
        quality_result=quality,
        temporal_result=temporal,
        target_belief=_belief(),
        gt_edit=EditOperation.ADD,
        evidence_cost=1.0,
        tool_cost=0.18,
        metadata={
            "source_transition_id": transition,
            "selected_by_model": True,
            "oracle_next_state_replay": False,
            "oracle_action_exported": False,
            "operation_update_threshold": 0.69,
            "test_assets_read": False,
        },
    )
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")


def test_grounded_tool_audit_passes_disjoint_data(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write(train, split="train", task="train-task", transition="train-transition")
    _write(val, split="val", task="val-task", transition="val-transition")
    result = audit(train, val)
    assert result["passed"] is True
    assert result["explicit_geospatial_tool_calls"] is True


def test_grounded_tool_audit_rejects_task_leakage(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write(train, split="train", task="same-task", transition="train-transition")
    _write(val, split="val", task="same-task", transition="val-transition")
    with pytest.raises(ValueError, match="leaks identities"):
        audit(train, val)
